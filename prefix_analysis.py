#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

# Standard
from collections import OrderedDict
from typing import List, Optional, Tuple, Union
import argparse
import json
import re

# Third Party
from tqdm import tqdm
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import torch

# Constants
DEFAULT_TOKENIZER = "meta-llama/Llama-3.1-8B"
DEFAULT_TOKENS_PER_GB = 8200  # Default for Llama-3.1; More details here: https://docs.lmcache.ai/getting_started/kv_cache_calculator.html
DEFAULT_POOL_SIZES_GB: List[Union[int, float, str]] = [
    "unlimited",
]


class LRUTokenPool:
    """
    Token pool with LRU eviction policy based on token count limit.
    """

    def __init__(self, max_tokens: float) -> None:
        self.max_tokens = max_tokens
        self.current_tokens = 0
        self.requests: OrderedDict[int, List[int]] = OrderedDict()

    def longest_prefix_len(self, tokens: List[int]) -> Tuple[int, int]:
        """
        Find longest prefix match and update LRU ordering.
        For request i (1-indexed):
        y[i] = y[i-1] + (len(tokens[i]) - max_shared_prefix(tokens[i], any previous))
        """
        best_len = 0
        best_id = -1

        for req_id, req_tokens in self.requests.items():
            common_len = 0
            for i in range(min(len(tokens), len(req_tokens))):
                if tokens[i] == req_tokens[i]:
                    common_len += 1
                else:
                    break

            if common_len > best_len:
                best_len = common_len
                best_id = req_id

        # Update LRU ordering
        if best_id != -1:
            self.requests.move_to_end(best_id)

        return best_len, best_id

    def longest_common_substring(
        self,
        request_id: int,
        token_tensor: torch.Tensor,
        tokens: List[int],
        *,
        input_len: Optional[int] = None,
        chunk_len: int = 16,
        stride_r: int = 16,
        chunk_batch: int = 512,
        return_details: bool = False,
    ) -> Tuple[int, float, Optional[List[dict]]]:
        """
        For token_tensor[request_id], chunk it and check whether each chunk
        appears contiguously in any previous request (token_tensor[:request_id]).
        Returns (total_tokens_matched, elapsed_seconds, match_details).
        
        If input_len is provided, only match against the first input_len tokens of the current request.
        Previous requests are matched against their full content (input + output).
        
        If return_details is True, match_details is a list of dicts with:
        - MatchStart: start position in current request (input portion only)
        - MatchEnd: end position in current request (input portion only)
        - PrevStep: matched request ID
        - PrevMatchStart: start position in matched request (can be in input or output)
        - PrevMatchEnd: end position in matched request (can be in input or output)
        """
        assert token_tensor.ndim == 2, "Expected [N, T] tensor"
        N, T = token_tensor.shape
        assert 0 <= request_id < N, "request_id out of range"

        if request_id == 0 or T < chunk_len:
            return 0, 0, [] if return_details else None

        r = token_tensor[request_id]  # [T]
        
        # Only process the input portion if input_len is specified
        if input_len is not None:
            r = r[:input_len]
            match_len = input_len
        else:
            r = r[: len(tokens)]
            match_len = len(tokens)

        # hotfix: Check if truncated length is sufficient for chunking
        if match_len < chunk_len:
            return 0, 0, [] if return_details else None

        # Only consider requests that are still in the pool (not evicted)
        valid_request_ids = [rid for rid in self.requests.keys() if rid < request_id]
        if not valid_request_ids:
            return 0, 0, [] if return_details else None

        Xprev = token_tensor[valid_request_ids]  # [num_valid, T]

        # Sliding windows for previous rows
        Xw = Xprev.unfold(dimension=1, size=chunk_len, step=1)  # [R, W, L]
        # Chunks of r
        r_chunks = r.unfold(dimension=0, size=chunk_len, step=stride_r)  # [C, L]
        if r_chunks.numel() == 0:
            return 0, 0, [] if return_details else None

        total_matched_chunks = 0
        # Track matches per request to find the best matching request
        request_match_counts: dict[int, int] = {}
        
        # Track detailed match information
        match_details = [] if return_details else None

        # Process in mini-batches to control memory
        for b in range(0, r_chunks.size(0), chunk_batch):
            rc = r_chunks[b : b + chunk_batch]  # [B, L]
            eq = Xw[:, :, None, :] == rc[None, None, :, :]
            full = eq.all(dim=-1)  # [R, W, B]
            # Count how many unique chunks matched (across all previous rows)
            matched_chunk_indices = torch.unique(full.nonzero(as_tuple=True)[2])
            total_matched_chunks += matched_chunk_indices.numel()

            # Track which requests contributed to matches
            if matched_chunk_indices.numel() > 0:
                nonzero_indices = full.nonzero(as_tuple=True)
                tensor_indices = nonzero_indices[0]  # [R] dimension (index into Xprev)
                window_indices = nonzero_indices[1]  # [W] dimension (position in prev request)
                chunk_indices = nonzero_indices[2]  # [B] dimension (chunk index in current request)

                # Count matches per request for matched chunks only
                # Map tensor index back to actual request ID
                for tensor_idx, window_idx, chunk_idx in zip(
                    tensor_indices.tolist(), window_indices.tolist(), chunk_indices.tolist(), strict=False
                ):
                    if chunk_idx in matched_chunk_indices.tolist():
                        actual_request_id = valid_request_ids[tensor_idx]
                        request_match_counts[actual_request_id] = (
                            request_match_counts.get(actual_request_id, 0) + 1
                        )
                        
                        # Store detailed match information
                        if return_details:
                            # Position in current request
                            match_start = b + chunk_idx * stride_r
                            match_end = match_start + chunk_len
                            # Position in previous request
                            prev_match_start = window_idx
                            prev_match_end = window_idx + chunk_len
                            
                            match_details.append({
                                "MatchStart": match_start,
                                "MatchEnd": match_end,
                                "PrevStep": actual_request_id,
                                "PrevMatchStart": prev_match_start,
                                "PrevMatchEnd": prev_match_end,
                            })

        total_tokens_matched = total_matched_chunks * chunk_len

        # Update LRU ordering for the best matching request
        if request_match_counts:
            best_id = max(request_match_counts, key=lambda x: request_match_counts[x])
            # The best_id is now the actual request ID from the original tensor
            if best_id in self.requests:
                self.requests.move_to_end(best_id)

        return total_tokens_matched, 0, match_details

    def add_request(
        self,
        request_id: int,
        tokens: List[int],
        token_tensor: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Add a request to the pool, evicting LRU entries if necessary.
        """
        # Evict until we have space
        while (
            self.current_tokens > 0
            and self.current_tokens + len(tokens) > self.max_tokens
        ):
            old_id, old_tokens = self.requests.popitem(last=False)
            self.current_tokens -= len(old_tokens)

            # substring matching case
            if token_tensor is not None:
                token_tensor[old_id, :] = 0

        # Add new request
        if self.current_tokens + len(tokens) <= self.max_tokens:
            self.requests[request_id] = tokens
            self.current_tokens += len(tokens)


def load_and_tokenize_inputs(
    jsonl_path: str, tokenizer_name: str = DEFAULT_TOKENIZER
) -> Tuple[List[List[int]], List[List[int]], List[int], List[int], torch.Tensor]:
    """
    Load and tokenize inputs and outputs from a JSONL file.

    Returns:
        Tuple of (input_sequences, output_sequences, input_lengths, output_lengths, combined_tensor)
        - input_sequences: List of input token lists
        - output_sequences: List of output token lists
        - input_lengths: List of input lengths
        - output_lengths: List of output lengths
        - combined_tensor: Padded 2D tensor (sequences, tokens) with input+output concatenated
          Sequences are padded with 0s to match the longest sequence.
    """
    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    print(f"Reading and tokenizing inputs and outputs from: {jsonl_path}")
    input_sequences = []
    output_sequences = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in tqdm(lines, desc="Tokenizing"):
        try:
            # Try standard JSON parsing first
            data = json.loads(line.strip())
            input_text = data.get("input", "")
            output_text = data.get("output", "")
            input_tokens = tokenizer.encode(input_text)
            output_tokens = tokenizer.encode(output_text) if output_text else []
            input_sequences.append(input_tokens)
            output_sequences.append(output_tokens)
        except json.JSONDecodeError:
            # Handle malformed JSON with nested unescaped quotes
            try:
                line_str = line.strip()
                
                # Find the start of the input value
                input_match = re.search(r'"input"\s*:\s*"', line_str)
                if not input_match:
                    print(f"Warning: Failed to process line: Could not find input field")
                    input_sequences.append([])
                    output_sequences.append([])
                    continue
                
                input_start = input_match.end()
                
                # Find the end - look for ", "output" or ", "session_id" patterns
                output_pattern = r'",\s*"output"\s*:'
                session_pattern = r'",\s*"session_id"\s*:'
                
                output_match = re.search(output_pattern, line_str[input_start:])
                session_match = re.search(session_pattern, line_str[input_start:])
                
                if output_match:
                    input_end = input_start + output_match.start()
                elif session_match:
                    input_end = input_start + session_match.start()
                else:
                    # Last resort: find the last quote before the closing brace
                    last_brace = line_str.rfind('}')
                    if last_brace > input_start:
                        input_end = line_str.rfind('"', input_start, last_brace)
                    else:
                        print(f"Warning: Failed to process line: Could not determine input end")
                        input_sequences.append([])
                        output_sequences.append([])
                        continue
                
                if input_end <= input_start:
                    print(f"Warning: Failed to process line: Invalid input boundaries")
                    input_sequences.append([])
                    output_sequences.append([])
                    continue
                
                # Extract the input text
                input_text = line_str[input_start:input_end]
                input_tokens = tokenizer.encode(input_text)
                
                # Try to extract output if it exists
                output_tokens = []
                if output_match:
                    output_start = input_start + output_match.end()
                    # Find output end (before session_id or closing brace)
                    session_in_tail = re.search(session_pattern, line_str[output_start:])
                    if session_in_tail:
                        output_end = output_start + session_in_tail.start()
                    else:
                        last_brace = line_str.rfind('}')
                        if last_brace > output_start:
                            output_end = line_str.rfind('"', output_start, last_brace)
                        else:
                            output_end = output_start
                    
                    if output_end > output_start:
                        output_text = line_str[output_start:output_end]
                        output_tokens = tokenizer.encode(output_text)
                
                input_sequences.append(input_tokens)
                output_sequences.append(output_tokens)
                    
            except Exception as e2:
                print(f"Warning: Failed to process line: {e2}")
                input_sequences.append([])
                output_sequences.append([])

    # Calculate lengths
    input_lengths = [len(seq) for seq in input_sequences]
    output_lengths = [len(seq) for seq in output_sequences]
    
    # Create combined sequences (input + output)
    combined_sequences = []
    for inp, out in zip(input_sequences, output_sequences):
        combined_sequences.append(inp + out)
    
    if combined_sequences:
        max_length = max(len(seq) for seq in combined_sequences)
        num_sequences = len(combined_sequences)

        # Create padded tensor (pad with 0s)
        combined_tensor = torch.zeros((num_sequences, max_length), dtype=torch.long)
        for i, seq in enumerate(combined_sequences):
            if seq:
                combined_tensor[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
    else:
        combined_tensor = torch.tensor([], dtype=torch.long)

    return input_sequences, output_sequences, input_lengths, output_lengths, combined_tensor


def calculate_hit_rate(
    token_sequences: List[List[int]],
    input_lengths: List[int],
    output_lengths: List[int],
    pool_size: Optional[int] = None,
    token_tensor: Optional[torch.Tensor] = None,
    method: str = "prefix",
    collect_details: bool = False,
) -> Tuple[float, Optional[List[dict]]]:
    # Use float('inf') for unlimited case to avoid eviction
    max_tokens = float("inf") if pool_size is None else pool_size
    pool = LRUTokenPool(max_tokens)

    total_tokens = 0
    hit_tokens = 0

    total_lcs_time_s = 0.0
    lcs_calls = 0
    
    # Store detailed match information for each step
    step_details = [] if collect_details else None

    for idx, tokens in tqdm(list(enumerate(token_sequences))):
        # Only count input tokens for hit rate (not output)
        input_len = input_lengths[idx]
        output_len = output_lengths[idx]
        total_tokens += input_len
        
        step_info = None
        if collect_details:
            step_info = {
                "StepID": idx,
                "InputLen": input_len,
                "OutputLen": output_len,
                "Matches": []
            }

        if method == "prefix":
            if idx > 0:
                # For prefix matching, only match the input portion
                input_only = tokens[:input_len]
                common, _ = pool.longest_prefix_len(input_only)
                hit_tokens += common
            pool.add_request(idx, tokens)
        elif method == "substring" and token_tensor is not None:
            if idx > 0:
                # For substring matching, only match the input portion
                common, elapsed, match_details = pool.longest_common_substring(
                    idx, token_tensor, tokens, input_len=input_len, return_details=collect_details
                )
                hit_tokens += common
                total_lcs_time_s += elapsed
                lcs_calls += 1
                
                if collect_details and match_details:
                    step_info["Matches"] = match_details
            pool.add_request(idx, tokens, token_tensor)
        else:
            raise ValueError(f"Invalid method: {method}")
        
        if collect_details:
            step_details.append(step_info)

    if method == "substring":
        avg_ms = (total_lcs_time_s / lcs_calls * 1000.0) if lcs_calls > 0 else 0.0
        print(
            f"  [Timing] longest_common_substring: total {total_lcs_time_s:.3f}s, "
            f"calls {lcs_calls}, avg {avg_ms:.2f} ms"
        )

    hit_rate = hit_tokens / total_tokens if total_tokens > 0 else 0.0
    return hit_rate, step_details


def analyze_hit_rates_across_pool_sizes(
    token_sequences: List[List[int]],
    input_lengths: List[int],
    output_lengths: List[int],
    pool_sizes_gb: List[Union[int, float, str]],
    tokens_per_gb: int,
    token_tensor: Optional[torch.Tensor] = None,
    collect_details: bool = False,
) -> Tuple[List[float], List[float], List[str], Optional[List[dict]]]:
    print("\nAnalyzing hit rates across pool sizes...")
    print("=" * 60)

    prefix_hit_rates = []
    substring_hit_rates = []
    x_labels = []
    substring_details = None

    for size_gb in pool_sizes_gb:
        if size_gb == "unlimited":
            size_tokens = None
            x_labels.append("∞")
            pool_desc = "unlimited"
            token_desc = ""
        else:
            size_tokens = int(size_gb * tokens_per_gb)
            x_labels.append(str(size_gb))
            pool_desc = f"{size_gb}GB"
            token_desc = f" ({size_tokens:,} tokens)"

        print(f"Testing pool size: {pool_desc}{token_desc}")

        # For every pool size round, we should start from fresh
        tensor_copy = token_tensor.clone() if token_tensor is not None else None

        prefix_hit_rate, _ = calculate_hit_rate(
            token_sequences, input_lengths, output_lengths, size_tokens, tensor_copy, method="prefix", collect_details=False
        )
        prefix_hit_rates.append(prefix_hit_rate)
        print(f"  Prefix: {prefix_hit_rate:.4f} ({prefix_hit_rate * 100:.2f}%)")

        substring_hit_rate, details = calculate_hit_rate(
            token_sequences, input_lengths, output_lengths, size_tokens, tensor_copy, method="substring", collect_details=collect_details
        )
        substring_hit_rates.append(substring_hit_rate)
        print(
            f"  Substring: {substring_hit_rate:.4f} ({substring_hit_rate * 100:.2f}%)\n"
        )
        
        # Only store details for the first (unlimited) pool size
        if collect_details and substring_details is None:
            substring_details = details

    print("=" * 60)
    return prefix_hit_rates, substring_hit_rates, x_labels, substring_details


def plot_hit_rates(
    prefix_hit_rates: List[float],
    substring_hit_rates: List[float],
    x_labels: List[str],
    output_path: str,
    num_requests: int,
    avg_token_length: float,
    tokenizer_name: str,
) -> None:
    """
    Generate and save the hit rate vs pool size plot comparing both methods.
    """
    plt.figure(figsize=(12, 7))

    # log scale
    x_values = []
    for label in x_labels:
        if label == "∞":
            x_values.append(1000)  # Use large value for unlimited
        else:
            x_values.append(float(label))

    # Plot prefix
    plt.plot(
        x_values,
        prefix_hit_rates,
        marker="o",
        linewidth=2,
        markersize=8,
        color="#2E86AB",
        label="Prefix Matching",
    )

    # Plot substring
    plt.plot(
        x_values,
        substring_hit_rates,
        marker="s",
        linewidth=2,
        markersize=8,
        color="#A23B72",
        label="Substring Matching",
    )

    plt.xlabel("Pool Size (GB)", fontsize=12, fontweight="bold")
    plt.ylabel("Hit Rate", fontsize=12, fontweight="bold")
    plt.title(
        "Cache Hit Rate vs Pool Size: Prefix vs Substring Matching",
        fontsize=14,
        fontweight="bold",
    )
    plt.xscale('log')
    plt.xticks(x_values, x_labels, rotation=45)
    plt.grid(True, alpha=0.3, linestyle="--")

    # Set y-axis limit based on max of both methods
    max_rate = max(max(prefix_hit_rates), max(substring_hit_rates))
    plt.ylim(0, min(1.0, max_rate * 1.1))
    plt.legend(loc="best", fontsize=10)

    # Annotate prefix matching rates (below the curve)
    for x_val, rate in zip(x_values, prefix_hit_rates, strict=False):
        plt.annotate(
            f"{rate * 100:.1f}%",
            xy=(x_val, rate),
            xytext=(0, -10),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color="#2E86AB",
        )

    # Annotate substring matching rates (above the curve)
    for x_val, rate in zip(x_values, substring_hit_rates, strict=False):
        plt.annotate(
            f"{rate * 100:.1f}%",
            xy=(x_val, rate),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color="#A23B72",
        )

    # Add statistics text box in bottom right
    stats_text = (
        f"Num of llm calls: {num_requests}\nAvg Token Length: {avg_token_length:.1f}\
    \nTokenizer: {tokenizer_name}"
    )
    plt.text(
        0.98,
        0.02,
        stats_text,
        transform=plt.gca().transAxes,
        fontsize=9,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(
            boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray", alpha=0.8
        ),
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to: {output_path}")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze prefix cache hit rates across different pool sizes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -i trace.jsonl
  %(prog)s -i trace.jsonl -o custom_output.png
  %(prog)s -i trace.jsonl --pool-sizes 1 2 4 8 16 unlimited
  %(prog)s -i trace.jsonl --log-matches matching_details.jsonl
        """,
    )

    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Path to input JSONL file (trace.jsonl)",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="prefix_cache_hit_rate.png",
        help="Path to output plot file (PNG) (default: prefix_cache_hit_rate.png)",
    )

    parser.add_argument(
        "--log-matches",
        type=str,
        default=None,
        help="Path to output JSONL file for detailed substring matching information (optional)",
    )

    parser.add_argument(
        "--tokenizer",
        type=str,
        default=DEFAULT_TOKENIZER,
        help=f"HuggingFace tokenizer model name (default: {DEFAULT_TOKENIZER})",
    )

    parser.add_argument(
        "--tokens-per-gb",
        type=int,
        default=DEFAULT_TOKENS_PER_GB,
        help=f"Conversion factor from GB to tokens "
        f"(default: {DEFAULT_TOKENS_PER_GB}). "
        "This should be adjusted when using a different tokenizer.",
    )

    parser.add_argument(
        "--pool-sizes",
        nargs="+",
        default=None,
        help='Pool sizes in GB to test (space-separated, can include "unlimited"). '
        f"Default: {' '.join(map(str, DEFAULT_POOL_SIZES_GB))}",
    )

    return parser.parse_args()


def parse_pool_sizes(
    pool_sizes_input: Optional[List[str]],
) -> List[Union[int, float, str]]:
    if pool_sizes_input is None:
        return DEFAULT_POOL_SIZES_GB

    parsed_sizes: List[Union[int, float, str]] = []
    for size in pool_sizes_input:
        if size.lower() == "unlimited":
            parsed_sizes.append("unlimited")
        else:
            try:
                parsed_sizes.append(float(size))
            except ValueError:
                raise ValueError(
                    f"Invalid pool size: {size}. Must be a number or 'unlimited'"
                ) from None

    return parsed_sizes


def write_matching_details_to_jsonl(details: List[dict], output_path: str) -> None:
    """
    Write detailed substring matching information to a JSONL file.
    Each line contains: StepID, InputLen, OutputLen, and a list of matches.
    """
    print(f"\nWriting matching details to: {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        for step_info in details:
            # Format the line according to user requirements
            line_data = {
                "StepID": step_info["StepID"],
                "InputLen": step_info["InputLen"],
                "OutputLen": step_info["OutputLen"],
                "Matches": step_info["Matches"]
            }
            f.write(json.dumps(line_data) + "\n")
    print(f"Wrote {len(details)} lines to {output_path}")


def main() -> None:
    args = parse_arguments()

    # Parse pool sizes
    pool_sizes_gb = parse_pool_sizes(args.pool_sizes)

    print("Configuration:")
    print(f"  Input: {args.input}")
    print(f"  Output: {args.output}")
    if args.log_matches:
        print(f"  Match logging: {args.log_matches}")
    print(f"  Tokenizer: {args.tokenizer}")
    print(f"  Tokens per GB: {args.tokens_per_gb}")
    print(f"  Pool sizes: {pool_sizes_gb}\n")

    # Load and tokenize inputs and outputs
    input_sequences, output_sequences, input_lengths, output_lengths, combined_tensor = \
        load_and_tokenize_inputs(args.input, args.tokenizer)
    
    # Combined sequences for processing
    combined_sequences = []
    for inp, out in zip(input_sequences, output_sequences):
        combined_sequences.append(inp + out)
    
    print(f"Loaded {len(combined_sequences)} requests")
    print(f"Combined tensor shape: {combined_tensor.shape} (input+output, padded with 0s)")
    print(f"First sequence: {combined_tensor[0]}")

    # Calculate statistics (only count input lengths for average)
    num_requests = len(combined_sequences)
    avg_input_length = sum(input_lengths) / num_requests if num_requests > 0 else 0
    avg_output_length = sum(output_lengths) / num_requests if num_requests > 0 else 0

    # Analyze hit rates using both methods
    # Collect detailed match information if user wants to log it
    collect_details = args.log_matches is not None
    prefix_hit_rates, substring_hit_rates, x_labels, substring_details = (
        analyze_hit_rates_across_pool_sizes(
            combined_sequences,
            input_lengths,
            output_lengths,
            pool_sizes_gb,
            args.tokens_per_gb,
            combined_tensor,
            collect_details=collect_details,
        )
    )

    # Write matching details to JSONL if requested
    if args.log_matches and substring_details:
        write_matching_details_to_jsonl(substring_details, args.log_matches)

    # Generate comparison plot
    plot_hit_rates(
        prefix_hit_rates,
        substring_hit_rates,
        x_labels,
        args.output,
        num_requests,
        avg_input_length,
        args.tokenizer,
    )
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
