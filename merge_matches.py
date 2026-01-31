#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""
Merge overlapping and adjacent substring matches from the prefix_analysis output.

This script reads a JSONL file containing substring matching information and merges
matches that are contiguous or overlapping.
"""

import argparse
import json
from typing import List, Dict
from collections import defaultdict


def merge_matches(matches: List[Dict]) -> List[Dict]:
    """
    Merge overlapping and adjacent matches for a single step.
    
    Strategy:
    1. Remove duplicate current regions - keep only the match to the most recent PrevStep
    2. Group remaining matches by PrevStep
    3. For each group, remove duplicates and contained matches
    4. Merge adjacent/overlapping matches that map to adjacent regions
    
    Args:
        matches: List of match dictionaries with keys:
                 MatchStart, MatchEnd, PrevStep, PrevMatchStart, PrevMatchEnd
    
    Returns:
        List of merged match dictionaries
    """
    if not matches:
        return []
    
    # Step 0: Deduplicate - keep only the most recent PrevStep for each current region
    by_current_region = defaultdict(list)
    for match in matches:
        key = (match["MatchStart"], match["MatchEnd"])
        by_current_region[key].append(match)
    
    deduplicated = []
    for region_key, region_matches in by_current_region.items():
        # Keep only the match with the highest PrevStep (most recent)
        most_recent = max(region_matches, key=lambda m: m["PrevStep"])
        deduplicated.append(most_recent)
    
    # Group matches by PrevStep
    by_prev_step = defaultdict(list)
    for match in deduplicated:
        by_prev_step[match["PrevStep"]].append(match)
    
    merged_matches = []
    
    for prev_step, step_matches in by_prev_step.items():
        # Sort by MatchStart, then by match length (longest first)
        step_matches.sort(key=lambda m: (m["MatchStart"], -(m["MatchEnd"] - m["MatchStart"])))
        
        # Step 1: Remove duplicates and contained matches
        # A match is "contained" if another match covers the same or larger range
        filtered = []
        for i, match in enumerate(step_matches):
            is_redundant = False
            for j, other in enumerate(step_matches):
                if i == j:
                    continue
                
                # Check if 'match' is contained within 'other' in BOTH current and previous
                curr_contained = (match["MatchStart"] >= other["MatchStart"] and 
                                 match["MatchEnd"] <= other["MatchEnd"])
                prev_contained = (match["PrevMatchStart"] >= other["PrevMatchStart"] and 
                                 match["PrevMatchEnd"] <= other["PrevMatchEnd"])
                
                if curr_contained and prev_contained:
                    # This match is redundant
                    is_redundant = True
                    break
            
            if not is_redundant:
                filtered.append(match)
        
        if not filtered:
            continue
        
        # Step 2: Sort filtered matches for merging
        # Sort by MatchStart, then PrevMatchStart
        filtered.sort(key=lambda m: (m["MatchStart"], m["PrevMatchStart"]))
        
        # Step 3: Merge adjacent/overlapping matches
        current_merged = filtered[0].copy()
        
        for i in range(1, len(filtered)):
            match = filtered[i]
            
            # Calculate gaps
            curr_gap = match["MatchStart"] - current_merged["MatchEnd"]
            prev_gap = match["PrevMatchStart"] - current_merged["PrevMatchEnd"]
            
            # Check if we can merge:
            # 1. Overlapping (negative gap) with aligned overlap
            # 2. Adjacent or small gap (0-16 tokens) with similar gaps
            # 3. Both positions should be moving forward consistently
            
            # For overlapping matches
            if curr_gap < 0 and prev_gap < 0:
                # Calculate overlap
                curr_overlap = -curr_gap
                prev_overlap = -prev_gap
                
                # Only merge if overlaps are similar (same content)
                if abs(curr_overlap - prev_overlap) <= 2:
                    current_merged["MatchEnd"] = max(current_merged["MatchEnd"], match["MatchEnd"])
                    current_merged["PrevMatchEnd"] = max(current_merged["PrevMatchEnd"], match["PrevMatchEnd"])
                else:
                    # Different overlap, save and start new
                    merged_matches.append(current_merged)
                    current_merged = match.copy()
            
            # For adjacent/small gap matches
            elif curr_gap >= 0 and prev_gap >= 0 and curr_gap <= 16 and prev_gap <= 16:
                # Check if gaps are similar (aligned continuation)
                if abs(curr_gap - prev_gap) <= 2:
                    # Extend the current match
                    current_merged["MatchEnd"] = match["MatchEnd"]
                    current_merged["PrevMatchEnd"] = match["PrevMatchEnd"]
                else:
                    # Not aligned, save and start new
                    merged_matches.append(current_merged)
                    current_merged = match.copy()
            
            else:
                # Too far apart or misaligned, save and start new
                merged_matches.append(current_merged)
                current_merged = match.copy()
        
        # Don't forget the last merged match
        merged_matches.append(current_merged)
    
    # Sort final merged matches by MatchStart
    merged_matches.sort(key=lambda m: m["MatchStart"])
    
    return merged_matches


def process_jsonl(input_path: str, output_path: str, verbose: bool = False) -> None:
    """
    Process the JSONL file and merge matches for each step.
    
    Args:
        input_path: Path to input JSONL file
        output_path: Path to output JSONL file with merged matches
        verbose: Print detailed progress
    """
    print(f"Reading from: {input_path}")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"Processing {len(lines)} steps...")
    
    total_original = 0
    total_merged = 0
    
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for line_num, line in enumerate(lines, 1):
            data = json.loads(line.strip())
            
            original_match_count = len(data["Matches"])
            total_original += original_match_count
            
            # Merge the matches
            merged = merge_matches(data["Matches"])
            
            merged_match_count = len(merged)
            total_merged += merged_match_count
            
            # Update the data with merged matches
            data["Matches"] = merged
            
            # Write to output
            f_out.write(json.dumps(data) + '\n')
            
            if verbose and (line_num % 5 == 0 or original_match_count != merged_match_count):
                reduction = 0 if original_match_count == 0 else (1 - merged_match_count / original_match_count) * 100
                print(f"  Step {data['StepID']}: {original_match_count} → {merged_match_count} matches ({reduction:.1f}% reduction)")
    
    reduction_pct = (1 - total_merged / total_original) * 100 if total_original > 0 else 0
    print(f"\nMerging complete!")
    print(f"Total original matches: {total_original:,}")
    print(f"Total merged matches: {total_merged:,}")
    print(f"Reduction: {reduction_pct:.1f}%")
    print(f"Output written to: {output_path}")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Merge overlapping and adjacent substring matches",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -i LOG_MATCHES.jsonl -o LOG_MATCHES_MERGED.jsonl
  %(prog)s -i LOG_MATCHES.jsonl -o LOG_MATCHES_MERGED.jsonl --verbose
        """,
    )
    
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Path to input JSONL file from prefix_analysis.py --log-matches",
    )
    
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Path to output JSONL file with merged matches",
    )
    
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print detailed progress information",
    )
    
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    process_jsonl(args.input, args.output, args.verbose)


if __name__ == "__main__":
    main()
