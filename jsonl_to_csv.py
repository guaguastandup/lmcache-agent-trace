#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""
Convert merged matches JSONL to CSV format.

Each row contains: StepID, InputLen, OutputLen, followed by match information.
Each match is represented by 5 numbers: MatchStart, MatchEnd, PrevStep, PrevMatchStart, PrevMatchEnd
"""

import argparse
import json
import csv


def jsonl_to_csv(input_path: str, output_path: str) -> None:
    """
    Convert JSONL matches to CSV format.
    
    Args:
        input_path: Path to input JSONL file
        output_path: Path to output CSV file
    """
    print(f"Reading from: {input_path}")
    
    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8', newline='') as f_out:
        
        csv_writer = csv.writer(f_out)
        
        for line_num, line in enumerate(f_in, 1):
            data = json.loads(line.strip())
            
            # Start row with StepID, InputLen, OutputLen
            row = [data["StepID"], data["InputLen"], data["OutputLen"]]
            
            # Add match information
            for match in data["Matches"]:
                row.extend([
                    match["MatchStart"],
                    match["MatchEnd"],
                    match["PrevStep"],
                    match["PrevMatchStart"],
                    match["PrevMatchEnd"]
                ])
            
            csv_writer.writerow(row)
    
    print(f"Converted {line_num} lines")
    print(f"Output written to: {output_path}")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert merged matches JSONL to CSV format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -i MERGED_MATCHES.jsonl -o matches.csv
        """,
    )
    
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Path to input JSONL file",
    )
    
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Path to output CSV file",
    )
    
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    jsonl_to_csv(args.input, args.output)


if __name__ == "__main__":
    main()
