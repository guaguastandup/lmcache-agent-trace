#!/usr/bin/env python3
import json
from pathlib import Path

# Find all JSONL files in miniswe directory
miniswe_dir = Path("/Users/xiaokun/Documents/all_trajectories/demo_data_final")
jsonl_files = sorted(miniswe_dir.glob("*.jsonl"))

print(f"Found {len(jsonl_files)} JSONL files")
print("=" * 80)

# Read all files into memory and collect statistics
all_lines = []
file_stats = []

for jsonl_file in jsonl_files:
    with open(jsonl_file, 'r') as f:
        lines = f.readlines()
        all_lines.append(lines)

        # Get last line character count
        last_line_chars = len(lines[-1]) if lines else 0

        file_stats.append({
            'name': jsonl_file.name,
            'line_count': len(lines),
            'last_line_chars': last_line_chars
        })

        print(f"  {jsonl_file.name}:")
        print(f"    Lines: {len(lines)}")
        print(f"    Last line characters: {last_line_chars}")

# Overall statistics
line_counts = [stat['line_count'] for stat in file_stats]
last_line_chars = [stat['last_line_chars'] for stat in file_stats]

print("\n" + "=" * 80)
print("OVERALL STATISTICS:")
print(f"  Total files: {len(jsonl_files)}")
print(f"  Max lines in a file: {max(line_counts)}")
print(f"  Min lines in a file: {min(line_counts)}")
print(f"  Average lines per file: {sum(line_counts) / len(line_counts):.1f}")
print(f"\n  Max characters in last line: {max(last_line_chars)}")
print(f"  Min characters in last line: {min(last_line_chars)}")
print(f"  Average characters in last line: {sum(last_line_chars) / len(last_line_chars):.1f}")

# Find files with max/min values
max_lines_file = max(file_stats, key=lambda x: x['line_count'])
min_lines_file = min(file_stats, key=lambda x: x['line_count'])
max_chars_file = max(file_stats, key=lambda x: x['last_line_chars'])
min_chars_file = min(file_stats, key=lambda x: x['last_line_chars'])

print(f"\n  File with most lines: {max_lines_file['name']} ({max_lines_file['line_count']} lines)")
print(f"  File with fewest lines: {min_lines_file['name']} ({min_lines_file['line_count']} lines)")
print(f"  File with longest last line: {max_chars_file['name']} ({max_chars_file['last_line_chars']} chars)")
print(f"  File with shortest last line: {min_chars_file['name']} ({min_chars_file['last_line_chars']} chars)")

# Find the maximum number of lines across all files
max_lines = max(len(lines) for lines in all_lines)
print("\n" + "=" * 80)

# Combine in round-robin fashion
combined = []
for i in range(max_lines):
    for lines in all_lines:
        if i < len(lines):
            combined.append(lines[i])

# Write combined output
output_file = "demo_data_combined.jsonl"
with open(output_file, 'w') as f:
    f.writelines(combined)

print(f"\nWrote {len(combined)} lines to {output_file}")
