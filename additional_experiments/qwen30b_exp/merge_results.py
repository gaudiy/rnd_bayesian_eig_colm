#!/usr/bin/env python3
"""
Merge results from multi-GPU runs into a single JSON file.

Usage:
    python merge_results.py results/results_1.json results/results_2.json -o results/results_merged.json
"""
import json
import argparse

def main():
    parser = argparse.ArgumentParser(description="Merge results from multiple GPU runs")
    parser.add_argument("files", nargs="+", help="JSON result files to merge")
    parser.add_argument("-o", "--output", default="results_merged.json",
                       help="Output file path")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("MERGING RESULTS")
    print("=" * 80)
    
    all_results = []
    
    for file_path in args.files:
        print(f"Reading {file_path}...")
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                all_results.extend(data)
                print(f"  Loaded {len(data)} results")
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
    
    print(f"\nTotal results: {len(all_results)}")
    
    # Sort by global_idx to maintain order
    all_results.sort(key=lambda x: x.get('global_idx', 0))
    
    # Save merged results
    print(f"Saving to {args.output}...")
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("=" * 80)
    print("MERGE COMPLETE")
    print("=" * 80)
    
    # Print summary statistics
    conditions = {}
    for r in all_results:
        cond = r.get('condition', 'unknown')
        if cond not in conditions:
            conditions[cond] = {'total': 0, 'correct': 0}
        conditions[cond]['total'] += 1
        if r.get('correct', False):
            conditions[cond]['correct'] += 1
    
    print("\nSummary by condition:")
    for cond, stats in sorted(conditions.items()):
        acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        print(f"  {cond}: {stats['correct']}/{stats['total']} = {acc:.1%}")
    
    print(f"\nMerged results saved to: {args.output}")
    print("=" * 80)


if __name__ == "__main__":
    main()