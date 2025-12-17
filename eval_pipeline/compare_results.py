#!/usr/bin/env python3
"""Compare evaluation results across model and baselines."""

import pandas as pd
import glob
import re
from pathlib import Path

def parse_dataset_size(filename):
    """Extract n and m from filename like 'evaluation_results_10_15_100000_*.csv'"""
    match = re.search(r'_(\d+)_(\d+)_\d+_', filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None

def load_and_summarize(filepath):
    """Load CSV and compute summary statistics."""
    df = pd.read_csv(filepath)

    summary = {
        'ef_pct': (df['envy_free'] == True).mean() * 100,
        'ef1_pct': (df['ef1'] == True).mean() * 100,
        'efx_pct': (df['efx'] == True).mean() * 100,
        'utility_pct': df['fraction_util_welfare'].mean() * 100,
        'nash_welfare_pct': df['fraction_nash_welfare'].mean() * 100,
    }

    return summary

def main():
    results_dir = Path('results')

    # Collect all results
    data = []

    for m in range(10, 21):
        dataset_pattern = f'10_{m}_100000'

        # Model results
        model_files = list(results_dir.glob(f'evaluation_results_{dataset_pattern}_best_from_sweep_*.csv'))
        if model_files:
            model_summary = load_and_summarize(model_files[0])
            model_summary['dataset'] = f'10_{m}'
            model_summary['type'] = 'Model'
            data.append(model_summary)

        # Random baseline
        random_files = list(results_dir.glob(f'evaluation_results_{dataset_pattern}_random.csv'))
        if random_files:
            random_summary = load_and_summarize(random_files[0])
            random_summary['dataset'] = f'10_{m}'
            random_summary['type'] = 'Random'
            data.append(random_summary)

        # RR baseline
        rr_files = list(results_dir.glob(f'evaluation_results_{dataset_pattern}_rr.csv'))
        if rr_files:
            rr_summary = load_and_summarize(rr_files[0])
            rr_summary['dataset'] = f'10_{m}'
            rr_summary['type'] = 'RR'
            data.append(rr_summary)

    # Create comparison DataFrame
    df = pd.DataFrame(data)

    # Print comparison table
    print("\n" + "="*120)
    print("EVALUATION COMPARISON: Model vs Baselines (All percentages)")
    print("="*120)
    print()

    for m in range(10, 21):
        dataset = f'10_{m}'
        subset = df[df['dataset'] == dataset]

        if len(subset) == 0:
            continue

        print(f"Dataset: {dataset} (n=10, m={m})")
        print("-" * 120)
        print(f"{'Method':<12} {'EF':>8} {'EF1':>8} {'EFx':>8} {'Utility':>10} {'Nash Welfare':>14}")
        print("-" * 120)

        for _, row in subset.iterrows():
            print(f"{row['type']:<12} {row['ef_pct']:>7.1f}% {row['ef1_pct']:>7.1f}% {row['efx_pct']:>7.1f}% "
                  f"{row['utility_pct']:>9.1f}% {row['nash_welfare_pct']:>13.1f}%")

        print()

    # Summary statistics
    print("="*120)
    print("AVERAGE ACROSS ALL DATASETS")
    print("="*120)

    avg_by_type = df.groupby('type')[['ef_pct', 'ef1_pct', 'efx_pct', 'utility_pct', 'nash_welfare_pct']].mean()

    print(f"{'Method':<12} {'EF':>8} {'EF1':>8} {'EFx':>8} {'Utility':>10} {'Nash Welfare':>14}")
    print("-" * 120)
    for method in ['Model', 'RR', 'Random']:
        if method in avg_by_type.index:
            row = avg_by_type.loc[method]
            print(f"{method:<12} {row['ef_pct']:>7.1f}% {row['ef1_pct']:>7.1f}% {row['efx_pct']:>7.1f}% "
                  f"{row['utility_pct']:>9.1f}% {row['nash_welfare_pct']:>13.1f}%")

    print()
    print("="*120)

    # Save detailed comparison to CSV
    comparison_file = results_dir / 'comparison_summary.csv'
    df.to_csv(comparison_file, index=False)
    print(f"\nDetailed comparison saved to: {comparison_file}")

if __name__ == '__main__':
    main()
