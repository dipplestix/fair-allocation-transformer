#!/usr/bin/env python3
"""Analyze results and create comparison plots."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_results():
    """Load all evaluation results"""
    results_dir = Path('results')
    data = []

    for m in range(10, 21):
        dataset_pattern = f'10_{m}_100000'

        # Model results
        model_files = list(results_dir.glob(f'evaluation_results_{dataset_pattern}_best_from_sweep_*.csv'))
        if model_files:
            df = pd.read_csv(model_files[0])
            summary = {
                'dataset': f'10_{m}',
                'm': m,
                'type': 'Model',
                'ef_pct': (df['envy_free'] == True).mean() * 100,
                'ef1_pct': (df['ef1'] == True).mean() * 100,
                'efx_pct': (df['efx'] == True).mean() * 100,
                'utility_pct': df['fraction_util_welfare'].mean() * 100,
                'nash_welfare_pct': df['fraction_nash_welfare'].mean() * 100,
                'avg_time_ms': df['inference_time'].mean() * 1000,  # Convert to ms
                'batch_size': df['batch_size'].iloc[0]
            }
            data.append(summary)

        # Model with swaps results
        model_swap_files = list(results_dir.glob(f'evaluation_results_{dataset_pattern}_best_from_sweep_*_with_swaps.csv'))
        if model_swap_files:
            df = pd.read_csv(model_swap_files[0])
            summary = {
                'dataset': f'10_{m}',
                'm': m,
                'type': 'Model+Swaps',
                'ef_pct': (df['envy_free'] == True).mean() * 100,
                'ef1_pct': (df['ef1'] == True).mean() * 100,
                'efx_pct': (df['efx'] == True).mean() * 100,
                'utility_pct': df['fraction_util_welfare'].mean() * 100,
                'nash_welfare_pct': df['fraction_nash_welfare'].mean() * 100,
                'avg_time_ms': df['inference_time'].mean() * 1000,  # Convert to ms
                'batch_size': df['batch_size'].iloc[0]
            }
            data.append(summary)

        # Random baseline
        random_files = list(results_dir.glob(f'evaluation_results_{dataset_pattern}_random.csv'))
        if random_files:
            df = pd.read_csv(random_files[0])
            summary = {
                'dataset': f'10_{m}',
                'm': m,
                'type': 'Random',
                'ef_pct': (df['envy_free'] == True).mean() * 100,
                'ef1_pct': (df['ef1'] == True).mean() * 100,
                'efx_pct': (df['efx'] == True).mean() * 100,
                'utility_pct': df['fraction_util_welfare'].mean() * 100,
                'nash_welfare_pct': df['fraction_nash_welfare'].mean() * 100,
                'avg_time_ms': df['inference_time'].mean() * 1000,
                'batch_size': df['batch_size'].iloc[0]
            }
            data.append(summary)

        # RR baseline
        rr_files = list(results_dir.glob(f'evaluation_results_{dataset_pattern}_rr.csv'))
        if rr_files:
            df = pd.read_csv(rr_files[0])
            summary = {
                'dataset': f'10_{m}',
                'm': m,
                'type': 'RR',
                'ef_pct': (df['envy_free'] == True).mean() * 100,
                'ef1_pct': (df['ef1'] == True).mean() * 100,
                'efx_pct': (df['efx'] == True).mean() * 100,
                'utility_pct': df['fraction_util_welfare'].mean() * 100,
                'nash_welfare_pct': df['fraction_nash_welfare'].mean() * 100,
                'avg_time_ms': df['inference_time'].mean() * 1000,
                'batch_size': df['batch_size'].iloc[0]
            }
            data.append(summary)

        # ECE baseline
        ece_files = list(results_dir.glob(f'evaluation_results_{dataset_pattern}_ece.csv'))
        if ece_files:
            df = pd.read_csv(ece_files[0])
            summary = {
                'dataset': f'10_{m}',
                'm': m,
                'type': 'ECE',
                'ef_pct': (df['envy_free'] == True).mean() * 100,
                'ef1_pct': (df['ef1'] == True).mean() * 100,
                'efx_pct': (df['efx'] == True).mean() * 100,
                'utility_pct': df['fraction_util_welfare'].mean() * 100,
                'nash_welfare_pct': df['fraction_nash_welfare'].mean() * 100,
                'avg_time_ms': df['inference_time'].mean() * 1000,
                'batch_size': df['batch_size'].iloc[0]
            }
            data.append(summary)

    return pd.DataFrame(data)

def print_runtime_comparison(df):
    """Print runtime comparison table"""
    print("\n" + "="*80)
    print("RUNTIME COMPARISON (ms per batch)")
    print("="*80)
    print(f"Note: Batch sizes may vary by method")
    print()

    for m in range(10, 21):
        subset = df[df['m'] == m]
        if len(subset) == 0:
            continue

        print(f"Dataset: 10_{m} (n=10, m={m})")
        print("-" * 80)
        print(f"{'Method':<12} {'Avg Time (ms)':<18} {'Batch Size':<12} {'Time per item (ms)':<20}")
        print("-" * 80)

        for _, row in subset.iterrows():
            time_per_item = row['avg_time_ms'] / row['batch_size']
            print(f"{row['type']:<12} {row['avg_time_ms']:>15.2f} {row['batch_size']:>11} {time_per_item:>19.4f}")

        print()

    # Average runtime
    print("="*80)
    print("AVERAGE RUNTIME ACROSS ALL DATASETS")
    print("="*80)
    avg_by_type = df.groupby('type')[['avg_time_ms', 'batch_size']].mean()

    print(f"{'Method':<12} {'Avg Time (ms)':<18} {'Avg Batch Size':<16} {'Time per item (ms)':<20}")
    print("-" * 80)
    for method in ['Model', 'Model+Swaps', 'RR', 'ECE', 'Random']:
        if method in avg_by_type.index:
            row = avg_by_type.loc[method]
            time_per_item = row['avg_time_ms'] / row['batch_size']
            print(f"{method:<12} {row['avg_time_ms']:>15.2f} {row['batch_size']:>15.1f} {time_per_item:>19.4f}")

    print()

def create_plots(df):
    """Create comparison plots"""
    # Set up the plot style
    plt.style.use('seaborn-v0_8-darkgrid')
    colors = {'Model': '#2E86AB', 'Model+Swaps': '#1A5F7A', 'RR': '#A23B72', 'ECE': '#06A77D', 'Random': '#F18F01'}
    markers = {'Model': 'o', 'Model+Swaps': '*', 'RR': 's', 'ECE': 'D', 'Random': '^'}

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Performance Metrics vs. Number of Items (n=10 agents)', fontsize=16, fontweight='bold')

    metrics = [
        ('ef_pct', 'Envy-Free (%)', axes[0, 0]),
        ('ef1_pct', 'EF1 (%)', axes[0, 1]),
        ('efx_pct', 'EFx (%)', axes[0, 2]),
        ('utility_pct', 'Utility Fraction (%)', axes[1, 0]),
        ('nash_welfare_pct', 'Nash Welfare Fraction (%)', axes[1, 1]),
        ('avg_time_ms', 'Average Runtime (ms per batch)', axes[1, 2])
    ]

    for metric, title, ax in metrics:
        for method in ['Model', 'Model+Swaps', 'RR', 'ECE', 'Random']:
            method_data = df[df['type'] == method].sort_values('m')
            if len(method_data) > 0:  # Only plot if data exists
                ax.plot(method_data['m'], method_data[metric],
                       label=method, color=colors[method], marker=markers[method],
                       linewidth=2, markersize=8 if method != 'Model+Swaps' else 12)

        ax.set_xlabel('Number of Items (m)', fontsize=11, fontweight='bold')
        ax.set_ylabel(title, fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(10, 21))

        # Set y-axis limits based on metric
        if metric in ['ef_pct', 'ef1_pct', 'efx_pct']:
            ax.set_ylim(-5, 105)
        elif metric in ['utility_pct', 'nash_welfare_pct']:
            ax.set_ylim(0, 105)

    plt.tight_layout()

    # Save figure
    output_path = Path('results/comparison_plots.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlots saved to: {output_path}")

    # Create separate runtime plot with better visibility
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    fig2.suptitle('Runtime Comparison vs. Number of Items', fontsize=14, fontweight='bold')

    for method in ['Model', 'Model+Swaps', 'RR', 'ECE', 'Random']:
        method_data = df[df['type'] == method].sort_values('m')
        if len(method_data) > 0:  # Only plot if data exists
            ax2.plot(method_data['m'], method_data['avg_time_ms'],
                   label=method, color=colors[method], marker=markers[method],
                   linewidth=2, markersize=8 if method != 'Model+Swaps' else 12)

    ax2.set_xlabel('Number of Items (m)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Average Runtime (ms per batch)', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(range(10, 21))

    plt.tight_layout()

    output_path2 = Path('results/runtime_comparison.png')
    plt.savefig(output_path2, dpi=300, bbox_inches='tight')
    print(f"Runtime plot saved to: {output_path2}")

def main():
    print("Loading evaluation results...")
    df = load_results()

    print(f"Loaded {len(df)} result sets")

    # Print runtime comparison
    print_runtime_comparison(df)

    # Create plots
    print("\nCreating comparison plots...")
    create_plots(df)

    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)

if __name__ == '__main__':
    main()
