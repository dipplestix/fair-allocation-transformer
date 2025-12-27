#!/usr/bin/env python3
"""
Compare Residual FATransformer + EF1 repair against baselines (Random, Random+EF1, RR).

Usage:
    python eval_pipeline/compare_residual_vs_baselines.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_results():
    """Load all evaluation results for comparison."""
    results_dir = Path('results')
    residual_dir = results_dir / 'residual'

    data = []

    # Load residual results
    residual_file = residual_dir / 'residual_all_sizes_results.csv'
    if residual_file.exists():
        residual_df = pd.read_csv(residual_file)
        for _, row in residual_df.iterrows():
            data.append({
                'm': row['m'],
                'type': row['model'],
                'ef_pct': row['ef_pct'],
                'ef1_pct': row['ef1_pct'],
                'efx_pct': row['efx_pct'],
                'utility_pct': row['utility_pct'],
                'nash_pct': row['nash_pct'],
                'avg_time_ms': row['avg_time_ms']
            })

    # Load baseline results for each m value
    for m in range(10, 31):
        dataset_pattern = f'10_{m}_100000'

        # Random baseline
        random_files = list(results_dir.glob(f'evaluation_results_{dataset_pattern}_random.csv'))
        if random_files:
            df = pd.read_csv(random_files[0])
            data.append({
                'm': m,
                'type': 'Random',
                'ef_pct': (df['envy_free'] == True).mean() * 100,
                'ef1_pct': (df['ef1'] == True).mean() * 100,
                'efx_pct': (df['efx'] == True).mean() * 100,
                'utility_pct': df['fraction_util_welfare'].mean() * 100,
                'nash_pct': df['fraction_nash_welfare'].mean() * 100,
                'avg_time_ms': df['inference_time'].mean() * 1000 / df['batch_size'].iloc[0]
            })

        # Random + EF1 repair
        random_ef1_files = list(results_dir.glob(f'evaluation_results_{dataset_pattern}_random_with_ef1_repair.csv'))
        if random_ef1_files:
            df = pd.read_csv(random_ef1_files[0])
            data.append({
                'm': m,
                'type': 'Random+EF1',
                'ef_pct': (df['envy_free'] == True).mean() * 100,
                'ef1_pct': (df['ef1'] == True).mean() * 100,
                'efx_pct': (df['efx'] == True).mean() * 100,
                'utility_pct': df['fraction_util_welfare'].mean() * 100,
                'nash_pct': df['fraction_nash_welfare'].mean() * 100,
                'avg_time_ms': df['inference_time'].mean() * 1000 / df['batch_size'].iloc[0]
            })

        # RR baseline
        rr_files = list(results_dir.glob(f'evaluation_results_{dataset_pattern}_rr.csv'))
        if rr_files:
            df = pd.read_csv(rr_files[0])
            data.append({
                'm': m,
                'type': 'RR',
                'ef_pct': (df['envy_free'] == True).mean() * 100,
                'ef1_pct': (df['ef1'] == True).mean() * 100,
                'efx_pct': (df['efx'] == True).mean() * 100,
                'utility_pct': df['fraction_util_welfare'].mean() * 100,
                'nash_pct': df['fraction_nash_welfare'].mean() * 100,
                'avg_time_ms': df['inference_time'].mean() * 1000 / df['batch_size'].iloc[0]
            })

    return pd.DataFrame(data)


def create_comparison_plots(df, output_dir):
    """Create comparison plots."""

    plt.style.use('seaborn-v0_8-darkgrid')

    # Colors and markers for each method
    colors = {
        'Residual+EF1': '#DC3545',  # Red
        'Random': '#6C757D',        # Gray
        'Random+EF1': '#17A2B8',    # Cyan
        'RR': '#28A745'             # Green
    }
    markers = {
        'Residual+EF1': 'o',
        'Random': '^',
        'Random+EF1': 'v',
        'RR': 's'
    }

    # Filter to only the methods we want to compare
    methods = ['Residual+EF1', 'Random', 'Random+EF1', 'RR']
    plot_df = df[df['type'].isin(methods)]

    # Main comparison plot (2x3 grid)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Residual+EF1 vs Baselines (Random, Random+EF1, RR)\nn=10 agents, varying items',
                 fontsize=14, fontweight='bold')

    metrics = [
        ('ef_pct', 'Envy-Free (%)', axes[0, 0]),
        ('ef1_pct', 'EF1 (%)', axes[0, 1]),
        ('efx_pct', 'EFx (%)', axes[0, 2]),
        ('utility_pct', 'Utility Fraction (%)', axes[1, 0]),
        ('nash_pct', 'Nash Welfare Fraction (%)', axes[1, 1]),
        ('avg_time_ms', 'Inference Time (ms/sample)', axes[1, 2])
    ]

    for method in methods:
        method_data = plot_df[plot_df['type'] == method].sort_values('m')
        if len(method_data) == 0:
            continue

        for metric, title, ax in metrics:
            ax.plot(method_data['m'], method_data[metric],
                   label=method, color=colors[method],
                   marker=markers[method], linewidth=2, markersize=6)

    for metric, title, ax in metrics:
        ax.set_xlabel('Number of Items (m)', fontsize=11)
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        if metric in ['ef_pct', 'ef1_pct', 'efx_pct']:
            ax.set_ylim(-5, 105)
        elif metric in ['utility_pct', 'nash_pct']:
            ax.set_ylim(50, 105)

    plt.tight_layout()
    plot_path = output_dir / 'residual_vs_baselines_all_metrics.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Main comparison plot saved to: {plot_path}")
    plt.close()

    # Focused comparison: EF1 and Nash welfare
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    fig2.suptitle('Residual+EF1 vs Baselines: Fairness and Efficiency', fontsize=14, fontweight='bold')

    # EF1 comparison
    ax = axes2[0]
    for method in methods:
        method_data = plot_df[plot_df['type'] == method].sort_values('m')
        if len(method_data) > 0:
            ax.plot(method_data['m'], method_data['ef1_pct'],
                   label=method, color=colors[method],
                   marker=markers[method], linewidth=2.5, markersize=8)
    ax.set_xlabel('Number of Items (m)', fontsize=12)
    ax.set_ylabel('EF1 Rate (%)', fontsize=12)
    ax.set_title('EF1 Fairness Guarantee', fontsize=12, fontweight='bold')
    ax.legend(fontsize=11)
    ax.set_ylim(-5, 105)
    ax.axhline(y=100, color='black', linestyle='--', alpha=0.3)
    ax.grid(True, alpha=0.3)

    # Nash welfare comparison
    ax = axes2[1]
    for method in methods:
        method_data = plot_df[plot_df['type'] == method].sort_values('m')
        if len(method_data) > 0:
            ax.plot(method_data['m'], method_data['nash_pct'],
                   label=method, color=colors[method],
                   marker=markers[method], linewidth=2.5, markersize=8)
    ax.set_xlabel('Number of Items (m)', fontsize=12)
    ax.set_ylabel('Nash Welfare (% of Optimal)', fontsize=12)
    ax.set_title('Nash Welfare Efficiency', fontsize=12, fontweight='bold')
    ax.legend(fontsize=11)
    ax.set_ylim(50, 105)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path2 = output_dir / 'residual_vs_baselines_ef1_nash.png'
    plt.savefig(plot_path2, dpi=300, bbox_inches='tight')
    print(f"EF1/Nash comparison saved to: {plot_path2}")
    plt.close()

    # Bar chart comparison at key sizes
    fig3, axes3 = plt.subplots(1, 3, figsize=(15, 5))
    fig3.suptitle('Performance Comparison at Key Dataset Sizes', fontsize=14, fontweight='bold')

    key_sizes = [10, 15, 20]
    key_data = plot_df[plot_df['m'].isin(key_sizes)]

    x = np.arange(len(key_sizes))
    width = 0.2

    for idx, (metric, title) in enumerate([('ef1_pct', 'EF1 Rate (%)'),
                                            ('nash_pct', 'Nash Welfare (%)'),
                                            ('utility_pct', 'Utility (%)')]):
        ax = axes3[idx]

        for i, method in enumerate(methods):
            method_data = key_data[key_data['type'] == method].sort_values('m')
            if len(method_data) > 0:
                values = method_data[metric].values
                offset = (i - len(methods)/2 + 0.5) * width
                ax.bar(x + offset, values, width, label=method, color=colors[method])

        ax.set_xlabel('Number of Items (m)', fontsize=11)
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(key_sizes)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

        if metric in ['ef1_pct']:
            ax.set_ylim(0, 110)

    plt.tight_layout()
    plot_path3 = output_dir / 'residual_vs_baselines_bar_chart.png'
    plt.savefig(plot_path3, dpi=300, bbox_inches='tight')
    print(f"Bar chart comparison saved to: {plot_path3}")
    plt.close()

    return plot_df


def print_summary(df):
    """Print summary comparison table."""

    methods = ['Residual+EF1', 'Random', 'Random+EF1', 'RR']

    print("\n" + "="*100)
    print("COMPARISON: Residual+EF1 vs Baselines")
    print("="*100)

    # Print for key sizes
    for m in [10, 15, 20]:
        print(f"\n--- m = {m} items ---")
        print(f"{'Method':<15} {'EF1%':<10} {'Nash%':<12} {'Utility%':<12} {'Time(ms)':<12}")
        print("-"*60)

        subset = df[(df['m'] == m) & (df['type'].isin(methods))]
        for method in methods:
            row = subset[subset['type'] == method]
            if len(row) > 0:
                row = row.iloc[0]
                print(f"{method:<15} {row['ef1_pct']:<10.1f} {row['nash_pct']:<12.2f} "
                      f"{row['utility_pct']:<12.2f} {row['avg_time_ms']:<12.4f}")

    # Average across all sizes where we have data
    print("\n" + "="*100)
    print("AVERAGE ACROSS ALL DATASET SIZES (m=10-20)")
    print("="*100)
    print(f"{'Method':<15} {'EF1%':<10} {'Nash%':<12} {'Utility%':<12}")
    print("-"*60)

    for method in methods:
        method_data = df[(df['type'] == method) & (df['m'] <= 20)]
        if len(method_data) > 0:
            print(f"{method:<15} {method_data['ef1_pct'].mean():<10.1f} "
                  f"{method_data['nash_pct'].mean():<12.2f} "
                  f"{method_data['utility_pct'].mean():<12.2f}")

    print("="*100)

    # Highlight key findings
    print("\nKEY FINDINGS:")

    residual_ef1 = df[df['type'] == 'Residual+EF1']
    rr_data = df[df['type'] == 'RR']
    random_ef1 = df[df['type'] == 'Random+EF1']

    if len(residual_ef1) > 0 and len(rr_data) > 0:
        # Compare at m=20 (training size)
        res_m20 = residual_ef1[residual_ef1['m'] == 20]
        rr_m20 = rr_data[rr_data['m'] == 20]

        if len(res_m20) > 0 and len(rr_m20) > 0:
            nash_diff = res_m20.iloc[0]['nash_pct'] - rr_m20.iloc[0]['nash_pct']
            print(f"  - At m=20: Residual+EF1 achieves {nash_diff:+.1f}% higher Nash welfare than RR")
            print(f"  - Residual+EF1: 100% EF1, {res_m20.iloc[0]['nash_pct']:.1f}% Nash")
            print(f"  - RR: 100% EF1, {rr_m20.iloc[0]['nash_pct']:.1f}% Nash")

    if len(residual_ef1) > 0 and len(random_ef1) > 0:
        common_m = set(residual_ef1['m']) & set(random_ef1['m'])
        if common_m:
            res_avg_nash = residual_ef1[residual_ef1['m'].isin(common_m)]['nash_pct'].mean()
            rand_avg_nash = random_ef1[random_ef1['m'].isin(common_m)]['nash_pct'].mean()
            print(f"  - Average Nash improvement over Random+EF1: {res_avg_nash - rand_avg_nash:+.1f}%")


def main():
    output_dir = Path('results/residual')
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading evaluation results...")
    df = load_results()

    print(f"Loaded {len(df)} result entries")

    # Print available data
    print("\nData available by method:")
    for method in df['type'].unique():
        m_values = sorted(df[df['type'] == method]['m'].unique())
        print(f"  {method}: m = {min(m_values)}-{max(m_values)} ({len(m_values)} sizes)")

    # Create plots
    print("\nGenerating comparison plots...")
    plot_df = create_comparison_plots(df, output_dir)

    # Print summary
    print_summary(df)

    # Save combined results
    csv_path = output_dir / 'residual_vs_baselines_comparison.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")


if __name__ == "__main__":
    main()
