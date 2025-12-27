#!/usr/bin/env python3
"""
Evaluate Residual FFTransformer on all dataset sizes (10_10 through 10_30).
Creates comprehensive plots of the results.

Usage:
    python eval_pipeline/evaluate_residual_all_sizes.py
    python eval_pipeline/evaluate_residual_all_sizes.py --max_samples 10000
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import torch
import time
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.calculations import (
    calculate_agent_bundle_values_batch,
    is_envy_free_batch,
    is_ef1_batch,
    is_efx_batch,
    utility_sum_batch,
    nash_welfare_batch
)
from utils.inference import get_model_allocations_batch


def load_residual_model():
    """Load the residual FFTransformer model."""
    from fftransformer.fftransformer_residual import FFTransformerResidual

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Parameters from the actual training run
    model = FFTransformerResidual(
        d_model=256, num_heads=8,
        num_output_layers=2, dropout=0.0,
        initial_temperature=1.0, final_temperature=0.01
    )

    weights_path = project_root / "checkpoints" / "residual" / "best_model.pt"
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()

    return model


def evaluate_dataset(model, data_file, batch_size=100, apply_ef1_repair=False,
                     ef1_repair_max_passes=10, max_samples=None):
    """Evaluate model on a single dataset."""

    data = np.load(data_file)
    matrices = data['matrices']
    nash_welfare_max = data['nash_welfare']
    util_welfare_max = data['util_welfare']

    if max_samples is not None and max_samples < len(matrices):
        matrices = matrices[:max_samples]
        nash_welfare_max = nash_welfare_max[:max_samples]
        util_welfare_max = util_welfare_max[:max_samples]

    n, m = matrices[0].shape

    all_envy_free = []
    all_ef1 = []
    all_efx = []
    all_util_fractions = []
    all_nash_fractions = []
    total_time = 0.0

    for i in range(0, len(matrices), batch_size):
        batch_end = min(i + batch_size, len(matrices))
        batch_matrices = matrices[i:batch_end]
        batch_nash_max = nash_welfare_max[i:batch_end]
        batch_util_max = util_welfare_max[i:batch_end]

        start_time = time.perf_counter()
        batch_allocations = get_model_allocations_batch(
            model, batch_matrices,
            apply_ef1_repair=apply_ef1_repair,
            ef1_repair_params={'max_passes': ef1_repair_max_passes}
        )
        end_time = time.perf_counter()
        total_time += (end_time - start_time)

        agent_bundle_values = calculate_agent_bundle_values_batch(batch_matrices, batch_allocations)

        all_envy_free.extend(is_envy_free_batch(agent_bundle_values))
        all_ef1.extend(is_ef1_batch(batch_matrices, batch_allocations, agent_bundle_values))
        all_efx.extend(is_efx_batch(batch_matrices, batch_allocations, agent_bundle_values))

        util_sums = utility_sum_batch(agent_bundle_values)
        nash_welfares = nash_welfare_batch(agent_bundle_values)

        all_util_fractions.extend(util_sums / batch_util_max)
        all_nash_fractions.extend(nash_welfares / batch_nash_max)

    return {
        'n': n,
        'm': m,
        'num_samples': len(matrices),
        'ef_pct': np.mean(all_envy_free) * 100,
        'ef1_pct': np.mean(all_ef1) * 100,
        'efx_pct': np.mean(all_efx) * 100,
        'utility_pct': np.mean(all_util_fractions) * 100,
        'nash_pct': np.mean(all_nash_fractions) * 100,
        'avg_time_ms': total_time / len(matrices) * 1000,
        'total_time': total_time
    }


def create_plots(results_df, output_dir):
    """Create comprehensive plots for residual model results."""

    plt.style.use('seaborn-v0_8-darkgrid')

    colors = {
        'Residual': '#F18F01',
        'Residual+EF1': '#DC3545'
    }
    markers = {
        'Residual': 'o',
        'Residual+EF1': 's'
    }

    # Main comparison plot (2x3 grid)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Residual FFTransformer Performance Across Dataset Sizes\n(n=10 agents, varying items)',
                 fontsize=14, fontweight='bold')

    metrics = [
        ('ef_pct', 'Envy-Free (%)', axes[0, 0]),
        ('ef1_pct', 'EF1 (%)', axes[0, 1]),
        ('efx_pct', 'EFx (%)', axes[0, 2]),
        ('utility_pct', 'Utility Fraction (%)', axes[1, 0]),
        ('nash_pct', 'Nash Welfare Fraction (%)', axes[1, 1]),
        ('avg_time_ms', 'Inference Time (ms/sample)', axes[1, 2])
    ]

    for model_type in ['Residual', 'Residual+EF1']:
        model_data = results_df[results_df['model'] == model_type].sort_values('m')

        for metric, title, ax in metrics:
            ax.plot(model_data['m'], model_data[metric],
                   label=model_type, color=colors[model_type],
                   marker=markers[model_type], linewidth=2, markersize=6)

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
    plot_path = output_dir / 'residual_performance_all_sizes.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Main plot saved to: {plot_path}")
    plt.close()

    # Separate detailed plots
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    fig2.suptitle('Residual Model: Fairness vs Efficiency Trade-off', fontsize=14, fontweight='bold')

    # EF1 rate comparison
    ax = axes2[0]
    for model_type in ['Residual', 'Residual+EF1']:
        model_data = results_df[results_df['model'] == model_type].sort_values('m')
        ax.plot(model_data['m'], model_data['ef1_pct'],
               label=model_type, color=colors[model_type],
               marker=markers[model_type], linewidth=2, markersize=8)
    ax.set_xlabel('Number of Items (m)', fontsize=12)
    ax.set_ylabel('EF1 Rate (%)', fontsize=12)
    ax.set_title('EF1 Fairness Rate', fontsize=12, fontweight='bold')
    ax.legend(fontsize=11)
    ax.set_ylim(-5, 105)
    ax.axhline(y=100, color='green', linestyle='--', alpha=0.5, label='100% EF1')
    ax.grid(True, alpha=0.3)

    # Nash welfare comparison
    ax = axes2[1]
    for model_type in ['Residual', 'Residual+EF1']:
        model_data = results_df[results_df['model'] == model_type].sort_values('m')
        ax.plot(model_data['m'], model_data['nash_pct'],
               label=model_type, color=colors[model_type],
               marker=markers[model_type], linewidth=2, markersize=8)
    ax.set_xlabel('Number of Items (m)', fontsize=12)
    ax.set_ylabel('Nash Welfare Fraction (%)', fontsize=12)
    ax.set_title('Nash Welfare (% of Optimal)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=11)
    ax.set_ylim(50, 105)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path2 = output_dir / 'residual_fairness_efficiency.png'
    plt.savefig(plot_path2, dpi=300, bbox_inches='tight')
    print(f"Fairness-efficiency plot saved to: {plot_path2}")
    plt.close()

    # Summary bar chart for key metrics at different sizes
    fig3, axes3 = plt.subplots(1, 3, figsize=(15, 5))
    fig3.suptitle('Residual Model Performance at Key Dataset Sizes', fontsize=14, fontweight='bold')

    key_sizes = [10, 15, 20, 25, 30]
    key_data = results_df[results_df['m'].isin(key_sizes)]

    x = np.arange(len(key_sizes))
    width = 0.35

    for idx, (metric, title) in enumerate([('ef1_pct', 'EF1 Rate (%)'),
                                            ('nash_pct', 'Nash Welfare (%)'),
                                            ('avg_time_ms', 'Time (ms/sample)')]):
        ax = axes3[idx]

        residual_data = key_data[key_data['model'] == 'Residual'].sort_values('m')[metric].values
        residual_ef1_data = key_data[key_data['model'] == 'Residual+EF1'].sort_values('m')[metric].values

        bars1 = ax.bar(x - width/2, residual_data, width, label='Residual', color=colors['Residual'])
        bars2 = ax.bar(x + width/2, residual_ef1_data, width, label='Residual+EF1', color=colors['Residual+EF1'])

        ax.set_xlabel('Number of Items (m)', fontsize=11)
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(key_sizes)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plot_path3 = output_dir / 'residual_key_sizes_comparison.png'
    plt.savefig(plot_path3, dpi=300, bbox_inches='tight')
    print(f"Key sizes comparison saved to: {plot_path3}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluate Residual FFTransformer on all dataset sizes')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size for evaluation')
    parser.add_argument('--max_samples', type=int, default=None, help='Maximum samples per dataset')
    parser.add_argument('--ef1_repair_passes', type=int, default=10, help='Max EF1 repair passes')
    parser.add_argument('--output_dir', type=str, default='results/residual', help='Output directory')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model once
    print("Loading Residual FFTransformer model...")
    model = load_residual_model()
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Find all datasets
    datasets_dir = project_root / "datasets"
    dataset_files = sorted(datasets_dir.glob("10_*_100000_dataset.npz"))

    print(f"\nFound {len(dataset_files)} datasets to evaluate")

    all_results = []

    for dataset_file in dataset_files:
        # Extract m from filename (e.g., "10_20_100000_dataset.npz" -> 20)
        parts = dataset_file.stem.split('_')
        m = int(parts[1])

        print(f"\n{'='*50}")
        print(f"Evaluating on dataset: n=10, m={m}")
        print(f"{'='*50}")

        # Evaluate without EF1 repair
        print("  Without EF1 repair...")
        result = evaluate_dataset(
            model, dataset_file,
            batch_size=args.batch_size,
            apply_ef1_repair=False,
            max_samples=args.max_samples
        )
        result['model'] = 'Residual'
        all_results.append(result)
        print(f"    EF1: {result['ef1_pct']:.1f}%, Nash: {result['nash_pct']:.2f}%")

        # Evaluate with EF1 repair
        print("  With EF1 repair...")
        result_ef1 = evaluate_dataset(
            model, dataset_file,
            batch_size=args.batch_size,
            apply_ef1_repair=True,
            ef1_repair_max_passes=args.ef1_repair_passes,
            max_samples=args.max_samples
        )
        result_ef1['model'] = 'Residual+EF1'
        all_results.append(result_ef1)
        print(f"    EF1: {result_ef1['ef1_pct']:.1f}%, Nash: {result_ef1['nash_pct']:.2f}%")

    # Create DataFrame
    results_df = pd.DataFrame(all_results)

    # Save results
    csv_path = output_dir / 'residual_all_sizes_results.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

    # Print summary table
    print("\n" + "="*100)
    print("RESIDUAL MODEL EVALUATION SUMMARY")
    print("="*100)
    print(f"{'m':<5} {'Model':<15} {'EF%':<8} {'EF1%':<8} {'EFx%':<8} {'Util%':<10} {'Nash%':<10} {'Time(ms)':<10}")
    print("-"*100)

    for _, row in results_df.sort_values(['m', 'model']).iterrows():
        print(f"{row['m']:<5} {row['model']:<15} {row['ef_pct']:<8.1f} {row['ef1_pct']:<8.1f} "
              f"{row['efx_pct']:<8.1f} {row['utility_pct']:<10.2f} {row['nash_pct']:<10.2f} "
              f"{row['avg_time_ms']:<10.4f}")

    print("="*100)

    # Create plots
    print("\nGenerating plots...")
    create_plots(results_df, output_dir)

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
