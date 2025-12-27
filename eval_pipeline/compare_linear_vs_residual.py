#!/usr/bin/env python3
"""
Compare Linear FATransformer vs Residual FATransformer performance.

This script evaluates both models (with and without EF1 repair) on a dataset
and generates comparison plots.

Usage:
    python eval_pipeline/compare_linear_vs_residual.py datasets/10_20_100000_dataset.npz
    python eval_pipeline/compare_linear_vs_residual.py datasets/10_20_100000_dataset.npz --max_samples 10000
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
from utils.ef1_repair import ef1_quick_repair_batch


def load_linear_model():
    """Load the linear FATransformer model."""
    from fatransformer.fatransformer import FATransformer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FATransformer(
        n=10, m=20, d_model=768, num_heads=16,
        num_output_layers=4, dropout=0.008020981126192437,
        initial_temperature=1.0, final_temperature=0.01
    )

    weights_path = project_root / "checkpoints" / "linear" / "best_model.pt"
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()

    return model, "Linear"


def load_residual_model():
    """Load the residual FATransformer model."""
    from fatransformer.fatransformer_residual import FATransformer as FATransformerResidual

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Parameters from the actual training run (from checkpoint config)
    model = FATransformerResidual(
        n=10, m=20, d_model=256, num_heads=8,
        num_output_layers=2, dropout=0.0,
        initial_temperature=1.0, final_temperature=0.01
    )

    weights_path = project_root / "checkpoints" / "residual" / "best_model.pt"
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()

    return model, "Residual"


def evaluate_model(model, model_name, matrices, nash_welfare_max, util_welfare_max,
                   batch_size=100, apply_ef1_repair=False, ef1_repair_max_passes=10):
    """Evaluate a model on the dataset."""

    suffix = f"{model_name}+EF1" if apply_ef1_repair else model_name

    all_results = []
    total_inference_time = 0.0

    for i in tqdm(range(0, len(matrices), batch_size), desc=f"Evaluating {suffix}"):
        batch_end = min(i + batch_size, len(matrices))
        batch_matrices = matrices[i:batch_end]
        batch_nash_max = nash_welfare_max[i:batch_end]
        batch_util_max = util_welfare_max[i:batch_end]

        # Get allocations
        start_time = time.perf_counter()
        batch_allocations = get_model_allocations_batch(
            model, batch_matrices,
            apply_ef1_repair=apply_ef1_repair,
            ef1_repair_params={'max_passes': ef1_repair_max_passes}
        )
        end_time = time.perf_counter()
        batch_time = end_time - start_time
        total_inference_time += batch_time

        # Calculate metrics
        agent_bundle_values = calculate_agent_bundle_values_batch(batch_matrices, batch_allocations)

        envy_free = is_envy_free_batch(agent_bundle_values)
        ef1 = is_ef1_batch(batch_matrices, batch_allocations, agent_bundle_values)
        efx = is_efx_batch(batch_matrices, batch_allocations, agent_bundle_values)
        util_sums = utility_sum_batch(agent_bundle_values)
        nash_welfares = nash_welfare_batch(agent_bundle_values)

        # Store results
        for j in range(len(batch_matrices)):
            result = {
                'model': suffix,
                'matrix_id': i + j,
                'envy_free': bool(envy_free[j]),
                'ef1': bool(ef1[j]),
                'efx': bool(efx[j]),
                'utility_sum': float(util_sums[j]),
                'nash_welfare': float(nash_welfares[j]),
                'max_nash_welfare': float(batch_nash_max[j]),
                'max_util_welfare': float(batch_util_max[j]),
                'fraction_util_welfare': float(util_sums[j]) / float(batch_util_max[j]) if batch_util_max[j] > 0 else 0,
                'fraction_nash_welfare': float(nash_welfares[j]) / float(batch_nash_max[j]) if batch_nash_max[j] > 0 else 0,
                'inference_time': batch_time,
                'batch_size': len(batch_matrices)
            }
            all_results.append(result)

    return pd.DataFrame(all_results), total_inference_time


def create_comparison_plot(results_df, output_path):
    """Create comparison plots for linear vs residual models."""

    plt.style.use('seaborn-v0_8-darkgrid')

    colors = {
        'Linear': '#2E86AB',
        'Linear+EF1': '#28A745',
        'Residual': '#F18F01',
        'Residual+EF1': '#DC3545'
    }

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Linear vs Residual FATransformer Comparison', fontsize=16, fontweight='bold')

    models = ['Linear', 'Linear+EF1', 'Residual', 'Residual+EF1']

    # Calculate summary stats for each model
    summary_data = []
    for model in models:
        model_df = results_df[results_df['model'] == model]
        if len(model_df) > 0:
            summary = {
                'model': model,
                'ef_pct': model_df['envy_free'].mean() * 100,
                'ef1_pct': model_df['ef1'].mean() * 100,
                'efx_pct': model_df['efx'].mean() * 100,
                'utility_pct': model_df['fraction_util_welfare'].mean() * 100,
                'nash_pct': model_df['fraction_nash_welfare'].mean() * 100,
                'avg_time_ms': model_df['inference_time'].mean() * 1000 / model_df['batch_size'].iloc[0]
            }
            summary_data.append(summary)

    summary_df = pd.DataFrame(summary_data)

    # Bar charts for each metric
    metrics = [
        ('ef_pct', 'Envy-Free (%)', axes[0, 0]),
        ('ef1_pct', 'EF1 (%)', axes[0, 1]),
        ('efx_pct', 'EFx (%)', axes[0, 2]),
        ('utility_pct', 'Utility Fraction (%)', axes[1, 0]),
        ('nash_pct', 'Nash Welfare Fraction (%)', axes[1, 1]),
        ('avg_time_ms', 'Inference Time (ms/sample)', axes[1, 2])
    ]

    x = np.arange(len(summary_df))

    for metric, title, ax in metrics:
        bars = ax.bar(x, summary_df[metric], color=[colors[m] for m in summary_df['model']])
        ax.set_xticks(x)
        ax.set_xticklabels(summary_df['model'], rotation=45, ha='right')
        ax.set_ylabel(title)
        ax.set_title(title, fontweight='bold')

        # Add value labels on bars
        for bar, val in zip(bars, summary_df[metric]):
            height = bar.get_height()
            ax.annotate(f'{val:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)

        if metric != 'avg_time_ms':
            ax.set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to: {output_path}")

    return summary_df


def print_summary_table(summary_df):
    """Print a formatted summary table."""
    print("\n" + "="*80)
    print("COMPARISON SUMMARY: Linear vs Residual FATransformer")
    print("="*80)
    print(f"{'Model':<15} {'EF%':<8} {'EF1%':<8} {'EFx%':<8} {'Util%':<10} {'Nash%':<10} {'Time(ms)':<10}")
    print("-"*80)

    for _, row in summary_df.iterrows():
        print(f"{row['model']:<15} {row['ef_pct']:<8.1f} {row['ef1_pct']:<8.1f} "
              f"{row['efx_pct']:<8.1f} {row['utility_pct']:<10.2f} {row['nash_pct']:<10.2f} "
              f"{row['avg_time_ms']:<10.4f}")

    print("="*80)

    # Calculate improvements
    linear_row = summary_df[summary_df['model'] == 'Linear'].iloc[0] if 'Linear' in summary_df['model'].values else None
    residual_row = summary_df[summary_df['model'] == 'Residual'].iloc[0] if 'Residual' in summary_df['model'].values else None

    if linear_row is not None and residual_row is not None:
        print("\nResidual vs Linear (without EF1 repair):")
        print(f"  Nash Welfare: {residual_row['nash_pct']:.2f}% vs {linear_row['nash_pct']:.2f}% "
              f"(diff: {residual_row['nash_pct'] - linear_row['nash_pct']:+.2f}%)")
        print(f"  Utility:      {residual_row['utility_pct']:.2f}% vs {linear_row['utility_pct']:.2f}% "
              f"(diff: {residual_row['utility_pct'] - linear_row['utility_pct']:+.2f}%)")
        print(f"  EF1 Rate:     {residual_row['ef1_pct']:.1f}% vs {linear_row['ef1_pct']:.1f}%")

        # Speed comparison
        speedup = linear_row['avg_time_ms'] / residual_row['avg_time_ms']
        print(f"  Speed:        Residual is {speedup:.1f}x {'faster' if speedup > 1 else 'slower'}")


def main():
    parser = argparse.ArgumentParser(description='Compare Linear vs Residual FATransformer')
    parser.add_argument('data_file', help='Input .npz dataset file')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size for evaluation')
    parser.add_argument('--max_samples', type=int, default=None, help='Maximum samples to evaluate')
    parser.add_argument('--ef1_repair_passes', type=int, default=10, help='Max EF1 repair passes')
    parser.add_argument('--output_dir', type=str, default='results/residual',
                       help='Output directory for results')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print(f"Loading dataset from {args.data_file}...")
    data = np.load(args.data_file)
    matrices = data['matrices']
    nash_welfare_max = data['nash_welfare']
    util_welfare_max = data['util_welfare']

    if args.max_samples is not None and args.max_samples < len(matrices):
        matrices = matrices[:args.max_samples]
        nash_welfare_max = nash_welfare_max[:args.max_samples]
        util_welfare_max = util_welfare_max[:args.max_samples]
        print(f"Limited to {args.max_samples} samples")

    print(f"Dataset: {len(matrices)} matrices, shape {matrices[0].shape}")

    # Load models
    print("\nLoading models...")
    linear_model, linear_name = load_linear_model()
    residual_model, residual_name = load_residual_model()

    linear_params = sum(p.numel() for p in linear_model.parameters())
    residual_params = sum(p.numel() for p in residual_model.parameters())
    print(f"Linear model: {linear_params:,} parameters")
    print(f"Residual model: {residual_params:,} parameters")

    # Evaluate all configurations
    all_results = []

    # Linear without EF1
    print("\n" + "="*50)
    df, time_linear = evaluate_model(
        linear_model, linear_name, matrices, nash_welfare_max, util_welfare_max,
        batch_size=args.batch_size, apply_ef1_repair=False
    )
    all_results.append(df)

    # Linear with EF1
    df, time_linear_ef1 = evaluate_model(
        linear_model, linear_name, matrices, nash_welfare_max, util_welfare_max,
        batch_size=args.batch_size, apply_ef1_repair=True, ef1_repair_max_passes=args.ef1_repair_passes
    )
    all_results.append(df)

    # Residual without EF1
    df, time_residual = evaluate_model(
        residual_model, residual_name, matrices, nash_welfare_max, util_welfare_max,
        batch_size=args.batch_size, apply_ef1_repair=False
    )
    all_results.append(df)

    # Residual with EF1
    df, time_residual_ef1 = evaluate_model(
        residual_model, residual_name, matrices, nash_welfare_max, util_welfare_max,
        batch_size=args.batch_size, apply_ef1_repair=True, ef1_repair_max_passes=args.ef1_repair_passes
    )
    all_results.append(df)

    # Combine results
    results_df = pd.concat(all_results, ignore_index=True)

    # Save detailed results
    csv_path = output_dir / "linear_vs_residual_comparison.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nDetailed results saved to: {csv_path}")

    # Create comparison plot
    plot_path = output_dir / "linear_vs_residual_comparison.png"
    summary_df = create_comparison_plot(results_df, plot_path)

    # Print summary
    print_summary_table(summary_df)

    # Save summary
    summary_path = output_dir / "linear_vs_residual_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
