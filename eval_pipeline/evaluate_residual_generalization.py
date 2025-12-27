#!/usr/bin/env python3
"""
Evaluate Residual FFTransformer on datasets with different agent counts (n).
Tests generalization beyond the training distribution (n=10, m=20).

Usage:
    python eval_pipeline/evaluate_residual_generalization.py datasets/5_10_100000_dataset.npz
    python eval_pipeline/evaluate_residual_generalization.py datasets/5_10_100000_dataset.npz --max_samples 10000
"""

import argparse
import numpy as np
import torch
import time
import sys
from pathlib import Path

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
    """Load the residual FFTransformer model (trained on n=10, m=20)."""
    from fftransformer.fftransformer_residual import FFTransformerResidual

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model parameters from training (n=10, m=20 was training size)
    # But the architecture is size-agnostic
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
    """Evaluate model on a dataset (handles any n, m)."""

    data = np.load(data_file)
    matrices = data['matrices']
    nash_welfare_max = data['nash_welfare']
    util_welfare_max = data['util_welfare']

    if max_samples is not None and max_samples < len(matrices):
        matrices = matrices[:max_samples]
        nash_welfare_max = nash_welfare_max[:max_samples]
        util_welfare_max = util_welfare_max[:max_samples]

    # Extract n, m from actual data shape
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


def main():
    parser = argparse.ArgumentParser(description='Evaluate Residual FFTransformer on any dataset size')
    parser.add_argument('dataset', type=str, help='Path to dataset .npz file')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size for evaluation')
    parser.add_argument('--max_samples', type=int, default=None, help='Maximum samples to evaluate')
    parser.add_argument('--ef1_repair_passes', type=int, default=10, help='Max EF1 repair passes')

    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Error: Dataset not found: {dataset_path}")
        sys.exit(1)

    # Load model once
    print("Loading Residual FFTransformer model...")
    print("  (trained on n=10, m=20)")
    model = load_residual_model()
    print(f"  Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Load dataset to get n, m
    data = np.load(dataset_path)
    n, m = data['matrices'][0].shape
    num_samples = len(data['matrices'])
    print(f"\nDataset: {dataset_path.name}")
    print(f"  Size: n={n} agents, m={m} items")
    print(f"  Samples: {num_samples:,}")
    if args.max_samples:
        print(f"  Using first {args.max_samples:,} samples")

    print(f"\n{'='*60}")
    print(f"Evaluating (n={n}, m={m})")
    print(f"{'='*60}")

    # Evaluate without EF1 repair
    print("\n  Without EF1 repair...")
    result = evaluate_dataset(
        model, dataset_path,
        batch_size=args.batch_size,
        apply_ef1_repair=False,
        max_samples=args.max_samples
    )
    print(f"    EF:      {result['ef_pct']:.2f}%")
    print(f"    EF1:     {result['ef1_pct']:.2f}%")
    print(f"    EFx:     {result['efx_pct']:.2f}%")
    print(f"    Utility: {result['utility_pct']:.2f}%")
    print(f"    Nash:    {result['nash_pct']:.2f}%")
    print(f"    Time:    {result['avg_time_ms']:.4f} ms/sample")

    # Evaluate with EF1 repair
    print("\n  With EF1 repair...")
    result_ef1 = evaluate_dataset(
        model, dataset_path,
        batch_size=args.batch_size,
        apply_ef1_repair=True,
        ef1_repair_max_passes=args.ef1_repair_passes,
        max_samples=args.max_samples
    )
    print(f"    EF:      {result_ef1['ef_pct']:.2f}%")
    print(f"    EF1:     {result_ef1['ef1_pct']:.2f}%")
    print(f"    EFx:     {result_ef1['efx_pct']:.2f}%")
    print(f"    Utility: {result_ef1['utility_pct']:.2f}%")
    print(f"    Nash:    {result_ef1['nash_pct']:.2f}%")
    print(f"    Time:    {result_ef1['avg_time_ms']:.4f} ms/sample")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Dataset: n={n}, m={m} (model trained on n=10, m=20)")
    print(f"\n{'Method':<15} {'EF1%':<10} {'Nash%':<12} {'Utility%':<12}")
    print("-"*50)
    print(f"{'Residual':<15} {result['ef1_pct']:<10.2f} {result['nash_pct']:<12.2f} {result['utility_pct']:<12.2f}")
    print(f"{'Residual+EF1':<15} {result_ef1['ef1_pct']:<10.2f} {result_ef1['nash_pct']:<12.2f} {result_ef1['utility_pct']:<12.2f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
