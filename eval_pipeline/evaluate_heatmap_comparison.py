#!/usr/bin/env python3
"""
Evaluate and compare methods on pre-generated heatmap datasets.
Computes (model+EF1) - (max_util+EF1) difference heatmaps.
"""

import argparse
import numpy as np
import torch
import sys
import json
from pathlib import Path
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.calculations import (
    calculate_agent_bundle_values_batch,
    utility_sum_batch,
    nash_welfare_batch
)
from utils.inference import get_model_allocations_batch
from utils.ef1_repair import ef1_quick_repair_batch


def load_residual_model():
    """Load the residual FFTransformer model."""
    from fftransformer.fftransformer_residual import FFTransformerResidual

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FFTransformerResidual(
        n=10, m=20, d_model=256, num_heads=8,
        num_output_layers=2, dropout=0.0,
        initial_temperature=1.0, final_temperature=0.01
    )

    weights_path = project_root / "checkpoints" / "residual" / "best_model.pt"
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()

    return model


def max_util_allocation(valuations):
    """Assign each item to the agent who values it most."""
    n_agents, m_items = valuations.shape
    allocation = np.zeros((n_agents, m_items), dtype=np.float64)
    for j in range(m_items):
        best_agent = np.argmax(valuations[:, j])
        allocation[best_agent, j] = 1
    return allocation


def max_util_allocation_batch(valuations_batch):
    """Batch version of max_util_allocation."""
    batch_size, n_agents, m_items = valuations_batch.shape
    allocations = np.zeros((batch_size, n_agents, m_items), dtype=np.float64)
    for b in range(batch_size):
        allocations[b] = max_util_allocation(valuations_batch[b])
    return allocations


def evaluate_both_methods(model, matrices, nash_max, util_max, batch_size=100):
    """Evaluate both model+EF1 and max_util+EF1, return metrics for both."""
    # Model + EF1
    model_util_fractions = []
    model_nash_fractions = []
    # Max util + EF1
    maxutil_util_fractions = []
    maxutil_nash_fractions = []

    for i in range(0, len(matrices), batch_size):
        batch_end = min(i + batch_size, len(matrices))
        batch_matrices = matrices[i:batch_end]
        batch_nash_max = nash_max[i:batch_end]
        batch_util_max = util_max[i:batch_end]

        # Model + EF1
        model_allocs = get_model_allocations_batch(model, batch_matrices)
        model_allocs_ef1 = ef1_quick_repair_batch(
            model_allocs.astype(np.float64),
            batch_matrices.astype(np.float64),
            max_passes=10
        )
        model_values = calculate_agent_bundle_values_batch(batch_matrices, model_allocs_ef1)
        model_util_fractions.extend(utility_sum_batch(model_values) / batch_util_max)
        model_nash_fractions.extend(nash_welfare_batch(model_values) / batch_nash_max)

        # Max util + EF1
        maxutil_allocs = max_util_allocation_batch(batch_matrices)
        maxutil_allocs_ef1 = ef1_quick_repair_batch(
            maxutil_allocs,
            batch_matrices.astype(np.float64),
            max_passes=10
        )
        maxutil_values = calculate_agent_bundle_values_batch(batch_matrices, maxutil_allocs_ef1)
        maxutil_util_fractions.extend(utility_sum_batch(maxutil_values) / batch_util_max)
        maxutil_nash_fractions.extend(nash_welfare_batch(maxutil_values) / batch_nash_max)

    return (
        np.mean(model_util_fractions) * 100,
        np.mean(model_nash_fractions) * 100,
        np.mean(maxutil_util_fractions) * 100,
        np.mean(maxutil_nash_fractions) * 100
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default='datasets/heatmap')
    parser.add_argument('--n_min', type=int, default=10)
    parser.add_argument('--n_max', type=int, default=30)
    parser.add_argument('--m_min', type=int, default=10)
    parser.add_argument('--m_max', type=int, default=30)
    parser.add_argument('--num_samples', type=int, default=1000)
    parser.add_argument('--output', type=str, default='results/heatmap_comparison.json')
    args = parser.parse_args()

    print("Loading model...")
    model = load_residual_model()
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Build list of (n, m) pairs where m >= n
    configs = []
    for n in range(args.n_min, args.n_max + 1):
        for m in range(max(n, args.m_min), args.m_max + 1):
            configs.append((n, m))

    print(f"\nTotal configurations: {len(configs)}")

    results = {
        'n_range': [args.n_min, args.n_max],
        'm_range': [args.m_min, args.m_max],
        'num_samples': args.num_samples,
        'model_ef1': {'utility': {}, 'nash': {}},
        'maxutil_ef1': {'utility': {}, 'nash': {}},
        'diff': {'utility': {}, 'nash': {}}  # model_ef1 - maxutil_ef1
    }

    dataset_dir = Path(args.dataset_dir)

    for n, m in tqdm(configs, desc="Evaluating"):
        # Load dataset
        dataset_file = dataset_dir / f"{n}_{m}_{args.num_samples}_dataset.npz"
        if not dataset_file.exists():
            tqdm.write(f"Warning: Dataset not found: {dataset_file}")
            continue

        data = np.load(dataset_file)
        matrices = data['matrices']
        nash_max = data['nash_welfare']
        util_max = data['util_welfare']

        # Evaluate both methods
        model_util, model_nash, maxutil_util, maxutil_nash = evaluate_both_methods(
            model, matrices, nash_max, util_max
        )

        key = f"{n},{m}"
        results['model_ef1']['utility'][key] = model_util
        results['model_ef1']['nash'][key] = model_nash
        results['maxutil_ef1']['utility'][key] = maxutil_util
        results['maxutil_ef1']['nash'][key] = maxutil_nash
        results['diff']['utility'][key] = model_util - maxutil_util
        results['diff']['nash'][key] = model_nash - maxutil_nash

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")

    # Print summary
    diff_util = [v for v in results['diff']['utility'].values()]
    diff_nash = [v for v in results['diff']['nash'].values()]
    print(f"\nDifference (Model+EF1) - (MaxUtil+EF1):")
    print(f"  Utility: min={min(diff_util):.2f}%, max={max(diff_util):.2f}%, mean={np.mean(diff_util):.2f}%")
    print(f"  Nash: min={min(diff_nash):.2f}%, max={max(diff_nash):.2f}%, mean={np.mean(diff_nash):.2f}%")


if __name__ == "__main__":
    main()
