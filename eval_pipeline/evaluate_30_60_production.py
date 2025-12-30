#!/usr/bin/env python3
"""
Evaluate the production 30x60 model on various problem sizes.

Runs:
1. Heatmap evaluation (n=10-30, m=10-30)
2. Comparison at 50x50, 100x100, 50x100
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

from fftransformer.fftransformer_residual import FFTransformerResidual
from utils.calculations import (
    calculate_agent_bundle_values_batch,
    utility_sum_batch,
    nash_welfare_batch
)
from utils.inference import get_model_allocations_batch
from utils.ef1_repair import ef1_quick_repair_batch


# Pool configs (must match training)
POOL_CONFIGS = {
    "row_only": {'row': ['mean', 'max', 'min'], 'column': [], 'global': []},
    "row_col": {'row': ['mean', 'max', 'min'], 'column': ['mean', 'max', 'min'], 'global': []},
    "row_global": {'row': ['mean', 'max', 'min'], 'column': [], 'global': ['mean', 'max', 'min']},
    "all": {'row': ['mean', 'max', 'min'], 'column': ['mean', 'max', 'min'], 'global': ['mean', 'max', 'min']},
    "row_col_mean": {'row': 'mean', 'column': 'mean', 'global': []},
}


def load_30_60_production_model():
    """Load the production 30x60 model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Architecture from best_30_60_sweep.yaml
    # Note: pool_config is hardcoded in FFTransformerResidual (row_col)
    model = FFTransformerResidual(
        d_model=512,
        num_heads=4,
        num_output_layers=3,
        num_encoder_layers=3,
        dropout=0.136,
        initial_temperature=0.73,
        final_temperature=0.051,
    )

    weights_path = project_root / "checkpoints" / "residual" / "from_sweep_s5x3r4a5" / "best_model.pt"
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    return model


def max_util_allocation_batch(valuations_batch):
    """Assign each item to the agent who values it most."""
    batch_size, n_agents, m_items = valuations_batch.shape
    allocations = np.zeros_like(valuations_batch)
    for b in range(batch_size):
        for j in range(m_items):
            best_agent = np.argmax(valuations_batch[b, :, j])
            allocations[b, best_agent, j] = 1
    return allocations


def round_robin_allocation_batch(valuations_batch):
    """Round robin allocation."""
    batch_size, n_agents, m_items = valuations_batch.shape
    allocations = np.zeros_like(valuations_batch)
    for b in range(batch_size):
        for j in range(m_items):
            agent = j % n_agents
            allocations[b, agent, j] = 1
    return allocations


def generate_matrices(n_agents, m_items, num_samples, seed=42):
    """Generate random valuation matrices."""
    np.random.seed(seed)
    return np.random.uniform(0, 1, size=(num_samples, n_agents, m_items)).astype(np.float32)


def evaluate_method(model, matrices, method='model', use_ef1=True, batch_size=50):
    """Evaluate a method on the given matrices."""
    all_utils = []
    all_nash = []

    for i in range(0, len(matrices), batch_size):
        batch_end = min(i + batch_size, len(matrices))
        batch_matrices = matrices[i:batch_end]

        if method == 'model':
            allocations = get_model_allocations_batch(model, batch_matrices)
        elif method == 'maxutil':
            allocations = max_util_allocation_batch(batch_matrices)
        elif method == 'rr':
            allocations = round_robin_allocation_batch(batch_matrices)
        else:
            raise ValueError(f"Unknown method: {method}")

        if use_ef1:
            allocations = ef1_quick_repair_batch(
                allocations.astype(np.float64),
                batch_matrices.astype(np.float64),
                max_passes=10
            )

        values = calculate_agent_bundle_values_batch(batch_matrices, allocations)
        all_utils.extend(utility_sum_batch(values))
        all_nash.extend(nash_welfare_batch(values))

    return np.array(all_utils), np.array(all_nash)


def run_comparison(model, n, m, num_samples=1000, batch_size=50, seed=42):
    """Run comparison at a specific problem size."""
    print(f"\n{'='*70}")
    print(f"Evaluating at {n} agents x {m} items ({num_samples} samples)")
    print('='*70)

    matrices = generate_matrices(n, m, num_samples, seed)

    results = {}

    print("  Model + EF1...")
    results['model'] = evaluate_method(model, matrices, 'model', use_ef1=True, batch_size=batch_size)

    print("  MaxUtil + EF1...")
    results['maxutil'] = evaluate_method(None, matrices, 'maxutil', use_ef1=True, batch_size=batch_size)

    print("  RR + EF1...")
    results['rr'] = evaluate_method(None, matrices, 'rr', use_ef1=True, batch_size=batch_size)

    # Print results
    print(f"\n{'Method':<15} {'Utility (mean±std)':<25} {'Nash (mean±std)':<25}")
    print("-"*65)

    for name, (utils, nash) in results.items():
        print(f"{name:<15} {np.mean(utils):.4f} ± {np.std(utils):.4f}       {np.mean(nash):.4f} ± {np.std(nash):.4f}")

    # Comparisons
    print("\nModel vs Baselines:")
    for baseline in ['maxutil', 'rr']:
        model_utils, model_nash = results['model']
        base_utils, base_nash = results[baseline]
        util_diff = (np.mean(model_utils) - np.mean(base_utils)) / np.mean(base_utils) * 100
        nash_diff = (np.mean(model_nash) - np.mean(base_nash)) / np.mean(base_nash) * 100
        win_util = np.mean(model_utils > base_utils) * 100
        win_nash = np.mean(model_nash > base_nash) * 100
        print(f"  vs {baseline}: Utility {util_diff:+.2f}% (wins {win_util:.1f}%), Nash {nash_diff:+.2f}% (wins {win_nash:.1f}%)")

    return results


def run_heatmap(model, output_path, n_min=10, n_max=30, m_min=10, m_max=30,
                num_samples=1000, batch_size=100, dataset_dir='datasets/heatmap'):
    """Run heatmap evaluation across a range of sizes using precomputed datasets."""
    print(f"\n{'='*70}")
    print(f"Running heatmap evaluation: n=[{n_min},{n_max}], m=[{m_min},{m_max}]")
    print('='*70)

    dataset_dir = Path(dataset_dir)

    # Build list of (n, m) pairs where m >= n
    configs = []
    for n in range(n_min, n_max + 1):
        for m in range(max(n, m_min), m_max + 1):
            configs.append((n, m))

    print(f"Total configurations: {len(configs)}")

    results = {
        'n_range': [n_min, n_max],
        'm_range': [m_min, m_max],
        'num_samples': num_samples,
        'model_name': 'production_30_60',
        'model_ef1': {'utility': {}, 'nash': {}},
        'maxutil_ef1': {'utility': {}, 'nash': {}},
        'rr_ef1': {'utility': {}, 'nash': {}},
        'diff_vs_maxutil': {'utility': {}, 'nash': {}},
        'diff_vs_rr': {'utility': {}, 'nash': {}}
    }

    for n, m in tqdm(configs, desc="Evaluating heatmap"):
        # Load precomputed dataset with optimal values
        dataset_file = dataset_dir / f"{n}_{m}_{num_samples}_dataset.npz"
        if not dataset_file.exists():
            tqdm.write(f"Warning: Dataset not found: {dataset_file}")
            continue

        data = np.load(dataset_file)
        matrices = data['matrices']
        nash_max = data['nash_welfare']  # Optimal nash for each sample
        util_max = data['util_welfare']  # Optimal utility for each sample

        # Model + EF1
        model_allocs = get_model_allocations_batch(model, matrices)
        model_allocs_ef1 = ef1_quick_repair_batch(
            model_allocs.astype(np.float64),
            matrices.astype(np.float64),
            max_passes=10
        )
        model_values = calculate_agent_bundle_values_batch(matrices, model_allocs_ef1)
        # Compute % of optimal for each sample, then average
        model_util_pct = np.mean(utility_sum_batch(model_values) / util_max) * 100
        model_nash_pct = np.mean(nash_welfare_batch(model_values) / nash_max) * 100

        # MaxUtil + EF1
        maxutil_allocs = max_util_allocation_batch(matrices)
        maxutil_allocs_ef1 = ef1_quick_repair_batch(
            maxutil_allocs,
            matrices.astype(np.float64),
            max_passes=10
        )
        maxutil_values = calculate_agent_bundle_values_batch(matrices, maxutil_allocs_ef1)
        maxutil_util_pct = np.mean(utility_sum_batch(maxutil_values) / util_max) * 100
        maxutil_nash_pct = np.mean(nash_welfare_batch(maxutil_values) / nash_max) * 100

        # RR + EF1
        rr_allocs = round_robin_allocation_batch(matrices)
        rr_allocs_ef1 = ef1_quick_repair_batch(
            rr_allocs,
            matrices.astype(np.float64),
            max_passes=10
        )
        rr_values = calculate_agent_bundle_values_batch(matrices, rr_allocs_ef1)
        rr_util_pct = np.mean(utility_sum_batch(rr_values) / util_max) * 100
        rr_nash_pct = np.mean(nash_welfare_batch(rr_values) / nash_max) * 100

        key = f"{n},{m}"
        # Store as % of optimal
        results['model_ef1']['utility'][key] = model_util_pct
        results['model_ef1']['nash'][key] = model_nash_pct
        results['maxutil_ef1']['utility'][key] = maxutil_util_pct
        results['maxutil_ef1']['nash'][key] = maxutil_nash_pct
        results['rr_ef1']['utility'][key] = rr_util_pct
        results['rr_ef1']['nash'][key] = rr_nash_pct
        # Percentage point differences (consistent with other scripts)
        results['diff_vs_maxutil']['utility'][key] = model_util_pct - maxutil_util_pct
        results['diff_vs_maxutil']['nash'][key] = model_nash_pct - maxutil_nash_pct
        results['diff_vs_rr']['utility'][key] = model_util_pct - rr_util_pct
        results['diff_vs_rr']['nash'][key] = model_nash_pct - rr_nash_pct

    # Save results
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nHeatmap results saved to {output_path}")

    # Print summary
    diff_maxutil_util = list(results['diff_vs_maxutil']['utility'].values())
    diff_maxutil_nash = list(results['diff_vs_maxutil']['nash'].values())
    diff_rr_util = list(results['diff_vs_rr']['utility'].values())
    diff_rr_nash = list(results['diff_vs_rr']['nash'].values())

    print(f"\nModel vs MaxUtil+EF1 (percentage points of optimal):")
    print(f"  Utility: min={min(diff_maxutil_util):.2f}, max={max(diff_maxutil_util):.2f}, mean={np.mean(diff_maxutil_util):.2f}")
    print(f"  Nash: min={min(diff_maxutil_nash):.2f}, max={max(diff_maxutil_nash):.2f}, mean={np.mean(diff_maxutil_nash):.2f}")

    print(f"\nModel vs RR+EF1 (percentage points of optimal):")
    print(f"  Utility: min={min(diff_rr_util):.2f}, max={max(diff_rr_util):.2f}, mean={np.mean(diff_rr_util):.2f}")
    print(f"  Nash: min={min(diff_rr_nash):.2f}, max={max(diff_rr_nash):.2f}, mean={np.mean(diff_rr_nash):.2f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate production 30x60 model")
    parser.add_argument('--heatmap', action='store_true', help='Run heatmap evaluation')
    parser.add_argument('--compare', nargs='+', type=str,
                       help='Run comparisons (e.g., --compare 50x50 100x100 50x100)')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of samples')
    parser.add_argument('--batch_size', type=int, default=50, help='Batch size')
    parser.add_argument('--heatmap_output', type=str,
                       default='results/heatmaps/heatmap_production_30_60.json')
    args = parser.parse_args()

    print("Loading production 30x60 model...")
    model = load_30_60_production_model()
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

    if args.heatmap:
        run_heatmap(model, args.heatmap_output, num_samples=args.num_samples,
                   batch_size=args.batch_size)

    if args.compare:
        for size_str in args.compare:
            parts = size_str.lower().split('x')
            if len(parts) == 2:
                n, m = int(parts[0]), int(parts[1])
                run_comparison(model, n, m, num_samples=args.num_samples,
                             batch_size=args.batch_size)
            else:
                print(f"Invalid size format: {size_str}. Use NxM format (e.g., 50x50)")

    if not args.heatmap and not args.compare:
        # Default: run all
        run_heatmap(model, args.heatmap_output, num_samples=args.num_samples,
                   batch_size=args.batch_size)
        for n, m in [(50, 50), (100, 100), (50, 100)]:
            run_comparison(model, n, m, num_samples=args.num_samples,
                         batch_size=args.batch_size)


if __name__ == "__main__":
    main()
