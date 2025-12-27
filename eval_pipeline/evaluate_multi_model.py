#!/usr/bin/env python3
"""
Evaluate and compare multiple models and baselines on various problem sizes.

Compares:
- Original residual model (trained on 10x20) + EF1
- New residual model (trained on 30x60) + EF1
- MaxUtil + EF1
- RR + EF1
"""

import argparse
import numpy as np
import torch
import sys
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


def load_model(checkpoint_path, device, d_model=128):
    """Load a FFTransformerResidual model from checkpoint."""
    from fftransformer.fftransformer_residual import FFTransformerResidual

    model = FFTransformerResidual(
        d_model=d_model, num_heads=8,
        num_output_layers=2, dropout=0.0,
        initial_temperature=1.0, final_temperature=0.01
    )

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
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


def round_robin_allocation(valuations):
    """Round robin allocation: agents take turns in order."""
    n_agents, m_items = valuations.shape
    allocation = np.zeros((n_agents, m_items), dtype=np.float64)
    for j in range(m_items):
        agent = j % n_agents
        allocation[agent, j] = 1
    return allocation


def max_util_allocation_batch(valuations_batch):
    batch_size = valuations_batch.shape[0]
    allocations = np.zeros_like(valuations_batch)
    for b in range(batch_size):
        allocations[b] = max_util_allocation(valuations_batch[b])
    return allocations


def round_robin_allocation_batch(valuations_batch):
    batch_size = valuations_batch.shape[0]
    allocations = np.zeros_like(valuations_batch)
    for b in range(batch_size):
        allocations[b] = round_robin_allocation(valuations_batch[b])
    return allocations


def generate_matrices(n_agents, m_items, num_samples, seed=42):
    """Generate random valuation matrices."""
    np.random.seed(seed)
    return np.random.uniform(0, 1, size=(num_samples, n_agents, m_items))


def evaluate_method(model, matrices, method='model', use_ef1=True, batch_size=50):
    """Evaluate a method and return utility and nash welfare."""
    all_utils = []
    all_nash = []

    for i in tqdm(range(0, len(matrices), batch_size), desc=f"Evaluating {method}", leave=False):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=50, help="Number of agents")
    parser.add_argument('--m', type=int, default=50, help="Number of items")
    parser.add_argument('--num_samples', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=50)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Evaluating on {args.n} agents, {args.m} items, {args.num_samples} samples")
    print(f"Device: {device}")

    # Load models
    print("\nLoading models...")

    model_10_20_path = project_root / "checkpoints" / "residual" / "best_model.pt"
    model_30_60_path = project_root / "checkpoints" / "residual_30_60" / "best_model.pt"

    model_10_20 = None
    model_30_60 = None

    if model_10_20_path.exists():
        try:
            model_10_20 = load_model(model_10_20_path, device, d_model=256)
            print(f"  Loaded 10x20 model: {sum(p.numel() for p in model_10_20.parameters()):,} params")
        except Exception as e:
            print(f"  Warning: Could not load 10x20 model: {e}")
    else:
        print(f"  Warning: 10x20 model not found at {model_10_20_path}")

    if model_30_60_path.exists():
        model_30_60 = load_model(model_30_60_path, device, d_model=128)
        print(f"  Loaded 30x60 model: {sum(p.numel() for p in model_30_60.parameters()):,} params")
    else:
        print(f"  Warning: 30x60 model not found at {model_30_60_path}")

    # Generate matrices
    print("\nGenerating valuation matrices...")
    matrices = generate_matrices(args.n, args.m, args.num_samples, args.seed)
    print(f"Generated {len(matrices)} matrices of shape {matrices[0].shape}")

    # Evaluate all methods
    results = {}

    print("\n" + "="*70)

    if model_10_20 is not None:
        print("Evaluating Model(10x20) + EF1...")
        results['model_10_20'] = evaluate_method(
            model_10_20, matrices, 'model', use_ef1=True, batch_size=args.batch_size
        )

    if model_30_60 is not None:
        print("Evaluating Model(30x60) + EF1...")
        results['model_30_60'] = evaluate_method(
            model_30_60, matrices, 'model', use_ef1=True, batch_size=args.batch_size
        )

    print("Evaluating MaxUtil + EF1...")
    results['maxutil'] = evaluate_method(
        None, matrices, 'maxutil', use_ef1=True, batch_size=args.batch_size
    )

    print("Evaluating RR + EF1...")
    results['rr'] = evaluate_method(
        None, matrices, 'rr', use_ef1=True, batch_size=args.batch_size
    )

    # Print results
    print("\n" + "="*70)
    print(f"RESULTS ({args.n}x{args.m}, {args.num_samples} samples)")
    print("="*70)

    print("\nRaw Values:")
    print(f"{'Method':<20} {'Utility (mean±std)':<25} {'Nash (mean±std)':<25}")
    print("-"*70)

    for name, (utils, nash) in results.items():
        print(f"{name:<20} {np.mean(utils):.4f} ± {np.std(utils):.4f}       {np.mean(nash):.4f} ± {np.std(nash):.4f}")

    # Comparisons
    print("\n" + "-"*70)
    print("Pairwise Comparisons (% difference):")
    print("-"*70)

    def compare(name1, name2):
        if name1 not in results or name2 not in results:
            return
        utils1, nash1 = results[name1]
        utils2, nash2 = results[name2]
        util_diff = (np.mean(utils1) - np.mean(utils2)) / np.mean(utils2) * 100
        nash_diff = (np.mean(nash1) - np.mean(nash2)) / np.mean(nash2) * 100
        win_util = np.mean(utils1 > utils2) * 100
        win_nash = np.mean(nash1 > nash2) * 100
        print(f"\n{name1} vs {name2}:")
        print(f"  Utility: {util_diff:+.2f}%  (wins {win_util:.1f}%)")
        print(f"  Nash:    {nash_diff:+.2f}%  (wins {win_nash:.1f}%)")

    # Compare models to baselines
    if 'model_10_20' in results:
        compare('model_10_20', 'maxutil')
        compare('model_10_20', 'rr')

    if 'model_30_60' in results:
        compare('model_30_60', 'maxutil')
        compare('model_30_60', 'rr')

    # Compare models to each other
    if 'model_10_20' in results and 'model_30_60' in results:
        compare('model_30_60', 'model_10_20')


if __name__ == "__main__":
    main()
