#!/usr/bin/env python3
"""
Evaluate and compare multiple models including the large 30x60 model.

Compares:
- Original residual model (10x20, d_model=256) + EF1
- Small 30x60 model (d_model=128) + EF1
- Large 30x60 model (d_model=256) + EF1
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


def load_model(checkpoint_path, device, d_model):
    """Load a FFTransformerResidual model from checkpoint."""
    from fftransformer.fftransformer_residual import FFTransformerResidual

    model = FFTransformerResidual(
        n=10, m=20, d_model=d_model, num_heads=8,
        num_output_layers=2, dropout=0.0,
        initial_temperature=1.0, final_temperature=0.01
    )

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    return model


def max_util_allocation_batch(valuations_batch):
    batch_size, n_agents, m_items = valuations_batch.shape
    allocations = np.zeros_like(valuations_batch)
    for b in range(batch_size):
        for j in range(m_items):
            best_agent = np.argmax(valuations_batch[b, :, j])
            allocations[b, best_agent, j] = 1
    return allocations


def round_robin_allocation_batch(valuations_batch):
    batch_size, n_agents, m_items = valuations_batch.shape
    allocations = np.zeros_like(valuations_batch)
    for b in range(batch_size):
        for j in range(m_items):
            agent = j % n_agents
            allocations[b, agent, j] = 1
    return allocations


def generate_matrices(n_agents, m_items, num_samples, seed=42):
    np.random.seed(seed)
    return np.random.uniform(0, 1, size=(num_samples, n_agents, m_items))


def evaluate_method(model, matrices, method='model', use_ef1=True, batch_size=50):
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
    parser.add_argument('--n', type=int, default=50)
    parser.add_argument('--m', type=int, default=50)
    parser.add_argument('--num_samples', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=50)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Evaluating on {args.n} agents, {args.m} items, {args.num_samples} samples")
    print(f"Device: {device}")

    # Load models
    print("\nLoading models...")

    models = {}

    # 10x20 model (d_model=256)
    path_10_20 = project_root / "checkpoints" / "residual" / "best_model.pt"
    if path_10_20.exists():
        try:
            models['model_10_20'] = load_model(path_10_20, device, d_model=256)
            print(f"  10x20 model: {sum(p.numel() for p in models['model_10_20'].parameters()):,} params")
        except Exception as e:
            print(f"  Warning: Could not load 10x20 model: {e}")

    # 30x60 small model (d_model=128)
    path_30_60_small = project_root / "checkpoints" / "residual_30_60" / "best_model.pt"
    if path_30_60_small.exists():
        try:
            models['model_30_60_small'] = load_model(path_30_60_small, device, d_model=128)
            print(f"  30x60 small: {sum(p.numel() for p in models['model_30_60_small'].parameters()):,} params")
        except Exception as e:
            print(f"  Warning: Could not load 30x60 small model: {e}")

    # 30x60 large model (d_model=256)
    path_30_60_large = project_root / "checkpoints" / "residual_30_60_large" / "best_model.pt"
    if path_30_60_large.exists():
        try:
            models['model_30_60_large'] = load_model(path_30_60_large, device, d_model=256)
            print(f"  30x60 large: {sum(p.numel() for p in models['model_30_60_large'].parameters()):,} params")
        except Exception as e:
            print(f"  Warning: Could not load 30x60 large model: {e}")

    # Generate matrices
    print("\nGenerating valuation matrices...")
    matrices = generate_matrices(args.n, args.m, args.num_samples, args.seed)
    print(f"Generated {len(matrices)} matrices of shape {matrices[0].shape}")

    # Evaluate all methods
    results = {}

    print("\n" + "="*70)

    for name, model in models.items():
        print(f"Evaluating {name} + EF1...")
        results[name] = evaluate_method(model, matrices, 'model', use_ef1=True, batch_size=args.batch_size)

    print("Evaluating MaxUtil + EF1...")
    results['maxutil'] = evaluate_method(None, matrices, 'maxutil', use_ef1=True, batch_size=args.batch_size)

    print("Evaluating RR + EF1...")
    results['rr'] = evaluate_method(None, matrices, 'rr', use_ef1=True, batch_size=args.batch_size)

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

    # Compare all models to baselines
    for model_name in ['model_10_20', 'model_30_60_small', 'model_30_60_large']:
        if model_name in results:
            compare(model_name, 'maxutil')

    # Compare models to each other
    if 'model_30_60_large' in results:
        if 'model_10_20' in results:
            compare('model_30_60_large', 'model_10_20')
        if 'model_30_60_small' in results:
            compare('model_30_60_large', 'model_30_60_small')


if __name__ == "__main__":
    main()
