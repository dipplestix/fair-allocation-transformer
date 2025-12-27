#!/usr/bin/env python3
"""
Evaluate and compare Model+EF1 vs MaxUtil+EF1 vs RR+EF1 on 100x100 problems.
Since Gurobi can't handle 100x100 for optimal Nash, we compare methods directly.
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


def load_residual_model():
    """Load the residual FATransformer model."""
    from fatransformer.fatransformer_residual import FATransformer as FATransformerResidual

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FATransformerResidual(
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

    for i in tqdm(range(0, len(matrices), batch_size), desc=f"Evaluating {method}"):
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
    parser.add_argument('--n', type=int, default=100)
    parser.add_argument('--m', type=int, default=100)
    parser.add_argument('--num_samples', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    print(f"Evaluating on {args.n} agents, {args.m} items, {args.num_samples} samples")

    print("\nLoading model...")
    model = load_residual_model()
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

    print("\nGenerating valuation matrices...")
    matrices = generate_matrices(args.n, args.m, args.num_samples, args.seed)
    print(f"Generated {len(matrices)} matrices of shape {matrices[0].shape}")

    # Evaluate all methods
    print("\n" + "="*60)
    print("Evaluating Model + EF1...")
    model_utils, model_nash = evaluate_method(model, matrices, 'model', use_ef1=True)

    print("\nEvaluating MaxUtil + EF1...")
    maxutil_utils, maxutil_nash = evaluate_method(model, matrices, 'maxutil', use_ef1=True)

    print("\nEvaluating RR + EF1...")
    rr_utils, rr_nash = evaluate_method(model, matrices, 'rr', use_ef1=True)

    # Print results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)

    print("\nUtility (raw values):")
    print(f"  Model+EF1:   mean={np.mean(model_utils):.4f}, std={np.std(model_utils):.4f}")
    print(f"  MaxUtil+EF1: mean={np.mean(maxutil_utils):.4f}, std={np.std(maxutil_utils):.4f}")
    print(f"  RR+EF1:      mean={np.mean(rr_utils):.4f}, std={np.std(rr_utils):.4f}")

    print("\nNash Welfare (raw values):")
    print(f"  Model+EF1:   mean={np.mean(model_nash):.4f}, std={np.std(model_nash):.4f}")
    print(f"  MaxUtil+EF1: mean={np.mean(maxutil_nash):.4f}, std={np.std(maxutil_nash):.4f}")
    print(f"  RR+EF1:      mean={np.mean(rr_nash):.4f}, std={np.std(rr_nash):.4f}")

    # Compare Model to baselines
    print("\n" + "-"*60)
    print("Model+EF1 vs MaxUtil+EF1:")
    util_diff = (model_utils - maxutil_utils) / maxutil_utils * 100
    nash_diff = (model_nash - maxutil_nash) / maxutil_nash * 100
    print(f"  Utility: {np.mean(util_diff):+.2f}% (min={np.min(util_diff):+.2f}%, max={np.max(util_diff):+.2f}%)")
    print(f"  Nash:    {np.mean(nash_diff):+.2f}% (min={np.min(nash_diff):+.2f}%, max={np.max(nash_diff):+.2f}%)")
    print(f"  Model wins utility: {np.sum(model_utils > maxutil_utils)}/{len(model_utils)} ({100*np.mean(model_utils > maxutil_utils):.1f}%)")
    print(f"  Model wins Nash:    {np.sum(model_nash > maxutil_nash)}/{len(model_nash)} ({100*np.mean(model_nash > maxutil_nash):.1f}%)")

    print("\nModel+EF1 vs RR+EF1:")
    util_diff_rr = (model_utils - rr_utils) / rr_utils * 100
    nash_diff_rr = (model_nash - rr_nash) / rr_nash * 100
    print(f"  Utility: {np.mean(util_diff_rr):+.2f}% (min={np.min(util_diff_rr):+.2f}%, max={np.max(util_diff_rr):+.2f}%)")
    print(f"  Nash:    {np.mean(nash_diff_rr):+.2f}% (min={np.min(nash_diff_rr):+.2f}%, max={np.max(nash_diff_rr):+.2f}%)")
    print(f"  Model wins utility: {np.sum(model_utils > rr_utils)}/{len(model_utils)} ({100*np.mean(model_utils > rr_utils):.1f}%)")
    print(f"  Model wins Nash:    {np.sum(model_nash > rr_nash)}/{len(model_nash)} ({100*np.mean(model_nash > rr_nash):.1f}%)")


if __name__ == "__main__":
    main()
