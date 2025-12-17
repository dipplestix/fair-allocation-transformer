#!/usr/bin/env python3
"""Test that model allocations are valid."""

import numpy as np
import sys
import torch
sys.path.insert(0, 'eval_pipeline')
from utils.load_model import load_model
from utils.inference import get_model_allocations_batch

def test_model_allocation():
    """Test that model generates valid allocations"""
    print("Testing model allocation generation")
    print("="*60)

    # Load model
    model, model_name = load_model('eval_pipeline/best_model_config.json')
    print(f"Loaded model: {model_name}")

    # Create a small batch of test matrices
    np.random.seed(42)
    N = 5
    n_agents, m_items = 10, 20

    valuation_matrices = np.random.uniform(0, 1, (N, n_agents, m_items))

    print(f"\nGenerating allocations for {N} matrices ({n_agents} agents, {m_items} items)...")

    # Get model allocations
    allocations = get_model_allocations_batch(model, valuation_matrices)

    print(f"Allocations shape: {allocations.shape}")
    print(f"Expected shape: ({N}, {n_agents}, {m_items})")

    # Verify allocations are valid
    for k in range(N):
        allocation = allocations[k]

        # Check that each item is assigned to exactly one agent
        col_sums = np.sum(allocation, axis=0)

        if not np.all(col_sums == 1):
            print(f"\n⚠ ERROR: Matrix {k} has invalid allocation!")
            print(f"Column sums (should all be 1): {col_sums}")
            print(f"Items assigned to 0 agents: {np.sum(col_sums == 0)}")
            print(f"Items assigned to >1 agents: {np.sum(col_sums > 1)}")

            # Print the problematic allocation
            print("\nAllocation matrix:")
            print(allocation)
        else:
            print(f"✓ Matrix {k}: valid allocation (each item assigned to exactly one agent)")

    print("\n" + "="*60)

    # Test with actual dataset
    print("\nTesting on actual dataset samples...")
    data = np.load('datasets/10_20_100000_dataset.npz')
    matrices = data['matrices'][:10]  # First 10 matrices

    allocations = get_model_allocations_batch(model, matrices)

    all_valid = True
    for k in range(len(matrices)):
        col_sums = np.sum(allocations[k], axis=0)
        if not np.all(col_sums == 1):
            print(f"⚠ ERROR: Dataset matrix {k} has invalid allocation!")
            all_valid = False
        else:
            print(f"✓ Dataset matrix {k}: valid")

    if all_valid:
        print("\n✓ All dataset allocations are valid!")

    print("\n" + "="*60)

if __name__ == '__main__':
    test_model_allocation()
