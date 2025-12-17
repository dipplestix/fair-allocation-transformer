#!/usr/bin/env python3
"""Test that dataset max welfare values are computed correctly."""

import numpy as np
import sys
sys.path.insert(0, 'eval_pipeline')
from utils.calculations import calculate_agent_bundle_values, utility_sum, nash_welfare
from utils.max_utility import best_nash_welfare, best_utilitarian_welfare

def test_dataset_sample():
    """Load a dataset and verify max welfare values for a few samples"""
    print("Testing dataset max welfare calculations")
    print("="*60)

    # Load a small dataset
    data = np.load('datasets/10_10_100000_dataset.npz')
    matrices = data['matrices']
    nash_welfare_max = data['nash_welfare']
    util_welfare_max = data['util_welfare']

    print(f"Dataset shape: {matrices.shape}")
    print(f"Testing first 3 matrices...\n")

    for i in range(min(3, len(matrices))):
        print(f"Matrix {i}:")
        print("-" * 60)

        valuation_matrix = matrices[i]

        # Recompute max util welfare
        computed_util_max = best_utilitarian_welfare(valuation_matrix)
        stored_util_max = util_welfare_max[i]

        print(f"Stored max util welfare: {stored_util_max:.6f}")
        print(f"Computed max util welfare: {computed_util_max:.6f}")

        if not np.isclose(computed_util_max, stored_util_max, rtol=1e-5):
            print(f"⚠ WARNING: Mismatch in max util welfare!")
        else:
            print("✓ Max util welfare matches")

        # Recompute max Nash welfare (this takes longer)
        print("Computing max Nash welfare...")
        computed_nash_max = best_nash_welfare(valuation_matrix, num_segments=200)
        stored_nash_max = nash_welfare_max[i]

        print(f"Stored max Nash welfare: {stored_nash_max:.6f}")
        print(f"Computed max Nash welfare: {computed_nash_max:.6f}")

        # Allow for approximation error in Nash welfare
        if not np.isclose(computed_nash_max, stored_nash_max, rtol=0.01):
            print(f"⚠ WARNING: Mismatch in max Nash welfare!")
            print(f"Relative error: {abs(computed_nash_max - stored_nash_max) / stored_nash_max * 100:.2f}%")
        else:
            print("✓ Max Nash welfare matches")

        print()

    print("="*60)
    print("Dataset verification complete!")

def test_best_utilitarian_calculation():
    """Verify best utilitarian welfare is computed correctly"""
    print("\nTesting best utilitarian welfare calculation")
    print("="*60)

    # Create a simple case
    valuation_matrix = np.array([
        [5.0, 1.0, 2.0],  # Agent 0
        [2.0, 6.0, 1.0],  # Agent 1
        [1.0, 2.0, 7.0]   # Agent 2
    ])

    print("Valuation matrix:")
    print(valuation_matrix)

    # Best allocation: Agent 0 gets item 0 (val=5), Agent 1 gets item 1 (val=6), Agent 2 gets item 2 (val=7)
    # Total utility = 5 + 6 + 7 = 18

    computed = best_utilitarian_welfare(valuation_matrix)
    expected = 5.0 + 6.0 + 7.0

    print(f"\nExpected best util welfare: {expected}")
    print(f"Computed best util welfare: {computed}")

    assert np.isclose(computed, expected), f"Mismatch: {computed} != {expected}"
    print("✓ Best utilitarian welfare correct")

    # Verify manually
    manual = np.sum(np.max(valuation_matrix, axis=0))
    print(f"Manual calculation (sum of column maxes): {manual}")
    assert np.isclose(manual, expected), f"Manual calculation mismatch"

    print("\n✓ Test passed!\n")

if __name__ == '__main__':
    test_best_utilitarian_calculation()
    test_dataset_sample()
