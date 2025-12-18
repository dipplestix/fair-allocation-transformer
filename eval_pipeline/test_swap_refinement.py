#!/usr/bin/env python3
"""Tests for swap-based envy elimination."""

import numpy as np
import sys
sys.path.insert(0, 'eval_pipeline')

from utils.swap_refinement import (
    swap_bundles, compute_envy_matrix, swap_based_envy_elimination,
    swap_based_refinement_batch
)
from utils.calculations import calculate_agent_bundle_values


def test_swap_validity():
    """Ensure swaps maintain allocation constraints"""
    print("Test 1: Swap validity")
    allocation = np.array([[1, 0, 1], [0, 1, 0]])
    swapped = swap_bundles(allocation, 0, 1)

    # Each item assigned exactly once
    assert np.all(swapped.sum(axis=0) == 1), "Items not assigned to exactly one agent"
    # Bundles actually swapped
    assert np.array_equal(swapped[0], allocation[1])
    assert np.array_equal(swapped[1], allocation[0])
    print("✓ Swaps maintain validity\n")


def test_envy_computation():
    """Verify envy matrix calculation"""
    print("Test 2: Envy computation")
    valuations = np.array([[3.0, 1.0, 2.0], [2.0, 3.0, 1.0]])
    allocation = np.array([[1, 0, 1], [0, 1, 0]])

    envy = compute_envy_matrix(valuations, allocation)

    # Agent 0 has {0, 2} valued at 3+2=5
    # Agent 1 has {1} valued at 3
    # Agent 0 values agent 1's bundle at 1
    # Agent 1 values agent 0's bundle at 2+1=3
    # Both should have 0 envy
    assert envy[0][1] == 0, f"Expected 0, got {envy[0][1]}"
    assert envy[1][0] == 0, f"Expected 0, got {envy[1][0]}"
    print("✓ Envy computation correct\n")


def test_envy_reduction():
    """Verify algorithm reduces envy"""
    print("Test 3: Envy reduction")
    np.random.seed(42)
    valuations = np.array([
        [4.0, 1.0, 2.0],
        [1.0, 4.0, 2.0],
        [2.0, 2.0, 4.0]
    ])
    # Bad allocation: agent 0 gets low-value items
    allocation = np.array([
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ])

    initial_envy = np.sum(compute_envy_matrix(valuations, allocation))
    refined = swap_based_envy_elimination(allocation, valuations, max_iterations=50)
    final_envy = np.sum(compute_envy_matrix(valuations, refined))

    print(f"  Initial envy: {initial_envy:.3f}")
    print(f"  Final envy: {final_envy:.3f}")
    assert final_envy <= initial_envy + 1e-6, "Envy increased!"
    print("✓ Algorithm reduces or maintains envy\n")


def test_convergence():
    """Algorithm terminates in reasonable time"""
    print("Test 4: Convergence")
    np.random.seed(123)
    valuations = np.random.rand(4, 6)
    # Random allocation
    allocation = np.zeros((4, 6))
    for item in range(6):
        agent = np.random.randint(0, 4)
        allocation[agent][item] = 1

    refined = swap_based_envy_elimination(
        allocation, valuations, max_iterations=100
    )

    assert refined is not None
    assert np.all(refined.sum(axis=0) == 1), "Invalid allocation after refinement"
    print("✓ Algorithm converges\n")


def test_batch_processing():
    """Test batch refinement"""
    print("Test 5: Batch processing")
    np.random.seed(456)
    N, n_agents, m_items = 5, 3, 4

    valuations = np.random.rand(N, n_agents, m_items)
    allocations = np.zeros((N, n_agents, m_items))
    for b in range(N):
        for item in range(m_items):
            agent = np.random.randint(0, n_agents)
            allocations[b][agent][item] = 1

    refined = swap_based_refinement_batch(allocations, valuations, max_iterations=50)

    # Check validity
    assert refined.shape == allocations.shape
    for b in range(N):
        assert np.all(refined[b].sum(axis=0) == 1), f"Invalid allocation in batch {b}"

    print("✓ Batch processing works\n")


if __name__ == '__main__':
    test_swap_validity()
    test_envy_computation()
    test_envy_reduction()
    test_convergence()
    test_batch_processing()

    print("="*60)
    print("ALL TESTS PASSED! ✓")
    print("="*60)
