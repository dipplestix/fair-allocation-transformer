#!/usr/bin/env python3
"""Test script to verify metric calculations are correct."""

import numpy as np
import sys
sys.path.insert(0, 'eval_pipeline')
from utils.calculations import (
    calculate_agent_bundle_values, calculate_agent_bundle_values_batch,
    is_envy_free, is_envy_free_batch,
    is_ef1, is_ef1_batch,
    is_efx, is_efx_batch,
    utility_sum, utility_sum_batch,
    nash_welfare, nash_welfare_batch
)

def test_simple_case():
    """Test with a simple 2-agent, 3-item case"""
    print("Test 1: Simple 2-agent, 3-item case")
    print("="*60)

    # Valuation matrix: Agent 0 highly values item 0 and 2, Agent 1 highly values item 1
    valuation_matrix = np.array([
        [3.0, 1.0, 2.0],  # Agent 0's valuations
        [1.0, 3.0, 2.0]   # Agent 1's valuations
    ])

    # Allocation: Agent 0 gets items 0 and 2, Agent 1 gets item 1
    allocation_matrix = np.array([
        [1, 0, 1],  # Agent 0's bundle
        [0, 1, 0]   # Agent 1's bundle
    ])

    print("Valuation matrix:")
    print(valuation_matrix)
    print("\nAllocation matrix:")
    print(allocation_matrix)

    # Calculate bundle values
    bundle_values = calculate_agent_bundle_values(valuation_matrix, allocation_matrix)
    print("\nBundle values (row i = how much agent i values each bundle):")
    print(bundle_values)
    print(f"Agent 0 values own bundle: {bundle_values[0][0]}")
    print(f"Agent 0 values agent 1's bundle: {bundle_values[0][1]}")
    print(f"Agent 1 values own bundle: {bundle_values[1][1]}")
    print(f"Agent 1 values agent 0's bundle: {bundle_values[1][0]}")

    # Expected: Agent 0 gets 3+2=5, Agent 1 gets 3
    assert bundle_values[0][0] == 5.0, f"Expected 5.0, got {bundle_values[0][0]}"
    assert bundle_values[1][1] == 3.0, f"Expected 3.0, got {bundle_values[1][1]}"

    # Check fairness
    ef = is_envy_free(bundle_values)
    ef1_result = is_ef1(valuation_matrix, allocation_matrix, bundle_values)
    efx_result = is_efx(valuation_matrix, allocation_matrix, bundle_values)

    print(f"\nEnvy-free: {ef}")
    print(f"EF1: {ef1_result}")
    print(f"EFx: {efx_result}")

    # Check welfare
    util = utility_sum(bundle_values)
    nash = nash_welfare(bundle_values)

    print(f"\nUtility sum: {util}")
    print(f"Nash welfare: {nash}")
    print(f"Expected Nash welfare: {np.sqrt(5*3):.4f}")

    assert abs(nash - np.sqrt(15)) < 1e-6, f"Nash welfare mismatch"

    print("\n✓ Test 1 passed!\n")

def test_unfair_allocation():
    """Test with an unfair allocation"""
    print("Test 2: Unfair allocation (one agent gets everything)")
    print("="*60)

    valuation_matrix = np.array([
        [3.0, 2.0, 1.0],
        [2.0, 3.0, 1.0]
    ])

    # Agent 0 gets all items, Agent 1 gets nothing
    allocation_matrix = np.array([
        [1, 1, 1],
        [0, 0, 0]
    ])

    print("Allocation: Agent 0 gets all items, Agent 1 gets nothing")

    bundle_values = calculate_agent_bundle_values(valuation_matrix, allocation_matrix)

    ef = is_envy_free(bundle_values)
    ef1_result = is_ef1(valuation_matrix, allocation_matrix, bundle_values)
    efx_result = is_efx(valuation_matrix, allocation_matrix, bundle_values)

    print(f"Envy-free: {ef}")
    print(f"EF1: {ef1_result}")
    print(f"EFx: {efx_result}")

    # Should all be False (unfair allocation)
    assert ef == False, "Should not be envy-free"
    assert ef1_result == False, "Should not be EF1"
    assert efx_result == False, "Should not be EFx"

    print("\n✓ Test 2 passed!\n")

def test_batch_operations():
    """Test batch operations match single operations"""
    print("Test 3: Batch operations consistency")
    print("="*60)

    # Create 3 random test cases
    np.random.seed(42)
    N = 3
    m, n = 4, 6

    valuation_matrices = np.random.uniform(0, 1, (N, m, n))

    # Create random valid allocations
    allocation_matrices = np.zeros((N, m, n), dtype=int)
    for k in range(N):
        for j in range(n):
            i = np.random.randint(0, m)
            allocation_matrices[k, i, j] = 1

    # Test bundle values
    bundle_values_batch = calculate_agent_bundle_values_batch(valuation_matrices, allocation_matrices)
    for k in range(N):
        bundle_values_single = calculate_agent_bundle_values(valuation_matrices[k], allocation_matrices[k])
        assert np.allclose(bundle_values_batch[k], bundle_values_single), \
            f"Bundle values mismatch for matrix {k}"
    print("✓ Bundle values batch matches single")

    # Test envy-free
    ef_batch = is_envy_free_batch(bundle_values_batch)
    for k in range(N):
        ef_single = is_envy_free(bundle_values_batch[k])
        assert ef_batch[k] == ef_single, f"EF mismatch for matrix {k}: {ef_batch[k]} vs {ef_single}"
    print("✓ Envy-free batch matches single")

    # Test EF1
    ef1_batch = is_ef1_batch(valuation_matrices, allocation_matrices, bundle_values_batch)
    for k in range(N):
        ef1_single = is_ef1(valuation_matrices[k], allocation_matrices[k], bundle_values_batch[k])
        assert ef1_batch[k] == ef1_single, f"EF1 mismatch for matrix {k}: {ef1_batch[k]} vs {ef1_single}"
    print("✓ EF1 batch matches single")

    # Test EFx
    efx_batch = is_efx_batch(valuation_matrices, allocation_matrices, bundle_values_batch)
    for k in range(N):
        efx_single = is_efx(valuation_matrices[k], allocation_matrices[k], bundle_values_batch[k])
        assert efx_batch[k] == efx_single, f"EFx mismatch for matrix {k}: {efx_batch[k]} vs {efx_single}"
    print("✓ EFx batch matches single")

    # Test utility sum
    util_batch = utility_sum_batch(bundle_values_batch)
    for k in range(N):
        util_single = utility_sum(bundle_values_batch[k])
        assert np.isclose(util_batch[k], util_single), f"Utility sum mismatch for matrix {k}"
    print("✓ Utility sum batch matches single")

    # Test Nash welfare
    nash_batch = nash_welfare_batch(bundle_values_batch)
    for k in range(N):
        nash_single = nash_welfare(bundle_values_batch[k])
        assert np.isclose(nash_batch[k], nash_single), \
            f"Nash welfare mismatch for matrix {k}: {nash_batch[k]} vs {nash_single}"
    print("✓ Nash welfare batch matches single")

    print("\n✓ Test 3 passed!\n")

def test_allocation_validity():
    """Test that allocations are valid (each item to exactly one agent)"""
    print("Test 4: Allocation validity check")
    print("="*60)

    from utils.inference import get_random_allocations_batch, get_rr_allocations_batch_old

    np.random.seed(42)
    N = 10
    m, n = 10, 15

    valuation_matrices = np.random.uniform(0, 1, (N, m, n))

    # Test random allocations
    random_allocs = get_random_allocations_batch(valuation_matrices)
    print(f"Random allocations shape: {random_allocs.shape}")
    print(f"Expected shape: ({N*5}, {m}, {n})")

    for k in range(random_allocs.shape[0]):
        # Check each item assigned to exactly one agent
        col_sums = np.sum(random_allocs[k], axis=0)
        assert np.all(col_sums == 1), f"Random allocation {k} invalid: items not assigned to exactly one agent"
    print("✓ All random allocations valid")

    # Test RR allocations
    rr_allocs = get_rr_allocations_batch_old(valuation_matrices)
    print(f"RR allocations shape: {rr_allocs.shape}")
    print(f"Expected shape: ({N}, {m}, {n})")

    for k in range(rr_allocs.shape[0]):
        col_sums = np.sum(rr_allocs[k], axis=0)
        assert np.all(col_sums == 1), f"RR allocation {k} invalid: items not assigned to exactly one agent"
    print("✓ All RR allocations valid")

    print("\n✓ Test 4 passed!\n")

def test_ef1_definition():
    """Test EF1 definition carefully"""
    print("Test 5: EF1 definition verification")
    print("="*60)

    # Create a case that is EF1 but not EF
    valuation_matrix = np.array([
        [4.0, 1.0, 1.0],  # Agent 0
        [1.0, 4.0, 1.0]   # Agent 1
    ])

    # Agent 0 gets item 0, Agent 1 gets items 1 and 2
    allocation_matrix = np.array([
        [1, 0, 0],
        [0, 1, 1]
    ])

    print("Valuation matrix:")
    print(valuation_matrix)
    print("\nAllocation: Agent 0 gets item 0, Agent 1 gets items 1,2")

    bundle_values = calculate_agent_bundle_values(valuation_matrix, allocation_matrix)
    print("\nBundle values:")
    print(f"Agent 0 values own bundle: {bundle_values[0][0]} (item 0)")
    print(f"Agent 0 values agent 1's bundle: {bundle_values[0][1]} (items 1,2)")
    print(f"Agent 1 values own bundle: {bundle_values[1][1]} (items 1,2)")
    print(f"Agent 1 values agent 0's bundle: {bundle_values[1][0]} (item 0)")

    ef = is_envy_free(bundle_values)
    ef1_result = is_ef1(valuation_matrix, allocation_matrix, bundle_values)

    print(f"\nEnvy-free: {ef}")
    print(f"EF1: {ef1_result}")

    # Check manually:
    # Agent 0: values own bundle = 4, values agent 1's bundle = 1+1 = 2
    # Agent 0 doesn't envy agent 1 (4 >= 2), so OK

    # Agent 1: values own bundle = 4+1 = 5, values agent 0's bundle = 1
    # Agent 1 doesn't envy agent 0 (5 >= 1), so OK

    # This should be envy-free!
    assert ef == True, "This allocation should be envy-free"
    assert ef1_result == True, "This allocation should be EF1"

    print("\n✓ Test 5 passed!\n")

def test_ece_allocation_validity():
    """Test that ECE allocations are valid"""
    print("Test 6: ECE allocation validity check")
    print("="*60)

    from utils.inference import get_ece_allocations_batch

    np.random.seed(42)
    N = 10
    m, n = 10, 15

    valuation_matrices = np.random.uniform(0, 1, (N, m, n))

    ece_allocs = get_ece_allocations_batch(valuation_matrices)
    print(f"ECE allocations shape: {ece_allocs.shape}")
    print(f"Expected shape: ({N}, {m}, {n})")

    for k in range(ece_allocs.shape[0]):
        col_sums = np.sum(ece_allocs[k], axis=0)
        assert np.all(col_sums == 1), \
            f"ECE allocation {k} invalid: items not assigned to exactly one agent"
    print("✓ All ECE allocations valid")

    print("\n✓ Test 6 passed!\n")

def test_ece_ef1_guarantee():
    """Test that ECE produces EF1 allocations (theoretically guaranteed)"""
    print("Test 7: ECE EF1 guarantee")
    print("="*60)

    from utils.inference import get_ece_allocation

    np.random.seed(123)
    # Test multiple random instances
    for trial in range(20):
        n_agents = np.random.randint(2, 8)
        m_items = np.random.randint(n_agents, n_agents * 3)

        valuation_matrix = np.random.uniform(0, 1, (n_agents, m_items))
        allocation = get_ece_allocation(valuation_matrix)

        # Check EF1
        bundle_values = calculate_agent_bundle_values(valuation_matrix, allocation)
        ef1_result = is_ef1(valuation_matrix, allocation, bundle_values)

        if not ef1_result:
            print(f"\nTrial {trial} FAILED:")
            print(f"Agents: {n_agents}, Items: {m_items}")
            assert False, "ECE should always produce EF1 allocations"

    print(f"✓ All 20 trials produced EF1 allocations")
    print("\n✓ Test 7 passed!\n")

if __name__ == '__main__':
    test_simple_case()
    test_unfair_allocation()
    test_batch_operations()
    test_allocation_validity()
    test_ef1_definition()
    test_ece_allocation_validity()
    test_ece_ef1_guarantee()

    print("="*60)
    print("ALL TESTS PASSED! ✓")
    print("="*60)
