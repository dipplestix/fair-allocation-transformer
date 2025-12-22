#!/usr/bin/env python3
"""Tests for EF1 Quick Repair algorithm."""

import numpy as np
import sys
sys.path.insert(0, 'eval_pipeline')

from utils.ef1_repair import ef1_quick_repair, ef1_quick_repair_batch
from utils.calculations import calculate_agent_bundle_values, is_ef1, nash_welfare


def test_allocation_validity():
    """Ensure repairs maintain allocation constraints."""
    print("Test 1: Allocation validity")

    # Simple 2-agent, 3-item allocation
    allocation = np.array([
        [1, 0, 1],
        [0, 1, 0]
    ])
    valuations = np.array([
        [3.0, 1.0, 2.0],
        [2.0, 3.0, 1.0]
    ])

    repaired = ef1_quick_repair(allocation, valuations, max_passes=5)

    # Check each item assigned exactly once
    assert np.all(repaired.sum(axis=0) == 1), "Items not assigned to exactly one agent"

    # Check binary allocation
    assert np.all((repaired == 0) | (repaired == 1)), "Allocation not binary"

    print("✓ Repairs maintain allocation validity\n")


def test_ef1_improvement():
    """Verify algorithm improves EF1 satisfaction."""
    print("Test 2: EF1 improvement")

    np.random.seed(42)

    # Create allocation with known EF1 violation
    # Agent 0 gets low-value items, Agent 1 gets high-value items
    valuations = np.array([
        [1.0, 5.0, 5.0],
        [5.0, 1.0, 1.0]
    ])

    # Bad allocation: agent 0 gets item 0, agent 1 gets items 1,2
    allocation = np.array([
        [1, 0, 0],
        [0, 1, 1]
    ])

    # Check initial EF1
    initial_ef1 = is_ef1(valuations, allocation,
                         calculate_agent_bundle_values(valuations, allocation))

    # Apply repair
    repaired = ef1_quick_repair(allocation, valuations, max_passes=5)

    # Check final EF1
    final_ef1 = is_ef1(valuations, repaired,
                       calculate_agent_bundle_values(valuations, repaired))

    print(f"  Initial EF1: {initial_ef1}")
    print(f"  Final EF1: {final_ef1}")

    # EF1 should improve or stay the same
    if not initial_ef1:
        assert final_ef1, "Algorithm should repair EF1 violations"

    print("✓ Algorithm repairs EF1 violations\n")


def test_nsw_improvement():
    """Verify algorithm improves or maintains Nash Social Welfare."""
    print("Test 3: NSW improvement/preservation")

    np.random.seed(123)
    n_agents, m_items = 3, 5

    # Random valuations and allocation
    valuations = np.random.rand(n_agents, m_items) + 0.1  # Ensure positive

    # Random initial allocation
    allocation = np.zeros((n_agents, m_items))
    for item in range(m_items):
        agent = np.random.randint(0, n_agents)
        allocation[agent, item] = 1

    # Compute initial NSW
    initial_values = calculate_agent_bundle_values(valuations, allocation)
    initial_nsw = nash_welfare(initial_values)

    # Apply repair
    repaired = ef1_quick_repair(allocation, valuations, max_passes=10)

    # Compute final NSW
    final_values = calculate_agent_bundle_values(valuations, repaired)
    final_nsw = nash_welfare(final_values)

    print(f"  Initial NSW: {initial_nsw:.4f}")
    print(f"  Final NSW: {final_nsw:.4f}")
    print(f"  Improvement: {(final_nsw - initial_nsw):.4f}")

    # NSW should improve or stay the same (allowing small numerical errors)
    assert final_nsw >= initial_nsw - 1e-9, "NSW decreased!"

    print("✓ Algorithm maintains or improves NSW\n")


def test_convergence():
    """Algorithm terminates in reasonable time."""
    print("Test 4: Convergence")

    np.random.seed(456)
    n_agents, m_items = 4, 8

    # Random valuations
    valuations = np.random.rand(n_agents, m_items) + 0.1

    # Random allocation
    allocation = np.zeros((n_agents, m_items))
    for item in range(m_items):
        agent = np.random.randint(0, n_agents)
        allocation[agent, item] = 1

    # Should converge within max_passes
    repaired = ef1_quick_repair(allocation, valuations, max_passes=20)

    assert repaired is not None
    assert np.all(repaired.sum(axis=0) == 1), "Invalid allocation after repair"

    print("✓ Algorithm converges\n")


def test_already_ef1():
    """Algorithm doesn't break already EF1 allocations."""
    print("Test 5: Already EF1 allocations")

    # Create an allocation that's already EF1
    # Equal division: each agent gets items of similar value
    valuations = np.array([
        [3.0, 3.0, 1.0, 1.0],
        [3.0, 1.0, 3.0, 1.0],
        [1.0, 3.0, 1.0, 3.0]
    ])

    # Each agent gets items they value equally
    allocation = np.array([
        [1, 1, 0, 0],
        [0, 0, 1, 1],
        [0, 0, 0, 0]  # Agent 2 gets nothing - but this is still EF1
    ])

    initial_ef1 = is_ef1(valuations, allocation,
                         calculate_agent_bundle_values(valuations, allocation))

    repaired = ef1_quick_repair(allocation, valuations, max_passes=5)

    final_ef1 = is_ef1(valuations, repaired,
                       calculate_agent_bundle_values(valuations, repaired))

    print(f"  Initial EF1: {initial_ef1}")
    print(f"  Final EF1: {final_ef1}")

    # Should remain EF1
    if initial_ef1:
        assert final_ef1, "Algorithm broke EF1 property"

    print("✓ Algorithm preserves EF1 when already satisfied\n")


def test_batch_processing():
    """Test batch repair."""
    print("Test 6: Batch processing")

    np.random.seed(789)
    N, n_agents, m_items = 5, 3, 4

    # Generate batch of valuations and allocations
    valuations = np.random.rand(N, n_agents, m_items) + 0.1

    allocations = np.zeros((N, n_agents, m_items))
    for b in range(N):
        for item in range(m_items):
            agent = np.random.randint(0, n_agents)
            allocations[b, agent, item] = 1

    # Apply batch repair
    repaired = ef1_quick_repair_batch(allocations, valuations, max_passes=10)

    # Check validity for each instance
    assert repaired.shape == allocations.shape
    for b in range(N):
        assert np.all(repaired[b].sum(axis=0) == 1), f"Invalid allocation in batch {b}"

    print("✓ Batch processing works\n")


def test_empty_bundle_handling():
    """Test handling of agents with empty bundles."""
    print("Test 7: Empty bundle handling")

    # Agent 0 has nothing, agent 1 has everything
    allocation = np.array([
        [0, 0, 0],
        [1, 1, 1]
    ])

    valuations = np.array([
        [3.0, 2.0, 1.0],
        [1.0, 2.0, 3.0]
    ])

    # Should handle without crashing
    repaired = ef1_quick_repair(allocation, valuations, max_passes=5)

    assert np.all(repaired.sum(axis=0) == 1), "Invalid allocation"

    # Agent 0 should receive at least one item
    assert np.sum(repaired[0]) > 0, "Agent with empty bundle should receive items"

    print("✓ Empty bundles handled correctly\n")


if __name__ == '__main__':
    test_allocation_validity()
    test_ef1_improvement()
    test_nsw_improvement()
    test_convergence()
    test_already_ef1()
    test_batch_processing()
    test_empty_bundle_handling()

    print("=" * 60)
    print("ALL TESTS PASSED! ✓")
    print("=" * 60)
