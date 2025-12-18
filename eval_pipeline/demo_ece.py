#!/usr/bin/env python3
"""
Step-by-step demonstration of Envy Cycle Elimination (ECE) algorithm.
Shows how the algorithm maintains EF1 throughout execution.
"""

import numpy as np
import sys
sys.path.insert(0, 'eval_pipeline')

from utils.inference import build_envy_graph, detect_and_remove_cycle
from utils.calculations import (
    calculate_agent_bundle_values,
    is_envy_free,
    is_ef1,
    is_efx,
    nash_welfare,
    utility_sum
)


def visualize_envy_graph(envy_graph, n_agents):
    """Print ASCII representation of envy graph."""
    print("  Envy Graph (i→j means i envies j):")
    has_envy = False
    for agent in range(n_agents):
        if envy_graph[agent]:
            has_envy = True
            envies = ", ".join([f"Agent {j}" for j in envy_graph[agent]])
            print(f"    Agent {agent} → {envies}")

    if not has_envy:
        print("    (No envy)")


def demo_ece():
    """Run ECE on a small example with detailed output."""
    print("="*70)
    print("ENVY CYCLE ELIMINATION (ECE) ALGORITHM DEMONSTRATION")
    print("="*70)
    print()

    # Create small example: 3 agents, 5 items
    np.random.seed(42)
    n_agents, m_items = 3, 5

    valuation_matrix = np.array([
        [4.2, 1.1, 2.5, 3.8, 1.9],  # Agent 0
        [2.1, 5.0, 1.2, 2.3, 3.6],  # Agent 1
        [3.5, 2.2, 4.8, 1.5, 2.9],  # Agent 2
    ])

    print("Valuation Matrix (3 agents, 5 items):")
    print("         Item0  Item1  Item2  Item3  Item4")
    for i in range(n_agents):
        items_str = "  ".join([f"{v:5.1f}" for v in valuation_matrix[i]])
        print(f"  Agent {i}:  {items_str}")
    print()
    print("Algorithm: Process items in order (0, 1, 2, 3, 4)")
    print("For each item, give it to an unenvied agent.")
    print("If no unenvied agent exists, remove envy cycles first.")
    print()

    # Track allocations manually for demonstration
    agent_bundles = [set() for _ in range(n_agents)]
    allocation_matrix = np.zeros((n_agents, m_items), dtype=int)

    # Process each item
    for item_idx in range(m_items):
        print("-"*70)
        print(f"ITERATION {item_idx + 1}: Allocating Item {item_idx}")
        print("-"*70)

        # Show current bundles
        print("Current bundles:")
        for i in range(n_agents):
            items = sorted(list(agent_bundles[i]))
            value = sum(valuation_matrix[i][j] for j in agent_bundles[i])
            if items:
                items_str = "{" + ", ".join([f"Item{j}" for j in items]) + "}"
                print(f"  Agent {i}: {items_str} (value: {value:.1f})")
            else:
                print(f"  Agent {i}: ∅ (value: 0.0)")
        print()

        # Build and visualize envy graph
        envy_graph = build_envy_graph(valuation_matrix, agent_bundles)
        visualize_envy_graph(envy_graph, n_agents)
        print()

        # Find unenvied agents
        envied_agents = set()
        for agent, envies_list in envy_graph.items():
            envied_agents.update(envies_list)

        unenvied_agents = [i for i in range(n_agents) if i not in envied_agents]

        # Handle cycle removal if needed
        cycles_removed = 0
        while len(unenvied_agents) == 0:
            print("  ⚠ No unenvied agent found - envy cycle detected!")
            cycle_removed = detect_and_remove_cycle(envy_graph, agent_bundles)

            if not cycle_removed:
                print("  Error: Could not find cycle to remove")
                unenvied_agents = [0]
                break

            cycles_removed += 1
            print(f"  ✓ Cycle {cycles_removed} removed via bundle exchange")

            # Show bundles after cycle removal
            print("  New bundles after cycle removal:")
            for i in range(n_agents):
                items = sorted(list(agent_bundles[i]))
                value = sum(valuation_matrix[i][j] for j in agent_bundles[i])
                if items:
                    items_str = "{" + ", ".join([f"Item{j}" for j in items]) + "}"
                    print(f"    Agent {i}: {items_str} (value: {value:.1f})")
                else:
                    print(f"    Agent {i}: ∅")

            # Rebuild envy graph
            envy_graph = build_envy_graph(valuation_matrix, agent_bundles)
            print("  Updated envy graph:")
            visualize_envy_graph(envy_graph, n_agents)

            # Recompute unenvied agents
            envied_agents = set()
            for agent, envies_list in envy_graph.items():
                envied_agents.update(envies_list)
            unenvied_agents = [i for i in range(n_agents) if i not in envied_agents]
            print()

        if unenvied_agents:
            print(f"  Unenvied agents: {unenvied_agents}")
            selected = min(unenvied_agents)
            if len(unenvied_agents) > 1:
                print(f"  Selected: Agent {selected} (tie-break: min index)")
            else:
                print(f"  Selected: Agent {selected}")

        print(f"\n  → Agent {selected} receives Item {item_idx} (value: {valuation_matrix[selected][item_idx]:.1f})")

        # Allocate item
        allocation_matrix[selected][item_idx] = 1
        agent_bundles[selected].add(item_idx)

        # Update allocation matrix for proper tracking
        allocation_matrix = np.zeros((n_agents, m_items), dtype=int)
        for agent_idx, bundle in enumerate(agent_bundles):
            for item in bundle:
                allocation_matrix[agent_idx][item] = 1

        print()

    # Final allocation analysis
    print("="*70)
    print("FINAL ALLOCATION")
    print("="*70)
    print()

    for i in range(n_agents):
        items = sorted(list(agent_bundles[i]))
        value = sum(valuation_matrix[i][j] for j in agent_bundles[i])
        items_str = "{" + ", ".join([f"Item{j}" for j in items]) + "}"
        print(f"  Agent {i}: {items_str} - Total value: {value:.1f}")

    print()

    # Verify allocation validity
    col_sums = np.sum(allocation_matrix, axis=0)
    valid = np.all(col_sums == 1)
    print(f"Allocation validity: {'✓' if valid else '✗'} (each item to exactly one agent)")
    print()

    # Check fairness properties
    bundle_values = calculate_agent_bundle_values(valuation_matrix, allocation_matrix)
    ef = is_envy_free(bundle_values)
    ef1_result = is_ef1(valuation_matrix, allocation_matrix, bundle_values)
    efx_result = is_efx(valuation_matrix, allocation_matrix, bundle_values)

    util_sum = utility_sum(bundle_values)
    nash_w = nash_welfare(bundle_values)

    print("Fairness Properties:")
    print(f"  Envy-Free (EF):  {'✓ YES' if ef else '✗ NO'}")
    print(f"  EF1:             {'✓ YES' if ef1_result else '✗ NO'} (GUARANTEED by ECE)")
    print(f"  EFx:             {'✓ YES' if efx_result else '✗ NO'}")
    print()

    print("Welfare Metrics:")
    print(f"  Utility Sum:     {util_sum:.2f}")
    print(f"  Nash Welfare:    {nash_w:.4f}")
    print()

    # Show detailed envy analysis if not EF
    if not ef:
        print("Envy Analysis (why not EF):")
        for i in range(n_agents):
            for j in range(n_agents):
                if i != j and bundle_values[i][j] > bundle_values[i][i]:
                    print(f"  Agent {i} envies Agent {j}: values their bundle at {bundle_values[i][j]:.1f} > own bundle {bundle_values[i][i]:.1f}")
        print()

    print("="*70)
    print()
    print("Key Takeaways:")
    print("  • ECE guarantees EF1 (envy-free up to one item)")
    print("  • May not achieve full envy-freeness (EF)")
    print("  • Maintains EF1 at every intermediate step, not just final allocation")
    print("  • Uses cycle detection and bundle swapping to ensure progress")
    print("="*70)


if __name__ == '__main__':
    demo_ece()
