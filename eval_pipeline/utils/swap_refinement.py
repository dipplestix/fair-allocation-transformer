"""Swap-based envy elimination for post-processing allocations.

This module implements a swap-based refinement algorithm that takes an existing
allocation and iteratively improves it by swapping bundles between envious agent
pairs. The algorithm balances envy reduction with Nash welfare preservation.

Based on concepts from: https://arxiv.org/pdf/2407.12461
Note: Paper focuses on subadditive valuations, but we apply to additive valuations.
"""

import numpy as np


def swap_bundles(allocation, agent_i, agent_j):
    """Swap entire bundles between two agents.

    Args:
        allocation: (n_agents, m_items) binary allocation matrix
        agent_i: Index of first agent
        agent_j: Index of second agent

    Returns:
        new_allocation: Allocation with bundles swapped between i and j
    """
    new_allocation = allocation.copy()
    new_allocation[[agent_i, agent_j]] = new_allocation[[agent_j, agent_i]]
    return new_allocation


def compute_envy_matrix(valuations, allocation):
    """Compute pairwise envy between all agents.

    Args:
        valuations: (n_agents, m_items) valuation matrix
        allocation: (n_agents, m_items) binary allocation matrix

    Returns:
        envy_matrix: (n_agents, n_agents) where envy[i][j] =
                     max(0, value_i(bundle_j) - value_i(bundle_i))
    """
    from utils.calculations import calculate_agent_bundle_values

    # Compute how much each agent values each other agent's bundle
    agent_bundle_values = calculate_agent_bundle_values(valuations, allocation)

    # Extract own values (diagonal)
    own_values = np.diag(agent_bundle_values)

    # Compute envy: value_i(bundle_j) - value_i(bundle_i)
    envy_matrix = np.maximum(0, agent_bundle_values - own_values[:, np.newaxis])

    # No self-envy
    np.fill_diagonal(envy_matrix, 0)

    return envy_matrix


def find_best_swap(valuations, allocation, envy_matrix, current_nash, welfare_weight=0.5):
    """Find the best swap that balances envy reduction and welfare preservation.

    Args:
        valuations: (n_agents, m_items) valuation matrix
        allocation: (n_agents, m_items) binary allocation matrix
        envy_matrix: (n_agents, n_agents) current envy matrix
        current_nash: Current Nash welfare value
        welfare_weight: Weight for welfare vs envy (0=pure envy, 1=pure welfare)

    Returns:
        best_swap: Tuple (i, j, new_allocation, new_envy, new_nash) or None
        best_score: Score of best swap found
    """
    from utils.calculations import calculate_agent_bundle_values, nash_welfare

    n_agents = allocation.shape[0]
    best_score = -np.inf
    best_swap = None
    current_total_envy = np.sum(envy_matrix)

    # Try all possible swaps
    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            # Skip if no envy between this pair (optimization)
            if envy_matrix[i][j] < 1e-6 and envy_matrix[j][i] < 1e-6:
                continue

            # Test swap
            test_allocation = swap_bundles(allocation, i, j)
            test_envy_matrix = compute_envy_matrix(valuations, test_allocation)
            test_total_envy = np.sum(test_envy_matrix)

            # Compute Nash welfare
            test_bundle_values = calculate_agent_bundle_values(valuations, test_allocation)
            test_nash = nash_welfare(test_bundle_values)

            # Score swap: weighted combination of envy reduction and welfare preservation
            envy_improvement = (current_total_envy - test_total_envy) / (current_total_envy + 1e-9)
            welfare_ratio = test_nash / (current_nash + 1e-9)

            score = welfare_weight * welfare_ratio + (1 - welfare_weight) * (1 + envy_improvement)

            if score > best_score:
                best_score = score
                best_swap = (i, j, test_allocation, test_total_envy, test_nash)

    return best_swap, best_score


def swap_based_envy_elimination(allocation, valuations, max_iterations=100,
                                welfare_weight=0.5, min_improvement=0.001):
    """Main swap-based refinement algorithm.

    Iteratively swaps bundles between envious agent pairs to reduce envy
    while preserving Nash welfare. Terminates when no beneficial swaps remain
    or maximum iterations reached.

    Args:
        allocation: (n_agents, m_items) initial binary allocation matrix
        valuations: (n_agents, m_items) valuation matrix
        max_iterations: Maximum number of swap iterations
        welfare_weight: Weight for welfare vs envy (0=pure envy, 1=pure welfare)
        min_improvement: Minimum score improvement to accept a swap

    Returns:
        refined_allocation: (n_agents, m_items) refined allocation with reduced envy
    """
    from utils.calculations import calculate_agent_bundle_values, nash_welfare

    current_allocation = allocation.copy()

    for iteration in range(max_iterations):
        # Compute current state
        envy_matrix = compute_envy_matrix(valuations, current_allocation)
        total_envy = np.sum(envy_matrix)

        # Check convergence (envy-free or negligible envy)
        if total_envy < 1e-6:
            break

        # Compute current Nash welfare
        bundle_values = calculate_agent_bundle_values(valuations, current_allocation)
        current_nash = nash_welfare(bundle_values)

        # Find best swap
        best_swap, best_score = find_best_swap(
            valuations, current_allocation, envy_matrix, current_nash, welfare_weight
        )

        # Apply swap if beneficial
        if best_swap is not None and best_score > 1.0 + min_improvement:
            i, j, new_allocation, new_envy, new_nash = best_swap
            current_allocation = new_allocation
        else:
            # No beneficial swap found, converged
            break

    return current_allocation


def swap_based_refinement_batch(allocations, valuations, **params):
    """Apply swap refinement to a batch of allocations.

    Args:
        allocations: (N, n_agents, m_items) batch of binary allocation matrices
        valuations: (N, n_agents, m_items) batch of valuation matrices
        **params: Parameters for swap_based_envy_elimination

    Returns:
        refined_allocations: (N, n_agents, m_items) batch of refined allocations
    """
    N, n_agents, m_items = allocations.shape
    refined_allocations = np.zeros_like(allocations)

    # Process each instance (hard to fully vectorize due to different convergence)
    for i in range(N):
        refined_allocations[i] = swap_based_envy_elimination(
            allocations[i], valuations[i], **params
        )

    return refined_allocations
