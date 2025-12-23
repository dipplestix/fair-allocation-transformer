"""EF1 Quick Repair Algorithm

Implements a post-processing algorithm that repairs EF1 violations by transferring
items from envied agents to envious agents. The item chosen for transfer is the one
that maximizes the improvement in Nash Social Welfare (log NSW).

Based on the EF1_quick_repair algorithm which iteratively checks for EF1 violations
and repairs them by transferring items that improve overall welfare.

This module provides both a pure numpy implementation and a faster numba-accelerated
version. The numba version is used by default when available (up to 88x faster).
"""

import numpy as np

# Try to import the fast numba implementation
try:
    from .ef1_repair_fast import (
        ef1_quick_repair_numba,
        ef1_quick_repair_batch_numba,
    )
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


def ef1_quick_repair(allocation, valuations, max_passes=10, use_numba=True):
    """Repair EF1 violations in an allocation by transferring items.

    For each agent pair (i, j), checks if agent i envies agent j even after
    removing j's most valuable item (from i's perspective). If so, transfers
    the item from j to i that maximizes log Nash Social Welfare improvement.

    Args:
        allocation: (n_agents, m_items) binary allocation matrix
        valuations: (n_agents, m_items) valuation matrix (all positive)
        max_passes: Maximum number of repair passes (default: 10)
        use_numba: Use numba-accelerated version if available (default: True)

    Returns:
        Repaired allocation matrix
    """
    # Use fast numba version if available
    if use_numba and _HAS_NUMBA:
        return ef1_quick_repair_numba(allocation, valuations, max_passes)

    n_agents, m_items = allocation.shape
    A = allocation.copy()

    for pass_num in range(max_passes):
        changed = False

        # Precompute utilities for all agents
        u = np.sum(valuations * A, axis=1)  # u[i] = sum of values in A[i]

        # Check all agent pairs for EF1 violations
        for i in range(n_agents):
            for j in range(n_agents):
                if i == j:
                    continue

                # Get items in j's bundle
                j_items = np.where(A[j] == 1)[0]

                if len(j_items) == 0:
                    continue  # j has no items to take

                # Check EF1 violation: i envies j even after removing j's best item
                # i's value for j's bundle
                value_i_for_j_bundle = np.sum(valuations[i, j_items])

                # i's value for j's best item (from i's perspective)
                max_item_value = np.max(valuations[i, j_items])

                # EF1 violation if: value_i_for_j_bundle - max_item_value > u[i]
                if value_i_for_j_bundle - max_item_value > u[i]:
                    # Find item to transfer that best improves log NSW
                    best_g = None
                    best_delta = -np.inf

                    for g in j_items:
                        # Only transfer if j would still have positive utility after
                        # (or if j would have 0 items left, we allow it)
                        new_u_j = u[j] - valuations[j, g]

                        # Skip if this would make j's utility non-positive and j has >1 items
                        if len(j_items) > 1 and new_u_j <= 0:
                            continue

                        new_u_i = u[i] + valuations[i, g]

                        # Compute log NSW improvement
                        # Special case: if agent i has empty bundle (u[i] = 0),
                        # prioritize giving them items by using a very large delta
                        if u[i] == 0:
                            # When u[i] = 0, any item significantly improves NSW
                            # Use new_u_i as the score (prefer higher value items)
                            delta = 1e10 + new_u_i  # Large positive value
                        elif u[j] > 0 and new_u_j > 0 and new_u_i > 0:
                            # Normal case: both agents have positive utility
                            # delta = log(new_u_i) + log(new_u_j) - log(u[i]) - log(u[j])
                            delta = (np.log(new_u_i) + np.log(new_u_j) -
                                   np.log(u[i]) - np.log(u[j]))
                        else:
                            continue  # Skip if log would be undefined

                        if delta > best_delta:
                            best_delta = delta
                            best_g = g

                    # Transfer the best item
                    if best_g is not None:
                        A[j, best_g] = 0
                        A[i, best_g] = 1
                        u[i] += valuations[i, best_g]
                        u[j] -= valuations[j, best_g]
                        changed = True

        # If no changes in this pass, we've converged
        if not changed:
            break

    return A


def ef1_quick_repair_batch(allocations, valuations, max_passes=10, use_numba=True):
    """Apply EF1 quick repair to a batch of allocations.

    Uses parallel processing with numba when available for significant speedup
    (up to 88x faster on large batches).

    Args:
        allocations: (N, n_agents, m_items) batch of allocation matrices
        valuations: (N, n_agents, m_items) batch of valuation matrices
        max_passes: Maximum number of repair passes per allocation
        use_numba: Use numba-accelerated version if available (default: True)

    Returns:
        Repaired allocations of shape (N, n_agents, m_items)
    """
    # Use fast numba version if available
    if use_numba and _HAS_NUMBA:
        return ef1_quick_repair_batch_numba(allocations, valuations, max_passes)

    N = allocations.shape[0]
    repaired = np.zeros_like(allocations)

    for i in range(N):
        repaired[i] = ef1_quick_repair(allocations[i], valuations[i], max_passes, use_numba=False)

    return repaired
