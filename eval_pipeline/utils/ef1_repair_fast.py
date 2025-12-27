"""Optimized EF1 Quick Repair Algorithm

Faster implementation using:
1. Numba JIT compilation for inner loops
2. Parallel batch processing
"""

import numpy as np
from numba import jit, prange
import multiprocessing


@jit(nopython=True, cache=True)
def _ef1_repair_numba(A, v, max_passes):
    """Numba-compiled core of EF1 repair algorithm."""
    n_agents, m_items = A.shape

    for _ in range(max_passes):
        made_change = False

        # Compute utilities
        u = np.zeros(n_agents)
        for i in range(n_agents):
            for g in range(m_items):
                u[i] += v[i, g] * A[i, g]

        # Check all agent pairs
        for i in range(n_agents):
            for j in range(n_agents):
                if i == j:
                    continue

                # Get items in j's bundle and compute values
                j_item_count = 0
                value_i_for_j = 0.0
                max_item_value = -1e10

                for g in range(m_items):
                    if A[j, g] > 0.5:
                        j_item_count += 1
                        value_i_for_j += v[i, g]
                        if v[i, g] > max_item_value:
                            max_item_value = v[i, g]

                if j_item_count == 0:
                    continue

                # Check EF1 violation
                if value_i_for_j - max_item_value > u[i]:
                    # Find best item to transfer
                    best_g = -1
                    best_delta = -1e20

                    for g in range(m_items):
                        if A[j, g] < 0.5:
                            continue

                        new_u_i = u[i] + v[i, g]
                        new_u_j = u[j] - v[j, g]

                        # Skip if would make j non-positive (unless last item)
                        if j_item_count > 1 and new_u_j <= 0:
                            continue

                        # Compute delta
                        if u[i] == 0:
                            delta = 1e10 + new_u_i
                        elif u[j] > 0 and new_u_j > 0 and new_u_i > 0:
                            delta = (np.log(new_u_i) + np.log(new_u_j)
                                    - np.log(u[i]) - np.log(u[j]))
                        else:
                            continue

                        if delta > best_delta:
                            best_delta = delta
                            best_g = g

                    # Transfer best item
                    if best_g >= 0:
                        A[j, best_g] = 0.0
                        A[i, best_g] = 1.0
                        u[i] += v[i, best_g]
                        u[j] -= v[j, best_g]
                        made_change = True

        if not made_change:
            break

    return A


@jit(nopython=True, parallel=True, cache=True)
def _ef1_repair_batch_numba(allocations, valuations, max_passes):
    """Numba-compiled parallel batch repair."""
    N = allocations.shape[0]
    result = np.empty_like(allocations)

    for b in prange(N):
        A = allocations[b].copy()
        v = valuations[b]
        result[b] = _ef1_repair_numba(A, v, max_passes)

    return result


def ef1_quick_repair_numba(allocation, valuations, max_passes=10):
    """Numba-accelerated EF1 repair.

    Args:
        allocation: (n_agents, m_items) binary allocation matrix
        valuations: (n_agents, m_items) valuation matrix
        max_passes: Maximum repair passes

    Returns:
        Repaired allocation matrix
    """
    A = allocation.astype(np.float64).copy()
    v = valuations.astype(np.float64)
    return _ef1_repair_numba(A, v, max_passes)


def ef1_quick_repair_batch_numba(allocations, valuations, max_passes=10):
    """Numba-accelerated parallel batch repair.

    Args:
        allocations: (N, n_agents, m_items) batch of allocations
        valuations: (N, n_agents, m_items) batch of valuations
        max_passes: Maximum repair passes

    Returns:
        Repaired allocations
    """
    A = allocations.astype(np.float64).copy()
    v = valuations.astype(np.float64)
    return _ef1_repair_batch_numba(A, v, max_passes)


def ef1_quick_repair_fast(allocation, valuations, max_passes=10):
    """Optimized EF1 repair using vectorized operations.

    Args:
        allocation: (n_agents, m_items) binary allocation matrix
        valuations: (n_agents, m_items) valuation matrix (all positive)
        max_passes: Maximum number of repair passes (default: 10)

    Returns:
        Repaired allocation matrix
    """
    n_agents, m_items = allocation.shape
    A = allocation.astype(np.float64).copy()
    v = valuations.astype(np.float64)

    for _ in range(max_passes):
        made_change_this_pass = False

        # Compute utilities for all agents: u[i] = sum of values in A[i]
        u = np.sum(v * A, axis=1)

        # Check all agent pairs for violations (like original)
        for i in range(n_agents):
            for j in range(n_agents):
                if i == j:
                    continue

                # Get items in j's bundle
                j_items = np.where(A[j] > 0)[0]

                if len(j_items) == 0:
                    continue

                # Check EF1 violation
                value_i_for_j_bundle = np.sum(v[i, j_items])
                max_item_value = np.max(v[i, j_items])

                if value_i_for_j_bundle - max_item_value > u[i]:
                    # Find best item to transfer
                    v_i_items = v[i, j_items]
                    v_j_items = v[j, j_items]
                    new_u_i = u[i] + v_i_items
                    new_u_j = u[j] - v_j_items

                    if u[i] == 0:
                        deltas = 1e10 + new_u_i
                    else:
                        valid = (new_u_j > 0) & (new_u_i > 0) & (u[i] > 0) & (u[j] > 0)
                        deltas = np.full(len(j_items), -np.inf)
                        if np.any(valid):
                            deltas[valid] = (np.log(new_u_i[valid]) + np.log(new_u_j[valid])
                                            - np.log(u[i]) - np.log(u[j]))

                    if len(j_items) > 1:
                        deltas[new_u_j <= 0] = -np.inf

                    best_idx = np.argmax(deltas)

                    if deltas[best_idx] > -np.inf:
                        best_g = j_items[best_idx]
                        A[j, best_g] = 0
                        A[i, best_g] = 1
                        # Update utilities immediately
                        u[i] += v[i, best_g]
                        u[j] -= v[j, best_g]
                        made_change_this_pass = True

        if not made_change_this_pass:
            break

    return A


def _repair_single(args):
    """Helper for parallel processing."""
    alloc, vals, max_passes = args
    return ef1_quick_repair_fast(alloc, vals, max_passes)


def ef1_quick_repair_batch_fast(allocations, valuations, max_passes=10, n_workers=None):
    """Parallel batch EF1 repair using multiprocessing.

    Args:
        allocations: (N, n_agents, m_items) batch of allocation matrices
        valuations: (N, n_agents, m_items) batch of valuation matrices
        max_passes: Maximum number of repair passes per allocation
        n_workers: Number of parallel workers (default: CPU count)

    Returns:
        Repaired allocations of shape (N, n_agents, m_items)
    """
    N = allocations.shape[0]

    if n_workers is None:
        n_workers = min(multiprocessing.cpu_count(), N)

    # For small batches, don't bother with parallelism
    if N <= 4 or n_workers <= 1:
        repaired = np.zeros_like(allocations)
        for i in range(N):
            repaired[i] = ef1_quick_repair_fast(allocations[i], valuations[i], max_passes)
        return repaired

    # Prepare arguments
    args = [(allocations[i], valuations[i], max_passes) for i in range(N)]

    # Use ThreadPoolExecutor instead of ProcessPoolExecutor to avoid pickling overhead
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(_repair_single, args))

    return np.array(results)


def ef1_quick_repair_batch_vectorized(allocations, valuations, max_passes=10):
    """Batch repair that processes instances independently but with vectorized inner ops.

    Args:
        allocations: (N, n_agents, m_items) batch of allocation matrices
        valuations: (N, n_agents, m_items) batch of valuation matrices
        max_passes: Maximum number of repair passes per allocation

    Returns:
        Repaired allocations of shape (N, n_agents, m_items)
    """
    N = allocations.shape[0]
    repaired = np.zeros_like(allocations)

    for i in range(N):
        repaired[i] = ef1_quick_repair_fast(allocations[i], valuations[i], max_passes)

    return repaired
