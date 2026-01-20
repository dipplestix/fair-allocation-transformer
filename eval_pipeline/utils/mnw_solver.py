"""
Enhanced MNW (Maximum Nash Welfare) solver with proper timeout, gap tracking, and diagnostics.

This module provides a rigorous MNW solver that:
- Sets explicit time limits and optimality gap tolerances
- Tracks and reports solver status (OPTIMAL, TIME_LIMIT, etc.)
- Returns diagnostic information for each solve
"""

import numpy as np
import gurobipy as gp
from gurobipy import GRB
from dataclasses import dataclass
from typing import Optional, Tuple
from enum import Enum


class SolverStatus(Enum):
    """Solver termination status."""
    OPTIMAL = "optimal"
    TIME_LIMIT = "time_limit"
    GAP_LIMIT = "gap_limit"
    SUBOPTIMAL = "suboptimal"
    INFEASIBLE = "infeasible"
    ERROR = "error"


@dataclass
class MNWSolveResult:
    """Result of an MNW solve with full diagnostics."""
    nash_welfare: Optional[float]
    allocation: Optional[np.ndarray]
    status: SolverStatus
    solve_time: float  # seconds
    mip_gap: Optional[float]  # relative gap at termination
    node_count: int
    is_provably_optimal: bool

    def to_dict(self) -> dict:
        return {
            'nash_welfare': self.nash_welfare,
            'status': self.status.value,
            'solve_time': self.solve_time,
            'mip_gap': self.mip_gap,
            'node_count': self.node_count,
            'is_provably_optimal': self.is_provably_optimal,
        }


def solve_mnw(
    valuation_matrix: np.ndarray,
    time_limit: float = 300.0,
    mip_gap: float = 0.001,
    num_segments: int = 200,
    return_allocation: bool = True,
    verbose: bool = False,
) -> MNWSolveResult:
    """
    Solve for Maximum Nash Welfare allocation with full diagnostics.

    Args:
        valuation_matrix: (n_agents, m_items) valuation matrix with values in [0, 1]
        time_limit: Maximum solve time in seconds (default: 300s = 5 min)
        mip_gap: Relative MIP gap tolerance (default: 0.1%)
        num_segments: PWL approximation segments for log function (default: 200)
        return_allocation: Whether to extract allocation matrix
        verbose: Print Gurobi output

    Returns:
        MNWSolveResult with nash_welfare, allocation, status, and diagnostics
    """
    m, n = valuation_matrix.shape  # m agents, n items

    # Scale valuation matrix for numerical stability
    scaled_valuation = valuation_matrix * 10

    try:
        model = gp.Model("mnw_solver")
        model.setParam('OutputFlag', 1 if verbose else 0)
        model.setParam('TimeLimit', time_limit)
        model.setParam('MIPGap', mip_gap)

        # Binary allocation variables: x[i,j] = 1 if agent i gets item j
        x = model.addVars(m, n, vtype=GRB.BINARY, name="x")

        # Each item assigned to exactly one agent
        for j in range(n):
            model.addConstr(gp.quicksum(x[i, j] for i in range(m)) == 1)

        # Utility bounds for PWL approximation
        lower_bound = 0.01  # Avoid log(0)
        upper_bound = 10 * n + 1  # Max possible utility after scaling

        # Agent utilities
        u = model.addVars(m, lb=lower_bound, ub=upper_bound, name="u")
        for i in range(m):
            model.addConstr(
                u[i] == gp.quicksum(scaled_valuation[i, j] * x[i, j] for j in range(n)) + 0.01
            )

        # Piecewise linear approximation of log
        log_u = model.addVars(m, lb=-float('inf'), name="log_u")
        breakpoints = np.linspace(lower_bound, upper_bound, num_segments)
        log_values = np.log(breakpoints)

        for i in range(m):
            model.addGenConstrPWL(u[i], log_u[i], breakpoints.tolist(), log_values.tolist())

        # Maximize sum of log utilities (equivalent to geometric mean)
        model.setObjective(gp.quicksum(log_u[i] for i in range(m)), GRB.MAXIMIZE)

        model.optimize()

        # Extract results based on status
        solve_time = model.Runtime
        node_count = int(model.NodeCount)

        # Map Gurobi status to our status enum
        if model.status == GRB.OPTIMAL:
            status = SolverStatus.OPTIMAL
            is_optimal = True
        elif model.status == GRB.TIME_LIMIT:
            status = SolverStatus.TIME_LIMIT
            is_optimal = False
        elif model.status == GRB.SUBOPTIMAL:
            status = SolverStatus.SUBOPTIMAL
            is_optimal = False
        elif model.status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD):
            status = SolverStatus.INFEASIBLE
            is_optimal = False
        else:
            status = SolverStatus.ERROR
            is_optimal = False

        # Get MIP gap if available
        try:
            gap = model.MIPGap
        except:
            gap = None

        # Check if we have a solution (even if not proven optimal)
        if model.SolCount > 0:
            sum_of_logs = model.objVal
            nash_welfare = np.exp(sum_of_logs / m) / 10  # Geometric mean, scaled back

            allocation = None
            if return_allocation:
                allocation = np.array([
                    [int(round(x[i, j].x)) for j in range(n)]
                    for i in range(m)
                ])

            return MNWSolveResult(
                nash_welfare=nash_welfare,
                allocation=allocation,
                status=status,
                solve_time=solve_time,
                mip_gap=gap,
                node_count=node_count,
                is_provably_optimal=is_optimal,
            )
        else:
            return MNWSolveResult(
                nash_welfare=None,
                allocation=None,
                status=status,
                solve_time=solve_time,
                mip_gap=None,
                node_count=node_count,
                is_provably_optimal=False,
            )

    except gp.GurobiError as e:
        return MNWSolveResult(
            nash_welfare=None,
            allocation=None,
            status=SolverStatus.ERROR,
            solve_time=0.0,
            mip_gap=None,
            node_count=0,
            is_provably_optimal=False,
        )


def solve_mnw_batch(
    valuation_matrices: np.ndarray,
    time_limit: float = 300.0,
    mip_gap: float = 0.001,
    num_segments: int = 200,
    return_allocations: bool = True,
    verbose: bool = False,
    progress: bool = True,
) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Solve MNW for a batch of valuation matrices with diagnostics.

    Args:
        valuation_matrices: (N, n_agents, m_items) batch of valuation matrices
        time_limit: Maximum solve time per instance in seconds
        mip_gap: Relative MIP gap tolerance
        num_segments: PWL approximation segments
        return_allocations: Whether to return allocation matrices
        verbose: Print Gurobi output
        progress: Show progress bar

    Returns:
        nash_values: (N,) array of Nash welfare values (NaN for failed solves)
        allocations: (N, n_agents, m_items) allocation matrices (if requested)
        diagnostics: List of MNWSolveResult objects with full diagnostics
    """
    from tqdm import tqdm

    N, n_agents, m_items = valuation_matrices.shape
    nash_values = np.full(N, np.nan)
    allocations = np.zeros((N, n_agents, m_items), dtype=int) if return_allocations else None
    diagnostics = []

    iterator = tqdm(range(N), desc="Solving MNW") if progress else range(N)

    for i in iterator:
        result = solve_mnw(
            valuation_matrices[i],
            time_limit=time_limit,
            mip_gap=mip_gap,
            num_segments=num_segments,
            return_allocation=return_allocations,
            verbose=verbose,
        )

        diagnostics.append(result)

        if result.nash_welfare is not None:
            nash_values[i] = result.nash_welfare
            if return_allocations and result.allocation is not None:
                allocations[i] = result.allocation

    return nash_values, allocations, diagnostics


def summarize_diagnostics(diagnostics: list) -> dict:
    """
    Summarize solver diagnostics for a batch of solves.

    Args:
        diagnostics: List of MNWSolveResult objects

    Returns:
        Dictionary with summary statistics
    """
    n_total = len(diagnostics)
    status_counts = {}
    for status in SolverStatus:
        status_counts[status.value] = sum(1 for d in diagnostics if d.status == status)

    n_optimal = status_counts[SolverStatus.OPTIMAL.value]
    n_timeout = status_counts[SolverStatus.TIME_LIMIT.value]
    n_with_solution = sum(1 for d in diagnostics if d.nash_welfare is not None)

    solve_times = [d.solve_time for d in diagnostics]
    gaps = [d.mip_gap for d in diagnostics if d.mip_gap is not None]

    return {
        'n_total': n_total,
        'n_provably_optimal': n_optimal,
        'n_timeout': n_timeout,
        'n_with_solution': n_with_solution,
        'pct_provably_optimal': 100 * n_optimal / n_total if n_total > 0 else 0,
        'pct_timeout': 100 * n_timeout / n_total if n_total > 0 else 0,
        'status_counts': status_counts,
        'solve_time_mean': np.mean(solve_times) if solve_times else 0,
        'solve_time_max': np.max(solve_times) if solve_times else 0,
        'solve_time_median': np.median(solve_times) if solve_times else 0,
        'mip_gap_mean': np.mean(gaps) if gaps else None,
        'mip_gap_max': np.max(gaps) if gaps else None,
    }
