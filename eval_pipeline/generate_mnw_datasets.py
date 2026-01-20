#!/usr/bin/env python3
"""
Generate MNW benchmark datasets with proper solver diagnostics.

This script generates datasets for multiple problem sizes with:
- Two valuation distributions: uniform and correlated preferences
- Full solver diagnostics (status, gap, time, optimality)
- Separate files for each configuration

Usage:
    python generate_mnw_datasets.py --output-dir datasets/mnw_benchmark
    python generate_mnw_datasets.py --sizes 5x10 10x20 --num-matrices 100
"""

import argparse
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Tuple, List
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent))

from utils.mnw_solver import solve_mnw, summarize_diagnostics, MNWSolveResult
from utils.max_utility import best_utilitarian_welfare


# =============================================================================
# Valuation Matrix Generators
# =============================================================================

def generate_uniform(n_agents: int, m_items: int) -> np.ndarray:
    """
    Generate valuation matrix with uniform random values in [0, 1].

    This is the standard baseline distribution used in fair allocation literature.
    """
    return np.random.uniform(0, 1, size=(n_agents, m_items))


def generate_correlated(
    n_agents: int,
    m_items: int,
    rank: int = 3,
    noise_scale: float = 0.1,
) -> np.ndarray:
    """
    Generate valuation matrix with correlated preferences via low-rank structure.

    Model: v_{i,o} = u_i^T x_o + epsilon, clipped to [0, 1]

    This captures scenarios where agents have similar underlying preferences
    (e.g., item quality) but with individual variation.

    Args:
        n_agents: Number of agents
        m_items: Number of items
        rank: Rank of the latent factor matrices (controls correlation strength)
        noise_scale: Standard deviation of Gaussian noise

    Returns:
        Valuation matrix with shape (n_agents, m_items), values in [0, 1]
    """
    # Latent factors: agents and items in shared latent space
    u = np.random.randn(n_agents, rank)  # Agent embeddings
    x = np.random.randn(rank, m_items)   # Item embeddings

    # Base valuations from inner products
    base = u @ x

    # Normalize to roughly [0, 1] range before adding noise
    base = (base - base.min()) / (base.max() - base.min() + 1e-8)

    # Add Gaussian noise
    noise = np.random.randn(n_agents, m_items) * noise_scale
    valuations = base + noise

    # Clip to [0, 1]
    valuations = np.clip(valuations, 0, 1)

    return valuations


# =============================================================================
# Dataset Generation
# =============================================================================

DISTRIBUTIONS = {
    'uniform': generate_uniform,
    'correlated': generate_correlated,
}


def parse_size(size_str: str) -> Tuple[int, int]:
    """Parse size string like '10x20' into (n_agents, m_items)."""
    parts = size_str.lower().split('x')
    if len(parts) != 2:
        raise ValueError(f"Invalid size format: {size_str}. Expected 'NxM' (e.g., '10x20')")
    return int(parts[0]), int(parts[1])


def generate_dataset(
    n_agents: int,
    m_items: int,
    num_matrices: int,
    distribution: str,
    time_limit: float,
    mip_gap: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[MNWSolveResult]]:
    """
    Generate a dataset of valuation matrices with MNW solutions.

    Returns:
        matrices: (N, n_agents, m_items) valuation matrices
        nash_values: (N,) MNW values (NaN if solve failed)
        util_values: (N,) max utilitarian welfare values
        diagnostics: List of MNWSolveResult with solver info
    """
    np.random.seed(seed)

    generator = DISTRIBUTIONS[distribution]

    matrices = []
    nash_values = []
    util_values = []
    diagnostics = []

    # Generate more matrices than needed to account for failures
    generated = 0
    attempts = 0
    max_attempts = num_matrices * 3

    pbar = tqdm(total=num_matrices, desc=f"Generating {n_agents}x{m_items} {distribution}")

    while generated < num_matrices and attempts < max_attempts:
        attempts += 1

        # Generate valuation matrix
        if distribution == 'correlated':
            # Rank scales with min(agents, items), capped at reasonable value
            rank = min(3, n_agents, m_items)
            V = generator(n_agents, m_items, rank=rank, noise_scale=0.1)
        else:
            V = generator(n_agents, m_items)

        # Solve MNW
        result = solve_mnw(
            V,
            time_limit=time_limit,
            mip_gap=mip_gap,
            num_segments=200,
            return_allocation=True,
            verbose=False,
        )

        # Only keep instances where we got a solution
        if result.nash_welfare is not None:
            matrices.append(V)
            nash_values.append(result.nash_welfare)
            util_values.append(best_utilitarian_welfare(V))
            diagnostics.append(result)
            generated += 1
            pbar.update(1)

    pbar.close()

    if generated < num_matrices:
        print(f"Warning: Only generated {generated}/{num_matrices} matrices "
              f"after {attempts} attempts")

    return (
        np.array(matrices),
        np.array(nash_values),
        np.array(util_values),
        diagnostics,
    )


def save_dataset(
    output_path: Path,
    matrices: np.ndarray,
    nash_values: np.ndarray,
    util_values: np.ndarray,
    diagnostics: List[MNWSolveResult],
    config: dict,
):
    """Save dataset with full metadata and diagnostics."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Compute summary statistics
    summary = summarize_diagnostics(diagnostics)

    # Save main data
    np.savez_compressed(
        output_path,
        matrices=matrices,
        nash_welfare=nash_values,
        util_welfare=util_values,
    )

    # Save diagnostics as JSON
    diagnostics_path = output_path.with_suffix('.diagnostics.json')
    diagnostics_data = {
        'config': config,
        'summary': summary,
        'per_instance': [d.to_dict() for d in diagnostics],
    }
    with open(diagnostics_path, 'w') as f:
        json.dump(diagnostics_data, f, indent=2)

    # Print summary
    print(f"\nDataset saved: {output_path}")
    print(f"  Matrices: {matrices.shape}")
    print(f"  Provably optimal: {summary['n_provably_optimal']}/{summary['n_total']} "
          f"({summary['pct_provably_optimal']:.1f}%)")
    print(f"  Timed out: {summary['n_timeout']}/{summary['n_total']} "
          f"({summary['pct_timeout']:.1f}%)")
    if summary['mip_gap_max'] is not None:
        print(f"  Max MIP gap: {summary['mip_gap_max']:.4f}")
    print(f"  Mean solve time: {summary['solve_time_mean']:.2f}s")
    print(f"  Max solve time: {summary['solve_time_max']:.2f}s")


def generate_all_datasets(
    output_dir: Path,
    sizes: List[str],
    distributions: List[str],
    num_matrices: int,
    time_limit: float,
    mip_gap: float,
    seed: int,
):
    """Generate all dataset combinations."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save generation config
    config = {
        'generated_at': datetime.now().isoformat(),
        'sizes': sizes,
        'distributions': distributions,
        'num_matrices': num_matrices,
        'time_limit_seconds': time_limit,
        'mip_gap': mip_gap,
        'seed': seed,
        'solver': 'Gurobi',
        'pwl_segments': 200,
    }

    config_path = output_dir / 'generation_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Generating datasets with config:")
    print(f"  Sizes: {sizes}")
    print(f"  Distributions: {distributions}")
    print(f"  Matrices per config: {num_matrices}")
    print(f"  Time limit: {time_limit}s")
    print(f"  MIP gap tolerance: {mip_gap}")
    print(f"  Output directory: {output_dir}")
    print()

    all_summaries = {}

    for size_str in sizes:
        n_agents, m_items = parse_size(size_str)

        for dist in distributions:
            print(f"\n{'='*60}")
            print(f"Generating: {n_agents}x{m_items} {dist}")
            print('='*60)

            # Generate unique seed for this combination
            combo_seed = seed + hash((n_agents, m_items, dist)) % 10000

            matrices, nash_values, util_values, diagnostics = generate_dataset(
                n_agents=n_agents,
                m_items=m_items,
                num_matrices=num_matrices,
                distribution=dist,
                time_limit=time_limit,
                mip_gap=mip_gap,
                seed=combo_seed,
            )

            # Filename: {agents}x{items}_{distribution}.npz
            filename = f"{n_agents}x{m_items}_{dist}.npz"
            output_path = output_dir / filename

            dataset_config = {
                **config,
                'n_agents': n_agents,
                'm_items': m_items,
                'distribution': dist,
                'actual_num_matrices': len(matrices),
            }

            save_dataset(
                output_path,
                matrices,
                nash_values,
                util_values,
                diagnostics,
                dataset_config,
            )

            # Track summary for final report
            summary = summarize_diagnostics(diagnostics)
            all_summaries[f"{n_agents}x{m_items}_{dist}"] = summary

    # Print final summary report
    print(f"\n{'='*60}")
    print("GENERATION COMPLETE - SUMMARY")
    print('='*60)
    print(f"\n{'Dataset':<25} {'Optimal':<12} {'Timeout':<12} {'Max Gap':<12} {'Max Time':<12}")
    print('-'*73)

    for name, summary in all_summaries.items():
        opt_str = f"{summary['n_provably_optimal']}/{summary['n_total']}"
        timeout_str = f"{summary['n_timeout']}/{summary['n_total']}"
        gap_str = f"{summary['mip_gap_max']:.4f}" if summary['mip_gap_max'] else "N/A"
        time_str = f"{summary['solve_time_max']:.1f}s"
        print(f"{name:<25} {opt_str:<12} {timeout_str:<12} {gap_str:<12} {time_str:<12}")

    # Check if any datasets have non-optimal solutions
    any_non_optimal = any(
        s['n_provably_optimal'] < s['n_total']
        for s in all_summaries.values()
    )

    if any_non_optimal:
        print("\n" + "!"*60)
        print("WARNING: Some instances are NOT provably optimal.")
        print("Consider relabeling as 'MNW Best Found' instead of 'MNW Optimum'.")
        print("See .diagnostics.json files for per-instance details.")
        print("!"*60)


def main():
    parser = argparse.ArgumentParser(
        description='Generate MNW benchmark datasets with solver diagnostics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate default sizes with both distributions
  python generate_mnw_datasets.py --output-dir datasets/mnw_benchmark

  # Generate specific sizes
  python generate_mnw_datasets.py --sizes 5x10 10x20 15x30 --num-matrices 500

  # Only uniform distribution with longer timeout
  python generate_mnw_datasets.py --distributions uniform --time-limit 600
        """
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='datasets/mnw_benchmark',
        help='Output directory for datasets (default: datasets/mnw_benchmark)'
    )
    parser.add_argument(
        '--sizes',
        nargs='+',
        default=['5x10', '10x20', '15x30', '20x40'],
        help='Problem sizes as NxM (agents x items). Default: 5x10 10x20 15x30 20x40'
    )
    parser.add_argument(
        '--distributions',
        nargs='+',
        choices=list(DISTRIBUTIONS.keys()),
        default=['uniform', 'correlated'],
        help='Valuation distributions to generate. Default: uniform correlated'
    )
    parser.add_argument(
        '--num-matrices',
        type=int,
        default=1000,
        help='Number of matrices per configuration (default: 1000)'
    )
    parser.add_argument(
        '--time-limit',
        type=float,
        default=300.0,
        help='Gurobi time limit per solve in seconds (default: 300)'
    )
    parser.add_argument(
        '--mip-gap',
        type=float,
        default=0.001,
        help='MIP gap tolerance (default: 0.001 = 0.1%%)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )

    args = parser.parse_args()

    generate_all_datasets(
        output_dir=Path(args.output_dir),
        sizes=args.sizes,
        distributions=args.distributions,
        num_matrices=args.num_matrices,
        time_limit=args.time_limit,
        mip_gap=args.mip_gap,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()
