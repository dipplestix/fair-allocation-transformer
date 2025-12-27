#!/usr/bin/env python3
"""
Generate datasets for all (n, m) combinations needed for heatmaps.
Uses multiprocessing for parallel generation.
"""

import argparse
import numpy as np
import sys
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.max_utility import best_nash_welfare, best_utilitarian_welfare


def generate_valuation_matrix(n_agents, m_items):
    return np.random.uniform(0, 1, size=(n_agents, m_items))


def generate_single_dataset(config, num_matrices, output_dir, seed=42):
    """Generate a single dataset for (n, m) configuration."""
    n, m = config

    np.random.seed(seed * m + n)

    matrices = []
    nash_values = []
    util_values = []

    for _ in range(num_matrices):
        valuation_matrix = generate_valuation_matrix(n, m)

        nash_welfare = best_nash_welfare(valuation_matrix, num_segments=200)
        if nash_welfare is None:
            continue

        util_welfare = best_utilitarian_welfare(valuation_matrix)

        matrices.append(valuation_matrix)
        nash_values.append(nash_welfare)
        util_values.append(util_welfare)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    filename = output_path / f"{n}_{m}_{num_matrices}_dataset.npz"
    np.savez_compressed(
        filename,
        matrices=np.array(matrices),
        nash_welfare=np.array(nash_values),
        util_welfare=np.array(util_values)
    )

    return (n, m, len(matrices))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=1000)
    parser.add_argument('--n_min', type=int, default=10)
    parser.add_argument('--n_max', type=int, default=30)
    parser.add_argument('--m_min', type=int, default=10)
    parser.add_argument('--m_max', type=int, default=30)
    parser.add_argument('--output_dir', type=str, default='datasets/heatmap')
    parser.add_argument('--workers', type=int, default=None, help='Number of parallel workers (default: CPU count)')
    args = parser.parse_args()

    if args.workers is None:
        args.workers = cpu_count()

    # Build list of (n, m) pairs where m >= n
    configs = []
    for n in range(args.n_min, args.n_max + 1):
        for m in range(max(n, args.m_min), args.m_max + 1):
            configs.append((n, m))

    print(f"Generating datasets for {len(configs)} configurations")
    print(f"Samples per config: {args.num_samples}")
    print(f"Output directory: {args.output_dir}")
    print(f"Using {args.workers} parallel workers")
    print()

    # Create partial function with fixed arguments
    generate_fn = partial(
        generate_single_dataset,
        num_matrices=args.num_samples,
        output_dir=args.output_dir
    )

    # Run in parallel with progress bar
    with Pool(args.workers) as pool:
        results = list(tqdm(
            pool.imap_unordered(generate_fn, configs),
            total=len(configs),
            desc="Generating datasets"
        ))

    # Report any issues
    for n, m, count in results:
        if count < args.num_samples * 0.9:
            print(f"Warning: Only got {count} valid samples for n={n}, m={m}")

    print(f"\nDone! Generated {len(configs)} dataset files in {args.output_dir}/")


if __name__ == "__main__":
    main()
