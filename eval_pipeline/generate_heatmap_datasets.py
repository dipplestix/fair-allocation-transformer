#!/usr/bin/env python3
"""
Generate datasets for all (n, m) combinations needed for heatmaps.
n = 10-30, m = 10-30, m >= n, 1000 samples each.

Saves to datasets/{n}_{m}_1000_dataset.npz
"""

import argparse
import numpy as np
import sys
from pathlib import Path
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.max_utility import best_nash_welfare, best_utilitarian_welfare


def generate_valuation_matrix(n_agents, m_items):
    return np.random.uniform(0, 1, size=(n_agents, m_items))


def generate_and_save_dataset(n_agents, n_items, num_matrices, output_dir, seed=42):
    """Generate dataset and save to disk."""
    np.random.seed(seed * n_items + n_agents)

    matrices = []
    nash_values = []
    util_values = []

    for _ in range(num_matrices):
        valuation_matrix = generate_valuation_matrix(n_agents, n_items)

        nash_welfare = best_nash_welfare(valuation_matrix, num_segments=200)
        if nash_welfare is None:
            continue

        util_welfare = best_utilitarian_welfare(valuation_matrix)

        matrices.append(valuation_matrix)
        nash_values.append(nash_welfare)
        util_values.append(util_welfare)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    filename = output_path / f"{n_agents}_{n_items}_{num_matrices}_dataset.npz"
    np.savez_compressed(
        filename,
        matrices=np.array(matrices),
        nash_welfare=np.array(nash_values),
        util_welfare=np.array(util_values)
    )

    return len(matrices)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=1000)
    parser.add_argument('--n_min', type=int, default=10)
    parser.add_argument('--n_max', type=int, default=30)
    parser.add_argument('--m_min', type=int, default=10)
    parser.add_argument('--m_max', type=int, default=30)
    parser.add_argument('--output_dir', type=str, default='datasets/heatmap')
    args = parser.parse_args()

    # Build list of (n, m) pairs where m >= n
    configs = []
    for n in range(args.n_min, args.n_max + 1):
        for m in range(max(n, args.m_min), args.m_max + 1):
            configs.append((n, m))

    print(f"Generating datasets for {len(configs)} configurations")
    print(f"Samples per config: {args.num_samples}")
    print(f"Output directory: {args.output_dir}")
    print()

    for n, m in tqdm(configs, desc="Generating datasets"):
        count = generate_and_save_dataset(n, m, args.num_samples, args.output_dir)
        if count < args.num_samples * 0.9:
            tqdm.write(f"Warning: Only got {count} valid samples for n={n}, m={m}")

    print(f"\nDone! Generated {len(configs)} dataset files in {args.output_dir}/")


if __name__ == "__main__":
    main()
