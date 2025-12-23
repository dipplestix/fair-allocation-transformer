"""
Generate PrefLib-based Datasets for Fair Allocation Evaluation

This script processes PrefLib voting datasets and converts them into valuation matrices
for fair allocation problems, with precomputed optimal welfare values.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from utils.max_utility import best_nash_welfare, best_utilitarian_welfare
from convert_preflib import load_preflib_dataset, get_dataset_name


def generate_preflib_dataset(
    dataset_dir: str,
    n_agents: int,
    n_items: int,
    num_matrices: int,
    output_file: str,
    method: str = 'borda',
    seed: int = 42
):
    """
    Generate dataset from PrefLib data with precomputed max welfare values.

    Args:
        dataset_dir: Path to PrefLib dataset directory containing .soi files
        n_agents: Number of agents per valuation matrix
        n_items: Number of items per valuation matrix
        num_matrices: Number of valuation matrices to generate
        output_file: Output .npz file path
        method: Valuation conversion method ('borda' or 'linear')
        seed: Random seed for reproducibility
    """
    print(f"="*80)
    print(f"Generating PrefLib-based dataset")
    print(f"Dataset: {dataset_dir}")
    print(f"Configuration: {n_agents} agents, {n_items} items, {num_matrices} matrices")
    print(f"Conversion method: {method}")
    print(f"="*80)

    # Load and convert PrefLib data
    print("\n[1/3] Loading and converting PrefLib data...")
    try:
        matrices = load_preflib_dataset(
            dataset_dir, n_agents, n_items, num_matrices, method, seed
        )
    except ValueError as e:
        print(f"\nError: {e}")
        sys.exit(1)

    print(f"Successfully created {len(matrices)} valuation matrices")

    # Compute optimal welfare values
    print(f"\n[2/3] Computing optimal welfare values...")

    nash_values = []
    util_values = []
    nash_times = []
    util_times = []

    total_start_time = time.time()

    for i, valuation_matrix in enumerate(tqdm(matrices, desc="Processing matrices")):
        # Compute Nash welfare
        nash_start = time.time()
        nash_welfare = best_nash_welfare(valuation_matrix, num_segments=200)

        if nash_welfare is None:
            print(f"\nWarning: No optimal solution found for matrix {i}. Skipping.")
            continue

        nash_times.append((time.time() - nash_start) * 1000)

        # Compute utilitarian welfare
        util_start = time.time()
        util_welfare = best_utilitarian_welfare(valuation_matrix)
        util_times.append((time.time() - util_start) * 1000)

        nash_values.append(nash_welfare)
        util_values.append(util_welfare)

    total_time = time.time() - total_start_time

    # Calculate statistics
    avg_nash_time = np.mean(nash_times)
    avg_util_time = np.mean(util_times)

    print(f"\n{'='*80}")
    print(f"Completed processing {len(matrices)} matrices")
    print(f"{'='*80}")

    # Save dataset
    print(f"\n[3/3] Saving dataset...")

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as compressed numpy archive
    print(f"Saving to: {output_path}")
    np.savez_compressed(
        output_path,
        matrices=np.array(matrices),        # shape: (num_matrices, n_agents, n_items)
        nash_welfare=np.array(nash_values), # shape: (num_matrices,)
        util_welfare=np.array(util_values), # shape: (num_matrices,)
    )

    # Save timing data to CSV
    timing_output_file = output_path.with_name(output_path.stem + '_timing.csv')
    print(f"Saving timing data to: {timing_output_file}")
    with open(timing_output_file, 'w') as f:
        f.write("matrix_index,nash_time_ms,util_time_ms\n")
        for i in range(len(matrices)):
            f.write(f"{i},{nash_times[i]:.4f},{util_times[i]:.4f}\n")

    print(f"\nDataset saved successfully!")
    print(f"Matrices shape: {np.array(matrices).shape}")
    print(f"Nash welfare shape: {np.array(nash_values).shape}")
    print(f"Utilitarian welfare shape: {np.array(util_values).shape}")

    print(f"\n{'='*80}")
    print(f"Timing Statistics")
    print(f"{'='*80}")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Average Nash welfare calculation time: {avg_nash_time:.2f}ms")
    print(f"Average utilitarian welfare calculation time: {avg_util_time:.2f}ms")
    print(f"Average total time per matrix: {(avg_nash_time + avg_util_time):.2f}ms")

    # Save timing statistics
    stats_output_file = output_path.with_name(output_path.stem + '_timing_stats.txt')
    print(f"Saving timing statistics to: {stats_output_file}")
    with open(stats_output_file, 'w') as f:
        f.write(f"Total processing time: {total_time:.2f} seconds\n")
        f.write(f"Average Nash welfare calculation time: {avg_nash_time:.2f}ms\n")
        f.write(f"Average utilitarian welfare calculation time: {avg_util_time:.2f}ms\n")
        f.write(f"Average total time per matrix: {(avg_nash_time + avg_util_time):.2f}ms\n")

    print(f"\n{'='*80}")
    print(f"Done! Dataset ready for evaluation.")
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate PrefLib-based dataset with precomputed max welfare',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate dataset from french-irv-2007 with 10 agents, 12 items
  python generate_preflib_dataset.py \\
    --dataset raw_datasets/00072_french-irv-2007 \\
    --agents 10 --items 12 --num_matrices 1000

  # Use linear conversion method
  python generate_preflib_dataset.py \\
    --dataset raw_datasets/00072_french-irv-2007 \\
    --agents 10 --items 12 --num_matrices 1000 \\
    --method linear

  # Custom output directory
  python generate_preflib_dataset.py \\
    --dataset raw_datasets/00072_french-irv-2007 \\
    --agents 10 --items 12 --num_matrices 1000 \\
    --output ../datasets/custom-dir/10_12_1000_dataset.npz
        """
    )

    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Path to PrefLib dataset directory (e.g., raw_datasets/00072_french-irv-2007)'
    )
    parser.add_argument(
        '--agents',
        type=int,
        default=10,
        help='Number of agents per matrix (default: 10)'
    )
    parser.add_argument(
        '--items',
        type=int,
        default=12,
        help='Number of items per matrix (default: 12)'
    )
    parser.add_argument(
        '--num_matrices',
        type=int,
        required=True,
        help='Number of valuation matrices to generate'
    )
    parser.add_argument(
        '--method',
        type=str,
        default='borda',
        choices=['borda', 'linear'],
        help='Ranking to valuation conversion method (default: borda)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=False,
        help='Output .npz file path (default: auto-generated based on parameters)'
    )

    args = parser.parse_args()

    # Generate default output filename if not provided
    if not args.output:
        dataset_name = get_dataset_name(args.dataset)
        # Standard naming format: {agents}_{items}_{num_matrices}_dataset.npz
        output_filename = f"{args.agents}_{args.items}_{args.num_matrices}_dataset.npz"
        # Save to method-specific subdirectory: datasets/{method}-{dataset_name}/
        output_dir = Path(__file__).parent.parent / "datasets" / f"{args.method}-{dataset_name}"
        args.output = str(output_dir / output_filename)

    # Ensure .npz extension
    if not args.output.endswith('.npz'):
        args.output += '.npz'

    # Run dataset generation
    generate_preflib_dataset(
        dataset_dir=args.dataset,
        n_agents=args.agents,
        n_items=args.items,
        num_matrices=args.num_matrices,
        output_file=args.output,
        method=args.method,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
