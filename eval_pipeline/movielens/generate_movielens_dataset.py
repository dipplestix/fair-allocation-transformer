"""
Generate MovieLens-based Datasets for Fair Allocation Evaluation

This script processes MovieLens rating datasets and converts them into valuation matrices
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
from convert_movielens import load_movielens_dataset


def generate_movielens_dataset(
    csv_path: str,
    n_agents: int,
    n_items: int,
    num_matrices: int,
    output_file: str,
    max_rating: float = 5.0,
    seed: int = 42
):
    """
    Generate dataset from MovieLens data with precomputed max welfare values.
    Ratings are directly normalized to [0, 1] range as valuations.

    Args:
        csv_path: Path to ratings.csv file
        n_agents: Number of agents per valuation matrix
        n_items: Number of items per valuation matrix
        num_matrices: Number of valuation matrices to generate
        output_file: Output .npz file path
        max_rating: Maximum rating value for normalization (default: 5.0)
        seed: Random seed for reproducibility
    """
    print(f"="*80)
    print(f"Generating MovieLens-based dataset")
    print(f"Dataset: {csv_path}")
    print(f"Configuration: {n_agents} agents, {n_items} items, {num_matrices} matrices")
    print(f"Rating normalization: ratings / {max_rating}")
    print(f"="*80)

    # Load and convert MovieLens data
    print(f"\n[1/3] Loading and converting MovieLens data...")
    try:
        matrices = load_movielens_dataset(
            csv_path, n_agents, n_items, num_matrices, max_rating, seed
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
        description='Generate MovieLens-based dataset with precomputed max welfare',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate dataset with 10 agents, 12 items
  python generate_movielens_dataset.py \\
    --csv raw_datasets/ratings.csv \\
    --agents 10 --items 12 --num_matrices 1000

  # Custom max rating (if using different scale)
  python generate_movielens_dataset.py \\
    --csv raw_datasets/ratings.csv \\
    --agents 10 --items 12 --num_matrices 1000 \\
    --max_rating 10.0

  # Custom output directory
  uv run generate_movielens_dataset.py \\
    --csv raw_datasets/ratings.csv \\
    --agents 10 --items 12 --num_matrices 1000 \\
    --output ../datasets/movielens-custom/10_12_1000_dataset.npz
        """
    )

    parser.add_argument(
        '--csv',
        type=str,
        required=True,
        help='Path to MovieLens ratings.csv file'
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
        '--max_rating',
        type=float,
        default=5.0,
        help='Maximum rating value for normalization (default: 5.0 for MovieLens)'
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
        output_filename = f"{args.agents}_{args.items}_{args.num_matrices}_dataset.npz"
        # Save to datasets directory
        output_dir = Path(__file__).parent.parent / "datasets" / "movielens"
        args.output = str(output_dir / output_filename)

    # Ensure .npz extension
    if not args.output.endswith('.npz'):
        args.output += '.npz'

    # Run dataset generation
    generate_movielens_dataset(
        csv_path=args.csv,
        n_agents=args.agents,
        n_items=args.items,
        num_matrices=args.num_matrices,
        output_file=args.output,
        max_rating=args.max_rating,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
