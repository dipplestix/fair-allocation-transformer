import argparse
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm
from utils.max_utility import best_nash_welfare, best_utilitarian_welfare, best_nash_welfare_bruteforce

def generate_valuation_matrix(n_agents, m_items):
    """Generate random valuation matrix with values between 0 and 1"""
    return np.random.uniform(0, 1, size=(n_agents, m_items))


def generate_correlated_valuation_matrix(n_agents, m_items, rank=3, noise_scale=0.1):
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
    return np.clip(valuations, 0, 1)


def generate_dataset(n_agents, n_items, num_matrices, output_file, seed=10, distribution='uniform'):
    """Generate dataset of valuation matrices with precomputed max welfare values

    Args:
        n_agents: Number of agents
        n_items: Number of items
        num_matrices: Number of matrices to generate
        output_file: Output file path
        seed: Random seed
        distribution: 'uniform' or 'correlated'
    """

    matrices = []
    nash_values = []
    util_values = []

    # Timing variables
    nash_times = []
    util_times = []
    generation_times = []

    # Set seed
    np.random.seed(seed * n_items + n_agents)

    print(f"Generating {num_matrices} valuation matrices ({n_agents} agents, {n_items} items, {distribution} distribution)...")

    total_start_time = time.time()

    for i in tqdm(range(num_matrices), desc="Processing matrices"):
        # Time valuation matrix generation
        gen_start = time.time()
        if distribution == 'correlated':
            rank = min(3, n_agents, n_items)
            valuation_matrix = generate_correlated_valuation_matrix(n_agents, n_items, rank=rank)
        else:
            valuation_matrix = generate_valuation_matrix(n_agents, n_items)
        generation_times.append((time.time() - gen_start) * 1000)

        # Time Nash welfare calculation
        nash_start = time.time()
        nash_welfare = best_nash_welfare(valuation_matrix, num_segments=200)
        if nash_welfare is None:
            # reprocess step i
            print(f"Warning: No optimal solution found for matrix {i}. Regenerating matrix.")
            i -= 1
            continue
        nash_times.append((time.time() - nash_start) * 1000)

        # nash_welfare_bruteforce = best_nash_welfare_bruteforce(valuation_matrix) # Uncomment to verify correctness

        # Time utilitarian welfare calculation
        util_start = time.time()
        util_welfare = best_utilitarian_welfare(valuation_matrix)
        util_times.append((time.time() - util_start) * 1000)

        # Store results
        matrices.append(valuation_matrix)
        nash_values.append(nash_welfare)
        util_values.append(util_welfare)

        # print(f"Nash approximation error: {abs(nash_welfare - nash_welfare_bruteforce)/nash_welfare_bruteforce * 100:.2f}%")

    total_time = time.time() - total_start_time

    # Calculate timing statistics
    avg_generation_time = np.mean(generation_times)
    avg_nash_time = np.mean(nash_times)
    avg_util_time = np.mean(util_times)
    
    
    print(f"\n======Completed generating {num_matrices} matrices======")

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as compressed numpy archive with timing data
    print(f"Saving dataset to {output_path}...")
    np.savez_compressed(output_path,
        matrices=np.array(matrices),        # shape: (num_matrices, n_agents, n_items)
        nash_welfare=np.array(nash_values), # shape: (num_matrices,)
        util_welfare=np.array(util_values), # shape: (num_matrices,)
    )

    # save timing data to csv
    timing_output_file = output_path.with_name(output_path.stem + '_timing.csv')
    print(f"Saving timing data to {timing_output_file}...")
    with open(timing_output_file, 'w') as f:
        f.write("matrix_index,generation_time_ms,nash_time_ms,util_time_ms\n")
        for i in range(num_matrices):
            f.write(f"{i},{generation_times[i]:.4f},{nash_times[i]:.4f},{util_times[i]:.4f}\n")

    print(f"Dataset saved successfully!")
    print(f"Matrices shape: {np.array(matrices).shape}")
    print(f"Nash welfare shape: {np.array(nash_values).shape}")
    print(f"Utilitarian welfare shape: {np.array(util_values).shape}")

    print(f"\n======Timing Statistics:======")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Average matrix generation time: {avg_generation_time:.2f}ms")
    print(f"Average Nash welfare calculation time: {avg_nash_time:.2f}ms")
    print(f"Average utilitarian welfare calculation time: {avg_util_time:.2f}ms")
    print(f"Average total time per matrix: {(avg_generation_time + avg_nash_time + avg_util_time):.2f}ms")

    # save timing statistics to a text file
    stats_output_file = output_path.with_name(output_path.stem + '_timing_stats.txt')
    print(f"Saving timing statistics to {stats_output_file}...")
    with open(stats_output_file, 'w') as f:
        f.write(f"Total processing time: {total_time:.2f} seconds\n")
        f.write(f"Average matrix generation time: {avg_generation_time:.2f}ms\n")
        f.write(f"Average Nash welfare calculation time: {avg_nash_time:.2f}ms\n")
        f.write(f"Average utilitarian welfare calculation time: {avg_util_time:.2f}ms\n")
        f.write(f"Average total time per matrix: {(avg_generation_time + avg_nash_time + avg_util_time):.2f}ms\n")

    

def main():
    parser = argparse.ArgumentParser(description='Generate dataset of valuation matrices with precomputed max welfare')
    parser.add_argument('--agents', type=int, default=10, help='Number of agents (default: 10)')
    parser.add_argument('--items', type=int, default=14, help='Number of items (default: 14)')
    parser.add_argument('--num_matrices', type=int, required=True, help='Number of valuation matrices to generate')
    parser.add_argument('--output', type=str, required=False, help='Output .npz file to save the dataset, default: dataset_<agents>_<items>_<num_matrices>_dataset.npz')
    parser.add_argument('--distribution', type=str, default='uniform', choices=['uniform', 'correlated'],
                        help='Valuation distribution: uniform (default) or correlated (low-rank + noise)')

    args = parser.parse_args()

    if not args.output:
        args.output = f"datasets/{args.agents}_{args.items}_{args.num_matrices}_{args.distribution}_dataset.npz"

    if not args.output.endswith('.npz'):
        args.output += '.npz'

    generate_dataset(args.agents, args.items, args.num_matrices, args.output, distribution=args.distribution)

if __name__ == "__main__":
    main()