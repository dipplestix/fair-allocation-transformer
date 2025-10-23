import argparse
import itertools
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm
from utils.max_utility import best_nash_welfare, best_utilitarian_welfare, best_nash_welfare_bruteforce

def generate_valuation_matrix(n_agents, m_items):
    """Generate random valuation matrix with values between 0 and 1"""
    return np.random.uniform(0, 1, size=(n_agents, m_items))


def generate_dataset(n_agents, n_items, num_matrices, output_file, seed=10):
    """Generate dataset of valuation matrices with precomputed max welfare values"""

    matrices = []
    nash_values = []
    util_values = []

    # Timing variables
    nash_times = []
    util_times = []
    generation_times = []

    # Set seed
    np.random.seed(seed * n_items + n_agents)

    print(f"Generating {num_matrices} valuation matrices ({n_agents} agents, {n_items} items)...")

    total_start_time = time.time()

    for i in tqdm(range(num_matrices), desc="Processing matrices"):
        # Time valuation matrix generation
        gen_start = time.time()
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



def generate_datasets(agent_counts, item_counts, num_matrices, output_dir, seed=10, pairwise=False):
    """Generate datasets for multiple (agents, items) combinations."""

    if not agent_counts:
        raise ValueError("agent_counts must contain at least one value")

    if not item_counts:
        raise ValueError("item_counts must contain at least one value")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if pairwise:
        if len(agent_counts) != len(item_counts):
            raise ValueError("When pairwise=True, agent_counts and item_counts must have the same length")
        combinations = list(zip(agent_counts, item_counts))
    else:
        combinations = list(itertools.product(agent_counts, item_counts))

    total = len(combinations)

    for index, (n_agents, n_items) in enumerate(combinations, start=1):
        print(f"\n====== Dataset {index}/{total}: {n_agents} agents, {n_items} items ======")
        output_file = output_dir / f"{n_agents}_{n_items}_{num_matrices}_dataset.npz"
        generate_dataset(n_agents, n_items, num_matrices, output_file, seed=seed)


def main():
    parser = argparse.ArgumentParser(description='Generate dataset of valuation matrices with precomputed max welfare')
    parser.add_argument('--agents', type=int, nargs='+', default=[10], help='One or more agent counts (default: 10)')
    parser.add_argument('--items', type=int, nargs='+', default=[14], help='One or more item counts (default: 14)')
    parser.add_argument('--num_matrices', type=int, required=True, help='Number of valuation matrices to generate')
    parser.add_argument('--output', type=str, required=False, help='Output .npz file (single dataset) or directory (multiple datasets). Default naming uses datasets/<agents>_<items>_<num_matrices>_dataset.npz')
    parser.add_argument('--seed', type=int, default=10, help='Base random seed (default: 10)')
    parser.add_argument('--pairwise', action='store_true', help='Pair each agents/items entry instead of generating every combination')

    args = parser.parse_args()

    if args.pairwise and len(args.agents) != len(args.items):
        parser.error('When using --pairwise, provide the same number of --agents and --items values.')

    if args.pairwise:
        total_combinations = len(args.agents)
    else:
        total_combinations = len(args.agents) * len(args.items)

    if total_combinations == 1:
        n_agents = args.agents[0]
        n_items = args.items[0]
        output_path = args.output or f"datasets/{n_agents}_{n_items}_{args.num_matrices}_dataset.npz"

        if not output_path.endswith('.npz'):
            output_path += '.npz'

        generate_dataset(n_agents, n_items, args.num_matrices, output_path, seed=args.seed)
    else:
        output_dir = args.output or 'datasets'
        output_dir_path = Path(output_dir)

        if output_dir_path.suffix == '.npz':
            parser.error('Provide a directory for --output when generating multiple datasets.')

        generate_datasets(args.agents, args.items, args.num_matrices, output_dir_path, seed=args.seed, pairwise=args.pairwise)

if __name__ == "__main__":
    main()
