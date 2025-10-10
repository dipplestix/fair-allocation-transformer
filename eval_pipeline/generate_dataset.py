import argparse
import time
import numpy as np
from tqdm import tqdm
from utils.max_utility import best_nash_welfare, best_utilitarian_welfare, best_nash_welfare_bruteforce

def generate_valuation_matrix(n_agents, m_items):
    """Generate random valuation matrix with values between 0 and 1"""
    return np.random.uniform(0, 1, size=(n_agents, m_items))


def generate_dataset(n_agents, n_items, num_matrices, output_file):
    """Generate dataset of valuation matrices with precomputed max welfare values"""

    matrices = []
    nash_values = []
    util_values = []

    # Timing variables
    nash_times = []
    util_times = []
    generation_times = []

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
    output_file = "datasets/" + output_file
    
    print(f"\n======Completed generating {num_matrices} matrices======")
    
    # Save as compressed numpy archive with timing data
    print(f"Saving dataset to {output_file}...")
    np.savez_compressed(output_file,
        matrices=np.array(matrices),        # shape: (num_matrices, n_agents, n_items)
        nash_welfare=np.array(nash_values), # shape: (num_matrices,)
        util_welfare=np.array(util_values), # shape: (num_matrices,)
    )

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

    

def main():
    parser = argparse.ArgumentParser(description='Generate dataset of valuation matrices with precomputed max welfare')
    parser.add_argument('--agents', type=int, default=10, help='Number of agents (default: 10)')
    parser.add_argument('--items', type=int, default=14, help='Number of items (default: 14)')
    parser.add_argument('--num_matrices', type=int, required=True, help='Number of valuation matrices to generate')
    parser.add_argument('--output', type=str, required=True, help='Output .npz filename')

    args = parser.parse_args()

    if not args.output.endswith('.npz'):
        args.output += '.npz'

    generate_dataset(args.agents, args.items, args.num_matrices, args.output)

if __name__ == "__main__":
    main()