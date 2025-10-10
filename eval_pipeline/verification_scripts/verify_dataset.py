import argparse
import numpy as np

def verify_dataset(dataset_file, num_samples=10):
    """Display random samples from the dataset for verification"""
    print(f"Loading dataset from {dataset_file}...")

    try:
        data = np.load(dataset_file, allow_pickle=True)
    except FileNotFoundError:
        print(f"Error: File {dataset_file} not found")
        return
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    matrices = data['matrices']
    nash_welfare = data['nash_welfare']
    util_welfare = data['util_welfare']

    total_matrices = len(matrices)
    print(f"Dataset contains {total_matrices} matrices")
    print(f"Matrix dimensions: {matrices[0].shape[0]} agents × {matrices[0].shape[1]} items")

    # Determine how many samples to show
    samples_to_show = min(num_samples, total_matrices)
    print(f"Showing {samples_to_show} random samples:\n")

    # Sample random indices
    if total_matrices <= num_samples:
        indices = list(range(total_matrices))
    else:
        indices = np.random.choice(total_matrices, size=num_samples, replace=False)
        indices = sorted(indices)

    for i, idx in enumerate(indices):
        matrix = matrices[idx]
        nash = nash_welfare[idx]
        util = util_welfare[idx]

        print(f"Sample {i+1} (Index {idx}):")
        print(f"Agents: {matrix.shape[0]}")
        print(f"Items: {matrix.shape[1]}")
        print(f"Valuation Matrix:")
        print(matrix)
        print(f"Max Utilitarian Welfare: {util}")
        print(f"Max Nash Welfare: {nash}")
        print()

def main():
    parser = argparse.ArgumentParser(description='Verify dataset by displaying random samples')
    parser.add_argument('dataset_file', help='Path to .npz dataset file')
    parser.add_argument('--samples', type=int, default=10, help='Number of random samples to show (default: 10)')

    args = parser.parse_args()

    if not args.dataset_file.endswith('.npz'):
        print("Warning: File should have .npz extension")

    verify_dataset(args.dataset_file, args.samples)

if __name__ == "__main__":
    main()