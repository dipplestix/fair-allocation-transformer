import argparse
import numpy as np

def verify_output(output_file, num_problems=10):
    """Display first n problems from evaluation output for verification"""
    print(f"Loading evaluation results from {output_file}...")

    try:
        data = np.load(output_file, allow_pickle=True)
    except FileNotFoundError:
        print(f"Error: File {output_file} not found")
        return
    except Exception as e:
        print(f"Error loading evaluation results: {e}")
        return

    valuation_matrices = data['valuation_matrices']
    allocation_matrices = data['allocation_matrices']
    envy_info = data['envy_info']
    utilities = data['utilities']
    max_utilities = data['max_utilities']
    fractions = data['fractions']

    total_problems = len(valuation_matrices)
    print(f"Evaluation results contain {total_problems} problems")
    print(f"Matrix dimensions: {valuation_matrices[0].shape[0]} agents × {valuation_matrices[0].shape[1]} items")

    # Determine how many problems to show
    problems_to_show = min(num_problems, total_problems)
    print(f"Showing first {problems_to_show} problems:\n")

    for i in range(problems_to_show):
        valuation = valuation_matrices[i]
        allocation = allocation_matrices[i]
        envy = envy_info[i]  # [envy_free, ef1, efx]
        util = utilities[i]  # [utility_sum, nash_welfare]
        max_util = max_utilities[i]  # [max_nash, max_util]
        frac = fractions[i]  # [fraction_util, fraction_nash]

        print(f"Problem {i+1}:")
        print(f"Agents: {valuation.shape[0]}")
        print(f"Items: {valuation.shape[1]}")
        print(f"Valuation Matrix:")
        print(repr(valuation))
        print(f"Generated Allocation:")
        print(repr(allocation))
        print(f"Envy Information:")
        print(f"  Envy-Free: {bool(envy[0])}")
        print(f"  EF1: {bool(envy[1])}")
        print(f"  EFx: {bool(envy[2])}")
        print(f"Utilities:")
        print(f"  Utility Sum: {util[0]:.3f}")
        print(f"  Nash Welfare: {util[1]:.3f}")
        print(f"Max Utilities:")
        print(f"  Max Nash Welfare: {max_util[0]:.3f}")
        print(f"  Max Utilitarian Welfare: {max_util[1]:.3f}")
        print(f"Fraction of Best Utility:")
        print(f"  Utilitarian Fraction: {frac[0]:.3f}")
        print(f"  Nash Fraction: {frac[1]:.3f}")
        print()

    # Summary statistics
    avg_util_fraction = np.mean(fractions[:, 0])
    avg_nash_fraction = np.mean(fractions[:, 1])
    proportion_ef = np.mean(envy_info[:, 0])
    proportion_ef1 = np.mean(envy_info[:, 1])
    proportion_efx = np.mean(envy_info[:, 2])
    print("======= Summary Statistics =======")
    print(f"Average Utilitarian Fraction: {avg_util_fraction:.3f}")
    print(f"Average Nash Fraction: {avg_nash_fraction:.3f}")
    print(f"Proportion Envy-Free: {proportion_ef:.3f}")
    print(f"Proportion EF1: {proportion_ef1:.3f}")
    print(f"Proportion EFx: {proportion_efx:.3f}")
    print("==================================")
def main():
    parser = argparse.ArgumentParser(description='Verify evaluation output by displaying first n problems')
    parser.add_argument('output_file', help='Path to evaluation output .npz file')
    parser.add_argument('--problems', type=int, default=10, help='Number of problems to show (default: 10)')

    args = parser.parse_args()

    if not args.output_file.endswith('.npz'):
        print("Warning: File should have .npz extension")

    verify_output(args.output_file, args.problems)

if __name__ == "__main__":
    main()