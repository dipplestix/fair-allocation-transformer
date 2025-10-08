import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils.calculations import calculate_agent_bundle_values, is_envy_free, is_ef1, is_efx, utility_sum, nash_welfare
from utils.calculations import calculate_agent_bundle_values_batch, is_envy_free_batch, utility_sum_batch, nash_welfare_batch, is_ef1_batch, is_efx_batch
from utils.inference import get_model_allocations_batch, get_random_allocations_batch, get_rr_allocations_batch, get_rr_allocations_batch_old
from load_model import load_model


def evaluate_single_allocation(valuation_matrix, allocation_matrix, max_nash_welfare, max_util_welfare):
    """Evaluate a single allocation with precomputed max welfare values"""
    agent_bundle_values = calculate_agent_bundle_values(valuation_matrix, allocation_matrix)

    results = {}

    # Fairness properties
    results['envy_free'] = is_envy_free(agent_bundle_values)
    results['ef1'] = is_ef1(valuation_matrix, allocation_matrix, agent_bundle_values)
    results['efx'] = is_efx(valuation_matrix, allocation_matrix, agent_bundle_values)

    # Welfare metrics
    current_util_sum = utility_sum(agent_bundle_values)
    current_nash = nash_welfare(agent_bundle_values)

    results['utility_sum'] = current_util_sum
    results['nash_welfare'] = current_nash

    # Fractions using precomputed max values
    results['fraction_util_welfare'] = current_util_sum / max_util_welfare if max_util_welfare > 0 else 0
    results['fraction_nash_welfare'] = current_nash / max_nash_welfare if max_nash_welfare > 0 else 0

    return results

def evaluate_batch_allocations(valuation_matrices, allocation_matrices, max_nash_welfares, max_util_welfares):
    """Evaluate a batch of allocations with precomputed max welfare values"""
    agent_bundle_values_batch = calculate_agent_bundle_values_batch(valuation_matrices, allocation_matrices)

    results_list = []

    # Fairness properties
    envy_free_array = is_envy_free_batch(agent_bundle_values_batch)
    ef1_array = is_ef1_batch(valuation_matrices, allocation_matrices, agent_bundle_values_batch)
    efx_array = is_efx_batch(valuation_matrices, allocation_matrices, agent_bundle_values_batch)

    # print number of true in each
    # print(f"Envy-free: {np.sum(envy_free_array)}/{len(envy_free_array)}")
    # print(f"EF1: {np.sum(ef1_array)}/{len(ef1_array)}")
    # print(f"EFx: {np.sum(efx_array)}/{len(efx_array)}")

    # Welfare metrics
    util_sums = utility_sum_batch(agent_bundle_values_batch)
    nash_welfares = nash_welfare_batch(agent_bundle_values_batch)

    # group results

    results_tensors = [envy_free_array, ef1_array, efx_array,
                       util_sums, nash_welfares]  # each shaped (N, )

    return results_tensors


def run_evaluation(data_file, output_csv, output_npz, batch_size=100, eval_type='model'):
    """Run evaluation on all matrices in the dataset"""
    print(f"Loading dataset from {data_file}...")
    data = np.load(data_file)

    data_file = data_file.split('/')[-1].split('_')
    print(f"Dataset parameters: Agents={data_file[0]}, Items={data_file[1]}, Size={data_file[2]}")

    matrices = data['matrices']
    nash_welfare_max = data['nash_welfare']
    util_welfare_max = data['util_welfare']

    print(f"Dataset loaded: {len(matrices)} matrices")
    print(f"Matrix shape: {matrices[0].shape}")

    if eval_type.startswith('model'):
        print(f"Loading model {eval_type}...")
        model = load_model(eval_type)
        print(f"Model {eval_type} loaded.")

    output_csv = "results/" + output_csv + "_" + data_file[0] + "_" + data_file[1] + "_" + data_file[2] + "_" + eval_type + ".csv"
    output_npz = "results/" + output_npz + "_" + data_file[0] + "_" + data_file[1] + "_" + data_file[2] + "_" + eval_type + ".npz"

    all_results = []
    all_valuations = []
    all_allocations = []
    all_envy_info = []
    all_utilities = []
    all_max_utilities = []
    all_fractions = []

    print("Running evaluations...")
    for i in tqdm(range(0, len(matrices), batch_size), desc="Evaluating matrices"):

        # batch processing
        batch_end = min(i + batch_size, len(matrices))
        batch_matrices = matrices[i:batch_end]
        batch_nash_max = nash_welfare_max[i:batch_end]
        batch_util_max = util_welfare_max[i:batch_end]

        if eval_type.startswith('model'):
            # Get model allocations for the batch
            batch_allocations = get_model_allocations_batch(model, batch_matrices)
            
        elif eval_type == 'rr':
            batch_allocations = get_rr_allocations_batch_old(batch_matrices)
            # since we are generating 5 allocations per matrix for random and rr, we need to repeat the max values and valuation matrices
            # batch_nash_max = np.repeat(batch_nash_max, 5)
            # batch_util_max = np.repeat(batch_util_max, 5)
            # batch_matrices = np.repeat(batch_matrices, 5, axis=0)
        else:  # random
            batch_allocations = get_random_allocations_batch(batch_matrices)
            # since we are generating 5 allocations per matrix for random and rr, we need to repeat the max values and valuation matrices
            batch_nash_max = np.repeat(batch_nash_max, 5)
            batch_util_max = np.repeat(batch_util_max, 5)
            batch_matrices = np.repeat(batch_matrices, 5, axis=0)

        # Evaluate the allocation
        results_tensors = evaluate_batch_allocations(batch_matrices, batch_allocations, batch_nash_max, batch_util_max)

        if eval_type in ['random']:
            # For random and rr, we generated 5 allocations per matrix, so we need to reshape results
            for j in range(len(results_tensors)):
                reshaped = results_tensors[j].reshape(batch_size, 5)  # (num_matrices_in_batch, 5)
                if j < 3:  # boolean arrays
                    results_tensors[j] = np.mean(reshaped, axis=1) >= 0.5  # majority vote
                else:  # numeric arrays
                    results_tensors[j] = np.mean(reshaped, axis=1)  # take mean value

        result_tensor = np.stack(results_tensors, axis=1)  # shape (num_matrices_in_batch, 5)
        # Convert boolean to int for easier storage
        result_tensor[:, :3] = result_tensor[:, :3].astype(int)
        # Append results to list
        for j in range(result_tensor.shape[0]):
            matrix_id = i + j
            num_agents, num_items = batch_matrices[j].shape
            nash_max = batch_nash_max[j]
            util_max = batch_util_max[j]
            allocation_matrix = batch_allocations[j]
            matrix = batch_matrices[j]

            result_dict = {
                'matrix_id': matrix_id,
                'num_agents': num_agents,
                'num_items': num_items,
                'max_nash_welfare': nash_max,
                'max_util_welfare': util_max,
                'envy_free': bool(result_tensor[j, 0]),
                'ef1': bool(result_tensor[j, 1]),
                'efx': bool(result_tensor[j, 2]),
                'utility_sum': float(result_tensor[j, 3]),
                'nash_welfare': float(result_tensor[j, 4]),
                'fraction_util_welfare': float(result_tensor[j, 3]) / util_max if util_max > 0 else 0,
                'fraction_nash_welfare': float(result_tensor[j, 4]) / nash_max if nash_max > 0 else 0
            }
            all_results.append(result_dict)

        # Store data for NPZ file
        all_valuations.append(matrix)
        all_allocations.append(allocation_matrix)
        all_envy_info.append([result_dict['envy_free'], result_dict['ef1'], result_dict['efx']])
        all_utilities.append([result_dict['utility_sum'], result_dict['nash_welfare']])
        all_max_utilities.append([nash_max, util_max])
        all_fractions.append([result_dict['fraction_util_welfare'], result_dict['fraction_nash_welfare']])

    # Create DataFrame and save CSV
    df = pd.DataFrame(all_results)

    # Reorder columns for better readability
    columns = ['matrix_id', 'num_agents', 'num_items', 'max_nash_welfare', 'max_util_welfare',
               'envy_free', 'ef1', 'efx',
               'utility_sum', 'nash_welfare',
               'fraction_util_welfare', 'fraction_nash_welfare']

    df = df[columns]
    df.to_csv(output_csv, index=False)

    # Save NPZ file with detailed allocation data
    print(f"Saving detailed results to {output_npz}...")
    np.savez_compressed(output_npz,
        valuation_matrices=np.array(all_valuations),
        allocation_matrices=np.array(all_allocations),
        envy_info=np.array(all_envy_info),  # [envy_free, ef1, efx]
        utilities=np.array(all_utilities),  # [utility_sum, nash_welfare]
        max_utilities=np.array(all_max_utilities),  # [max_nash, max_util]
        fractions=np.array(all_fractions)  # [fraction_util, fraction_nash]
    )

    print(f"Evaluation complete! CSV saved to {output_csv}, NPZ saved to {output_npz}")
    print(f"Processed {len(df)} allocations")

    # Summary statistics
    print("\n========Summary Statistics:========")
    print(f"Envy-free allocations: {df['envy_free'].sum()} ({df['envy_free'].mean()*100:.1f}%)")
    print(f"EF1 allocations: {df['ef1'].sum()} ({df['ef1'].mean()*100:.1f}%)")
    print(f"EFx allocations: {df['efx'].sum()} ({df['efx'].mean()*100:.1f}%)")
    print(f"Average fraction of best utility: {df['fraction_util_welfare'].mean():.3f}")
    print(f"Average fraction of best Nash welfare: {df['fraction_nash_welfare'].mean():.3f}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate model allocations on precomputed dataset')
    parser.add_argument('data_file', help='Input .npz file with matrices and max welfare values')
    parser.add_argument('--output_csv', default='evaluation_results', help='Output CSV filename')
    parser.add_argument('--output_npz', default='evaluation_results', help='Output NPZ filename')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size for processing (default: 100)')
    parser.add_argument('--eval_type', default='model0', help='Type of evaluation: model[#n], random, or round robin (rr) (default: model0)')

    args = parser.parse_args()

    if not args.data_file.endswith('.npz'):
        print("Error: Input file must be a .npz file")
        return

    run_evaluation(args.data_file, args.output_csv, args.output_npz, args.batch_size, args.eval_type)

if __name__ == "__main__":
    main()