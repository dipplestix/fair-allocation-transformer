import argparse
import os
import pandas as pd
import glob

def parse_filename(filename):
    """Parse filename to extract agents, items, matrices, and inference type"""
    # Remove path and extension
    basename = os.path.basename(filename).replace('.csv', '')

    # Expected format: evaluation_results_agents_items_matrices_inference
    parts = basename.split('_')
    if len(parts) >= 6 and parts[0] == 'evaluation' and parts[1] == 'results':
        try:
            agents = int(parts[2])
            items = int(parts[3])
            matrices = int(parts[4])
            inference = parts[5]
            return agents, items, matrices, inference
        except ValueError:
            return None
    return None

def print_summary_stats(df, agents=None, items=None, inference=None):
    """Print summary statistics for a dataframe"""
    if agents and items and inference:
        print(f"========Summary Statistics for Agents = {agents}, Items = {items}, Inference type = {inference}:========")
    elif inference:
        print(f"========Summary Statistics for Inference type = {inference}:========")

    print(f"Envy-free allocations: {df['envy_free'].sum()} ({df['envy_free'].mean()*100:.1f}%)")
    print(f"EF1 allocations: {df['ef1'].sum()} ({df['ef1'].mean()*100:.1f}%)")
    print(f"EFx allocations: {df['efx'].sum()} ({df['efx'].mean()*100:.1f}%)")
    print(f"Average fraction of best utility: {df['fraction_util_welfare'].mean():.3f}")
    print(f"Average fraction of best Nash welfare: {df['fraction_nash_welfare'].mean():.3f}")
    print()

def create_summary_table(file_data, agent_item_pairs, inference_types):
    """Create summary table and save as CSV"""
    table_data = []

    for agents, items in sorted(agent_item_pairs):
        row_data = {'Agent-Item': f"{agents}-{items}"}

        for inference in inference_types:
            key = (agents, items, inference)
            if key in file_data:
                df = pd.read_csv(file_data[key])

                # Calculate metrics with 3 decimal places for percentages
                ef_pct = df['envy_free'].mean() * 100
                ef_sd = df['envy_free'].std() * 100
                ef1_pct = df['ef1'].mean() * 100
                ef1_sd = df['ef1'].std() * 100
                efx_pct = df['efx'].mean() * 100
                efx_sd = df['efx'].std() * 100
                util_frac = df['fraction_util_welfare'].mean()
                util_sd = df['fraction_util_welfare'].std()
                nash_frac = df['fraction_nash_welfare'].mean()
                nash_sd = df['fraction_nash_welfare'].std()

                row_data[f'{inference}_EF'] = f"{ef_pct:.3f}%"
                row_data[f'{inference}_EF_SD'] = f"{ef_sd:.3f}%"
                row_data[f'{inference}_EF1'] = f"{ef1_pct:.3f}%"
                row_data[f'{inference}_EF1_SD'] = f"{ef1_sd:.3f}%"
                row_data[f'{inference}_EFx'] = f"{efx_pct:.3f}%"
                row_data[f'{inference}_EFx_SD'] = f"{efx_sd:.3f}%"
                row_data[f'{inference}_Util_Frac'] = f"{util_frac:.3f}"
                row_data[f'{inference}_Util_Frac_SD'] = f"{util_sd:.3f}"
                row_data[f'{inference}_Nash_Frac'] = f"{nash_frac:.3f}"
                row_data[f'{inference}_Nash_Frac_SD'] = f"{nash_sd:.3f}"
            else:
                # Fill with empty values if data not found
                row_data[f'{inference}_EF'] = "N/A"
                row_data[f'{inference}_EF_SD'] = "N/A"
                row_data[f'{inference}_EF1'] = "N/A"
                row_data[f'{inference}_EF1_SD'] = "N/A"
                row_data[f'{inference}_EFx'] = "N/A"
                row_data[f'{inference}_EFx_SD'] = "N/A"
                row_data[f'{inference}_Util_Frac'] = "N/A"
                row_data[f'{inference}_Util_Frac_SD'] = "N/A"
                row_data[f'{inference}_Nash_Frac'] = "N/A"
                row_data[f'{inference}_Nash_Frac_SD'] = "N/A"

        table_data.append(row_data)

    # Create DataFrame and save
    summary_df = pd.DataFrame(table_data)
    summary_df.to_csv('summary.csv', index=False)
    print("Summary table saved to summary.csv")

def summarize_results(results_folder, inference_types, verbose, table_mode):
    """Summarize results from CSV files in the specified folder"""

    # Find all CSV files in the folder
    csv_files = glob.glob(os.path.join(results_folder, "*.csv"))

    if not csv_files:
        print(f"No CSV files found in {results_folder}")
        return

    # Parse filenames and group by (agents, items, inference)
    file_data = {}

    for csv_file in csv_files:
        parsed = parse_filename(csv_file)
        if parsed:
            agents, items, matrices, inference = parsed
            if inference in inference_types:
                key = (agents, items, inference)
                file_data[key] = csv_file

    if not file_data:
        print(f"No valid CSV files found for inference types: {inference_types}")
        return

    # Get unique agent/item combinations and inference types
    agent_item_pairs = set((agents, items) for agents, items, _ in file_data.keys())
    found_inference_types = set(inference for _, _, inference in file_data.keys())

    if table_mode:
        create_summary_table(file_data, agent_item_pairs, inference_types)
        return

    if verbose:
        # Verbose output: show each agent/item combination for each inference type
        for agents, items in sorted(agent_item_pairs):
            for inference in inference_types:
                key = (agents, items, inference)
                if key in file_data:
                    df = pd.read_csv(file_data[key])
                    print_summary_stats(df, agents, items, inference)
                else:
                    print(f"Warning: didn't find agent/item csv file for inference type {inference} (agents={agents}, items={items})")

    # Non-verbose: aggregate across all agent/item pairs for each inference type
    for inference in inference_types:
        if inference in found_inference_types:
            all_dfs = []
            for agents, items in agent_item_pairs:
                key = (agents, items, inference)
                if key in file_data:
                    df = pd.read_csv(file_data[key])
                    all_dfs.append(df)

            if all_dfs:
                combined_df = pd.concat(all_dfs, ignore_index=True)
                print_summary_stats(combined_df, inference=inference)
        else:
            print(f"Warning: didn't find agent/item csv file for inference type {inference}")

def main():
    parser = argparse.ArgumentParser(description='Summarize evaluation results from CSV files')
    parser.add_argument('--folder', default='results/', help='Folder containing CSV files (default: results/)')
    parser.add_argument('--inference_types', nargs='+', default=['model0', 'rr', 'random'],
                       help='Inference types to include (default: model0 rr random)')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='Show detailed breakdown by agent/item pairs (default: True)')
    parser.add_argument('--no-verbose', dest='verbose', action='store_false',
                       help='Show only aggregated summaries per inference type')
    parser.add_argument('--table', action='store_true',
                       help='Create summary table and save as summary.csv')

    args = parser.parse_args()

    if not os.path.exists(args.folder):
        print(f"Error: Folder {args.folder} does not exist")
        return

    summarize_results(args.folder, args.inference_types, args.verbose, args.table)

if __name__ == "__main__":
    main()