from evaluation import run_evaluation
import argparse
import glob
import os

def main():
    parser = argparse.ArgumentParser(description='Auto run evaluations on datasets')
    parser.add_argument('folder', type=str, help='Folder containing datasets (.npz files)')
    parser.add_argument('--num_agents_list', type=str, nargs='+', default=None, help='List of specific agent counts to evaluate (default: all .npz files in folder)')
    parser.add_argument('--num_item_list', type=str, nargs='+', default=None, help='List of specific item counts to evaluate (default: all .npz files in folder)')
    parser.add_argument('--num_matrices_list', type=str, nargs='+', default=None, help='Number of matrices to evaluate from each file (default: all)')
    parser.add_argument('--batch_size', type=int, default=100, help='Number of allocations and inference to run in parallel (default: 100)')
    parser.add_argument('--eval_type', default='model0', help='Type of evaluation: model[#n], random, or round robin (rr) (default: model0)')
    args = parser.parse_args()

    # traverse the folder to get all .npz files and filter based on num_agents_list and num_item_list
    npz_files = glob.glob(os.path.join(args.folder, "*.npz"))
    if not npz_files:
        print(f"No .npz files found in {args.folder}")
        return

    # parse num_agents_list and num_item_list to integers
    if args.num_agents_list is not None:
        args.num_agents_list = [int(x) for x in args.num_agents_list]
    if args.num_item_list is not None:
        args.num_item_list = [int(x) for x in args.num_item_list]
    if args.num_matrices_list is not None:
        args.num_matrices_list = [int(x) for x in args.num_matrices_list]

    # Filter based on num_agents_list and num_item_list
    filtered_files = []
    for npz_file in npz_files:
        filename = os.path.basename(npz_file)
        parts = filename.split('_')
        if len(parts) < 3:
            continue
        try:
            num_agents = int(parts[0])
            num_items = int(parts[1])
            num_matrices = int(parts[2])
        except ValueError:
            continue

        if (args.num_agents_list is None or num_agents in args.num_agents_list) and (args.num_item_list is None or num_items in args.num_item_list) and (args.num_matrices_list is None or num_matrices in args.num_matrices_list):
            filtered_files.append(npz_file)

    print(f"Found {len(filtered_files)} matching .npz files for evaluation.")

    if not filtered_files:
        print("No matching .npz files found")
        return
    
    for npz_file in filtered_files:
        print(f"Running evaluation on {npz_file} with eval_type={args.eval_type} and batch_size={args.batch_size}")
        run_evaluation(npz_file, 'evaluation_results', 'evaluation_results', args.batch_size, args.eval_type)

if __name__ == "__main__":
    main()