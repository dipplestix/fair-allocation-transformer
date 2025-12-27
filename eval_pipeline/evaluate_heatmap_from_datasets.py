#!/usr/bin/env python3
"""
Evaluate model (and optionally model+EF1) on pre-generated heatmap datasets.
Much faster since optimal values are already computed.
"""

import argparse
import numpy as np
import torch
import sys
import json
from pathlib import Path
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.calculations import (
    calculate_agent_bundle_values_batch,
    utility_sum_batch,
    nash_welfare_batch
)
from utils.inference import get_model_allocations_batch
from utils.ef1_repair import ef1_quick_repair_batch


def load_residual_model():
    """Load the residual FFTransformer model."""
    from fftransformer.fftransformer_residual import FFTransformerResidual

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FFTransformerResidual(
        d_model=256, num_heads=8,
        num_output_layers=2, dropout=0.0,
        initial_temperature=1.0, final_temperature=0.01
    )

    weights_path = project_root / "checkpoints" / "residual" / "best_model.pt"
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()

    return model


def evaluate_on_dataset(model, matrices, nash_max, util_max, use_ef1=False, batch_size=100):
    """Evaluate model on dataset, return utility% and nash%."""
    all_util_fractions = []
    all_nash_fractions = []

    for i in range(0, len(matrices), batch_size):
        batch_end = min(i + batch_size, len(matrices))
        batch_matrices = matrices[i:batch_end]
        batch_nash_max = nash_max[i:batch_end]
        batch_util_max = util_max[i:batch_end]

        # Get model allocations
        batch_allocations = get_model_allocations_batch(model, batch_matrices)

        # Optionally apply EF1 repair
        if use_ef1:
            batch_allocations = ef1_quick_repair_batch(
                batch_allocations.astype(np.float64),
                batch_matrices.astype(np.float64),
                max_passes=10
            )

        agent_bundle_values = calculate_agent_bundle_values_batch(batch_matrices, batch_allocations)

        util_sums = utility_sum_batch(agent_bundle_values)
        nash_welfares = nash_welfare_batch(agent_bundle_values)

        all_util_fractions.extend(util_sums / batch_util_max)
        all_nash_fractions.extend(nash_welfares / batch_nash_max)

    return np.mean(all_util_fractions) * 100, np.mean(all_nash_fractions) * 100


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default='datasets/heatmap')
    parser.add_argument('--n_min', type=int, default=10)
    parser.add_argument('--n_max', type=int, default=30)
    parser.add_argument('--m_min', type=int, default=10)
    parser.add_argument('--m_max', type=int, default=30)
    parser.add_argument('--num_samples', type=int, default=1000)
    parser.add_argument('--ef1', action='store_true', help='Apply EF1 repair after model')
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    # Set default output based on ef1 flag
    if args.output is None:
        if args.ef1:
            args.output = 'results/heatmap_data_ef1.json'
        else:
            args.output = 'results/heatmap_data.json'

    print("Loading model...")
    model = load_residual_model()
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Build list of (n, m) pairs where m >= n
    configs = []
    for n in range(args.n_min, args.n_max + 1):
        for m in range(max(n, args.m_min), args.m_max + 1):
            configs.append((n, m))

    print(f"\nTotal configurations: {len(configs)}")
    print(f"Using EF1 repair: {args.ef1}")

    results = {
        'n_range': [args.n_min, args.n_max],
        'm_range': [args.m_min, args.m_max],
        'num_samples': args.num_samples,
        'method': 'model_ef1_repair' if args.ef1 else 'model_only',
        'utility': {},
        'nash': {}
    }

    dataset_dir = Path(args.dataset_dir)

    for n, m in tqdm(configs, desc="Evaluating"):
        # Load dataset
        dataset_file = dataset_dir / f"{n}_{m}_{args.num_samples}_dataset.npz"
        if not dataset_file.exists():
            tqdm.write(f"Warning: Dataset not found: {dataset_file}")
            continue

        data = np.load(dataset_file)
        matrices = data['matrices']
        nash_max = data['nash_welfare']
        util_max = data['util_welfare']

        # Evaluate
        util_pct, nash_pct = evaluate_on_dataset(model, matrices, nash_max, util_max, use_ef1=args.ef1)

        key = f"{n},{m}"
        results['utility'][key] = util_pct
        results['nash'][key] = nash_pct

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
