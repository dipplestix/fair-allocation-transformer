#!/usr/bin/env python3
"""
Evaluate and compare 30x60 model vs 10x20 model on heatmap datasets.
Computes (30x60 model + EF1) - (10x20 model + EF1) difference.
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


def load_model(checkpoint_path, device, d_model):
    """Load a FATransformerResidual model."""
    from fatransformer.fatransformer_residual import FATransformer as FATransformerResidual

    model = FATransformerResidual(
        n=10, m=20, d_model=d_model, num_heads=8,
        num_output_layers=2, dropout=0.0,
        initial_temperature=1.0, final_temperature=0.01
    )

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    return model


def evaluate_both_models(model_30_60, model_10_20, matrices, nash_max, util_max, batch_size=100):
    """Evaluate both models and return metrics."""
    model_30_60_util = []
    model_30_60_nash = []
    model_10_20_util = []
    model_10_20_nash = []

    for i in range(0, len(matrices), batch_size):
        batch_end = min(i + batch_size, len(matrices))
        batch_matrices = matrices[i:batch_end]
        batch_nash_max = nash_max[i:batch_end]
        batch_util_max = util_max[i:batch_end]

        # 30x60 Model + EF1
        allocs_30_60 = get_model_allocations_batch(model_30_60, batch_matrices)
        allocs_30_60_ef1 = ef1_quick_repair_batch(
            allocs_30_60.astype(np.float64),
            batch_matrices.astype(np.float64),
            max_passes=10
        )
        values_30_60 = calculate_agent_bundle_values_batch(batch_matrices, allocs_30_60_ef1)
        model_30_60_util.extend(utility_sum_batch(values_30_60) / batch_util_max)
        model_30_60_nash.extend(nash_welfare_batch(values_30_60) / batch_nash_max)

        # 10x20 Model + EF1
        allocs_10_20 = get_model_allocations_batch(model_10_20, batch_matrices)
        allocs_10_20_ef1 = ef1_quick_repair_batch(
            allocs_10_20.astype(np.float64),
            batch_matrices.astype(np.float64),
            max_passes=10
        )
        values_10_20 = calculate_agent_bundle_values_batch(batch_matrices, allocs_10_20_ef1)
        model_10_20_util.extend(utility_sum_batch(values_10_20) / batch_util_max)
        model_10_20_nash.extend(nash_welfare_batch(values_10_20) / batch_nash_max)

    return (
        np.mean(model_30_60_util) * 100,
        np.mean(model_30_60_nash) * 100,
        np.mean(model_10_20_util) * 100,
        np.mean(model_10_20_nash) * 100
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default='datasets/heatmap')
    parser.add_argument('--n_min', type=int, default=10)
    parser.add_argument('--n_max', type=int, default=30)
    parser.add_argument('--m_min', type=int, default=10)
    parser.add_argument('--m_max', type=int, default=30)
    parser.add_argument('--num_samples', type=int, default=1000)
    parser.add_argument('--output', type=str, default='results/heatmap_model_comparison.json')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading models...")
    model_30_60 = load_model(
        project_root / "checkpoints" / "residual_30_60" / "best_model.pt",
        device, d_model=128
    )
    print(f"  30x60 model: {sum(p.numel() for p in model_30_60.parameters()):,} params")

    model_10_20 = load_model(
        project_root / "checkpoints" / "residual" / "best_model.pt",
        device, d_model=256
    )
    print(f"  10x20 model: {sum(p.numel() for p in model_10_20.parameters()):,} params")

    # Build list of (n, m) pairs where m >= n
    configs = []
    for n in range(args.n_min, args.n_max + 1):
        for m in range(max(n, args.m_min), args.m_max + 1):
            configs.append((n, m))

    print(f"\nTotal configurations: {len(configs)}")

    results = {
        'n_range': [args.n_min, args.n_max],
        'm_range': [args.m_min, args.m_max],
        'num_samples': args.num_samples,
        'model_30_60_ef1': {'utility': {}, 'nash': {}},
        'model_10_20_ef1': {'utility': {}, 'nash': {}},
        'diff': {'utility': {}, 'nash': {}}  # 30x60 - 10x20
    }

    dataset_dir = Path(args.dataset_dir)

    for n, m in tqdm(configs, desc="Evaluating"):
        dataset_file = dataset_dir / f"{n}_{m}_{args.num_samples}_dataset.npz"
        if not dataset_file.exists():
            tqdm.write(f"Warning: Dataset not found: {dataset_file}")
            continue

        data = np.load(dataset_file)
        matrices = data['matrices']
        nash_max = data['nash_welfare']
        util_max = data['util_welfare']

        m30_util, m30_nash, m10_util, m10_nash = evaluate_both_models(
            model_30_60, model_10_20, matrices, nash_max, util_max
        )

        key = f"{n},{m}"
        results['model_30_60_ef1']['utility'][key] = m30_util
        results['model_30_60_ef1']['nash'][key] = m30_nash
        results['model_10_20_ef1']['utility'][key] = m10_util
        results['model_10_20_ef1']['nash'][key] = m10_nash
        results['diff']['utility'][key] = m30_util - m10_util
        results['diff']['nash'][key] = m30_nash - m10_nash

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")

    # Print summary
    diff_util = list(results['diff']['utility'].values())
    diff_nash = list(results['diff']['nash'].values())
    print(f"\nDifference (30x60 model) - (10x20 model):")
    print(f"  Utility: min={min(diff_util):.2f}%, max={max(diff_util):.2f}%, mean={np.mean(diff_util):.2f}%")
    print(f"  Nash: min={min(diff_nash):.2f}%, max={max(diff_nash):.2f}%, mean={np.mean(diff_nash):.2f}%")


if __name__ == "__main__":
    main()
