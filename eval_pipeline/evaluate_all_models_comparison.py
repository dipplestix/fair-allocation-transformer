#!/usr/bin/env python3
"""
Evaluate and compare multiple models against baselines on heatmap datasets.

Compares:
- 10-20 Model + EF1 repair
- 30-60 Model + EF1 repair
- Multi-objective Model + EF1 repair
- MaxUtil + EF1 repair
- Round-Robin (RR)
- Envy Cycle Elimination (ECE)

All on datasets from n=10-30, m=10-30 where m >= n.
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

from fftransformer.fftransformer_residual import FFTransformerResidual
from utils.calculations import (
    calculate_agent_bundle_values_batch,
    utility_sum_batch,
    nash_welfare_batch
)
from utils.inference import (
    get_model_allocations_batch,
    get_max_util_allocations_batch,
    get_rr_allocations_batch,
    get_ece_allocations_batch
)
from utils.ef1_repair import ef1_quick_repair_batch


def remap_legacy_state_dict(state_dict, num_encoder_layers):
    """Remap keys from legacy single-block format to ModuleList format.

    Legacy format: agent_transformer.attn.*
    New format: agent_transformer.0.attn.*
    """
    new_state_dict = {}

    for key, value in state_dict.items():
        new_key = key

        # Check if this is a legacy key that needs remapping
        # Legacy keys don't have the .0. index for agent_transformer and item_transformer
        if key.startswith('agent_transformer.') and not key.startswith('agent_transformer.0'):
            # Only remap if num_encoder_layers == 1 (legacy single block)
            if num_encoder_layers == 1:
                new_key = 'agent_transformer.0.' + key[len('agent_transformer.'):]
        elif key.startswith('item_transformer.') and not key.startswith('item_transformer.0'):
            if num_encoder_layers == 1:
                new_key = 'item_transformer.0.' + key[len('item_transformer.'):]

        new_state_dict[new_key] = value

    return new_state_dict


def load_model(checkpoint_path, d_model, num_heads, num_output_layers,
               num_encoder_layers=1, dropout=0.0, device=None, legacy_format=False):
    """Load a FFTransformerResidual model with specified architecture.

    Args:
        legacy_format: If True, remap keys from legacy single-block format
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FFTransformerResidual(
        d_model=d_model,
        num_heads=num_heads,
        num_output_layers=num_output_layers,
        num_encoder_layers=num_encoder_layers,
        dropout=dropout,
        initial_temperature=1.0,
        final_temperature=0.01
    )

    state_dict = torch.load(checkpoint_path, map_location=device)

    # Check if we need to remap legacy format
    if legacy_format or ('agent_transformer.attn.q_proj.weight' in state_dict and
                         num_encoder_layers == 1):
        state_dict = remap_legacy_state_dict(state_dict, num_encoder_layers)

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model


def evaluate_model_with_ef1(model, matrices, nash_max, util_max, batch_size=100):
    """Evaluate model with EF1 repair."""
    util_fractions = []
    nash_fractions = []

    for i in range(0, len(matrices), batch_size):
        batch_end = min(i + batch_size, len(matrices))
        batch_matrices = matrices[i:batch_end]
        batch_nash_max = nash_max[i:batch_end]
        batch_util_max = util_max[i:batch_end]

        # Model inference
        allocs = get_model_allocations_batch(model, batch_matrices)

        # EF1 repair
        allocs_ef1 = ef1_quick_repair_batch(
            allocs.astype(np.float64),
            batch_matrices.astype(np.float64),
            max_passes=10
        )

        # Calculate metrics
        values = calculate_agent_bundle_values_batch(batch_matrices, allocs_ef1)
        util_fractions.extend(utility_sum_batch(values) / batch_util_max)
        nash_fractions.extend(nash_welfare_batch(values) / batch_nash_max)

    return np.mean(util_fractions) * 100, np.mean(nash_fractions) * 100


def evaluate_maxutil_with_ef1(matrices, nash_max, util_max, batch_size=100):
    """Evaluate MaxUtil baseline with EF1 repair."""
    util_fractions = []
    nash_fractions = []

    for i in range(0, len(matrices), batch_size):
        batch_end = min(i + batch_size, len(matrices))
        batch_matrices = matrices[i:batch_end]
        batch_nash_max = nash_max[i:batch_end]
        batch_util_max = util_max[i:batch_end]

        # MaxUtil allocation
        allocs = get_max_util_allocations_batch(batch_matrices)

        # EF1 repair
        allocs_ef1 = ef1_quick_repair_batch(
            allocs.astype(np.float64),
            batch_matrices.astype(np.float64),
            max_passes=10
        )

        # Calculate metrics
        values = calculate_agent_bundle_values_batch(batch_matrices, allocs_ef1)
        util_fractions.extend(utility_sum_batch(values) / batch_util_max)
        nash_fractions.extend(nash_welfare_batch(values) / batch_nash_max)

    return np.mean(util_fractions) * 100, np.mean(nash_fractions) * 100


def evaluate_rr(matrices, nash_max, util_max, batch_size=100):
    """Evaluate Round-Robin baseline (no EF1 repair)."""
    util_fractions = []
    nash_fractions = []

    for i in range(0, len(matrices), batch_size):
        batch_end = min(i + batch_size, len(matrices))
        batch_matrices = matrices[i:batch_end]
        batch_nash_max = nash_max[i:batch_end]
        batch_util_max = util_max[i:batch_end]

        # RR allocation (returns n_permutations allocations per matrix)
        allocs = get_rr_allocations_batch(batch_matrices, n_permutations=1)

        # Calculate metrics
        values = calculate_agent_bundle_values_batch(batch_matrices, allocs)
        util_fractions.extend(utility_sum_batch(values) / batch_util_max)
        nash_fractions.extend(nash_welfare_batch(values) / batch_nash_max)

    return np.mean(util_fractions) * 100, np.mean(nash_fractions) * 100


def evaluate_ece(matrices, nash_max, util_max, batch_size=100):
    """Evaluate ECE baseline (already EF1 by design)."""
    util_fractions = []
    nash_fractions = []

    for i in range(0, len(matrices), batch_size):
        batch_end = min(i + batch_size, len(matrices))
        batch_matrices = matrices[i:batch_end]
        batch_nash_max = nash_max[i:batch_end]
        batch_util_max = util_max[i:batch_end]

        # ECE allocation
        allocs = get_ece_allocations_batch(batch_matrices)

        # Calculate metrics
        values = calculate_agent_bundle_values_batch(batch_matrices, allocs)
        util_fractions.extend(utility_sum_batch(values) / batch_util_max)
        nash_fractions.extend(nash_welfare_batch(values) / batch_nash_max)

    return np.mean(util_fractions) * 100, np.mean(nash_fractions) * 100


def main():
    parser = argparse.ArgumentParser(description="Compare models and baselines on heatmap datasets")
    parser.add_argument('--dataset_dir', type=str, default='datasets/heatmap',
                        help='Directory containing heatmap datasets')
    parser.add_argument('--n_min', type=int, default=10, help='Minimum number of agents')
    parser.add_argument('--n_max', type=int, default=30, help='Maximum number of agents')
    parser.add_argument('--m_min', type=int, default=10, help='Minimum number of items')
    parser.add_argument('--m_max', type=int, default=30, help='Maximum number of items')
    parser.add_argument('--num_samples', type=int, default=1000, help='Expected samples per dataset')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size for evaluation')
    parser.add_argument('--output', type=str, default='results/model_comparison_heatmap.json',
                        help='Output JSON file path')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load models
    print("\nLoading models...")

    # 10-20 Model (d_model=256, legacy format)
    model_10x20 = load_model(
        project_root / "checkpoints" / "residual" / "best_model.pt",
        d_model=256, num_heads=8, num_output_layers=2, num_encoder_layers=1, dropout=0.0,
        device=device, legacy_format=True
    )
    print(f"  10-20 Model: {sum(p.numel() for p in model_10x20.parameters()):,} parameters")

    # 30-60 Model (d_model=128, legacy format - actually 1 encoder layer despite config)
    model_30x60 = load_model(
        project_root / "checkpoints" / "residual_30_60" / "best_model.pt",
        d_model=128, num_heads=8, num_output_layers=2, num_encoder_layers=1, dropout=0.099,
        device=device, legacy_format=True
    )
    print(f"  30-60 Model: {sum(p.numel() for p in model_30x60.parameters()):,} parameters")

    # Multi-objective Model (same arch as 10-20, new format)
    model_multi_obj = load_model(
        project_root / "checkpoints" / "multi_size_strategies" / "multi_objective" / "best_model.pt",
        d_model=256, num_heads=8, num_output_layers=2, num_encoder_layers=1, dropout=0.0,
        device=device, legacy_format=False
    )
    print(f"  Multi-objective Model: {sum(p.numel() for p in model_multi_obj.parameters()):,} parameters")

    # Build list of (n, m) pairs where m >= n
    configs = []
    for n in range(args.n_min, args.n_max + 1):
        for m in range(max(n, args.m_min), args.m_max + 1):
            configs.append((n, m))

    print(f"\nTotal configurations: {len(configs)}")

    # Initialize results
    results = {
        'n_range': [args.n_min, args.n_max],
        'm_range': [args.m_min, args.m_max],
        'num_samples': args.num_samples,
        'methods': ['model_10x20_ef1', 'model_30x60_ef1', 'model_multi_objective_ef1',
                    'maxutil_ef1', 'rr', 'ece'],
        'model_10x20_ef1': {'utility': {}, 'nash': {}},
        'model_30x60_ef1': {'utility': {}, 'nash': {}},
        'model_multi_objective_ef1': {'utility': {}, 'nash': {}},
        'maxutil_ef1': {'utility': {}, 'nash': {}},
        'rr': {'utility': {}, 'nash': {}},
        'ece': {'utility': {}, 'nash': {}},
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

        key = f"{n},{m}"

        # Evaluate 10-20 Model + EF1
        util_10x20, nash_10x20 = evaluate_model_with_ef1(
            model_10x20, matrices, nash_max, util_max, args.batch_size
        )
        results['model_10x20_ef1']['utility'][key] = util_10x20
        results['model_10x20_ef1']['nash'][key] = nash_10x20

        # Evaluate 30-60 Model + EF1
        util_30x60, nash_30x60 = evaluate_model_with_ef1(
            model_30x60, matrices, nash_max, util_max, args.batch_size
        )
        results['model_30x60_ef1']['utility'][key] = util_30x60
        results['model_30x60_ef1']['nash'][key] = nash_30x60

        # Evaluate Multi-objective Model + EF1
        util_multi, nash_multi = evaluate_model_with_ef1(
            model_multi_obj, matrices, nash_max, util_max, args.batch_size
        )
        results['model_multi_objective_ef1']['utility'][key] = util_multi
        results['model_multi_objective_ef1']['nash'][key] = nash_multi

        # Evaluate MaxUtil + EF1
        util_maxutil, nash_maxutil = evaluate_maxutil_with_ef1(
            matrices, nash_max, util_max, args.batch_size
        )
        results['maxutil_ef1']['utility'][key] = util_maxutil
        results['maxutil_ef1']['nash'][key] = nash_maxutil

        # Evaluate RR
        util_rr, nash_rr = evaluate_rr(matrices, nash_max, util_max, args.batch_size)
        results['rr']['utility'][key] = util_rr
        results['rr']['nash'][key] = nash_rr

        # Evaluate ECE
        util_ece, nash_ece = evaluate_ece(matrices, nash_max, util_max, args.batch_size)
        results['ece']['utility'][key] = util_ece
        results['ece']['nash'][key] = nash_ece

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY (Mean across all configurations)")
    print("=" * 80)

    for method in results['methods']:
        nash_values = list(results[method]['nash'].values())
        util_values = list(results[method]['utility'].values())
        if nash_values and util_values:
            print(f"\n{method}:")
            print(f"  Nash Welfare:  mean={np.mean(nash_values):.2f}%, "
                  f"min={np.min(nash_values):.2f}%, max={np.max(nash_values):.2f}%")
            print(f"  Util Welfare:  mean={np.mean(util_values):.2f}%, "
                  f"min={np.min(util_values):.2f}%, max={np.max(util_values):.2f}%")


if __name__ == "__main__":
    main()
