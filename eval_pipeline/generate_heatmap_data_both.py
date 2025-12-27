#!/usr/bin/env python3
"""
Generate datasets and evaluate both:
1. Residual model only
2. Residual model + EF1 repair

Saves both to separate JSON files in one pass (avoids recomputing optimal values).
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
from utils.max_utility import best_nash_welfare, best_utilitarian_welfare
from utils.ef1_repair import ef1_quick_repair_batch


def generate_valuation_matrix(n_agents, m_items):
    return np.random.uniform(0, 1, size=(n_agents, m_items))


def generate_dataset_fast(n_agents, n_items, num_matrices, seed=42):
    """Generate dataset without saving to disk."""
    np.random.seed(seed * n_items + n_agents)

    matrices = []
    nash_values = []
    util_values = []

    for _ in range(num_matrices):
        valuation_matrix = generate_valuation_matrix(n_agents, n_items)

        nash_welfare = best_nash_welfare(valuation_matrix, num_segments=200)
        if nash_welfare is None:
            continue

        util_welfare = best_utilitarian_welfare(valuation_matrix)

        matrices.append(valuation_matrix)
        nash_values.append(nash_welfare)
        util_values.append(util_welfare)

    return np.array(matrices), np.array(nash_values), np.array(util_values)


def load_residual_model():
    """Load the residual FATransformer model."""
    from fatransformer.fatransformer_residual import FATransformer as FATransformerResidual

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FATransformerResidual(
        n=10, m=20, d_model=256, num_heads=8,
        num_output_layers=2, dropout=0.0,
        initial_temperature=1.0, final_temperature=0.01
    )

    weights_path = project_root / "checkpoints" / "residual" / "best_model.pt"
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()

    return model


def evaluate_on_data_both(model, matrices, nash_welfare_max, util_welfare_max, batch_size=100):
    """Evaluate model and model+EF1 on dataset, return both sets of metrics."""
    # Model only
    all_util_fractions = []
    all_nash_fractions = []
    # Model + EF1
    all_util_fractions_ef1 = []
    all_nash_fractions_ef1 = []

    for i in range(0, len(matrices), batch_size):
        batch_end = min(i + batch_size, len(matrices))
        batch_matrices = matrices[i:batch_end]
        batch_nash_max = nash_welfare_max[i:batch_end]
        batch_util_max = util_welfare_max[i:batch_end]

        # Get model allocations
        batch_allocations = get_model_allocations_batch(model, batch_matrices)

        # Evaluate model only
        agent_bundle_values = calculate_agent_bundle_values_batch(batch_matrices, batch_allocations)
        util_sums = utility_sum_batch(agent_bundle_values)
        nash_welfares = nash_welfare_batch(agent_bundle_values)
        all_util_fractions.extend(util_sums / batch_util_max)
        all_nash_fractions.extend(nash_welfares / batch_nash_max)

        # Apply EF1 repair and evaluate
        batch_allocations_repaired = ef1_quick_repair_batch(
            batch_allocations.astype(np.float64),
            batch_matrices.astype(np.float64),
            max_passes=10
        )
        agent_bundle_values_ef1 = calculate_agent_bundle_values_batch(batch_matrices, batch_allocations_repaired)
        util_sums_ef1 = utility_sum_batch(agent_bundle_values_ef1)
        nash_welfares_ef1 = nash_welfare_batch(agent_bundle_values_ef1)
        all_util_fractions_ef1.extend(util_sums_ef1 / batch_util_max)
        all_nash_fractions_ef1.extend(nash_welfares_ef1 / batch_nash_max)

    return (
        np.mean(all_util_fractions) * 100,
        np.mean(all_nash_fractions) * 100,
        np.mean(all_util_fractions_ef1) * 100,
        np.mean(all_nash_fractions_ef1) * 100
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=1000)
    parser.add_argument('--n_min', type=int, default=10)
    parser.add_argument('--n_max', type=int, default=30)
    parser.add_argument('--m_min', type=int, default=10)
    parser.add_argument('--m_max', type=int, default=30)
    parser.add_argument('--output_dir', type=str, default='results')
    args = parser.parse_args()

    print("Loading model...")
    model = load_residual_model()
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Build list of (n, m) pairs where m >= n
    configs = []
    for n in range(args.n_min, args.n_max + 1):
        for m in range(max(n, args.m_min), args.m_max + 1):
            configs.append((n, m))

    print(f"\nTotal configurations: {len(configs)}")
    print(f"Samples per config: {args.num_samples}")

    results_model = {
        'n_range': [args.n_min, args.n_max],
        'm_range': [args.m_min, args.m_max],
        'num_samples': args.num_samples,
        'method': 'model_only',
        'utility': {},
        'nash': {}
    }

    results_ef1 = {
        'n_range': [args.n_min, args.n_max],
        'm_range': [args.m_min, args.m_max],
        'num_samples': args.num_samples,
        'method': 'model_ef1_repair',
        'utility': {},
        'nash': {}
    }

    for n, m in tqdm(configs, desc="Evaluating configs"):
        # Generate dataset
        matrices, nash_max, util_max = generate_dataset_fast(n, m, args.num_samples)

        if len(matrices) < args.num_samples * 0.9:
            print(f"\nWarning: Only got {len(matrices)} valid samples for n={n}, m={m}")

        # Evaluate both
        util_pct, nash_pct, util_pct_ef1, nash_pct_ef1 = evaluate_on_data_both(
            model, matrices, nash_max, util_max
        )

        key = f"{n},{m}"
        results_model['utility'][key] = util_pct
        results_model['nash'][key] = nash_pct
        results_ef1['utility'][key] = util_pct_ef1
        results_ef1['nash'][key] = nash_pct_ef1

    # Save results
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(output_path / 'heatmap_data.json', 'w') as f:
        json.dump(results_model, f, indent=2)
    print(f"\nModel results saved to {output_path / 'heatmap_data.json'}")

    with open(output_path / 'heatmap_data_ef1.json', 'w') as f:
        json.dump(results_ef1, f, indent=2)
    print(f"Model+EF1 results saved to {output_path / 'heatmap_data_ef1.json'}")


if __name__ == "__main__":
    main()
