#!/usr/bin/env python3
"""
Generate datasets and evaluate residual model + EF1 repair for heatmap visualization.
n = 10-30, m = 10-30, m >= n, 1000 samples each.
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


def evaluate_on_data_with_ef1(model, matrices, nash_welfare_max, util_welfare_max, batch_size=100):
    """Evaluate model + EF1 repair on dataset, return utility% and nash%."""
    all_util_fractions = []
    all_nash_fractions = []

    for i in range(0, len(matrices), batch_size):
        batch_end = min(i + batch_size, len(matrices))
        batch_matrices = matrices[i:batch_end]
        batch_nash_max = nash_welfare_max[i:batch_end]
        batch_util_max = util_welfare_max[i:batch_end]

        # Get model allocations
        batch_allocations = get_model_allocations_batch(model, batch_matrices)

        # Apply EF1 repair
        batch_allocations_repaired = ef1_quick_repair_batch(
            batch_allocations.astype(np.float64),
            batch_matrices.astype(np.float64),
            max_passes=10
        )

        agent_bundle_values = calculate_agent_bundle_values_batch(batch_matrices, batch_allocations_repaired)

        util_sums = utility_sum_batch(agent_bundle_values)
        nash_welfares = nash_welfare_batch(agent_bundle_values)

        all_util_fractions.extend(util_sums / batch_util_max)
        all_nash_fractions.extend(nash_welfares / batch_nash_max)

    return np.mean(all_util_fractions) * 100, np.mean(all_nash_fractions) * 100


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=1000)
    parser.add_argument('--n_min', type=int, default=10)
    parser.add_argument('--n_max', type=int, default=30)
    parser.add_argument('--m_min', type=int, default=10)
    parser.add_argument('--m_max', type=int, default=30)
    parser.add_argument('--output', type=str, default='results/heatmap_data_ef1.json')
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

    results = {
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

        # Evaluate with EF1 repair
        util_pct, nash_pct = evaluate_on_data_with_ef1(model, matrices, nash_max, util_max)

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
