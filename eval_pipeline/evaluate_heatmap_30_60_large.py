#!/usr/bin/env python3
"""
Evaluate 30x60 LARGE model (d_model=256) on heatmap datasets.
"""

import argparse
import numpy as np
import torch
import sys
import json
from pathlib import Path
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.calculations import (
    calculate_agent_bundle_values_batch,
    utility_sum_batch,
    nash_welfare_batch
)
from utils.inference import get_model_allocations_batch
from utils.ef1_repair import ef1_quick_repair_batch


def load_model_30_60_large():
    from fftransformer.fftransformer_residual import FFTransformerResidual
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FFTransformerResidual(
        d_model=256, num_heads=8,
        num_output_layers=2, dropout=0.0,
        initial_temperature=1.0, final_temperature=0.01
    )
    weights_path = project_root / "checkpoints" / "residual_30_60_large" / "best_model.pt"
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def max_util_allocation_batch(valuations_batch):
    batch_size, n_agents, m_items = valuations_batch.shape
    allocations = np.zeros((batch_size, n_agents, m_items), dtype=np.float64)
    for b in range(batch_size):
        for j in range(m_items):
            best_agent = np.argmax(valuations_batch[b, :, j])
            allocations[b, best_agent, j] = 1
    return allocations


def round_robin_allocation_batch(valuations_batch):
    batch_size, n_agents, m_items = valuations_batch.shape
    allocations = np.zeros((batch_size, n_agents, m_items), dtype=np.float64)
    for b in range(batch_size):
        for j in range(m_items):
            allocations[b, j % n_agents, j] = 1
    return allocations


def evaluate_all_methods(model, matrices, nash_max, util_max, batch_size=100):
    model_util, model_nash = [], []
    maxutil_util, maxutil_nash = [], []
    rr_util, rr_nash = [], []

    for i in range(0, len(matrices), batch_size):
        batch_end = min(i + batch_size, len(matrices))
        batch = matrices[i:batch_end]
        b_nash_max = nash_max[i:batch_end]
        b_util_max = util_max[i:batch_end]

        # Model + EF1
        allocs = get_model_allocations_batch(model, batch)
        allocs_ef1 = ef1_quick_repair_batch(allocs.astype(np.float64), batch.astype(np.float64), max_passes=10)
        vals = calculate_agent_bundle_values_batch(batch, allocs_ef1)
        model_util.extend(utility_sum_batch(vals) / b_util_max)
        model_nash.extend(nash_welfare_batch(vals) / b_nash_max)

        # MaxUtil + EF1
        allocs = max_util_allocation_batch(batch)
        allocs_ef1 = ef1_quick_repair_batch(allocs, batch.astype(np.float64), max_passes=10)
        vals = calculate_agent_bundle_values_batch(batch, allocs_ef1)
        maxutil_util.extend(utility_sum_batch(vals) / b_util_max)
        maxutil_nash.extend(nash_welfare_batch(vals) / b_nash_max)

        # RR + EF1
        allocs = round_robin_allocation_batch(batch)
        allocs_ef1 = ef1_quick_repair_batch(allocs, batch.astype(np.float64), max_passes=10)
        vals = calculate_agent_bundle_values_batch(batch, allocs_ef1)
        rr_util.extend(utility_sum_batch(vals) / b_util_max)
        rr_nash.extend(nash_welfare_batch(vals) / b_nash_max)

    return (
        np.mean(model_util) * 100, np.mean(model_nash) * 100,
        np.mean(maxutil_util) * 100, np.mean(maxutil_nash) * 100,
        np.mean(rr_util) * 100, np.mean(rr_nash) * 100
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default='datasets/heatmap')
    parser.add_argument('--n_min', type=int, default=10)
    parser.add_argument('--n_max', type=int, default=30)
    parser.add_argument('--m_min', type=int, default=10)
    parser.add_argument('--m_max', type=int, default=30)
    parser.add_argument('--num_samples', type=int, default=1000)
    parser.add_argument('--output', type=str, default='results/heatmap_30_60_large_model.json')
    args = parser.parse_args()

    print("Loading 30x60 LARGE model...")
    model = load_model_30_60_large()
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

    configs = [(n, m) for n in range(args.n_min, args.n_max + 1)
               for m in range(max(n, args.m_min), args.m_max + 1)]
    print(f"\nTotal configurations: {len(configs)}")

    results = {
        'n_range': [args.n_min, args.n_max],
        'm_range': [args.m_min, args.m_max],
        'num_samples': args.num_samples,
        'model_name': 'residual_30_60_large',
        'model_ef1': {'utility': {}, 'nash': {}},
        'maxutil_ef1': {'utility': {}, 'nash': {}},
        'rr_ef1': {'utility': {}, 'nash': {}},
        'diff_vs_maxutil': {'utility': {}, 'nash': {}},
        'diff_vs_rr': {'utility': {}, 'nash': {}}
    }

    dataset_dir = Path(args.dataset_dir)

    for n, m in tqdm(configs, desc="Evaluating"):
        dataset_file = dataset_dir / f"{n}_{m}_{args.num_samples}_dataset.npz"
        if not dataset_file.exists():
            continue

        data = np.load(dataset_file)
        model_util, model_nash, maxutil_util, maxutil_nash, rr_util, rr_nash = \
            evaluate_all_methods(model, data['matrices'], data['nash_welfare'], data['util_welfare'])

        key = f"{n},{m}"
        results['model_ef1']['utility'][key] = model_util
        results['model_ef1']['nash'][key] = model_nash
        results['maxutil_ef1']['utility'][key] = maxutil_util
        results['maxutil_ef1']['nash'][key] = maxutil_nash
        results['rr_ef1']['utility'][key] = rr_util
        results['rr_ef1']['nash'][key] = rr_nash
        results['diff_vs_maxutil']['utility'][key] = model_util - maxutil_util
        results['diff_vs_maxutil']['nash'][key] = model_nash - maxutil_nash
        results['diff_vs_rr']['utility'][key] = model_util - rr_util
        results['diff_vs_rr']['nash'][key] = model_nash - rr_nash

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")
    diff_maxutil = list(results['diff_vs_maxutil']['utility'].values())
    diff_rr = list(results['diff_vs_rr']['utility'].values())
    print(f"\nvs MaxUtil: min={min(diff_maxutil):.2f}%, max={max(diff_maxutil):.2f}%, mean={np.mean(diff_maxutil):.2f}%")
    print(f"vs RR: min={min(diff_rr):.2f}%, max={max(diff_rr):.2f}%, mean={np.mean(diff_rr):.2f}%")


if __name__ == "__main__":
    main()
