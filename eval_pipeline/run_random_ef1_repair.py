#!/usr/bin/env python3
"""Run EF1 repair on random allocations to see how much the repair does vs the model."""

import numpy as np
import pandas as pd
import sys
import time
from pathlib import Path

sys.path.insert(0, 'eval_pipeline')

from utils.ef1_repair import ef1_quick_repair_batch
from utils.calculations import (
    calculate_agent_bundle_values,
    is_envy_free,
    is_ef1,
    is_efx,
    nash_welfare,
    utility_sum,
)


def generate_random_allocations(valuations, n_samples):
    """Generate random allocations for each valuation matrix."""
    n_agents, m_items = valuations.shape[1], valuations.shape[2]
    allocations = np.zeros((n_samples, n_agents, m_items))

    for i in range(n_samples):
        for item in range(m_items):
            agent = np.random.randint(0, n_agents)
            allocations[i, agent, item] = 1

    return allocations


def evaluate_allocations(valuations, allocations, max_nash, max_util):
    """Evaluate a batch of allocations."""
    results = []

    for i in range(len(allocations)):
        vals = valuations[i]
        alloc = allocations[i]

        bundle_values = calculate_agent_bundle_values(vals, alloc)

        results.append({
            'envy_free': is_envy_free(bundle_values),
            'ef1': is_ef1(vals, alloc, bundle_values),
            'efx': is_efx(vals, alloc, bundle_values),
            'utility_sum': utility_sum(bundle_values),
            'nash_welfare': nash_welfare(bundle_values),
            'fraction_util_welfare': utility_sum(bundle_values) / max_util[i] if max_util[i] > 0 else 0,
            'fraction_nash_welfare': nash_welfare(bundle_values) / max_nash[i] if max_nash[i] > 0 else 0,
        })

    return results


def main():
    results_dir = Path('results')

    print("Running EF1 Repair on Random Allocations")
    print("=" * 70)

    for m in range(10, 21):
        dataset_path = Path(f'datasets/10_{m}_100000_dataset.npz')

        if not dataset_path.exists():
            print(f"Skipping m={m} - dataset not found")
            continue

        print(f"\nProcessing m={m}...")

        # Load dataset
        data = np.load(dataset_path)
        valuations = data['matrices']
        max_nash = data['nash_welfare']
        max_util = data['util_welfare']

        n_samples = len(valuations)
        n_agents, m_items = valuations.shape[1], valuations.shape[2]

        all_results = []
        batch_size = 100

        start_time = time.time()

        for batch_start in range(0, n_samples, batch_size):
            batch_end = min(batch_start + batch_size, n_samples)
            batch_vals = valuations[batch_start:batch_end]
            batch_max_nash = max_nash[batch_start:batch_end]
            batch_max_util = max_util[batch_start:batch_end]

            # Generate random allocations
            random_allocs = generate_random_allocations(batch_vals, len(batch_vals))

            # Apply EF1 repair
            batch_start_time = time.time()
            repaired_allocs = ef1_quick_repair_batch(random_allocs, batch_vals, max_passes=10)
            batch_time = time.time() - batch_start_time

            # Evaluate
            batch_results = evaluate_allocations(batch_vals, repaired_allocs, batch_max_nash, batch_max_util)

            for j, res in enumerate(batch_results):
                res['matrix_id'] = batch_start + j
                res['num_agents'] = n_agents
                res['num_items'] = m_items
                res['max_nash_welfare'] = batch_max_nash[j]
                res['max_util_welfare'] = batch_max_util[j]
                res['inference_time'] = batch_time
                res['batch_size'] = len(batch_vals)
                all_results.append(res)

            if (batch_start // batch_size) % 100 == 0:
                print(f"  Processed {batch_end}/{n_samples}...")

        total_time = time.time() - start_time

        # Save results
        df = pd.DataFrame(all_results)
        output_path = results_dir / f'evaluation_results_10_{m}_100000_random_with_ef1_repair.csv'
        df.to_csv(output_path, index=False)

        # Print summary
        ef1_rate = df['ef1'].mean() * 100
        nsw_rate = df['fraction_nash_welfare'].mean() * 100
        print(f"  m={m}: EF1={ef1_rate:.2f}%, NSW={nsw_rate:.2f}%, Time={total_time:.1f}s")

    print("\n" + "=" * 70)
    print("Done! Results saved to results/evaluation_results_*_random_with_ef1_repair.csv")


if __name__ == '__main__':
    main()
