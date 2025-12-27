#!/usr/bin/env python3
"""
Plot comparison heatmaps: (30x60 model + EF1) - (10x20 model + EF1).
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_and_plot(data_file, output_dir='results'):
    with open(data_file, 'r') as f:
        data = json.load(f)

    n_min, n_max = data['n_range']
    m_min, m_max = data['m_range']

    n_vals = list(range(n_min, n_max + 1))
    m_vals = list(range(m_min, m_max + 1))

    # Create matrices for heatmaps (NaN where m < n)
    diff_util_matrix = np.full((len(n_vals), len(m_vals)), np.nan)
    diff_nash_matrix = np.full((len(n_vals), len(m_vals)), np.nan)

    for key, val in data['diff']['utility'].items():
        n, m = map(int, key.split(','))
        i = n - n_min
        j = m - m_min
        diff_util_matrix[i, j] = val

    for key, val in data['diff']['nash'].items():
        n, m = map(int, key.split(','))
        i = n - n_min
        j = m - m_min
        diff_nash_matrix[i, j] = val

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    def plot_heatmap(matrix, title, filename):
        fig, ax = plt.subplots(figsize=(12, 10))
        vmax = max(abs(np.nanmin(matrix)), abs(np.nanmax(matrix)))
        im = ax.imshow(matrix, cmap='RdBu', aspect='auto',
                       vmin=-vmax, vmax=vmax, origin='lower')
        ax.set_xticks(range(len(m_vals)))
        ax.set_xticklabels(m_vals)
        ax.set_yticks(range(len(n_vals)))
        ax.set_yticklabels(n_vals)
        ax.set_xlabel('m (items)', fontsize=12)
        ax.set_ylabel('n (agents)', fontsize=12)
        ax.set_title(title, fontsize=14)
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('% Difference', fontsize=12)

        # Add text annotations
        for i in range(len(n_vals)):
            for j in range(len(m_vals)):
                if not np.isnan(matrix[i, j]):
                    val = matrix[i, j]
                    text_color = 'white' if abs(val) > vmax * 0.5 else 'black'
                    ax.text(j, i, f'{val:.1f}',
                           ha='center', va='center', fontsize=6, color=text_color)

        plt.tight_layout()
        plt.savefig(output_path / filename, dpi=150)
        plt.close()
        print(f"Saved: {output_path / filename}")

    plot_heatmap(
        diff_util_matrix,
        'Utility % Diff: (Model 30x60 + EF1) - (Model 10x20 + EF1)',
        'heatmap_model_comparison_utility.png'
    )
    plot_heatmap(
        diff_nash_matrix,
        'Nash % Diff: (Model 30x60 + EF1) - (Model 10x20 + EF1)',
        'heatmap_model_comparison_nash.png'
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='results/heatmap_model_comparison.json')
    parser.add_argument('--output', type=str, default='results')
    args = parser.parse_args()

    load_and_plot(args.input, args.output)
