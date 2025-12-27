#!/usr/bin/env python3
"""
Plot comparison heatmaps for 30x60 model.
Shows (Model_30_60+EF1) - (MaxUtil+EF1) and (Model_30_60+EF1) - (RR+EF1).
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
    diff_maxutil_util = np.full((len(n_vals), len(m_vals)), np.nan)
    diff_maxutil_nash = np.full((len(n_vals), len(m_vals)), np.nan)
    diff_rr_util = np.full((len(n_vals), len(m_vals)), np.nan)
    diff_rr_nash = np.full((len(n_vals), len(m_vals)), np.nan)

    for key, val in data['diff_vs_maxutil']['utility'].items():
        n, m = map(int, key.split(','))
        i = n - n_min
        j = m - m_min
        diff_maxutil_util[i, j] = val

    for key, val in data['diff_vs_maxutil']['nash'].items():
        n, m = map(int, key.split(','))
        i = n - n_min
        j = m - m_min
        diff_maxutil_nash[i, j] = val

    for key, val in data['diff_vs_rr']['utility'].items():
        n, m = map(int, key.split(','))
        i = n - n_min
        j = m - m_min
        diff_rr_util[i, j] = val

    for key, val in data['diff_vs_rr']['nash'].items():
        n, m = map(int, key.split(','))
        i = n - n_min
        j = m - m_min
        diff_rr_nash[i, j] = val

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    def plot_heatmap(matrix, title, filename, cmap='RdBu'):
        fig, ax = plt.subplots(figsize=(12, 10))
        vmax = max(abs(np.nanmin(matrix)), abs(np.nanmax(matrix)))
        im = ax.imshow(matrix, cmap=cmap, aspect='auto',
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

    # Plot all 4 heatmaps
    plot_heatmap(
        diff_maxutil_util,
        'Utility % Diff: Model(30x60)+EF1 - MaxUtil+EF1',
        'heatmap_30_60_vs_maxutil_utility.png'
    )
    plot_heatmap(
        diff_maxutil_nash,
        'Nash % Diff: Model(30x60)+EF1 - MaxUtil+EF1',
        'heatmap_30_60_vs_maxutil_nash.png'
    )
    plot_heatmap(
        diff_rr_util,
        'Utility % Diff: Model(30x60)+EF1 - RR+EF1',
        'heatmap_30_60_vs_rr_utility.png'
    )
    plot_heatmap(
        diff_rr_nash,
        'Nash % Diff: Model(30x60)+EF1 - RR+EF1',
        'heatmap_30_60_vs_rr_nash.png'
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='results/heatmap_30_60_model.json')
    parser.add_argument('--output', type=str, default='results')
    args = parser.parse_args()

    load_and_plot(args.input, args.output)
