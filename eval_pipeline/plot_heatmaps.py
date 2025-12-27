#!/usr/bin/env python3
"""
Plot utility% and nash% heatmaps from generated data.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_and_plot(data_file, output_dir='results', suffix='', title_suffix=''):
    with open(data_file, 'r') as f:
        data = json.load(f)

    n_min, n_max = data['n_range']
    m_min, m_max = data['m_range']

    n_vals = list(range(n_min, n_max + 1))
    m_vals = list(range(m_min, m_max + 1))

    # Create matrices for heatmaps (NaN where m < n)
    util_matrix = np.full((len(n_vals), len(m_vals)), np.nan)
    nash_matrix = np.full((len(n_vals), len(m_vals)), np.nan)

    for key, val in data['utility'].items():
        n, m = map(int, key.split(','))
        i = n - n_min
        j = m - m_min
        util_matrix[i, j] = val

    for key, val in data['nash'].items():
        n, m = map(int, key.split(','))
        i = n - n_min
        j = m - m_min
        nash_matrix[i, j] = val

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Determine title based on data or override
    base_title = title_suffix if title_suffix else 'Residual FATransformer'

    # Plot utility heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(util_matrix, cmap='RdYlGn', aspect='auto',
                   vmin=80, vmax=100, origin='lower')
    ax.set_xticks(range(len(m_vals)))
    ax.set_xticklabels(m_vals)
    ax.set_yticks(range(len(n_vals)))
    ax.set_yticklabels(n_vals)
    ax.set_xlabel('m (items)', fontsize=12)
    ax.set_ylabel('n (agents)', fontsize=12)
    ax.set_title(f'Utility % ({base_title})', fontsize=14)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Utility %', fontsize=12)

    # Add text annotations
    for i in range(len(n_vals)):
        for j in range(len(m_vals)):
            if not np.isnan(util_matrix[i, j]):
                text_color = 'white' if util_matrix[i, j] < 90 else 'black'
                ax.text(j, i, f'{util_matrix[i, j]:.1f}',
                       ha='center', va='center', fontsize=6, color=text_color)

    plt.tight_layout()
    util_filename = f'heatmap_utility{suffix}.png'
    plt.savefig(output_path / util_filename, dpi=150)
    plt.close()
    print(f"Saved: {output_path / util_filename}")

    # Plot nash heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(nash_matrix, cmap='RdYlGn', aspect='auto',
                   vmin=80, vmax=100, origin='lower')
    ax.set_xticks(range(len(m_vals)))
    ax.set_xticklabels(m_vals)
    ax.set_yticks(range(len(n_vals)))
    ax.set_yticklabels(n_vals)
    ax.set_xlabel('m (items)', fontsize=12)
    ax.set_ylabel('n (agents)', fontsize=12)
    ax.set_title(f'Nash Welfare % ({base_title})', fontsize=14)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Nash %', fontsize=12)

    # Add text annotations
    for i in range(len(n_vals)):
        for j in range(len(m_vals)):
            if not np.isnan(nash_matrix[i, j]):
                text_color = 'white' if nash_matrix[i, j] < 90 else 'black'
                ax.text(j, i, f'{nash_matrix[i, j]:.1f}',
                       ha='center', va='center', fontsize=6, color=text_color)

    plt.tight_layout()
    nash_filename = f'heatmap_nash{suffix}.png'
    plt.savefig(output_path / nash_filename, dpi=150)
    plt.close()
    print(f"Saved: {output_path / nash_filename}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='results/heatmap_data.json')
    parser.add_argument('--output', type=str, default='results')
    parser.add_argument('--suffix', type=str, default='', help='Suffix for output filenames')
    parser.add_argument('--title', type=str, default='', help='Title suffix for plots')
    args = parser.parse_args()

    load_and_plot(args.input, args.output, args.suffix, args.title)
