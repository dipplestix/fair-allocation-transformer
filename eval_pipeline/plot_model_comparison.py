#!/usr/bin/env python3
"""
Generate comprehensive figures for model comparison results.

Creates:
1. Bar chart comparing mean performance
2. Heatmaps for each method
3. Difference heatmaps (model - baselines)
4. Performance vs problem size plots
5. Distribution plots
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from collections import defaultdict

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['figure.dpi'] = 150


def load_results(results_path):
    """Load results from JSON file."""
    with open(results_path) as f:
        return json.load(f)


def parse_key(key):
    """Parse 'n,m' key to integers."""
    n, m = key.split(',')
    return int(n), int(m)


def create_heatmap_data(results_dict, n_range, m_range):
    """Convert results dict to 2D array for heatmap."""
    n_min, n_max = n_range
    m_min, m_max = m_range

    # Create empty array (NaN for missing values)
    data = np.full((n_max - n_min + 1, m_max - m_min + 1), np.nan)

    for key, value in results_dict.items():
        n, m = parse_key(key)
        if n_min <= n <= n_max and m_min <= m <= m_max:
            data[n - n_min, m - m_min] = value

    return data


def plot_bar_comparison(results, output_dir):
    """Create bar chart comparing mean performance across methods."""
    methods = results['methods']

    # Calculate means
    nash_means = []
    util_means = []
    nash_stds = []
    util_stds = []

    for method in methods:
        nash_values = list(results[method]['nash'].values())
        util_values = list(results[method]['utility'].values())
        nash_means.append(np.mean(nash_values))
        util_means.append(np.mean(util_values))
        nash_stds.append(np.std(nash_values))
        util_stds.append(np.std(util_values))

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(methods))
    width = 0.35

    # Define colors
    colors_nash = ['#2ecc71', '#27ae60', '#1abc9c', '#3498db', '#9b59b6', '#e74c3c']
    colors_util = ['#a8e6cf', '#81c784', '#80cbc4', '#90caf9', '#ce93d8', '#ef9a9a']

    bars1 = ax.bar(x - width/2, nash_means, width, label='Nash Welfare',
                   color=colors_nash, yerr=nash_stds, capsize=3, alpha=0.9)
    bars2 = ax.bar(x + width/2, util_means, width, label='Utilitarian Welfare',
                   color=colors_util, yerr=util_stds, capsize=3, alpha=0.9)

    # Customize
    ax.set_ylabel('Mean % of Optimal')
    ax.set_title('Model Comparison: Mean Welfare Across All Problem Sizes')
    ax.set_xticks(x)

    # Format method names
    method_labels = [m.replace('_', ' ').replace('ef1', '+ EF1').title() for m in methods]
    ax.set_xticklabels(method_labels, rotation=15, ha='right')
    ax.legend(loc='lower right')
    ax.set_ylim(70, 100)

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / 'bar_comparison.png', bbox_inches='tight')
    plt.savefig(output_dir / 'bar_comparison.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: bar_comparison.png/pdf")


def plot_all_heatmaps(results, output_dir):
    """Create heatmaps for each method."""
    methods = results['methods']
    n_range = results['n_range']
    m_range = results['m_range']

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    # Use consistent color range
    vmin, vmax = 70, 100

    method_titles = {
        'model_10x20_ef1': '10-20 Model + EF1',
        'model_30x60_ef1': '30-60 Model + EF1',
        'model_multi_objective_ef1': 'Multi-Objective + EF1',
        'maxutil_ef1': 'MaxUtil + EF1',
        'rr': 'Round-Robin',
        'ece': 'ECE'
    }

    for idx, method in enumerate(methods):
        ax = axes[idx]
        data = create_heatmap_data(results[method]['nash'], n_range, m_range)

        # Mask upper triangle (where m < n)
        mask = np.triu(np.ones_like(data, dtype=bool), k=1)
        data_masked = np.ma.masked_where(mask.T, data)

        im = ax.imshow(data_masked, cmap='RdYlGn', vmin=vmin, vmax=vmax,
                       origin='lower', aspect='auto')

        ax.set_title(method_titles.get(method, method))
        ax.set_xlabel('Number of Items (m)')
        ax.set_ylabel('Number of Agents (n)')

        # Set ticks
        n_ticks = list(range(0, n_range[1] - n_range[0] + 1, 5))
        m_ticks = list(range(0, m_range[1] - m_range[0] + 1, 5))
        ax.set_xticks(m_ticks)
        ax.set_yticks(n_ticks)
        ax.set_xticklabels([m_range[0] + t for t in m_ticks])
        ax.set_yticklabels([n_range[0] + t for t in n_ticks])

    # Add colorbar
    fig.colorbar(im, ax=axes, label='Nash Welfare (% of Optimal)', shrink=0.8)

    plt.suptitle('Nash Welfare Heatmaps by Method', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'heatmaps_all_methods.png', bbox_inches='tight')
    plt.savefig(output_dir / 'heatmaps_all_methods.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: heatmaps_all_methods.png/pdf")


def plot_difference_heatmaps(results, output_dir):
    """Create difference heatmaps (best model - each baseline)."""
    n_range = results['n_range']
    m_range = results['m_range']

    # Get best model performance (use multi-objective as representative)
    best_model = 'model_multi_objective_ef1'
    baselines = ['maxutil_ef1', 'rr', 'ece']
    baseline_titles = {
        'maxutil_ef1': 'vs MaxUtil + EF1',
        'rr': 'vs Round-Robin',
        'ece': 'vs ECE'
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, baseline in enumerate(baselines):
        ax = axes[idx]

        # Calculate difference
        model_data = create_heatmap_data(results[best_model]['nash'], n_range, m_range)
        baseline_data = create_heatmap_data(results[baseline]['nash'], n_range, m_range)
        diff_data = model_data - baseline_data

        # Mask upper triangle
        mask = np.triu(np.ones_like(diff_data, dtype=bool), k=1)
        diff_masked = np.ma.masked_where(mask.T, diff_data)

        # Use diverging colormap centered at 0
        vmax = max(abs(np.nanmin(diff_data)), abs(np.nanmax(diff_data)))
        if baseline == 'ece':
            vmax = 50  # ECE has larger differences

        im = ax.imshow(diff_masked, cmap='RdYlGn', vmin=-vmax, vmax=vmax,
                       origin='lower', aspect='auto')

        ax.set_title(f'Multi-Objective Model {baseline_titles[baseline]}')
        ax.set_xlabel('Number of Items (m)')
        ax.set_ylabel('Number of Agents (n)')

        # Set ticks
        n_ticks = list(range(0, n_range[1] - n_range[0] + 1, 5))
        m_ticks = list(range(0, m_range[1] - m_range[0] + 1, 5))
        ax.set_xticks(m_ticks)
        ax.set_yticks(n_ticks)
        ax.set_xticklabels([m_range[0] + t for t in m_ticks])
        ax.set_yticklabels([n_range[0] + t for t in n_ticks])

        # Add colorbar for each
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Improvement (%)')

    plt.suptitle('Nash Welfare Improvement: Multi-Objective Model vs Baselines', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'heatmaps_difference.png', bbox_inches='tight')
    plt.savefig(output_dir / 'heatmaps_difference.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: heatmaps_difference.png/pdf")


def plot_performance_vs_size(results, output_dir):
    """Plot performance vs problem size (n + m)."""
    methods = results['methods']

    # Group by total size
    size_to_nash = defaultdict(lambda: defaultdict(list))

    for method in methods:
        for key, value in results[method]['nash'].items():
            n, m = parse_key(key)
            total_size = n + m
            size_to_nash[method][total_size].append(value)

    # Calculate means per size
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = {
        'model_10x20_ef1': '#2ecc71',
        'model_30x60_ef1': '#27ae60',
        'model_multi_objective_ef1': '#1abc9c',
        'maxutil_ef1': '#3498db',
        'rr': '#9b59b6',
        'ece': '#e74c3c'
    }

    labels = {
        'model_10x20_ef1': '10-20 Model + EF1',
        'model_30x60_ef1': '30-60 Model + EF1',
        'model_multi_objective_ef1': 'Multi-Objective + EF1',
        'maxutil_ef1': 'MaxUtil + EF1',
        'rr': 'Round-Robin',
        'ece': 'ECE'
    }

    for method in methods:
        sizes = sorted(size_to_nash[method].keys())
        means = [np.mean(size_to_nash[method][s]) for s in sizes]
        stds = [np.std(size_to_nash[method][s]) for s in sizes]

        ax.plot(sizes, means, 'o-', label=labels[method], color=colors[method],
                linewidth=2, markersize=4, alpha=0.8)
        ax.fill_between(sizes,
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        alpha=0.15, color=colors[method])

    ax.set_xlabel('Problem Size (n + m)')
    ax.set_ylabel('Nash Welfare (% of Optimal)')
    ax.set_title('Nash Welfare vs Problem Size')
    ax.legend(loc='lower left', ncol=2)
    ax.set_ylim(60, 100)

    plt.tight_layout()
    plt.savefig(output_dir / 'performance_vs_size.png', bbox_inches='tight')
    plt.savefig(output_dir / 'performance_vs_size.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: performance_vs_size.png/pdf")


def plot_distribution(results, output_dir):
    """Create box plots showing distribution of results."""
    methods = results['methods']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Prepare data
    nash_data = [list(results[m]['nash'].values()) for m in methods]
    util_data = [list(results[m]['utility'].values()) for m in methods]

    labels = ['10-20\n+EF1', '30-60\n+EF1', 'Multi-Obj\n+EF1',
              'MaxUtil\n+EF1', 'RR', 'ECE']

    colors = ['#2ecc71', '#27ae60', '#1abc9c', '#3498db', '#9b59b6', '#e74c3c']

    # Nash welfare box plot
    bp1 = ax1.boxplot(nash_data, labels=labels, patch_artist=True)
    for patch, color in zip(bp1['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax1.set_ylabel('Nash Welfare (% of Optimal)')
    ax1.set_title('Nash Welfare Distribution')
    ax1.set_ylim(30, 100)

    # Utilitarian welfare box plot
    bp2 = ax2.boxplot(util_data, labels=labels, patch_artist=True)
    for patch, color in zip(bp2['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax2.set_ylabel('Utilitarian Welfare (% of Optimal)')
    ax2.set_title('Utilitarian Welfare Distribution')
    ax2.set_ylim(30, 100)

    plt.suptitle('Distribution of Welfare Across Problem Configurations', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'distribution_boxplot.png', bbox_inches='tight')
    plt.savefig(output_dir / 'distribution_boxplot.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: distribution_boxplot.png/pdf")


def plot_model_comparison_heatmaps(results, output_dir):
    """Create side-by-side heatmaps comparing the three models."""
    n_range = results['n_range']
    m_range = results['m_range']

    models = ['model_10x20_ef1', 'model_30x60_ef1', 'model_multi_objective_ef1']
    titles = ['10-20 Trained Model + EF1', '30-60 Trained Model + EF1', 'Multi-Objective Model + EF1']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    vmin, vmax = 91, 99

    for idx, (model, title) in enumerate(zip(models, titles)):
        ax = axes[idx]
        data = create_heatmap_data(results[model]['nash'], n_range, m_range)

        # Mask upper triangle
        mask = np.triu(np.ones_like(data, dtype=bool), k=1)
        data_masked = np.ma.masked_where(mask.T, data)

        im = ax.imshow(data_masked, cmap='RdYlGn', vmin=vmin, vmax=vmax,
                       origin='lower', aspect='auto')

        ax.set_title(title)
        ax.set_xlabel('Number of Items (m)')
        ax.set_ylabel('Number of Agents (n)')

        # Set ticks
        n_ticks = list(range(0, n_range[1] - n_range[0] + 1, 5))
        m_ticks = list(range(0, m_range[1] - m_range[0] + 1, 5))
        ax.set_xticks(m_ticks)
        ax.set_yticks(n_ticks)
        ax.set_xticklabels([m_range[0] + t for t in m_ticks])
        ax.set_yticklabels([n_range[0] + t for t in n_ticks])

    # Add colorbar
    fig.colorbar(im, ax=axes, label='Nash Welfare (% of Optimal)', shrink=0.8)

    plt.suptitle('Nash Welfare Comparison: Three Training Strategies', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'heatmaps_models_only.png', bbox_inches='tight')
    plt.savefig(output_dir / 'heatmaps_models_only.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: heatmaps_models_only.png/pdf")


def plot_items_per_agent(results, output_dir):
    """Plot performance vs items-per-agent ratio."""
    methods = results['methods']

    # Group by m/n ratio
    ratio_to_nash = defaultdict(lambda: defaultdict(list))

    for method in methods:
        for key, value in results[method]['nash'].items():
            n, m = parse_key(key)
            ratio = m / n
            ratio_to_nash[method][round(ratio, 2)].append(value)

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = {
        'model_10x20_ef1': '#2ecc71',
        'model_30x60_ef1': '#27ae60',
        'model_multi_objective_ef1': '#1abc9c',
        'maxutil_ef1': '#3498db',
        'rr': '#9b59b6',
        'ece': '#e74c3c'
    }

    labels = {
        'model_10x20_ef1': '10-20 Model + EF1',
        'model_30x60_ef1': '30-60 Model + EF1',
        'model_multi_objective_ef1': 'Multi-Objective + EF1',
        'maxutil_ef1': 'MaxUtil + EF1',
        'rr': 'Round-Robin',
        'ece': 'ECE'
    }

    for method in methods:
        ratios = sorted(ratio_to_nash[method].keys())
        means = [np.mean(ratio_to_nash[method][r]) for r in ratios]

        ax.plot(ratios, means, 'o-', label=labels[method], color=colors[method],
                linewidth=2, markersize=4, alpha=0.8)

    ax.set_xlabel('Items per Agent Ratio (m/n)')
    ax.set_ylabel('Nash Welfare (% of Optimal)')
    ax.set_title('Nash Welfare vs Items-per-Agent Ratio')
    ax.legend(loc='lower right', ncol=2)
    ax.set_ylim(60, 100)

    plt.tight_layout()
    plt.savefig(output_dir / 'performance_vs_ratio.png', bbox_inches='tight')
    plt.savefig(output_dir / 'performance_vs_ratio.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: performance_vs_ratio.png/pdf")


def plot_summary_table(results, output_dir):
    """Create a summary table as an image."""
    methods = results['methods']

    # Calculate statistics
    rows = []
    for method in methods:
        nash_values = list(results[method]['nash'].values())
        util_values = list(results[method]['utility'].values())
        rows.append([
            method.replace('_', ' ').replace('ef1', '+ EF1').title(),
            f"{np.mean(nash_values):.2f}%",
            f"{np.min(nash_values):.2f}%",
            f"{np.max(nash_values):.2f}%",
            f"{np.mean(util_values):.2f}%",
            f"{np.min(util_values):.2f}%",
            f"{np.max(util_values):.2f}%"
        ])

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axis('off')

    columns = ['Method', 'Nash Mean', 'Nash Min', 'Nash Max',
               'Util Mean', 'Util Min', 'Util Max']

    table = ax.table(cellText=rows, colLabels=columns, loc='center',
                     cellLoc='center', colColours=['#3498db']*7)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    # Style header
    for i in range(len(columns)):
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Highlight best values
    nash_means = [float(r[1].replace('%', '')) for r in rows]
    util_means = [float(r[4].replace('%', '')) for r in rows]
    best_nash_idx = np.argmax(nash_means)
    best_util_idx = np.argmax(util_means)

    table[(best_nash_idx + 1, 1)].set_facecolor('#a8e6cf')
    table[(best_util_idx + 1, 4)].set_facecolor('#a8e6cf')

    plt.title('Summary Statistics: Welfare as % of Optimal', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(output_dir / 'summary_table.png', bbox_inches='tight')
    plt.savefig(output_dir / 'summary_table.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: summary_table.png/pdf")


def main():
    # Paths
    results_path = Path('results/model_comparison_heatmap.json')
    output_dir = Path('results/figures')
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading results...")
    results = load_results(results_path)

    print("\nGenerating figures:")

    # Generate all figures
    plot_bar_comparison(results, output_dir)
    plot_all_heatmaps(results, output_dir)
    plot_difference_heatmaps(results, output_dir)
    plot_model_comparison_heatmaps(results, output_dir)
    plot_performance_vs_size(results, output_dir)
    plot_items_per_agent(results, output_dir)
    plot_distribution(results, output_dir)
    plot_summary_table(results, output_dir)

    print(f"\nAll figures saved to: {output_dir}/")


if __name__ == "__main__":
    main()
