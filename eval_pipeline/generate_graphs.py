import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def parse_value(val):
    """Parse value, handling percentage strings"""
    if isinstance(val, str):
        if '%' in val:
            return float(val.strip('%')) / 100
        return float(val)
    return val

def create_plot(df, metric_name, ylabel, filename):
    """Create a single plot for a metric"""
    plt.figure(figsize=(10, 6))

    inference_types = ['model', 'rr', 'random']
    colors = {'model': '#2E86AB', 'rr': '#A23B72', 'random': '#F18F01'}
    markers = {'model': 'o', 'rr': 's', 'random': '^'}

    for inference in inference_types:
        col_mean = f'{inference}_{metric_name}'
        col_std = f'{inference}_{metric_name}_SD'

        y_values = df[col_mean].apply(parse_value).values
        y_std = df[col_std].apply(parse_value).values

        plt.errorbar(df['items'], y_values, yerr=y_std,
                    label=inference.upper() if inference != 'model' else 'Model',
                    marker=markers[inference], markersize=8,
                    color=colors[inference], linewidth=2,
                    capsize=5, capthick=2, alpha=0.8)

    plt.xlabel('Number of Items', fontsize=12, fontweight='bold')
    plt.ylabel(ylabel, fontsize=12, fontweight='bold')
    plt.title(f'{ylabel} vs Number of Items', fontsize=14, fontweight='bold', pad=20)
    plt.legend(loc='best', frameon=True, shadow=True, fontsize=11)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {filename}")

def generate_graphs():
    """Generate all plots from summary.csv"""
    df = pd.read_csv('summary.csv')

    # Create graphs directory
    os.makedirs('graphs', exist_ok=True)

    # Extract number of items from Agent-Item column
    df['items'] = df['Agent-Item'].apply(lambda x: int(x.split('-')[1]))
    df = df.sort_values('items')

    # Create plots for each metric
    metrics = [
        ('EF', 'Envy-Free Proportion', 'graphs/ef_plot.png'),
        ('EF1', 'EF1 Proportion', 'graphs/ef1_plot.png'),
        ('EFx', 'EFx Proportion', 'graphs/efx_plot.png'),
        ('Util_Frac', 'Utilitarian Welfare Fraction', 'graphs/util_frac_plot.png'),
        ('Nash_Frac', 'Nash Welfare Fraction', 'graphs/nash_frac_plot.png')
    ]

    for metric_name, ylabel, filename in metrics:
        create_plot(df, metric_name, ylabel, filename)

    print("\nAll plots generated successfully in graphs/ directory!")

if __name__ == "__main__":
    generate_graphs()