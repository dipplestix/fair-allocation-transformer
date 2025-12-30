#!/usr/bin/env python3
"""Analyze results from variable size training sweep and create comparison plots."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json
import argparse

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


def load_sweep_results(sweep_id: str, project: str, entity: str = None):
    """Load results from a W&B sweep.

    Args:
        sweep_id: W&B sweep ID
        project: W&B project name
        entity: W&B entity (optional)

    Returns:
        DataFrame with sweep results
    """
    if not HAS_WANDB:
        raise ImportError("wandb is required to load sweep results")

    api = wandb.Api()

    if entity:
        sweep_path = f"{entity}/{project}/{sweep_id}"
    else:
        sweep_path = f"{project}/{sweep_id}"

    sweep = api.sweep(sweep_path)
    runs = sweep.runs

    data = []
    for run in runs:
        if run.state != "finished":
            continue

        config = run.config
        summary = run.summary

        row = {
            'run_id': run.id,
            'run_name': run.name,
            # Model params
            'd_model': config.get('d_model'),
            'num_heads': config.get('num_heads'),
            'num_output_layers': config.get('num_output_layers'),
            'num_encoder_layers': config.get('num_encoder_layers'),
            'dropout': config.get('dropout'),
            'pool_config_name': config.get('pool_config_name'),
            'residual_scale_init': config.get('residual_scale_init'),
            # Training params
            'lr': config.get('lr'),
            'weight_decay': config.get('weight_decay'),
            'batch_size': config.get('batch_size'),
            'initial_temperature': config.get('initial_temperature'),
            'final_temperature': config.get('final_temperature'),
            # Results
            'best_nash_welfare': summary.get('best_nash_welfare'),
            'final_residual_scale': summary.get('final_residual_scale'),
            'num_parameters': summary.get('num_parameters'),
            # Validation by size
            'val_nash_welfare_10x15': summary.get('val_nash_welfare_10x15'),
            'val_nash_welfare_25x35': summary.get('val_nash_welfare_25x35'),
            'val_nash_welfare_40x50': summary.get('val_nash_welfare_40x50'),
        }
        data.append(row)

    return pd.DataFrame(data)


def load_local_results(results_dir: Path):
    """Load results from local CSV files."""
    data = []

    # Look for variable size results
    for csv_file in results_dir.glob('variable_size_*.csv'):
        df = pd.read_csv(csv_file)
        # Parse the filename to get metadata
        parts = csv_file.stem.split('_')
        # Expected format: variable_size_<run_id>.csv

        for _, row in df.iterrows():
            data.append(row.to_dict())

    return pd.DataFrame(data)


def create_sweep_analysis_plots(df: pd.DataFrame, output_dir: Path):
    """Create analysis plots for sweep results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.style.use('seaborn-v0_8-darkgrid')

    # 1. Nash welfare distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(df['best_nash_welfare'].dropna(), bins=20, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Best Nash Welfare (avg across sizes)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Distribution of Best Nash Welfare Across Sweep Runs', fontsize=14, fontweight='bold')
    ax.axvline(df['best_nash_welfare'].max(), color='red', linestyle='--', label=f'Best: {df["best_nash_welfare"].max():.4f}')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'nash_welfare_distribution.png', dpi=300)
    plt.close()

    # 2. Model size vs performance
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(df['num_parameters'], df['best_nash_welfare'],
                        c=df['d_model'], cmap='viridis', alpha=0.7, s=50)
    ax.set_xlabel('Number of Parameters', fontsize=12)
    ax.set_ylabel('Best Nash Welfare', fontsize=12)
    ax.set_title('Model Size vs Performance', fontsize=14, fontweight='bold')
    plt.colorbar(scatter, label='d_model')
    plt.tight_layout()
    plt.savefig(output_dir / 'model_size_vs_performance.png', dpi=300)
    plt.close()

    # 3. Performance by problem size
    fig, ax = plt.subplots(figsize=(10, 6))

    sizes = ['10x15', '25x35', '40x50']
    cols = ['val_nash_welfare_10x15', 'val_nash_welfare_25x35', 'val_nash_welfare_40x50']

    box_data = [df[col].dropna() for col in cols]
    bp = ax.boxplot(box_data, labels=sizes, patch_artist=True)

    colors = ['#2E86AB', '#28A745', '#A23B72']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xlabel('Problem Size (n x m)', fontsize=12)
    ax.set_ylabel('Validation Nash Welfare', fontsize=12)
    ax.set_title('Nash Welfare by Problem Size', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_by_size.png', dpi=300)
    plt.close()

    # 4. Hyperparameter importance (correlation with best nash welfare)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    hp_cols = ['d_model', 'num_heads', 'num_output_layers', 'lr', 'residual_scale_init', 'dropout']
    for ax, col in zip(axes.flatten(), hp_cols):
        if col in df.columns and df[col].notna().sum() > 0:
            ax.scatter(df[col], df['best_nash_welfare'], alpha=0.5)
            ax.set_xlabel(col, fontsize=10)
            ax.set_ylabel('Best Nash Welfare', fontsize=10)

            # Add correlation
            valid_mask = df[col].notna() & df['best_nash_welfare'].notna()
            if valid_mask.sum() > 2:
                corr = df.loc[valid_mask, col].corr(df.loc[valid_mask, 'best_nash_welfare'])
                ax.set_title(f'Corr: {corr:.3f}', fontsize=10)

    plt.suptitle('Hyperparameter Importance', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'hyperparameter_importance.png', dpi=300)
    plt.close()

    # 5. Pool config comparison
    fig, ax = plt.subplots(figsize=(10, 6))

    pool_configs = df['pool_config_name'].dropna().unique()
    pool_means = [df[df['pool_config_name'] == pc]['best_nash_welfare'].mean() for pc in pool_configs]
    pool_stds = [df[df['pool_config_name'] == pc]['best_nash_welfare'].std() for pc in pool_configs]

    x = np.arange(len(pool_configs))
    ax.bar(x, pool_means, yerr=pool_stds, capsize=5, alpha=0.7, edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(pool_configs, rotation=45, ha='right')
    ax.set_xlabel('Pool Configuration', fontsize=12)
    ax.set_ylabel('Mean Best Nash Welfare', fontsize=12)
    ax.set_title('Performance by Pool Configuration', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'pool_config_comparison.png', dpi=300)
    plt.close()

    print(f"Plots saved to: {output_dir}")


def print_best_configs(df: pd.DataFrame, n: int = 5):
    """Print the best configurations from the sweep."""
    print("\n" + "=" * 80)
    print(f"TOP {n} CONFIGURATIONS BY NASH WELFARE")
    print("=" * 80)

    top_df = df.nlargest(n, 'best_nash_welfare')

    for i, (_, row) in enumerate(top_df.iterrows(), 1):
        print(f"\n--- Rank {i} ---")
        print(f"Run ID: {row['run_id']}")
        print(f"Best Nash Welfare (avg): {row['best_nash_welfare']:.6f}")
        print(f"  - 10x15: {row.get('val_nash_welfare_10x15', 'N/A'):.6f}" if pd.notna(row.get('val_nash_welfare_10x15')) else "  - 10x15: N/A")
        print(f"  - 25x35: {row.get('val_nash_welfare_25x35', 'N/A'):.6f}" if pd.notna(row.get('val_nash_welfare_25x35')) else "  - 25x35: N/A")
        print(f"  - 40x50: {row.get('val_nash_welfare_40x50', 'N/A'):.6f}" if pd.notna(row.get('val_nash_welfare_40x50')) else "  - 40x50: N/A")
        print(f"\nModel Architecture:")
        print(f"  d_model: {int(row['d_model'])}")
        print(f"  num_heads: {int(row['num_heads'])}")
        print(f"  num_output_layers: {int(row['num_output_layers'])}")
        print(f"  num_encoder_layers: {int(row['num_encoder_layers'])}")
        print(f"  pool_config_name: {row['pool_config_name']}")
        print(f"  dropout: {row['dropout']:.4f}")
        print(f"\nTraining:")
        print(f"  lr: {row['lr']:.6f}")
        print(f"  weight_decay: {row['weight_decay']:.6f}")
        print(f"  batch_size: {int(row['batch_size'])}")
        print(f"  residual_scale_init: {row['residual_scale_init']:.4f}")
        print(f"  final_residual_scale: {row['final_residual_scale']:.4f}")
        print(f"  initial_temperature: {row['initial_temperature']:.4f}")
        print(f"  final_temperature: {row['final_temperature']:.6f}")
        print(f"\nParameters: {int(row['num_parameters']):,}")


def save_best_config(df: pd.DataFrame, output_path: Path):
    """Save the best configuration to a YAML file."""
    best_row = df.loc[df['best_nash_welfare'].idxmax()]

    config = {
        'model_type': 'FFTransformerResidual_VariableSize',
        'd_model': int(best_row['d_model']),
        'num_heads': int(best_row['num_heads']),
        'num_output_layers': int(best_row['num_output_layers']),
        'num_encoder_layers': int(best_row['num_encoder_layers']),
        'dropout': float(best_row['dropout']),
        'pool_config_name': best_row['pool_config_name'],
        'residual_scale_init': float(best_row['residual_scale_init']),
        'lr': float(best_row['lr']),
        'weight_decay': float(best_row['weight_decay']),
        'batch_size': int(best_row['batch_size']),
        'initial_temperature': float(best_row['initial_temperature']),
        'final_temperature': float(best_row['final_temperature']),
        'n_min': 10,
        'n_max': 50,
        'm_max': 50,
        'steps': 100000,
        'grad_clip_norm': 1.0,
        'patience': 50,
        'min_delta': 1e-5,
        'val_every': 1000,
        'val_size': 500,
        'sweep_best_nash_welfare': float(best_row['best_nash_welfare']),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as JSON for compatibility
    with open(output_path.with_suffix('.json'), 'w') as f:
        json.dump(config, f, indent=2)

    # Also save as YAML if pyyaml is available
    try:
        import yaml
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"\nBest config saved to: {output_path}")
    except ImportError:
        print(f"\nBest config saved to: {output_path.with_suffix('.json')}")


def main():
    parser = argparse.ArgumentParser(description="Analyze variable size sweep results")
    parser.add_argument("--sweep-id", type=str, help="W&B sweep ID to load results from")
    parser.add_argument("--project", type=str, default="fa-transformer-variable-size-sweep",
                       help="W&B project name")
    parser.add_argument("--entity", type=str, default=None, help="W&B entity")
    parser.add_argument("--local-dir", type=str, default="results/variable_size",
                       help="Local directory to load results from (if not using W&B)")
    parser.add_argument("--output-dir", type=str, default="results/variable_size",
                       help="Output directory for plots")
    parser.add_argument("--save-best-config", action="store_true",
                       help="Save the best configuration to a YAML file")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if args.sweep_id:
        print(f"Loading results from W&B sweep: {args.sweep_id}")
        df = load_sweep_results(args.sweep_id, args.project, args.entity)
    else:
        print(f"Loading results from local directory: {args.local_dir}")
        df = load_local_results(Path(args.local_dir))

    if len(df) == 0:
        print("No results found!")
        return

    print(f"Loaded {len(df)} runs")

    # Print summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print(f"Total runs: {len(df)}")
    print(f"Best Nash Welfare: {df['best_nash_welfare'].max():.6f}")
    print(f"Mean Nash Welfare: {df['best_nash_welfare'].mean():.6f}")
    print(f"Std Nash Welfare: {df['best_nash_welfare'].std():.6f}")

    # Print best configs
    print_best_configs(df, n=5)

    # Create plots
    print("\nCreating analysis plots...")
    create_sweep_analysis_plots(df, output_dir)

    # Save best config
    if args.save_best_config:
        save_best_config(df, output_dir / "best_from_variable_size_sweep.yaml")

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
