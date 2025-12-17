#!/usr/bin/env python
"""
Export the best configuration from a Weights & Biases sweep.

This script fetches the best run from a sweep and creates a production
training config file ready to use with training/train.py.

Usage:
    python tools/export_best_sweep.py <sweep_id> --project <project> --entity <entity>
    python tools/export_best_sweep.py abc123xyz --project fa-transformer-sweep
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import wandb
except ImportError:
    print("Error: wandb is required. Install with: pip install wandb")
    sys.exit(1)

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    print("Warning: PyYAML not installed. Will output JSON instead.")
    print("Install with: pip install pyyaml")

import json


def get_run_metric(run, metric_name: str, default=None):
    """
    Safely retrieve a metric from a wandb Run with multiple fallback strategies.

    This function tries multiple approaches to access summary metrics to handle
    cases where the wandb API returns data in unexpected formats.

    Args:
        run: wandb Run object from API
        metric_name: Name of the metric to retrieve
        default: Default value if metric not found

    Returns:
        Metric value or default
    """
    # Primary approach: use the summary property
    try:
        summary = run.summary
        if summary is not None and hasattr(summary, 'get'):
            result = summary.get(metric_name)
            if result is not None:
                return result
    except (TypeError, AttributeError, RuntimeError):
        # Log but continue to fallbacks
        pass

    # Fallback 1: try summary_metrics attribute (direct dict access)
    try:
        summary_metrics = getattr(run, 'summary_metrics', None)
        if summary_metrics is not None and isinstance(summary_metrics, dict):
            result = summary_metrics.get(metric_name)
            if result is not None:
                return result
    except (TypeError, AttributeError):
        pass

    # Fallback 2: try internal _attrs (last resort)
    try:
        summary_dict = getattr(run, '_attrs', {}).get('summaryMetrics', {})
        if isinstance(summary_dict, dict):
            result = summary_dict.get(metric_name)
            if result is not None:
                return result
    except (TypeError, AttributeError):
        pass

    return default


def get_run_config_value(run, config_key: str, default=None):
    """
    Safely retrieve a config value from a wandb Run with multiple fallback strategies.

    This function tries multiple approaches to access config values to handle
    cases where the wandb API returns data in unexpected formats.

    Args:
        run: wandb Run object from API
        config_key: Name of the config parameter to retrieve
        default: Default value if config not found

    Returns:
        Config value or default
    """
    # Primary approach: use the config attribute
    try:
        config = run.config
        if config is not None and hasattr(config, 'get'):
            result = config.get(config_key)
            if result is not None:
                return result
        elif config is not None and isinstance(config, dict):
            result = config.get(config_key)
            if result is not None:
                return result
    except (TypeError, AttributeError, RuntimeError):
        pass

    # Fallback 1: try _attrs['config'] (internal access)
    try:
        config_dict = getattr(run, '_attrs', {}).get('config', {})
        if isinstance(config_dict, dict):
            result = config_dict.get(config_key)
            if result is not None:
                return result
    except (TypeError, AttributeError):
        pass

    return default


def get_best_sweep_config(sweep_id: str, project: str, entity: str | None = None):
    """Get the best configuration from a wandb sweep."""
    api = wandb.Api()

    # Construct sweep path
    if entity:
        sweep_path = f"{entity}/{project}/{sweep_id}"
    else:
        sweep_path = f"{project}/{sweep_id}"

    print(f"Fetching sweep: {sweep_path}")

    try:
        sweep = api.sweep(sweep_path)
    except Exception as e:
        print(f"Error fetching sweep: {e}")
        print(f"\nMake sure:")
        print(f"  1. You're logged in: wandb login")
        print(f"  2. Sweep ID is correct: {sweep_id}")
        print(f"  3. Project name is correct: {project}")
        if entity:
            print(f"  4. Entity is correct: {entity}")
        sys.exit(1)

    # Get the best run
    print("Finding best run...")
    best_run = sweep.best_run()

    if best_run is None:
        print("Error: No runs found in sweep")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"Best run found!")
    print(f"{'='*60}")
    print(f"Run name: {best_run.name}")
    print(f"Run ID: {best_run.id}")
    print(f"Run URL: {best_run.url}")

    # Get metrics - use defensive access with fallbacks
    nash_welfare = get_run_metric(best_run, 'nash_welfare', 'N/A')
    best_nash = get_run_metric(best_run, 'best_nash_welfare', nash_welfare)
    early_stop = get_run_metric(best_run, 'early_stop_step')

    if nash_welfare != 'N/A':
        print(f"Nash welfare: {nash_welfare}")
    if best_nash != 'N/A' and best_nash != nash_welfare:
        print(f"Best Nash welfare: {best_nash}")
    if early_stop:
        print(f"Early stopped at step: {early_stop}")

    print(f"{'='*60}\n")

    # Extract config - use defensive access with fallbacks
    config = {
        # Model architecture
        'n': get_run_config_value(best_run, 'n', 10),
        'm': get_run_config_value(best_run, 'm', 20),
        'd_model': get_run_config_value(best_run, 'd_model', 768),
        'num_heads': get_run_config_value(best_run, 'num_heads', 12),
        'num_output_layers': get_run_config_value(best_run, 'num_output_layers', 4),
        'dropout': get_run_config_value(best_run, 'dropout', 0.0),

        # Training hyperparameters
        'lr': get_run_config_value(best_run, 'lr', 1e-4),
        'weight_decay': get_run_config_value(best_run, 'weight_decay', 1e-2),
        'batch_size': get_run_config_value(best_run, 'batch_size', 512),
        'steps': 100000,  # Use longer for production (sweep uses 20k)

        # Temperature
        'initial_temperature': get_run_config_value(best_run, 'initial_temperature', 1.0),
        'final_temperature': get_run_config_value(best_run, 'final_temperature', 0.01),

        # Training settings
        'grad_clip_norm': get_run_config_value(best_run, 'grad_clip_norm', 1.0),
        'seed': 42,

        # Checkpointing
        'checkpoint_dir': f'checkpoints/from_sweep_{best_run.id[:8]}',
        'checkpoint_every': 5000,
        'keep_checkpoints': 3,

        # Validation
        'val_every': 1000,
        'val_size': 1000,

        # Early stopping
        'patience': 50,
        'min_delta': 1e-5,

        # Wandb
        'wandb_project': 'fa-transformer-production',
        'run_name': f'production_from_sweep_{best_run.id[:8]}',
    }

    # Add metadata as comments (if YAML)
    try:
        best_nash_value = float(best_nash) if best_nash != 'N/A' else None
    except (ValueError, TypeError):
        best_nash_value = None

    metadata = {
        '_sweep_id': sweep_id,
        '_sweep_url': sweep.url,
        '_best_run_id': best_run.id,
        '_best_run_url': best_run.url,
        '_best_nash_welfare': best_nash_value,
    }

    return config, metadata, best_run


def save_config(config: dict, metadata: dict, output_path: Path):
    """Save config to file (YAML or JSON)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if HAS_YAML and (output_path.suffix in ['.yaml', '.yml']):
        # Save as YAML with comments
        with open(output_path, 'w') as f:
            f.write(f"# Best configuration from wandb sweep\n")
            f.write(f"# Sweep ID: {metadata['_sweep_id']}\n")
            f.write(f"# Sweep URL: {metadata['_sweep_url']}\n")
            f.write(f"# Best run: {metadata['_best_run_id']}\n")
            f.write(f"# Best run URL: {metadata['_best_run_url']}\n")
            if metadata['_best_nash_welfare']:
                f.write(f"# Best Nash welfare: {metadata['_best_nash_welfare']:.6f}\n")
            f.write(f"#\n")
            f.write(f"# To train with this config:\n")
            f.write(f"#   python training/train.py --config {output_path}\n")
            f.write(f"\n")
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        print(f"✓ Config saved to: {output_path}")
    else:
        # Save as JSON
        output_data = {**metadata, **config}
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"✓ Config saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Export best config from wandb sweep",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export from sweep (auto-detect entity)
  python tools/export_best_sweep.py abc123xyz --project fa-transformer-sweep

  # Specify entity
  python tools/export_best_sweep.py abc123xyz --project my-project --entity my-team

  # Custom output location
  python tools/export_best_sweep.py abc123xyz --project my-project -o configs/my_best.yaml
        """
    )

    parser.add_argument("sweep_id", help="Wandb sweep ID")
    parser.add_argument("--project", required=True, help="Wandb project name")
    parser.add_argument("--entity", default=None, help="Wandb entity (defaults to your user)")
    parser.add_argument("-o", "--output", default=None,
                       help="Output file path (default: configs/best_from_sweep.yaml)")

    args = parser.parse_args()

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        # Default to YAML if available, JSON otherwise
        if HAS_YAML:
            output_path = Path("configs/best_from_sweep.yaml")
        else:
            output_path = Path("configs/best_from_sweep.json")

    # Get best config
    config, metadata, best_run = get_best_sweep_config(
        args.sweep_id, args.project, args.entity
    )

    # Save to file
    save_config(config, metadata, output_path)

    # Print configuration summary
    print(f"\n{'='*60}")
    print("Configuration Summary")
    print(f"{'='*60}")
    print(f"Model: n={config['n']}, m={config['m']}, d_model={config['d_model']}")
    print(f"Architecture: heads={config['num_heads']}, layers={config['num_output_layers']}")
    print(f"Optimization: lr={config['lr']:.2e}, wd={config['weight_decay']:.2e}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Temperature: {config['initial_temperature']:.2f} → {config['final_temperature']:.4f}")
    print(f"Training steps: {config['steps']:,} (increased from sweep's 20k)")
    print(f"{'='*60}\n")

    # Print next steps
    print("Next steps:")
    print(f"  1. Review config: cat {output_path}")
    print(f"  2. Start training: python training/train.py --config {output_path}")
    print(f"  3. Monitor: https://wandb.ai/{args.entity or 'your-entity'}/{config['wandb_project']}")
    print()


if __name__ == "__main__":
    main()
