#!/usr/bin/env python
"""
Production training script for FFTransformerResidual with variable size inputs.

For each training step, n is drawn from Uniform(10, 50) and m is drawn from Uniform(n, 50).
This trains a single size-agnostic model that can handle varying problem sizes.

Use after identifying best hyperparameters from variable_size_sweep.py.
Supports longer training runs, checkpointing, and resuming.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import torch
import torch.optim as optim

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

# Ensure local imports work
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fftransformer.fftransformer_residual import FFTransformerResidual  # noqa: E402
from fftransformer.helpers import get_nash_welfare  # noqa: E402

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


# Size sampling range
N_MIN = 10
N_MAX = 50
M_MAX = 50

# Pool config options (same as in bayesian_sweep_residual.py)
POOL_CONFIGS = {
    "row_only": {'row': ['mean', 'max', 'min'], 'column': [], 'global': []},
    "row_col": {'row': ['mean', 'max', 'min'], 'column': ['mean', 'max', 'min'], 'global': []},
    "row_global": {'row': ['mean', 'max', 'min'], 'column': [], 'global': ['mean', 'max', 'min']},
    "all": {'row': ['mean', 'max', 'min'], 'column': ['mean', 'max', 'min'], 'global': ['mean', 'max', 'min']},
    "row_col_mean": {'row': 'mean', 'column': 'mean', 'global': []},
}


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Production training for FFTransformerResidual with variable sizes"
    )

    # Config options
    parser.add_argument("--config", type=str, help="Path to YAML/JSON config file")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")

    # Model hyperparameters (can override config)
    parser.add_argument("--d-model", type=int, default=256, help="Model dimension")
    parser.add_argument("--num-heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--num-output-layers", type=int, default=2, help="Number of output layers")
    parser.add_argument("--num-encoder-layers", type=int, default=1, help="Number of encoder layers")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate")

    # Residual-specific parameters
    parser.add_argument("--pool-config-name", type=str, default="row_col",
                       choices=list(POOL_CONFIGS.keys()),
                       help="Pool configuration name")
    parser.add_argument("--residual-scale-init", type=float, default=0.1,
                       help="Initial residual scale value")

    # Training hyperparameters
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="Weight decay")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--steps", type=int, default=100000, help="Total training steps")

    # Temperature
    parser.add_argument("--initial-temperature", type=float, default=1.0)
    parser.add_argument("--final-temperature", type=float, default=0.01)

    # Training settings
    parser.add_argument("--grad-clip-norm", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    # Checkpointing
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/variable_size",
                       help="Directory to save checkpoints")
    parser.add_argument("--checkpoint-every", type=int, default=5000,
                       help="Save checkpoint every N steps")
    parser.add_argument("--keep-checkpoints", type=int, default=3,
                       help="Number of recent checkpoints to keep")

    # Validation
    parser.add_argument("--val-every", type=int, default=1000,
                       help="Validate every N steps")
    parser.add_argument("--val-size", type=int, default=500,
                       help="Number of validation examples per size")

    # Early stopping
    parser.add_argument("--patience", type=int, default=50,
                       help="Early stopping patience")
    parser.add_argument("--min-delta", type=float, default=1e-5,
                       help="Minimum improvement for early stopping")

    # Wandb
    parser.add_argument("--wandb-project", type=str, default="fa-transformer-variable-size-production")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None, help="Wandb run name")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")

    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load config from YAML or JSON file."""
    with open(config_path) as f:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            if not HAS_YAML:
                raise ImportError("PyYAML is required for YAML config files. Install with: pip install pyyaml")
            return yaml.safe_load(f)
        else:
            return json.load(f)


def save_checkpoint(
    checkpoint_dir: Path,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    config: Dict[str, Any],
    best_metric: float,
    keep_last: int = 3,
):
    """Save training checkpoint and manage checkpoint cleanup."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / f"checkpoint_{step:07d}.pt"

    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'config': config,
        'best_metric': best_metric,
        'rng_state': torch.get_rng_state(),
        'model_type': 'FFTransformerResidual_VariableSize',
    }, checkpoint_path)

    # Clean up old checkpoints
    checkpoints = sorted(checkpoint_dir.glob("checkpoint_*.pt"))
    if len(checkpoints) > keep_last:
        for old_ckpt in checkpoints[:-keep_last]:
            old_ckpt.unlink()

    print(f"Saved checkpoint: {checkpoint_path}")
    return checkpoint_path


def load_checkpoint(checkpoint_path: str):
    """Load checkpoint for resuming training."""
    return torch.load(checkpoint_path)


def sample_problem_size():
    """Sample n from Uniform(10, 50) and m from Uniform(n, 50)."""
    n = torch.randint(N_MIN, N_MAX + 1, (1,)).item()
    m = torch.randint(n, M_MAX + 1, (1,)).item()
    return n, m


def validate(model, val_sets, device):
    """Run validation on fixed validation sets at multiple sizes.

    Args:
        model: The model to validate
        val_sets: List of (n, m, valuations) tuples
        device: Device to run on

    Returns:
        dict: Validation Nash welfare for each size and average
    """
    was_training = model.training
    current_temp = model.temperature

    model.eval()

    results = {}
    nash_welfares = []

    with torch.no_grad():
        for val_n, val_m, val_valuations in val_sets:
            val_allocations = model(val_valuations)
            val_nw = get_nash_welfare(
                val_valuations, val_allocations, reduction="mean"
            ).item()
            results[f"val_nash_welfare_{val_n}x{val_m}"] = val_nw
            nash_welfares.append(val_nw)

    results["val_nash_welfare_avg"] = sum(nash_welfares) / len(nash_welfares)

    # Restore original state
    if was_training:
        model.train()
    model.temperature = current_temp

    return results


def create_model(config: Dict[str, Any], device: torch.device):
    """Create FFTransformerResidual model from config."""
    model = FFTransformerResidual(
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        num_output_layers=config['num_output_layers'],
        num_encoder_layers=config.get('num_encoder_layers', 1),
        dropout=config['dropout'],
        initial_temperature=config['initial_temperature'],
        final_temperature=config['final_temperature'],
    ).to(device)

    # Override residual scale init if specified
    residual_scale_init = config.get('residual_scale_init', 0.1)
    with torch.no_grad():
        model.residual_scale.fill_(residual_scale_init)

    return model


def train(config: Dict[str, Any]):
    """Main training loop with variable size inputs."""

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    torch.manual_seed(config['seed'])

    checkpoint_dir = Path(config['checkpoint_dir'])

    # Initialize wandb
    use_wandb = HAS_WANDB and not config.get('no_wandb')
    if use_wandb:
        wandb.init(
            project=config['wandb_project'],
            entity=config.get('wandb_entity'),
            name=config.get('run_name'),
            config=config,
        )
    elif not HAS_WANDB and not config.get('no_wandb'):
        print("Warning: wandb not installed. Install with: pip install wandb")

    # Create model
    model = create_model(config, device)

    print(f"Model created: {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Residual scale init: {model.residual_scale.item():.4f}")
    print(f"Variable size training: n in [{N_MIN}, {N_MAX}], m in [n, {M_MAX}]")

    # Optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['steps']
    )

    # Create fixed validation sets at different sizes
    val_seed = 0
    torch.manual_seed(val_seed)
    val_sets = []
    # Validation sizes: small (10,15), medium (25,35), large (40,50)
    val_sizes = [(10, 15), (25, 35), (40, 50)]
    for n, m in val_sizes:
        val_valuations = torch.rand(config['val_size'], n, m, device=device)
        val_sets.append((n, m, val_valuations))
    print(f"Validation sets: {val_sizes}")
    torch.manual_seed(config['seed'])

    # Resume from checkpoint if specified
    start_step = 0
    best_metric = float('-inf')
    if config.get('resume'):
        print(f"Resuming from checkpoint: {config['resume']}")
        ckpt = load_checkpoint(config['resume'])
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        torch.set_rng_state(ckpt['rng_state'])
        start_step = ckpt['step'] + 1
        best_metric = ckpt['best_metric']
        print(f"Resumed from step {start_step}, best metric: {best_metric:.6f}")

    # Early stopping
    steps_without_improvement = 0

    # Training loop
    print(f"Starting training for {config['steps']} steps...")
    for step in range(start_step, config['steps']):
        model.train()

        # Update temperature with cosine annealing (starts high, ends low)
        progress = step / config['steps']
        current_temp = config['final_temperature'] + \
            (config['initial_temperature'] - config['final_temperature']) * \
            (1 + torch.cos(torch.tensor(torch.pi * progress))) / 2
        model.update_temperature(current_temp.item())

        # Sample variable problem size for this batch
        n, m = sample_problem_size()

        # Generate batch
        valuations = torch.rand(
            config['batch_size'], n, m, device=device
        )

        # Forward pass
        allocation = model(valuations)
        nash_welfare = get_nash_welfare(
            valuations, allocation, reduction="mean"
        )
        loss = -nash_welfare

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        if config.get('grad_clip_norm'):
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=config['grad_clip_norm']
            )
        optimizer.step()
        scheduler.step()

        # Logging
        metrics = {
            "step": step,
            "loss": loss.item(),
            "nash_welfare": nash_welfare.item(),
            "train_n": n,
            "train_m": m,
            "lr": scheduler.get_last_lr()[0],
            "temperature": current_temp.item(),
            "residual_scale": model.residual_scale.item(),
        }

        # Validation
        if step % config['val_every'] == 0:
            val_results = validate(model, val_sets, device)
            metrics.update(val_results)

            avg_val_nw = val_results["val_nash_welfare_avg"]

            # Check for improvement
            if avg_val_nw > best_metric + config['min_delta']:
                best_metric = avg_val_nw
                steps_without_improvement = 0

                # Save best model as full checkpoint (for resuming training)
                best_checkpoint_path = checkpoint_dir / "best_checkpoint.pt"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                torch.save({
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'config': config,
                    'best_metric': best_metric,
                    'rng_state': torch.get_rng_state(),
                    'model_type': 'FFTransformerResidual_VariableSize',
                }, best_checkpoint_path)

                # Also save just the model weights for easy inference
                best_weights_path = checkpoint_dir / "best_model.pt"
                torch.save(model.state_dict(), best_weights_path)

                print(f"Step {step}: New best avg validation Nash welfare: {best_metric:.6f}")
                for size_n, size_m in val_sizes:
                    print(f"  {size_n}x{size_m}: {val_results[f'val_nash_welfare_{size_n}x{size_m}']:.6f}")
                if use_wandb:
                    wandb.save(str(best_checkpoint_path))
                    wandb.save(str(best_weights_path))
            else:
                steps_without_improvement += 1

        # Log to console every 100 steps
        if step % 100 == 0:
            print(f"Step {step}/{config['steps']}: loss={loss.item():.6f}, "
                  f"nw={nash_welfare.item():.6f} (n={n}, m={m}), "
                  f"residual_scale={model.residual_scale.item():.4f}, "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")

        if use_wandb:
            wandb.log(metrics, step=step)

        # Checkpoint
        if step % config['checkpoint_every'] == 0 and step > 0:
            save_checkpoint(
                checkpoint_dir, step, model, optimizer, scheduler,
                config, best_metric, config['keep_checkpoints']
            )

        # Early stopping
        if steps_without_improvement >= config['patience']:
            print(f"Early stopping at step {step} (no improvement for {config['patience']} validation checks)")
            if use_wandb:
                wandb.log({
                    "early_stop_step": step,
                    "best_val_nash_welfare_avg": best_metric,
                    "final_residual_scale": model.residual_scale.item(),
                }, step=step)
            break

    # Final checkpoint
    print("Training complete. Saving final checkpoint...")
    save_checkpoint(
        checkpoint_dir, step, model, optimizer, scheduler,
        config, best_metric, config['keep_checkpoints']
    )

    print(f"\nBest avg validation Nash welfare: {best_metric:.6f}")
    print(f"Final residual scale: {model.residual_scale.item():.4f}")
    print(f"Best checkpoint (for resuming): {checkpoint_dir / 'best_checkpoint.pt'}")
    print(f"Best model weights (for inference): {checkpoint_dir / 'best_model.pt'}")

    if use_wandb:
        wandb.finish()


def main():
    args = parse_args()

    # Build config
    config = vars(args)

    # Load from file if specified - file config overrides argparse defaults
    if args.config:
        file_config = load_config(args.config)
        for key, value in file_config.items():
            key_normalized = key.replace('-', '_')
            config[key_normalized] = value

    train(config)


if __name__ == "__main__":
    main()
