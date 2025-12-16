#!/usr/bin/env python
"""
Production training script for FATransformer.

Use after identifying best hyperparameters from bayesian_sweep.py.
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

from fatransformer.fatransformer import FATransformer  # noqa: E402
from fatransformer.helpers import get_nash_welfare  # noqa: E402

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Production training for FATransformer"
    )

    # Config options
    parser.add_argument("--config", type=str, help="Path to YAML/JSON config file")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")

    # Model hyperparameters (can override config)
    parser.add_argument("--n", type=int, default=10, help="Number of agents")
    parser.add_argument("--m", type=int, default=20, help="Number of items")
    parser.add_argument("--d-model", type=int, default=768, help="Model dimension")
    parser.add_argument("--num-heads", type=int, default=12, help="Number of attention heads")
    parser.add_argument("--num-output-layers", type=int, default=4, help="Number of output layers")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate")

    # Training hyperparameters
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="Weight decay")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size")
    parser.add_argument("--steps", type=int, default=100000, help="Total training steps")

    # Temperature
    parser.add_argument("--initial-temperature", type=float, default=1.0)
    parser.add_argument("--final-temperature", type=float, default=0.01)

    # Training settings
    parser.add_argument("--grad-clip-norm", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    # Checkpointing
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                       help="Directory to save checkpoints")
    parser.add_argument("--checkpoint-every", type=int, default=5000,
                       help="Save checkpoint every N steps")
    parser.add_argument("--keep-checkpoints", type=int, default=3,
                       help="Number of recent checkpoints to keep")

    # Validation
    parser.add_argument("--val-every", type=int, default=1000,
                       help="Validate every N steps")
    parser.add_argument("--val-size", type=int, default=1000,
                       help="Number of validation examples")

    # Early stopping
    parser.add_argument("--patience", type=int, default=50,
                       help="Early stopping patience")
    parser.add_argument("--min-delta", type=float, default=1e-5,
                       help="Minimum improvement for early stopping")

    # Wandb
    parser.add_argument("--wandb-project", type=str, default="fa-transformer-production")
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


def validate(model, n, m, val_size, device):
    """Run validation on fixed validation set."""
    model.eval()
    with torch.no_grad():
        val_valuations = torch.rand(val_size, n, m, device=device)
        val_allocations = model(val_valuations)
        val_nash_welfare = get_nash_welfare(
            val_valuations, val_allocations, reduction="mean"
        )
    model.train()
    return val_nash_welfare.item()


def train(config: Dict[str, Any]):
    """Main training loop."""

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
    model = FATransformer(
        n=config['n'],
        m=config['m'],
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        num_output_layers=config['num_output_layers'],
        dropout=config['dropout'],
        initial_temperature=config['initial_temperature'],
        final_temperature=config['final_temperature'],
    ).to(device)

    print(f"Model created: {sum(p.numel() for p in model.parameters())} parameters")

    # Optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['steps']
    )

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

        # Generate batch
        valuations = torch.rand(
            config['batch_size'], config['n'], config['m'], device=device
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
            "lr": scheduler.get_last_lr()[0],
        }

        # Validation
        if step % config['val_every'] == 0:
            val_nw = validate(
                model, config['n'], config['m'],
                config['val_size'], device
            )
            metrics['val_nash_welfare'] = val_nw

            # Check for improvement
            if val_nw > best_metric + config['min_delta']:
                best_metric = val_nw
                steps_without_improvement = 0
                # Save best model
                best_path = checkpoint_dir / "best_model.pt"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), best_path)
                print(f"Step {step}: New best validation Nash welfare: {best_metric:.6f}")
                if use_wandb:
                    wandb.save(str(best_path))
            else:
                steps_without_improvement += 1

        # Log to console every 100 steps
        if step % 100 == 0:
            print(f"Step {step}/{config['steps']}: loss={loss.item():.6f}, "
                  f"nw={nash_welfare.item():.6f}, lr={scheduler.get_last_lr()[0]:.2e}")

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
                    "best_val_nash_welfare": best_metric,
                }, step=step)
            break

    # Final checkpoint
    print("Training complete. Saving final checkpoint...")
    save_checkpoint(
        checkpoint_dir, step, model, optimizer, scheduler,
        config, best_metric, config['keep_checkpoints']
    )

    print(f"Best validation Nash welfare: {best_metric:.6f}")
    print(f"Best model saved to: {checkpoint_dir / 'best_model.pt'}")

    if use_wandb:
        wandb.finish()


def main():
    args = parse_args()

    # Build config
    config = vars(args)

    # Load from file if specified
    if args.config:
        file_config = load_config(args.config)
        # CLI args override file config
        for key, value in file_config.items():
            key_normalized = key.replace('-', '_')
            if key_normalized not in config or config[key_normalized] is None:
                config[key_normalized] = value

    train(config)


if __name__ == "__main__":
    main()
