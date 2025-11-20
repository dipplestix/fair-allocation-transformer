"""
Bayesian hyperparameter sweep for FATransformer using Weights & Biases.

To create the sweep and launch an agent locally:

    python training/bayesian_sweep.py --create
    wandb agent <entity/project>/<sweep_id>

Alternatively, run `python training/bayesian_sweep.py --run-agent`
to both create the sweep (if needed) and start an agent process.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.optim as optim

# Prefer the fast protobuf runtime when available, but fall back to pure Python
# if a newer protobuf (>3.x) is installed that is incompatible with this wandb build.
try:
    import google.protobuf  # type: ignore
except ModuleNotFoundError:
    pass
else:
    version_tokens = []
    for token in google.protobuf.__version__.split("."):
        if token.isdigit():
            version_tokens.append(int(token))
        else:
            break
    if tuple(version_tokens) >= (4, 0, 0) and os.environ.get(
        "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"
    ) is None:
        os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

os.environ.setdefault("WANDB_DISABLE_SERVICE", "true")

import wandb


REPO_ROOT = Path(__file__).resolve().parents[1]

# Ensure local imports work without installing the package.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fatransformer.fatransformer import FATransformer  # noqa: E402
from fatransformer.helpers import get_nash_welfare  # noqa: E402


DEFAULT_PROJECT = os.environ.get("WANDB_PROJECT", "fa-transformer-sweep")
DEFAULT_ENTITY = os.environ.get("WANDB_ENTITY")


def make_sweep_config() -> Dict[str, Any]:
    """Return the Bayesian sweep configuration."""
    return {
        "method": "bayes",
        "metric": {"goal": "maximize", "name": "nash_welfare"},
        "parameters": {
            "d_model": {"values": [768]},
            "num_heads": {"values": [4, 8, 12, 16]},
            "num_output_layers": {"values": [1, 2, 3, 4, 5]},
            "dropout": {"min": 0.0, "max": 0.01},
            "lr": {
                "distribution": "log_uniform_values",
                "min": 3e-5,
                "max": 3e-4,
            },
            "weight_decay": {
                "distribution": "log_uniform_values",
                "min": 1e-5,
                "max": 5e-2,
            },
            "batch_size": {"values": [512, 1024, 2048]},
            "steps": {"value": 20000},
            "n": {"value": 10},
            "m": {"value": 20},
            "initial_temperature": {
                "min": 0.5,
                "max": 2.0,
            },
            "final_temperature": {
                "distribution": "log_uniform_values",
                "min": 0.001,
                "max": 0.01,
            },
            "grad_clip_norm": {"value": 1.0},
            "patience": {"value": 20},
            "min_delta": {"value": 1e-5},
        },
    }


def train(config: Optional[Dict[str, Any]] = None) -> None:
    """Train a single model instance under the sweep configuration."""
    with wandb.init(config=config, settings=wandb.Settings(start_method="thread")) as run:
        cfg = run.config

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(cfg.get("seed", 0))

        model = FATransformer(
            n=cfg.n,
            m=cfg.m,
            d_model=cfg.d_model,
            num_heads=cfg.num_heads,
            num_output_layers=cfg.num_output_layers,
            dropout=cfg.dropout,
            initial_temperature=cfg.initial_temperature,
            final_temperature=cfg.final_temperature,
        ).to(device)

        optimizer = optim.AdamW(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.steps
        )

        patience = cfg.patience
        min_delta = cfg.min_delta
        best_metric = float("-inf")
        steps_without_improvement = 0

        for step in range(cfg.steps):
            model.train()
            valuations = torch.rand(cfg.batch_size, cfg.n, cfg.m, device=device)
            allocation = model(valuations)
            nash_welfare = get_nash_welfare(
                valuations, allocation, reduction="mean"
            )
            loss = -nash_welfare

            optimizer.zero_grad()
            loss.backward()
            if cfg.grad_clip_norm:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=cfg.grad_clip_norm
                )
            optimizer.step()
            scheduler.step()

            current_lr = scheduler.get_last_lr()[0]
            wandb.log(
                {
                    "step": step,
                    "loss": loss.item(),
                    "nash_welfare": nash_welfare.item(),
                    "lr": current_lr,
                },
                step=step,
            )

            metric_value = nash_welfare.item()
            if metric_value > best_metric + min_delta:
                best_metric = metric_value
                steps_without_improvement = 0
            else:
                steps_without_improvement += 1

            if steps_without_improvement >= patience:
                wandb.log(
                    {
                        "early_stop_step": step,
                        "best_nash_welfare": best_metric,
                    },
                    step=step,
                )
                break


def create_sweep(project: str, entity: Optional[str]) -> str:
    """Register the sweep on W&B and return the sweep ID."""
    sweep_config = make_sweep_config()
    sweep_id = wandb.sweep(
        sweep_config,
        project=project,
        entity=entity,
    )
    print(f"Created sweep {sweep_id} in project {project}.")
    return sweep_id


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Bayesian W&B sweep for the Fair Allocation Transformer."
    )
    parser.add_argument(
        "--project",
        default=DEFAULT_PROJECT,
        help="W&B project name (defaults to WANDB_PROJECT env or 'fa-transformer-sweep').",
    )
    parser.add_argument(
        "--entity",
        default=DEFAULT_ENTITY,
        help="W&B entity (defaults to WANDB_ENTITY env if set).",
    )
    parser.add_argument(
        "--create",
        action="store_true",
        help="Create the sweep and print the sweep ID.",
    )
    parser.add_argument(
        "--run-agent",
        action="store_true",
        help="Launch a local sweep agent. Implies --create if no sweep id provided.",
    )
    parser.add_argument(
        "--sweep-id",
        help="Existing sweep id to run the agent against (skips creation).",
    )
    parser.add_argument(
        "--count",
        type=int,
        help="Optional limit on number of runs for the local agent.",
    )
    parser.add_argument(
        "--enable-service",
        action="store_true",
        help="Use the wandb service backend (enabled by default in newer wandb).",
    )

    args = parser.parse_args(argv)

    sweep_id = args.sweep_id
    if (args.create or args.run_agent) and not sweep_id:
        sweep_id = create_sweep(args.project, args.entity)

    if args.run_agent:
        if args.enable_service:
            os.environ.pop("WANDB_DISABLE_SERVICE", None)
        if not sweep_id:
            raise ValueError(
                "Cannot launch agent without a sweep id. Use --create or provide --sweep-id."
            )
        wandb.agent(
            sweep_id,
            function=train,
            project=args.project,
            entity=args.entity,
            count=args.count,
        )
    elif args.create and not args.run_agent:
        # Creation already printed the sweep id.
        return
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
