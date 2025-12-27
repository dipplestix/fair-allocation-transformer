"""
Bayesian hyperparameter sweep for FFTransformerResidual trained on n=30, m=60.

To run 50 sweeps:
    python training/bayesian_sweep_30_60.py --run-agent --count 50
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

try:
    import google.protobuf
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

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fftransformer.exchangeable_layer import ExchangeableLayer
from fftransformer.attention_blocks import FFSelfAttentionBlock, FFCrossAttentionBlock
from fftransformer.helpers import get_nash_welfare

DEFAULT_PROJECT = os.environ.get("WANDB_PROJECT", "fa-transformer-30-60-sweep")
DEFAULT_ENTITY = os.environ.get("WANDB_ENTITY")

POOL_CONFIGS = {
    "row_only": {'row': ['mean', 'max', 'min'], 'column': [], 'global': []},
    "row_col": {'row': ['mean', 'max', 'min'], 'column': ['mean', 'max', 'min'], 'global': []},
    "row_global": {'row': ['mean', 'max', 'min'], 'column': [], 'global': ['mean', 'max', 'min']},
    "all": {'row': ['mean', 'max', 'min'], 'column': ['mean', 'max', 'min'], 'global': ['mean', 'max', 'min']},
    "row_col_mean": {'row': 'mean', 'column': 'mean', 'global': []},
}


class FFTransformerResidualSweep(nn.Module):
    def __init__(
        self,
        n: int,
        m: int,
        d_model: int,
        num_heads: int,
        num_output_layers: int = 1,
        dropout: float = 0.0,
        initial_temperature: float = 1.0,
        final_temperature: float = 0.01,
        pool_config_name: str = "row_col",
        residual_scale_init: float = 0.1,
        num_encoder_layers: int = 1,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_output_layers = num_output_layers
        self.dropout = dropout
        self.n = n
        self.m = m
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.temperature = initial_temperature

        pool_config = POOL_CONFIGS.get(pool_config_name, POOL_CONFIGS["row_col"])
        self.agent_proj = ExchangeableLayer(1, d_model, pool_config=pool_config)
        self.item_proj = ExchangeableLayer(1, d_model, pool_config=pool_config)

        self.residual_scale = nn.Parameter(torch.tensor(residual_scale_init))

        self.agent_transformer = nn.ModuleList(
            [FFSelfAttentionBlock(d_model, num_heads, dropout) for _ in range(num_encoder_layers)]
        )
        self.item_transformer = nn.ModuleList(
            [FFSelfAttentionBlock(d_model, num_heads, dropout) for _ in range(num_encoder_layers)]
        )
        self.item_agent_transformer = FFCrossAttentionBlock(d_model, num_heads, dropout)
        self.output_transformer = nn.ModuleList(
            [FFSelfAttentionBlock(d_model, num_heads, dropout) for _ in range(num_output_layers)]
        )
        self.o_norm = nn.RMSNorm(d_model)

    def update_temperature(self, temperature: float):
        self.temperature = temperature

    def eval(self):
        super().eval()
        self.temperature = self.final_temperature
        return self

    def forward(self, x: torch.Tensor):
        B, n, m = x.shape

        x_agent = self.agent_proj(x)
        x_agent = x_agent.mean(dim=3).permute(0, 2, 1)

        x_item = self.item_proj(x.permute(0, 2, 1))
        x_item = x_item.mean(dim=3).permute(0, 2, 1)

        for layer in self.agent_transformer:
            x_agent = layer(x_agent)
        for layer in self.item_transformer:
            x_item = layer(x_item)
        x_output = self.item_agent_transformer(x_item, x_agent)
        for layer in self.output_transformer:
            x_output = layer(x_output)
        x_output = self.o_norm(x_output)

        bilinear_out = torch.matmul(x_output, x_agent.transpose(1, 2))
        residual = x.permute(0, 2, 1)
        x_output = bilinear_out + self.residual_scale * residual

        x_output = F.softmax(x_output / self.temperature, dim=-1)
        return x_output


def make_sweep_config() -> Dict[str, Any]:
    """Return the Bayesian sweep configuration for 30x60 model."""
    return {
        "method": "bayes",
        "metric": {"goal": "maximize", "name": "best_nash_welfare"},
        "parameters": {
            # Model architecture - optimized for larger problem size
            "d_model": {"values": [128, 256, 384, 512]},
            "num_heads": {"values": [4, 8, 16]},
            "num_output_layers": {"values": [1, 2, 3]},
            "num_encoder_layers": {"values": [1, 2, 3]},
            "dropout": {"min": 0.0, "max": 0.15},

            # Residual-specific
            "pool_config_name": {"values": ["row_only", "row_col", "row_global", "all"]},
            "residual_scale_init": {
                "distribution": "log_uniform_values",
                "min": 0.01,
                "max": 1.0,
            },

            # Training - lower LR range for stability with larger models
            "lr": {
                "distribution": "log_uniform_values",
                "min": 5e-5,
                "max": 1e-3,
            },
            "weight_decay": {
                "distribution": "log_uniform_values",
                "min": 1e-5,
                "max": 1e-2,
            },
            # Smaller batch sizes due to larger problem size
            "batch_size": {"values": [64, 128, 256]},
            "steps": {"value": 15000},  # More steps for larger problem

            # Problem size - FIXED at 30x60
            "n": {"value": 30},
            "m": {"value": 60},

            # Temperature
            "initial_temperature": {"min": 0.5, "max": 2.0},
            "final_temperature": {
                "distribution": "log_uniform_values",
                "min": 0.001,
                "max": 0.1,
            },

            # Training settings
            "grad_clip_norm": {"value": 1.0},
            "patience": {"value": 40},  # More patience
            "min_delta": {"value": 1e-5},
            "val_every": {"value": 300},
            "val_size": {"value": 500},  # Smaller val size for speed
        },
    }


def train(config: Optional[Dict[str, Any]] = None) -> None:
    with wandb.init(config=config, settings=wandb.Settings(start_method="thread")) as run:
        cfg = run.config

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        seed = cfg.get("seed", 0)
        torch.manual_seed(seed)

        model = FFTransformerResidualSweep(
            n=cfg.n,
            m=cfg.m,
            d_model=cfg.d_model,
            num_heads=cfg.num_heads,
            num_output_layers=cfg.num_output_layers,
            num_encoder_layers=cfg.num_encoder_layers,
            dropout=cfg.dropout,
            initial_temperature=cfg.initial_temperature,
            final_temperature=cfg.final_temperature,
            pool_config_name=cfg.pool_config_name,
            residual_scale_init=cfg.residual_scale_init,
        ).to(device)

        num_params = sum(p.numel() for p in model.parameters())
        wandb.log({"num_parameters": num_params})
        print(f"Model parameters: {num_params:,}")

        optimizer = optim.AdamW(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.steps
        )

        # Fixed validation set
        torch.manual_seed(0)
        val_valuations = torch.rand(cfg.val_size, cfg.n, cfg.m, device=device)
        torch.manual_seed(seed)

        patience = cfg.patience
        min_delta = cfg.min_delta
        best_metric = float("-inf")
        steps_without_improvement = 0

        for step in range(cfg.steps):
            model.train()

            progress = step / cfg.steps
            current_temp = cfg.final_temperature + \
                (cfg.initial_temperature - cfg.final_temperature) * \
                (1 + torch.cos(torch.tensor(torch.pi * progress))) / 2
            model.update_temperature(current_temp.item())

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

            log_dict = {
                "step": step,
                "loss": loss.item(),
                "train_nash_welfare": nash_welfare.item(),
                "lr": current_lr,
                "temperature": current_temp.item(),
                "residual_scale": model.residual_scale.item(),
            }

            if step % cfg.val_every == 0 or step == cfg.steps - 1:
                model.eval()
                with torch.no_grad():
                    val_allocation = model(val_valuations)
                    val_nash_welfare = get_nash_welfare(
                        val_valuations, val_allocation, reduction="mean"
                    ).item()
                model.train()

                log_dict["val_nash_welfare"] = val_nash_welfare

                if val_nash_welfare > best_metric + min_delta:
                    best_metric = val_nash_welfare
                    steps_without_improvement = 0
                    log_dict["best_nash_welfare"] = best_metric
                else:
                    steps_without_improvement += 1

                print(f"Step {step}: val_nw={val_nash_welfare:.4f}, best={best_metric:.4f}")

            wandb.log(log_dict, step=step)

            if steps_without_improvement >= patience:
                wandb.log({
                    "early_stop_step": step,
                    "best_nash_welfare": best_metric,
                    "final_residual_scale": model.residual_scale.item(),
                }, step=step)
                print(f"Early stopping at step {step}")
                break

        wandb.log({
            "best_nash_welfare": best_metric,
            "final_residual_scale": model.residual_scale.item(),
        })


def create_sweep(project: str, entity: Optional[str]) -> str:
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
        description="Bayesian W&B sweep for FFTransformer on 30x60 problems."
    )
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--entity", default=DEFAULT_ENTITY)
    parser.add_argument("--create", action="store_true")
    parser.add_argument("--run-agent", action="store_true")
    parser.add_argument("--sweep-id", help="Existing sweep id")
    parser.add_argument("--count", type=int, help="Number of runs")
    parser.add_argument("--enable-service", action="store_true")

    args = parser.parse_args(argv)

    sweep_id = args.sweep_id
    if (args.create or args.run_agent) and not sweep_id:
        sweep_id = create_sweep(args.project, args.entity)

    if args.run_agent:
        if args.enable_service:
            os.environ.pop("WANDB_DISABLE_SERVICE", None)
        if not sweep_id:
            raise ValueError("Cannot launch agent without a sweep id.")
        wandb.agent(
            sweep_id,
            function=train,
            project=args.project,
            entity=args.entity,
            count=args.count,
        )
    elif args.create and not args.run_agent:
        return
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
