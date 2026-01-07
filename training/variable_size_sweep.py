"""
Variable size training sweep for FFTransformerResidual using Weights & Biases.

For each training step, n is drawn from Uniform(10, 50) and m is drawn from Uniform(n, 50).
This trains a single size-agnostic model that can handle varying problem sizes.

To create the sweep and launch an agent locally:

    python training/variable_size_sweep.py --create
    wandb agent <entity/project>/<sweep_id>

Alternatively, run `python training/variable_size_sweep.py --run-agent`
to both create the sweep (if needed) and start an agent process.
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

from fftransformer.exchangeable_layer import ExchangeableLayer  # noqa: E402
from fftransformer.attention_blocks import FFSelfAttentionBlock, FFCrossAttentionBlock  # noqa: E402
from fftransformer.helpers import get_nash_welfare  # noqa: E402


DEFAULT_PROJECT = os.environ.get("WANDB_PROJECT", "fa-transformer-variable-size-sweep")
DEFAULT_ENTITY = os.environ.get("WANDB_ENTITY")

# Size sampling range
N_MIN = 10
N_MAX = 50
M_MAX = 50

# Pool config options to sweep over
POOL_CONFIGS = {
    "row_only": {'row': ['mean', 'max', 'min'], 'column': [], 'global': []},
    "row_col": {'row': ['mean', 'max', 'min'], 'column': ['mean', 'max', 'min'], 'global': []},
    "row_global": {'row': ['mean', 'max', 'min'], 'column': [], 'global': ['mean', 'max', 'min']},
    "all": {'row': ['mean', 'max', 'min'], 'column': ['mean', 'max', 'min'], 'global': ['mean', 'max', 'min']},
    "row_col_mean": {'row': 'mean', 'column': 'mean', 'global': []},
}


class FFTransformerResidualVariableSize(nn.Module):
    """
    FFTransformer with exchangeable layers and residual connection from input.
    Configurable for hyperparameter sweeps. Size-agnostic (no n, m parameters).
    Designed for variable size training where n and m change each batch.
    """
    def __init__(
        self,
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
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.temperature = initial_temperature

        pool_config = POOL_CONFIGS.get(pool_config_name, POOL_CONFIGS["row_col"])
        self.agent_proj = ExchangeableLayer(1, d_model, pool_config=pool_config)
        self.item_proj = ExchangeableLayer(1, d_model, pool_config=pool_config)

        # Learnable residual scale
        self.residual_scale = nn.Parameter(torch.tensor(residual_scale_init))

        # Encoder layers (can be deeper)
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
        """Update the temperature parameter for softmax scaling"""
        self.temperature = temperature

    def eval(self):
        """Set model to evaluation mode and use final temperature"""
        super().eval()
        self.temperature = self.final_temperature
        return self

    def forward(self, x: torch.Tensor):
        B, n, m = x.shape

        # Exchangeable projections
        x_agent = self.agent_proj(x)  # (B, d_model, n, m)
        x_agent = x_agent.mean(dim=3).permute(0, 2, 1)  # (B, n, d_model)

        x_item = self.item_proj(x.permute(0, 2, 1))  # (B, d_model, m, n)
        x_item = x_item.mean(dim=3).permute(0, 2, 1)  # (B, m, d_model)

        # Transformer processing
        for layer in self.agent_transformer:
            x_agent = layer(x_agent)
        for layer in self.item_transformer:
            x_item = layer(x_item)
        x_output = self.item_agent_transformer(x_item, x_agent)
        for layer in self.output_transformer:
            x_output = layer(x_output)
        x_output = self.o_norm(x_output)

        # Bilinear output + residual from input
        bilinear_out = torch.matmul(x_output, x_agent.transpose(1, 2))  # (B, m, n)
        residual = x.permute(0, 2, 1)  # (B, m, n)
        x_output = bilinear_out + self.residual_scale * residual

        x_output = F.softmax(x_output / self.temperature, dim=-1)
        return x_output


def sample_problem_size(device: torch.device):
    """Sample n from Uniform(10, 50) and m from Uniform(n, 50)."""
    n = torch.randint(N_MIN, N_MAX + 1, (1,)).item()
    m = torch.randint(n, M_MAX + 1, (1,)).item()
    return n, m


def make_sweep_config() -> Dict[str, Any]:
    """Return the Bayesian sweep configuration for variable size training."""
    return {
        "method": "bayes",
        "metric": {"goal": "maximize", "name": "best_nash_welfare"},
        "parameters": {
            # Model architecture
            # Note: num_heads must divide d_model, so we use [4, 8, 16] which divide all d_model values
            "d_model": {"values": [128, 256, 512]},
            "num_heads": {"values": [4, 8, 16]},
            "num_output_layers": {"values": [1, 2, 3, 4]},
            "num_encoder_layers": {"values": [1, 2, 3]},
            "dropout": {"min": 0.0, "max": 0.1},

            # Residual-specific
            "pool_config_name": {"values": ["row_only", "row_col", "row_global", "all", "row_col_mean"]},
            "residual_scale_init": {
                "distribution": "log_uniform_values",
                "min": 0.01,
                "max": 1.0,
            },

            # Training
            "lr": {
                "distribution": "log_uniform_values",
                "min": 1e-4,
                "max": 3e-3,
            },
            "weight_decay": {
                "distribution": "log_uniform_values",
                "min": 1e-5,
                "max": 5e-2,
            },
            "batch_size": {"values": [64, 128, 256]},
            "steps": {"value": 20000},

            # Temperature
            "initial_temperature": {
                "min": 0.5,
                "max": 2.0,
            },
            "final_temperature": {
                "distribution": "log_uniform_values",
                "min": 0.001,
                "max": 0.1,
            },

            # Training settings
            "grad_clip_norm": {"value": 1.0},
            "patience": {"value": 50},
            "min_delta": {"value": 1e-5},
            "val_every": {"value": 200},
            "val_size": {"value": 500},
        },
    }


def train(config: Optional[Dict[str, Any]] = None) -> None:
    """Train a single model instance under the sweep configuration."""
    with wandb.init(config=config, settings=wandb.Settings(start_method="thread")) as run:
        cfg = run.config

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        seed = cfg.get("seed", 0)
        torch.manual_seed(seed)

        model = FFTransformerResidualVariableSize(
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

        optimizer = optim.AdamW(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.steps
        )

        # Generate multiple fixed validation sets at different sizes
        torch.manual_seed(0)
        val_sets = []
        # Validation sizes: small (10,15), medium (25,35), large (40,50)
        val_sizes = [(10, 15), (25, 35), (40, 50)]
        for n, m in val_sizes:
            val_valuations = torch.rand(cfg.val_size, n, m, device=device)
            val_sets.append((n, m, val_valuations))
        torch.manual_seed(seed)

        patience = cfg.patience
        min_delta = cfg.min_delta
        best_metric = float("-inf")
        steps_without_improvement = 0

        for step in range(cfg.steps):
            model.train()

            # Temperature annealing
            progress = step / cfg.steps
            current_temp = cfg.final_temperature + \
                (cfg.initial_temperature - cfg.final_temperature) * \
                (1 + torch.cos(torch.tensor(torch.pi * progress))) / 2
            model.update_temperature(current_temp.item())

            # Sample variable problem size for this batch
            n, m = sample_problem_size(device)

            valuations = torch.rand(cfg.batch_size, n, m, device=device)
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

            # Log training metrics
            log_dict = {
                "step": step,
                "loss": loss.item(),
                "train_nash_welfare": nash_welfare.item(),
                "train_n": n,
                "train_m": m,
                "lr": current_lr,
                "temperature": current_temp.item(),
                "residual_scale": model.residual_scale.item(),
            }

            # Validation on multiple sizes
            if step % cfg.val_every == 0 or step == cfg.steps - 1:
                model.eval()
                val_nash_welfares = []
                with torch.no_grad():
                    for val_n, val_m, val_valuations in val_sets:
                        val_allocation = model(val_valuations)
                        val_nw = get_nash_welfare(
                            val_valuations, val_allocation, reduction="mean"
                        ).item()
                        val_nash_welfares.append(val_nw)
                        log_dict[f"val_nash_welfare_{val_n}x{val_m}"] = val_nw

                # Average validation Nash welfare across sizes
                avg_val_nash_welfare = sum(val_nash_welfares) / len(val_nash_welfares)
                log_dict["val_nash_welfare_avg"] = avg_val_nash_welfare
                model.train()

                # Early stopping based on average validation
                if avg_val_nash_welfare > best_metric + min_delta:
                    best_metric = avg_val_nash_welfare
                    steps_without_improvement = 0
                    log_dict["best_nash_welfare"] = best_metric
                else:
                    steps_without_improvement += 1

            wandb.log(log_dict, step=step)

            if steps_without_improvement >= patience:
                wandb.log(
                    {
                        "early_stop_step": step,
                        "best_nash_welfare": best_metric,
                        "final_residual_scale": model.residual_scale.item(),
                    },
                    step=step,
                )
                break

        # Final logging
        wandb.log({
            "best_nash_welfare": best_metric,
            "final_residual_scale": model.residual_scale.item(),
        })


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
        description="Variable size W&B sweep for the Fair Allocation Transformer (Residual variant)."
    )
    parser.add_argument(
        "--project",
        default=DEFAULT_PROJECT,
        help="W&B project name (defaults to WANDB_PROJECT env or 'fa-transformer-variable-size-sweep').",
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
