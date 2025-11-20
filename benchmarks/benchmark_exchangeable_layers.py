"""Benchmark ExchangeableLayer projection implementations.

This script compares the projection implemented with ``nn.Linear`` in
:class:`fatransformer.exchangeable_layer.ExchangeableLayer` against an
alternative 1x1 convolution based projection.  Both versions share the
same pooling front-end so the benchmark isolates the cost of the
projection step.

For each method, we also print the total number of parameters in the layer.

Usage example
-------------
Run the default benchmark on CPU::

    python benchmarks/benchmark_exchangeable_layers.py --device cpu

Run on CUDA with custom tensor sizes::

    python benchmarks/benchmark_exchangeable_layers.py \
        --device cuda --spatial-sizes 8x8,16x16,32x32

"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.utils.benchmark as benchmark

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fatransformer.exchangeable_layer import ExchangeableLayer, PoolLayer


@dataclass(frozen=True)
class BenchmarkConfig:
    batch_size: int
    spatial_sizes: Sequence[Tuple[int, int]]
    in_channels: int
    out_channels: int
    device: torch.device
    dtype: torch.dtype
    min_run_time: float


def parse_spatial_sizes(argument: str) -> List[Tuple[int, int]]:
    sizes: List[Tuple[int, int]] = []
    for item in argument.split(","):
        item = item.strip()
        if not item:
            continue
        if "x" not in item:
            raise argparse.ArgumentTypeError(
                f"Invalid spatial size '{item}'. Expected the form HxW (e.g. '16x32')."
            )
        height_str, width_str = item.lower().split("x", maxsplit=1)
        try:
            height = int(height_str)
            width = int(width_str)
        except ValueError as exc:  # pragma: no cover - defensive parsing
            raise argparse.ArgumentTypeError(
                f"Invalid integers in spatial size '{item}'."
            ) from exc
        if height <= 0 or width <= 0:
            raise argparse.ArgumentTypeError(
                f"Spatial sizes must be positive. Received {height}x{width}."
            )
        sizes.append((height, width))
    if not sizes:
        raise argparse.ArgumentTypeError("At least one spatial size must be provided.")
    return sizes


class ExchangeableLayerConv1x1(nn.Module):
    """ExchangeableLayer variant that replaces the linear projection with 1x1 conv."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pool_config: dict | None = None,
        activation: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if pool_config is None:
            pool_config = {"row": "mean", "column": "mean", "global": "mean"}
        if activation is None:
            activation = nn.GELU()

        total_aggs = 0
        for agg_funcs in pool_config.values():
            if isinstance(agg_funcs, str):
                total_aggs += 1
            else:
                total_aggs += len(agg_funcs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.proj_channels = in_channels + total_aggs * in_channels

        self.pool_layer = PoolLayer(pool_config)
        self.proj = nn.Conv2d(self.proj_channels, out_channels, kernel_size=1, bias=True)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - benchmark utility
        # Input is already in BCHW format
        x = self.pool_layer(x)
        x = self.activation(self.proj(x))
        return x

def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for the inputs.")
    parser.add_argument(
        "--spatial-sizes",
        type=parse_spatial_sizes,
        default=parse_spatial_sizes("8x8,16x16,32x32"),
        help="Comma separated list of HxW spatial sizes to benchmark.",
    )
    parser.add_argument("--in-channels", type=int, default=8, help="Number of input channels.")
    parser.add_argument("--out-channels", type=int, default=16, help="Number of output channels.")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run the benchmark on (cpu or cuda).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="Data type for the tensors.",
    )
    parser.add_argument(
        "--min-run-time",
        type=float,
        default=1.0,
        help="Minimum number of seconds per benchmark measurement.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> BenchmarkConfig:
    parser = make_parser()
    args = parser.parse_args(argv)

    try:
        device = torch.device(args.device)
    except RuntimeError as exc:  # pragma: no cover - depends on runtime
        parser.error(str(exc))

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    return BenchmarkConfig(
        batch_size=args.batch_size,
        spatial_sizes=args.spatial_sizes,
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        device=device,
        dtype=dtype,
        min_run_time=args.min_run_time,
    )


def make_inputs(
    batch_size: int,
    height: int,
    width: int,
    channels: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.randn(batch_size, channels, height, width, device=device, dtype=dtype)


def _forward_no_grad(layer: nn.Module, x: torch.Tensor) -> None:
    with torch.no_grad():
        layer(x)


def benchmark_layers(config: BenchmarkConfig) -> Iterable[benchmark.Measurement]:
    torch.manual_seed(0)
    results: List[benchmark.Measurement] = []

    linear_layer = ExchangeableLayer(
        config.in_channels,
        config.out_channels,
    ).to(config.device, dtype=config.dtype)
    conv_layer = ExchangeableLayerConv1x1(
        config.in_channels,
        config.out_channels,
    ).to(config.device, dtype=config.dtype)

    linear_layer.eval()
    conv_layer.eval()

    # Print parameter counts for each layer
    print(f"Linear projection total parameters: {count_parameters(linear_layer):,}")
    print(f"1x1 convolution total parameters: {count_parameters(conv_layer):,}")
    print("")

    for height, width in config.spatial_sizes:
        x = make_inputs(config.batch_size, height, width, config.in_channels, config.device, config.dtype)

        # Warm up both layers once outside of the timed region.
        _forward_no_grad(linear_layer, x)
        _forward_no_grad(conv_layer, x)

        label = f"B{config.batch_size}-C{config.in_channels}->C{config.out_channels}"
        sub_label = f"{height}x{width}"

        results.append(
            benchmark.Timer(
                stmt="_forward_no_grad(layer, x)",
                globals={"_forward_no_grad": _forward_no_grad, "layer": linear_layer, "x": x},
                label=label,
                sub_label=sub_label,
                description="Linear projection",
            ).blocked_autorange(min_run_time=config.min_run_time)
        )

        results.append(
            benchmark.Timer(
                stmt="_forward_no_grad(layer, x)",
                globals={"_forward_no_grad": _forward_no_grad, "layer": conv_layer, "x": x},
                label=label,
                sub_label=sub_label,
                description="1x1 convolution",
            ).blocked_autorange(min_run_time=config.min_run_time)
        )

    return results


def main(argv: Sequence[str] | None = None) -> None:  # pragma: no cover - CLI entrypoint
    config = parse_args(argv)
    measurements = list(benchmark_layers(config))
    compare = benchmark.Compare(measurements)
    compare.print()


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
