# FairFormer: A Transformer for Discrete Fair Division

![FairFormer mascot](ff.png)

This repository contains the research code accompanying the working paper
"FairFormer: A transformer architecture for discrete fair division". The model
implements the two-tower, symmetry-aware transformer described in our draft
ICML submission and provides utilities for computing Nash welfare during
training.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Fair Division Problem](#fair-division-problem)
3. [Architecture](#architecture)
4. [Installation](#installation)
5. [Model Variants](#model-variants)
6. [Training](#training)
7. [Evaluation Pipeline](#evaluation-pipeline)
8. [API Reference](#api-reference)
9. [Fairness Metrics](#fairness-metrics)
10. [Project Structure](#project-structure)
11. [Citation](#citation)
12. [License](#license)

---

## Quick Start

Get started in 5 minutes:

```bash
# Clone and install
git clone https://github.com/<username>/fair-allocation-transformer.git
cd fair-allocation-transformer

# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install project (creates venv automatically)
uv sync

# Run a simple example
uv run python -c "
import torch
from fftransformer import FFTransformerResidual, get_nash_welfare

# Create model (size-agnostic, works with any n agents and m items)
model = FFTransformerResidual(d_model=256, num_heads=8, num_output_layers=2)

# Generate random valuations and compute allocation
valuations = torch.rand(4, 10, 20)  # batch of 4
allocations = model(valuations)
nw = get_nash_welfare(valuations, allocations)
print(f'Nash Welfare: {nw.item():.4f}')
"
```

---

## Fair Division Problem

**Problem**: Allocate `m` indivisible items among `n` agents with heterogeneous valuations.

**Input**: Valuation matrix `V ∈ R^(n×m)` where `V[i,j]` = agent `i`'s value for item `j`

**Output**: Allocation matrix `A ∈ [0,1]^(m×n)` where `A[j,i]` = probability item `j` goes to agent `i`
- Each row sums to 1 (each item allocated to exactly one agent)
- Fractional allocations represent lottery distributions

**Objective**: Maximize Nash welfare (geometric mean of agent utilities) while learning patterns from data.

### Why Neural Networks?

Traditional fair division algorithms (MNW, round-robin) are:
- Computationally expensive for large instances
- Don't generalize across problem sizes
- Can't leverage patterns in valuation distributions

FairFormer learns to produce high-welfare allocations efficiently through:
- Amortized inference (fast at test time)
- Pattern recognition in valuation structures
- Differentiable allocation via softmax with temperature annealing

---

## Architecture

### FFTransformerResidual (Recommended)

The recommended architecture uses a two-tower design with exchangeable layers and a learnable residual connection:

```
                        ┌─────────────────────────────────────────────────────────┐
                        │              INPUT: Valuations (B, n, m)                │
                        │         n = agents, m = items, B = batch                │
                        └───────────────────────────┬─────────────────────────────┘
                                                    │
                        ┌───────────────────────────┼───────────────────────────┐
                        │                           │                           │
                        ▼                           │                           ▼
         ┌──────────────────────────┐               │          ┌──────────────────────────┐
         │   ExchangeableLayer      │               │          │   ExchangeableLayer      │
         │      (agent_proj)        │               │          │      (item_proj)         │
         │  1 → d_model channels    │               │          │  1 → d_model channels    │
         │  + row/col pooling stats │               │          │  + row/col pooling stats │
         └───────────┬──────────────┘               │          └───────────┬──────────────┘
                     │                              │                      │
                     ▼                              │                      ▼
         ┌──────────────────────────┐               │          ┌──────────────────────────┐
         │     Mean over items      │               │          │     Mean over agents     │
         │    (B, n, d_model)       │               │          │    (B, m, d_model)       │
         └───────────┬──────────────┘               │          └───────────┬──────────────┘
                     │                              │                      │
                     ▼                              │                      ▼
         ┌──────────────────────────┐               │          ┌──────────────────────────┐
         │   Self-Attention (×L)    │               │          │   Self-Attention (×L)    │
         │   agent_transformer      │               │          │   item_transformer       │
         │   L = num_encoder_layers │               │          │   L = num_encoder_layers │
         └───────────┬──────────────┘               │          └───────────┬──────────────┘
                     │                              │                      │
                     │         x_agent              │                      │  x_item
                     │       (B, n, d)              │                      │ (B, m, d)
                     │                              │                      │
                     ├──────────────────────────────┼─────────────────────►│
                     │                              │                      ▼
                     │                              │          ┌──────────────────────────┐
                     │                              │          │    Cross-Attention       │
                     │                              │          │  items attend to agents  │
                     │                              │          │ item_agent_transformer   │
                     │                              │          └───────────┬──────────────┘
                     │                              │                      │
                     │                              │                      ▼
                     │                              │          ┌──────────────────────────┐
                     │                              │          │   Self-Attention (×K)    │
                     │                              │          │   output_transformer     │
                     │                              │          │   K = num_output_layers  │
                     │                              │          └───────────┬──────────────┘
                     │                              │                      │
                     │                              │                      ▼
                     │                              │          ┌──────────────────────────┐
                     │                              │          │       RMSNorm            │
                     │                              │          │      x_output            │
                     │                              │          │    (B, m, d_model)       │
                     │                              │          └───────────┬──────────────┘
                     │                              │                      │
                     │                              │                      │
                     ▼                              │                      ▼
         ┌───────────────────────────────────────────────────────────────────┐
         │                      Bilinear Product                             │
         │                   x_output @ x_agent.T                            │
         │                        (B, m, n)                                  │
         └───────────────────────────────┬───────────────────────────────────┘
                                         │
                                         │
                                         ▼                      ▼ (from input)
         ┌───────────────────────────────────────────────────────────────────┐
         │                           ADD                                     │
         │   bilinear_out + residual_scale × input.permute(0,2,1)           │
         │                     (learnable scalar)                            │
         └───────────────────────────────┬───────────────────────────────────┘
                                         │
                                         ▼
                        ┌────────────────────────────────────┐
                        │   Softmax(x / temperature, dim=-1) │
                        │   (normalize over agents)          │
                        └────────────────┬───────────────────┘
                                         │
                                         ▼
                        ┌────────────────────────────────────┐
                        │    OUTPUT: Allocation (B, m, n)    │
                        │   Each item's fractional alloc     │
                        │   to each agent (sums to 1)        │
                        └────────────────────────────────────┘
```

**Key design principles**:

1. **Dual-stream encoding**: Agents and items are encoded separately with exchangeable layers that respect permutation equivariance
2. **Exchangeable pooling**: Each stream aggregates row/column statistics (mean, max, min) to capture global context while maintaining exchangeability
3. **Cross-attention fusion**: Items attend to agent representations to learn item-agent compatibility
4. **Bilinear output**: The final allocation scores come from `x_output @ x_agent.T`, creating an (m × n) compatibility matrix
5. **Residual connection**: Raw input valuations are added (scaled by learnable `residual_scale`) before softmax - this significantly boosts performance by giving the model direct access to the input signal
6. **Temperature-scaled softmax**: Annealed during training (high→low) to transition from soft to hard allocations

### Attention Blocks

- **FFSelfAttentionBlock**: RMSNorm → Multi-Head Attention → GLU (Gated Linear Unit)
- **FFCrossAttentionBlock**: Separate normalization for queries and keys/values
- **GLU**: SiLU-gated feedforward (8/3 expansion ratio)

### Temperature Annealing

During training:
- `initial_temperature=1.0`: Smooth distributions, easier gradients
- Gradually decrease temperature
- `final_temperature=0.01`: Near one-hot allocations (eval mode)

This balances exploration (learning) with exploitation (discrete allocations).

---

## Installation

### Requirements
- Python 3.10+
- PyTorch 2.1+
- CUDA 12.x (optional, for GPU acceleration)

### Install

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install project dependencies
uv sync
```

### Dependencies

Core dependencies (from `pyproject.toml`):
- `torch>=2.8.0` - Neural network framework
- `numpy>=2.3.2` - Numerical operations
- `wandb>=0.22.0` - Experiment tracking
- `gurobipy>=12.0.3` - Optimal welfare computation (eval)
- `tqdm>=4.67.1` - Progress bars
- `matplotlib`, `pandas`, `scipy` - Visualization and analysis

### Verify Installation

```bash
uv run python -c "from fftransformer import FFTransformerResidual; print('✓ Installation successful')"
```

---

## Model Variants

FairFormer includes three variants - use **FFTransformerResidual** for best performance.

### FFTransformerResidual (Recommended)

**File**: `fftransformer/fftransformer_residual.py`

**When to use**: Default choice for all fair division problems

**Features**:
- ExchangeableLayer projections with pooling statistics
- Bilinear output layer (dot product of item & agent embeddings)
- Learnable residual connection from input valuations
- Configurable encoder depth (`num_encoder_layers`)
- Size-agnostic: works on any (n, m) at inference time

**Import**:
```python
from fftransformer import FFTransformerResidual
```

### FFTransformer (Linear Baseline)

**File**: `fftransformer/fftransformer.py`

**When to use**: Fixed-size problems, comparison baseline

**Features**:
- Linear projections for agent/item embeddings
- Simpler architecture, faster training
- First approach that demonstrated size-agnostic input/output handling

**Import**:
```python
from fftransformer import FFTransformer
```

### FFTransformerExchangeable (Experimental)

**File**: `fftransformer/fftransformer_exchangeable.py`

**When to use**: Research on permutation equivariance, experimental only

**Features**:
- `ExchangeableLayer` with pooling operations (row/column/global mean/min/max)
- Bilinear output layer (dot product of item & agent embeddings)
- Enforces stricter equivariance properties

**Import**:
```python
from fftransformer import FFTransformerExchangeable
```

### Comparison Table

| Feature | Residual (Recommended) | Linear | Exchangeable |
|---------|------------------------|--------|--------------|
| Projection layers | ExchangeableLayer | Linear | ExchangeableLayer |
| Output layer | Bilinear + Residual | Linear | Bilinear |
| Residual connection | ✓ Yes | No | No |
| Performance | ✓ Best | Good | Experimental |
| Size-agnostic | ✓ Yes | Limited | ✓ Yes |
| Recommended use | Production | Baseline | Research |

---

## Training

### Hyperparameter Sweep with Weights & Biases

The recommended way to train FairFormer is using Bayesian optimization via W&B:

```bash
# 1. Create sweep and launch agent
uv run python training/bayesian_sweep_residual.py --run-agent --count 50

# Or create sweep first, then run agent separately
uv run python training/bayesian_sweep_residual.py --create
wandb agent <entity/project>/<sweep_id>
```

### Extracting Best Configuration

After your sweep completes, export the best configuration:

```bash
# Export best config from sweep
uv run python tools/export_best_sweep_residual.py <sweep_id> --project fa-transformer-residual-sweep

# Example
uv run python tools/export_best_sweep_residual.py abc123xyz --project fa-transformer-residual-sweep --entity my-team
```

This creates `configs/best_from_sweep_residual.yaml` with:
- All hyperparameters from the best run
- Increased steps to 100k for production
- Ready to use with `train_residual.py`

### Sweep Configuration

The sweep (defined in `bayesian_sweep_residual.py`) optimizes:

**Architecture**:
- `d_model`: [128, 256, 512, 768]
- `num_heads`: [4, 8, 16]
- `num_output_layers`: [1, 2, 3, 4]
- `num_encoder_layers`: [1, 2, 3]
- `dropout`: [0.0, 0.1]
- `pool_config_name`: ["row_only", "row_col", "row_global", "all"]
- `residual_scale_init`: log-uniform [0.01, 1.0]

**Training**:
- `lr`: log-uniform [1e-4, 3e-3]
- `weight_decay`: log-uniform [1e-5, 5e-2]
- `batch_size`: [256, 512, 1024]
- `steps`: 10000

**Temperature**:
- `initial_temperature`: [0.5, 2.0]
- `final_temperature`: log-uniform [0.001, 0.1]

**Early stopping**:
- `patience`: 30 steps
- `min_delta`: 1e-5

### Training Loop

Each run:
1. Generates random valuation matrices
2. Computes allocations via forward pass
3. Calculates Nash welfare (optimization objective)
4. Backpropagates `-nash_welfare` (maximize welfare = minimize negative welfare)
5. Clips gradients (norm=1.0)
6. Logs to W&B: loss, Nash welfare, learning rate, residual_scale

### Production Training

After identifying best hyperparameters from the sweep, run production training:

```bash
# Option 1: Use config file
uv run python training/train_residual.py --config configs/best_from_sweep_residual.yaml

# Option 2: Specify parameters via CLI
uv run python training/train_residual.py \
    --n 10 --m 20 \
    --d-model 256 --num-heads 8 --num-output-layers 2 \
    --lr 0.0001 --weight-decay 0.01 \
    --steps 100000

# Resume from checkpoint
uv run python training/train_residual.py --resume checkpoints/residual/checkpoint_50000.pt
```

**Key Features**:
- Checkpoint saving every N steps (default: 5000)
- Best model tracking based on validation metric
- Resume training from checkpoints
- Configurable early stopping
- Full wandb integration

Checkpoints are saved to `checkpoints/residual/` directory by default. The best model is saved as `checkpoints/residual/best_model.pt`.

---

## Evaluation Pipeline

The `eval_pipeline/` directory provides comprehensive fairness evaluation. See [`eval_pipeline/README.md`](eval_pipeline/README.md) for detailed documentation.

### Workflow

```
1. Generate Dataset → 2. Evaluate Model → 3. Summarize Results
   (generate_dataset.py)  (evaluation.py)     (summarize_results.py)
```

### Step 1: Generate Dataset

Create valuation matrices with precomputed optimal welfare:

```bash
cd eval_pipeline
uv run python generate_dataset.py --agents 10 --items 14 --num_matrices 100000
```

**Output**: `10_14_100000_dataset.npz` containing:
- `matrices`: Valuation matrices (100000 × 10 × 14)
- `nash_welfare`: Optimal Nash welfare (computed via Gurobi)
- `util_welfare`: Optimal utilitarian welfare
- Timing statistics (`.csv` and `.txt`)

### Step 2: Evaluate Model

Run model inference and compute fairness metrics:

```bash
uv run python evaluation.py 10_14_100000_dataset.npz \
  --eval_type model \
  --model_config best_model_config.json \
  --batch_size 100
```

### Baselines

Compare against:
- `--eval_type random`: Random allocations
- `--eval_type rr`: Round-robin allocations (greedy, EF1 guaranteed)
- MaxUtil: Each item to highest-valuing agent
- MaxNash: Gurobi LP for optimal Nash welfare
- ECE: Envy Cycle Elimination

### EF1 Repair Post-Processing

EF1 repair fixes EF1 violations in any allocation:

```bash
# Model + EF1 Repair
./run_ef1_repair.sh

# MaxUtil + EF1 Repair
./run_max_util_ef1_repair.sh
```

---

## API Reference

### FFTransformerResidual

```python
class FFTransformerResidual(
    d_model: int,                    # Embedding dimension
    num_heads: int,                  # Attention heads
    num_output_layers: int = 1,      # Output transformer layers
    num_encoder_layers: int = 1,     # Encoder transformer layers
    dropout: float = 0.0,            # Dropout rate
    initial_temperature: float = 1.0,# Starting softmax temperature
    final_temperature: float = 0.01  # Eval mode temperature
)
```

The model is **size-agnostic** - it works with any number of agents `n` and items `m` at inference time.

**Methods**:

- `forward(x: torch.Tensor) -> torch.Tensor`
  - Input: `x` of shape `(batch, n, m)` - valuation matrices
  - Output: `(batch, m, n)` - allocation probabilities

- `update_temperature(temperature: float)`
  - Manually set softmax temperature (used during training)

- `eval()`
  - Sets model to eval mode and temperature to `final_temperature`

**Example**:
```python
model = FFTransformerResidual(d_model=256, num_heads=8, num_output_layers=2)
valuations = torch.rand(4, 10, 20)  # 10 agents, 20 items
allocations = model(valuations)  # (4, 20, 10)
```

### Helper Functions

#### get_nash_welfare

```python
def get_nash_welfare(
    u: torch.Tensor,              # Valuations (B, n, m)
    allocation: torch.Tensor,     # Allocations (B, m, n)
    reduction: str = "mean"       # "mean" | "sum" | "none"
) -> torch.Tensor
```

Computes Nash welfare (geometric mean of agent utilities).

**Returns**:
- `reduction="mean"`: Scalar (mean across batch)
- `reduction="sum"`: Scalar (sum across batch)
- `reduction="none"`: Tensor of shape `(B,)` (per-example)

**Example**:
```python
nw = get_nash_welfare(valuations, allocations, reduction="mean")
loss = -nw  # Maximize Nash welfare
```

### Components

For building custom architectures:

- `FFSelfAttentionBlock(d_model, num_heads, dropout)`: Self-attention + GLU
- `FFCrossAttentionBlock(d_model, num_heads, dropout)`: Cross-attention + GLU
- `MHA(d_model, num_heads, dropout)`: Multi-head attention
- `GLU(input_dim, intermediate_dim, output_dim)`: Gated linear unit

---

## Fairness Metrics

The evaluation pipeline computes standard fairness properties from economics.

### Notation

- Agent `i`'s **utility** from allocation `A`: `u_i(A) = Σ_j V[i,j] · A[j,i]`
- Agent `i` **envies** agent `k` if `u_i(A_k) > u_i(A_i)` (prefers `k`'s bundle)

### Envy-Freeness (EF)

**Definition**: No agent envies any other agent

**Formal**: `∀i,k: u_i(A_i) ≥ u_i(A_k)`

**Interpretation**: Strongest fairness guarantee, rarely achievable with indivisible items

### Envy-Free up to One item (EF1)

**Definition**: Envy can be eliminated by removing at most one item from the envied bundle

**Formal**: `∀i,k: ∃j ∈ A_k: u_i(A_i) ≥ u_i(A_k \ {j})`

**Interpretation**: Relaxed fairness notion, always exists for additive valuations

### Envy-Free up to any item (EFx)

**Definition**: Envy can be eliminated by removing ANY item from the envied bundle

**Formal**: `∀i,k: ∀j ∈ A_k: u_i(A_i) ≥ u_i(A_k \ {j})`

**Interpretation**: Stronger than EF1, existence not guaranteed for general valuations

### Nash Welfare (NW)

**Definition**: Geometric mean of agent utilities

**Formal**: `NW(A) = (∏_i u_i(A))^(1/n)`

**Interpretation**:
- Balances efficiency (total utility) with equity (distribution)
- Maximizing NW often yields fair allocations
- Scale-invariant and Pareto-efficient

### Utilitarian Welfare

**Definition**: Sum of agent utilities

**Formal**: `UW(A) = Σ_i u_i(A)`

**Interpretation**: Pure efficiency metric, ignores fairness

### Efficiency Fractions

Evaluation compares model allocations to optimal:

- **Nash welfare fraction**: `model_NW / optimal_NW`
- **Util welfare fraction**: `model_UW / optimal_UW`

Values close to 1.0 indicate near-optimal performance.

---

## Project Structure

```
fair-allocation-transformer/
├── fftransformer/              # Main model package
│   ├── __init__.py            # Package exports
│   ├── fftransformer.py       # Linear FFTransformer
│   ├── fftransformer_exchangeable.py  # Experimental variant
│   ├── fftransformer_residual.py      # RECOMMENDED variant
│   ├── attention_blocks.py    # Self/Cross attention blocks
│   ├── model_components.py    # GLU, MHA components
│   ├── exchangeable_layer.py  # Permutation-equivariant layers
│   └── helpers.py             # Nash welfare computation
│
├── training/                   # Training scripts
│   ├── train_residual.py      # Production training for residual model
│   ├── bayesian_sweep_residual.py  # W&B hyperparameter sweep
│   └── bayesian_sweep_30_60.py     # Sweep for 30x60 problems
│
├── eval_pipeline/              # Evaluation framework
│   ├── README.md              # Detailed evaluation docs
│   ├── generate_dataset.py    # Create test datasets with optimal welfare
│   ├── evaluation.py          # Run model inference + fairness metrics
│   ├── summarize_results.py   # Aggregate evaluation statistics
│   ├── auto_run_evals.py      # Batch evaluation runner
│   └── utils/                 # Helper utilities
│
├── configs/                    # Configuration files
│   ├── best_from_sweep_residual.yaml  # Best hyperparameters
│   └── residual_30_60.yaml           # Config for 30x60 training
│
├── benchmarks/                 # Performance benchmarks
│
├── README.md                   # This file
├── pyproject.toml             # Dependencies and package config
├── LICENSE                    # Apache License 2.0
└── .gitignore                 # Git ignore rules
```

### Key Directories

- **fftransformer/**: Core model implementation, use `from fftransformer import FFTransformerResidual`
- **training/**: Use `train_residual.py` for production training
- **eval_pipeline/**: Complete evaluation workflow, see `eval_pipeline/README.md`
- **configs/**: Best hyperparameter configurations

---

## Citation

If you use FairFormer in your research, please cite:

```bibtex
@article{fairformer2026,
  title={FairFormer: A Transformer Architecture for Discrete Fair Division},
  author={[Author Names]},
  journal={ICML},
  year={2026},
  note={Draft submission}
}
```

**Status**: Draft submission to ICML 2026.

---

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with PyTorch and Weights & Biases
- Fairness metrics implementation based on economics literature
- Gurobi optimization for computing optimal welfare baselines
