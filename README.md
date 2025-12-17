# FairFormer: A Transformer for Discrete Fair Division

![FairFormer mascot](ff.png)

This repository contains the research code accompanying the working paper
"FairFormer: A transformer architecture for discrete fair division". The model
implements the two-tower, symmetry-aware transformer described in the draft
AAMAS 2026 submission and provides utilities for computing Nash welfare during
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
11. [Troubleshooting](#troubleshooting)
12. [Citation](#citation)
13. [License](#license)

---

## Quick Start

Get started in 5 minutes:

```bash
# Clone and install
git clone https://github.com/dipplestix/fair-allocation-transformer.git
cd fair-allocation-transformer

# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install project (creates venv automatically)
uv sync

# Run a simple example
uv run python -c "
import torch
from fatransformer import FATransformer, get_nash_welfare

# Create model for 10 agents, 20 items
model = FATransformer(n=10, m=20, d_model=256, num_heads=8, num_output_layers=2)

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

### Two-Tower Design

FairFormer uses a symmetric two-tower architecture that respects the problem's symmetries:

```
Valuation Matrix (n × m)
         ↓
    ┌────┴────┐
    ↓         ↓
Agent Tower  Item Tower
(n × d)      (m × d)
    ↓         ↓
Self-Attn   Self-Attn
    ↓         ↓
    └────┬────┘
         ↓
   Cross Attention
         ↓
   Output Layers
         ↓
Softmax (temperature)
         ↓
  Allocation (m × n)
```

**Key design principles**:

1. **Agent Tower**: Projects each agent's valuation vector → embedding, applies self-attention
2. **Item Tower**: Projects each item's valuation vector (across agents) → embedding, applies self-attention
3. **Cross Attention**: Items attend to agents to compute allocation scores
4. **Temperature Annealing**: Starts with smooth softmax (exploration), anneals to near-discrete (exploitation)

### Attention Blocks

- **FASelfAttentionBlock**: RMSNorm → Multi-Head Attention → GLU (Gated Linear Unit)
- **FACrossAttentionBlock**: Separate normalization for queries and keys/values
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

### Method 1: uv (recommended)

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install project dependencies
uv sync
```

### Method 2: pip install (alternative)

```bash
pip install -e .
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
uv run python -c "from fatransformer import FATransformer; print('✓ Installation successful')"
```

---

## Model Variants

FairFormer includes two variants - use the **standard** version for most applications.

### Standard FATransformer (Recommended)

**File**: `fatransformer/fatransformer.py`

**When to use**: Default choice for all fair division problems

**Features**:
- Linear projections for agent/item embeddings
- Proven performance on benchmarks
- Simpler, faster, more stable

**Import**:
```python
from fatransformer import FATransformer
```

### Exchangeable FATransformer (Experimental)

**File**: `fatransformer/fatransformer_exchangeable.py`

**When to use**: Research on permutation equivariance, experimental only

**Features**:
- `ExchangeableLayer` with pooling operations (row/column/global mean/min/max)
- Bilinear output layer (dot product of item & agent embeddings)
- Enforces stricter equivariance properties

**Import**:
```python
from fatransformer import FATransformerExchangeable
```

**Note**: The exchangeable variant is currently experimental. The standard FATransformer typically achieves better performance and training stability.

### Comparison Table

| Feature | Standard | Exchangeable |
|---------|----------|--------------|
| Projection layers | Linear | ExchangeableLayer |
| Output layer | Linear | Bilinear |
| Performance | ✓ Proven | Experimental |
| Training stability | ✓ High | Variable |
| Equivariance guarantees | Empirical | Theoretical |
| Recommended use | Production | Research |

---

## Training

### Hyperparameter Sweep with Weights & Biases

The recommended way to train FairFormer is using Bayesian optimization via W&B:

```bash
# 1. Create sweep
python training/bayesian_sweep.py --create --project my-fa-project

# 2. Launch agent (copy sweep ID from step 1)
wandb agent <entity/project>/<sweep_id>

# Or combine both steps
python training/bayesian_sweep.py --run-agent --project my-fa-project
```

### Extracting Best Configuration

After your sweep completes, export the best configuration:

```bash
# Export best config from sweep
python tools/export_best_sweep.py <sweep_id> --project fa-transformer-sweep

# Example
python tools/export_best_sweep.py abc123xyz --project fa-transformer-sweep --entity my-team
```

This creates `configs/best_from_sweep.yaml` with:
- All hyperparameters from the best run
- Increased steps to 100k for production
- Ready to use with `train.py`

### Sweep Configuration

The sweep (defined in `bayesian_sweep.py`) optimizes:

**Architecture**:
- `d_model`: 768 (fixed for this sweep)
- `num_heads`: [4, 8, 12, 16]
- `num_output_layers`: [1, 2, 3, 4, 5]
- `dropout`: [0.0, 0.01]

**Training**:
- `lr`: log-uniform [3e-5, 3e-4]
- `weight_decay`: log-uniform [1e-5, 5e-2]
- `batch_size`: [512, 1024, 2048]
- `steps`: 20000

**Temperature**:
- `initial_temperature`: [0.5, 2.0]
- `final_temperature`: log-uniform [0.001, 0.01]

**Early stopping**:
- `patience`: 20 steps
- `min_delta`: 1e-5

### Training Loop

Each run:
1. Generates random valuation matrices
2. Computes allocations via forward pass
3. Calculates Nash welfare (optimization objective)
4. Backpropagates `-nash_welfare` (maximize welfare = minimize negative welfare)
5. Clips gradients (norm=1.0)
6. Logs to W&B: loss, Nash welfare, learning rate

### Monitoring Training

W&B dashboard shows:
- Nash welfare over time (primary metric)
- Learning rate schedule (cosine annealing)
- Early stopping triggers

### Production Training

After identifying best hyperparameters from the sweep, run production training with the dedicated training script:

```bash
# Option 1: Use config file
python training/train.py --config configs/best_config.yaml

# Option 2: Specify parameters via CLI
python training/train.py \
    --n 10 --m 20 \
    --d-model 768 --num-heads 12 --num-output-layers 4 \
    --lr 0.0001 --weight-decay 0.01 \
    --steps 100000

# Resume from checkpoint
python training/train.py --resume checkpoints/checkpoint_50000.pt
```

**Key Features**:
- Checkpoint saving every N steps (default: 5000)
- Best model tracking based on validation metric
- Resume training from checkpoints
- Configurable early stopping
- Full wandb integration

**Config File Example** (`configs/example_config.yaml`):
See `configs/example_config.yaml` for a complete configuration template with all parameters documented.

Checkpoints are saved to `checkpoints/` directory by default. The best model is saved as `checkpoints/best_model.pt`.

### Manual Training

For custom training loops, see `training/train.py` or `training/bayesian_sweep.py` as templates. For interactive experimentation, see notebooks in `notebooks/training/`.

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
python generate_dataset.py --agents 10 --items 14 --num_matrices 100000
```

**Output**: `10_14_100000_dataset.npz` containing:
- `matrices`: Valuation matrices (100000 × 10 × 14)
- `nash_welfare`: Optimal Nash welfare (computed via Gurobi)
- `util_welfare`: Optimal utilitarian welfare
- Timing statistics (`.csv` and `.txt`)

### Step 2: Evaluate Model

Run model inference and compute fairness metrics:

```bash
python evaluation.py 10_14_100000_dataset.npz \
  --eval_type model \
  --model_config my_model_config.json \
  --batch_size 100
```

**Model config** (`my_model_config.json`):
```json
{
  "model_name": "fatransformer_v1",
  "model_weight_path": "../models/fatransformer_10_14.pt",
  "model_def_path": "../fatransformer/fatransformer.py",
  "model_class_def": "FATransformer",
  "n": 10,
  "m": 14,
  "d_model": 768,
  "num_heads": 12,
  "num_output_layers": 4,
  "initial_temperature": 1.0,
  "final_temperature": 0.01
}
```

**Output**: CSV with columns:
- `matrix_id`, `num_agents`, `num_items`
- `envy_free`, `ef1`, `efx` (boolean fairness properties)
- `utility_sum`, `nash_welfare` (allocation quality)
- `fraction_util_welfare`, `fraction_nash_welfare` (efficiency ratios vs optimal)
- `inference_time`

### Step 3: Summarize Results

Aggregate statistics across evaluation runs:

```bash
python summarize_results.py --folder results/ --inference_types model random
```

**Output**: Summary statistics (mean, std, median) for:
- Fairness property rates (% EF, EF1, EFx)
- Welfare metrics
- Efficiency fractions

### Baselines

Compare against:
- `--eval_type random`: Random allocations
- `--eval_type rr`: Round-robin allocations

### Auto-Run Multiple Evaluations

```bash
python auto_run_evals.py datasets/ \
  --num_agents_list 10 \
  --num_items_list 10 14 20 \
  --num_matrices_list 10000 \
  --eval_type model \
  --model_config config.json
```

### Batch Evaluation Scripts

For comprehensive model evaluation across multiple dataset sizes and baselines, use the shell scripts:

#### 1. Evaluate Model Across All Datasets

```bash
cd eval_pipeline
./run_all_evaluations.sh
```

**What it does**:
- Evaluates your trained model on datasets from `10_10` to `10_20` (10 agents, 10-20 items)
- Uses `best_model_config.json` for model configuration
- Batch size: 100 instances per batch
- Saves results to `results/evaluation_results_10_{m}_100000_best_from_sweep_*.csv`

**Customize**:
- Edit the script to change dataset range (default: `m in {10..20}`)
- Modify `--batch_size` for different processing batches
- Update model config path if using different configuration

#### 2. Evaluate Random Baseline

```bash
./run_random_baseline.sh
```

**What it does**:
- Generates 5 random allocations per instance
- Averages fairness and welfare metrics
- Runs on same datasets (10-20 items)
- Saves to `results/evaluation_results_10_{m}_100000_random.csv`

#### 3. Evaluate Round-Robin Baseline

```bash
./run_rr_baseline.sh
```

**What it does**:
- Implements greedy round-robin allocation (agents pick in order)
- Each agent selects their most preferred available item
- Guaranteed to satisfy EF1 property
- Saves to `results/evaluation_results_10_{m}_100000_rr.csv`

#### 4. Generate Comparison Summary

After running evaluations, create comparison tables:

```bash
cd eval_pipeline
uv run compare_results.py
```

**Output**:
- Prints detailed comparison tables to console
- Shows EF, EF1, EFx, Utility, and Nash Welfare percentages
- Breaks down by dataset size (m=10 to m=20)
- Saves summary to `results/comparison_summary.csv`

**Example output**:
```
Dataset: 10_20 (n=10, m=20)
Method             EF      EF1      EFx    Utility   Nash Welfare
----------------------------------------------------------------
Model            8.3%    70.8%    31.8%      97.9%          97.6%
Random           0.0%     0.0%     0.0%      55.0%           9.2%
RR               6.1%   100.0%    94.6%      92.8%          94.4%
```

#### 5. Create Performance Visualizations

Generate plots comparing metrics across dataset sizes:

```bash
uv run analyze_and_plot.py
```

**Output**:
- `results/comparison_plots.png`: 6-panel plot showing all metrics vs. number of items
  - Envy-Free (EF) %
  - EF1 %
  - EFx %
  - Utility Fraction %
  - Nash Welfare Fraction %
  - Average Runtime (ms)
- `results/runtime_comparison.png`: Detailed runtime analysis

**What the plots show**:
- How each method performs as problem size increases (m=10 to m=20)
- Model generalization from training distribution (m=20)
- Speed vs. performance tradeoffs

#### Complete Evaluation Workflow

Run the full evaluation pipeline:

```bash
cd eval_pipeline

# 1. Evaluate model
./run_all_evaluations.sh

# 2. Evaluate baselines (can run in parallel)
./run_random_baseline.sh &
./run_rr_baseline.sh &
wait

# 3. Generate comparison and plots
uv run compare_results.py
uv run analyze_and_plot.py

# Results available in results/ directory
ls -lh results/
```

**Expected runtime**:
- Model evaluation: ~5-10 minutes (depends on GPU)
- Random baseline: ~1 minute (very fast)
- Round-robin baseline: ~3-5 minutes (greedy selection)
- Analysis & plotting: <1 minute

#### Script Configuration

All scripts process datasets matching pattern `datasets/10_{m}_100000_dataset.npz`. To modify:

**Change dataset range** (e.g., only m=15 to m=18):
```bash
# Edit the shell script
for m in {15..18}; do
    dataset="datasets/10_${m}_100000_dataset.npz"
    ...
done
```

**Change batch size**:
```bash
# In the shell scripts, modify:
uv run eval_pipeline/evaluation.py "$dataset" \
    --eval_type model \
    --model_config eval_pipeline/best_model_config.json \
    --batch_size 256  # Change from default 100
```

**Use different model config**:
```bash
# In run_all_evaluations.sh, change:
--model_config eval_pipeline/my_custom_config.json
```

---

## API Reference

### FATransformer

```python
class FATransformer(
    n: int,                          # Number of agents
    m: int,                          # Number of items
    d_model: int,                    # Embedding dimension
    num_heads: int,                  # Attention heads
    num_output_layers: int = 1,      # Output transformer layers
    dropout: float = 0.0,            # Dropout rate
    initial_temperature: float = 1.0,# Starting softmax temperature
    final_temperature: float = 0.01  # Eval mode temperature
)
```

**Methods**:

- `forward(x: torch.Tensor) -> torch.Tensor`
  - Input: `x` of shape `(batch, n, m)` - valuation matrices
  - Output: `(batch, m, n)` - allocation probabilities
  - Handles variable `m` via padding (if `m < self.m`)

- `update_temperature(temperature: float)`
  - Manually set softmax temperature (used during training)

- `eval()`
  - Sets model to eval mode and temperature to `final_temperature`

**Example**:
```python
model = FATransformer(n=10, m=20, d_model=256, num_heads=8, num_output_layers=2)
valuations = torch.rand(4, 10, 20)
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

- `FASelfAttentionBlock(d_model, num_heads, dropout)`: Self-attention + GLU
- `FACrossAttentionBlock(d_model, num_heads, dropout)`: Cross-attention + GLU
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
├── fatransformer/              # Main model package
│   ├── __init__.py            # Package exports
│   ├── fatransformer.py       # Standard FATransformer (RECOMMENDED)
│   ├── fatransformer_exchangeable.py  # Experimental variant
│   ├── attention_blocks.py    # Self/Cross attention blocks
│   ├── model_components.py    # GLU, MHA components
│   ├── exchangeable_layer.py  # Permutation-equivariant layers
│   └── helpers.py             # Nash welfare computation
│
├── training/                   # Training scripts
│   └── bayesian_sweep.py      # W&B hyperparameter sweep
│
├── eval_pipeline/              # Evaluation framework
│   ├── README.md              # Detailed evaluation docs
│   ├── generate_dataset.py    # Create test datasets with optimal welfare
│   ├── evaluation.py          # Run model inference + fairness metrics
│   ├── summarize_results.py   # Aggregate evaluation statistics
│   ├── auto_run_evals.py      # Batch evaluation runner
│   └── utils/                 # Helper utilities
│
├── notebooks/                  # Experimental notebooks
│   ├── README.md              # Notebook documentation
│   ├── training/              # Training experiments
│   ├── set_transformer/       # Alternative architecture tests
│   └── test_exchange.ipynb    # Exchangeable layer experiments
│
├── set_transformer/            # Experimental baseline (Set Transformer)
│   ├── __init__.py
│   └── set_transformer.py     # MAB, SAB, ISAB, PMA blocks
│
├── benchmarks/                 # Performance benchmarks
│   └── benchmark_exchangeable_layers.py
│
├── README.md                   # This file
├── pyproject.toml             # Dependencies and package config
├── LICENSE                    # Apache License 2.0
└── .gitignore                 # Git ignore rules
```

### Key Directories

- **fatransformer/**: Core model implementation, use `from fatransformer import FATransformer`
- **training/**: Use `bayesian_sweep.py` for hyperparameter tuning
- **eval_pipeline/**: Complete evaluation workflow, see `eval_pipeline/README.md`
- **notebooks/**: Research notebooks, not for production
- **set_transformer/**: Alternative architecture for comparison

---

## Troubleshooting

### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'fatransformer'`

**Solution**: Install package dependencies:
```bash
uv sync
# Or with pip: pip install -e .
```

### CUDA Out of Memory

**Problem**: `RuntimeError: CUDA out of memory`

**Solutions**:
1. Reduce batch size: `--batch_size 256` (default 512)
2. Reduce model size: `d_model=512` instead of 768
3. Use gradient accumulation (modify training script)
4. Use CPU: Model works on CPU, just slower

### Gurobi License (Eval Pipeline)

**Problem**: `GurobiError: Model is too large for size-limited license`

**Solution**: The evaluation pipeline uses Gurobi to compute optimal welfare. Options:
1. Get free academic license: https://www.gurobi.com/academia/
2. Use smaller datasets (`--num_matrices 1000` instead of 100000)
3. Skip optimal welfare computation (modify `generate_dataset.py`)

### W&B Login Issues

**Problem**: `wandb: ERROR Failed to authenticate`

**Solution**:
```bash
wandb login
# Enter your W&B API key from https://wandb.ai/authorize
```

### Temperature Not Decreasing

**Problem**: Allocations remain smooth during training

**Solution**:
- Ensure you call `model.update_temperature(new_temp)` during training
- In eval mode, `model.eval()` automatically sets `temperature = final_temperature`
- Check temperature schedule in training loop

### Poor Performance

**Problem**: Low Nash welfare or fairness metrics

**Debugging**:
1. Check temperature: Should anneal to ~0.01 for discrete allocations
2. Verify input shape: `(batch, n, m)` valuations → `(batch, m, n)` allocations
3. Ensure Nash welfare is negated in loss: `loss = -get_nash_welfare(...)`
4. Check learning rate: Typical range [3e-5, 3e-4]
5. Increase training steps or model capacity

### Notebooks Not Running

**Problem**: Old notebooks fail with import errors

**Solution**: Notebooks in `notebooks/` are experimental snapshots. For stable code:
- Use `fatransformer` package imports
- Run `training/bayesian_sweep.py` for training
- Use `eval_pipeline/` for evaluation

---

## Citation

If you use FairFormer in your research, please cite:

```bibtex
@article{fairformer2026,
  title={FairFormer: A Transformer Architecture for Discrete Fair Division},
  author={[Author Names]},
  journal={AAMAS},
  year={2026},
  note={Working paper, preprint in preparation}
}
```

**Status**: Draft submission to AAMAS 2026. Preprint link will be added upon publication.

---

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with PyTorch and Weights & Biases
- Fairness metrics implementation based on economics literature
- Gurobi optimization for computing optimal welfare baselines
