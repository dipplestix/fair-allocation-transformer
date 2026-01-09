# Pretrained FairFormer Models

This directory contains pretrained FairFormer (FFTransformerResidual) checkpoints that achieve state-of-the-art performance on discrete fair division tasks.

## Available Models

| Model | File | Training Size | Nash Welfare | Util Welfare | Description |
|-------|------|---------------|--------------|--------------|-------------|
| 10x20 Model | `model_10x20.pt` | 10 agents, 20 items | 96.62% | 95.57% | Trained on fixed 10x20 instances |
| 30x60 Model | `model_30x60.pt` | 30 agents, 60 items | 96.81% | 95.76% | Trained on fixed 30x60 instances |
| Multi-Objective | `model_multi_objective.pt` | Mixed sizes | 96.70% | 95.88% | Trained on 10x15, 25x35, 40x50 |

*Welfare values are percentages of optimal, averaged across 231 test configurations (n=10-30, m=10-30).*

## Model Configurations

### 10x20 Model
```python
config = {
    'd_model': 256,
    'num_heads': 8,
    'num_output_layers': 2,
    'num_encoder_layers': 1,
    'dropout': 0.0
}
```

### 30x60 Model
```python
config = {
    'd_model': 128,
    'num_heads': 8,
    'num_output_layers': 2,
    'num_encoder_layers': 3,
    'dropout': 0.099
}
```

### Multi-Objective Model
```python
config = {
    'd_model': 256,
    'num_heads': 8,
    'num_output_layers': 2,
    'num_encoder_layers': 1,
    'dropout': 0.0
}
```

## Usage

### Basic Usage

```python
import torch
from fftransformer import FFTransformerResidual

# Load the multi-objective model (recommended for general use)
model = FFTransformerResidual(
    d_model=256,
    num_heads=8,
    num_output_layers=2,
    num_encoder_layers=1,
    dropout=0.0
)
model.load_state_dict(torch.load('pretrained/model_multi_objective.pt', weights_only=True))
model.eval()

# Generate allocations for any problem size
valuations = torch.rand(1, 15, 25)  # 15 agents, 25 items
allocations = model(valuations)  # (1, 25, 15) - soft allocations

# Convert to discrete allocations
discrete_alloc = torch.zeros_like(allocations)
discrete_alloc.scatter_(2, allocations.argmax(dim=2, keepdim=True), 1)
```

### With EF1 Repair (Recommended)

For guaranteed EF1 fairness, apply EF1 repair post-processing:

```python
from eval_pipeline.utils.ef1_repair import ef1_quick_repair_batch

# Get model allocations
model.eval()
with torch.no_grad():
    soft_alloc = model(valuations)
    discrete_alloc = torch.zeros_like(soft_alloc)
    discrete_alloc.scatter_(2, soft_alloc.argmax(dim=2, keepdim=True), 1)

# Apply EF1 repair
repaired_alloc = ef1_quick_repair_batch(
    valuations.numpy(),
    discrete_alloc.numpy()
)
```

### Loading 30x60 Model

The 30x60 model has a different architecture (3 encoder layers):

```python
model_30x60 = FFTransformerResidual(
    d_model=128,
    num_heads=8,
    num_output_layers=2,
    num_encoder_layers=3,  # Different from other models
    dropout=0.099
)
model_30x60.load_state_dict(torch.load('pretrained/model_30x60.pt', weights_only=True))
model_30x60.eval()
```

## Performance Comparison

All models with EF1 repair significantly outperform classical baselines:

| Method | Nash Welfare | Util Welfare |
|--------|-------------|--------------|
| **30-60 Model + EF1** | **96.81%** | 95.76% |
| **Multi-Obj + EF1** | 96.70% | **95.88%** |
| **10-20 Model + EF1** | 96.62% | 95.57% |
| MaxUtil + EF1 | 93.89% | 93.56% |
| Round-Robin | 93.70% | 92.26% |
| ECE | 79.32% | 78.96% |

## Which Model to Use?

- **Multi-Objective Model**: Best general-purpose choice. Most consistent across problem sizes.
- **30x60 Model**: Slightly higher Nash welfare. Good for larger problems.
- **10x20 Model**: Fastest inference (smaller d_model would be even faster). Good for small problems.

All models generalize well to problem sizes from 10x10 to 30x30 (and beyond with some degradation).

## Citation

If you use these pretrained models, please cite:

```bibtex
@article{fairformer2026,
  title={FairFormer: A Transformer Architecture for Discrete Fair Division},
  author={[Author Names]},
  journal={ICML},
  year={2026}
}
```
