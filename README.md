# FairFormer: A Transformer for Discrete Fair Division

![FairFormer mascot](ff.png)

This repository contains the research code accompanying the working paper
"FairFormer: A transformer architecture for discrete fair division". The model
implements the two-tower, symmetry-aware transformer described in the draft
AAMAS 2026 submission and provides utilities for computing Nash welfare during
training.

## Repository structure

- `fatransformer/fatransformer.py` – neural architecture for mapping valuation
  matrices to allocation distributions.
- `fatransformer/model_components.py` – reusable multi-head attention (MHA) and
  gated linear unit (GLU) blocks used throughout the model.
- `fatransformer/helpers.py` – helper utilities, including Nash welfare
  computation for batched allocations.

## Installation

Create a fresh Python environment (Python 3.10+) and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The code has been tested with PyTorch 2.1 on CUDA 12.x, but the architecture is
backend-agnostic and will also run on CPU-only installations.

## Usage

```python
import torch
from fatransformer.fatransformer import FATransformer

# n agents, m items
n_agents, m_items = 10, 12
model = FATransformer(
    n=n_agents,
    m=m_items,
    d_model=256,
    num_heads=8,
    num_output_layers=2,
    initial_temperature=1.0,
    final_temperature=0.01,
)

valuations = torch.rand(4, n_agents, m_items)  # batch of valuation matrices
allocations = model(valuations)                # (batch, m, n)
```

During training, call `model.update_temperature(value)` to anneal the softmax
temperature. Switching to evaluation mode via `model.eval()` automatically sets
`temperature` to `final_temperature`, producing near one-hot allocations.

## Nash welfare utility

Use `fatransformer.helpers.get_nash_welfare` to compute the (geometric-mean)
Nash welfare of batched allocations:

```python
from fatransformer.helpers import get_nash_welfare

nw = get_nash_welfare(valuations, allocations, reduction="mean")
```

The helper function returns the mean Nash welfare across the batch by default,
and supports `"sum"` and `"none"` reductions.

## Citing

If you use this code, please cite the FairFormer working paper once available.
A preprint draft is currently in preparation for AAMAS 2026.
