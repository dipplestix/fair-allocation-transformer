# Experimental Notebooks

This directory contains Jupyter notebooks for experimentation and analysis.

## Training Notebooks (`training/`)

- **faformer_sweep.ipynb**: Hyperparameter sweep experiments for FATransformer
- **fatransformer.ipynb**: Core model training experiments and analysis
- **fatransformer_conv.ipynb**: Convolutional architecture experiments

## Set Transformer (`set_transformer/`)

- **set_transformer_test.ipynb**: Testing alternative set-based architecture

## Exchangeable Experiments

- **test_exchange.ipynb**: Experiments with exchangeable layer variants and permutation equivariance

## Algorithm Examples

- **ece_example.md**: Detailed walkthrough of the Envy-Cycle Elimination algorithm with 3 agents and 10 items, demonstrating how ECE achieves EF1 allocations step-by-step

## Usage

These notebooks are for research and development. For production training, use:
- `training/bayesian_sweep.py` for hyperparameter optimization
- Evaluation pipeline in `eval_pipeline/` for systematic benchmarking
