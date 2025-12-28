# PrefLib Dataset Integration

This module enables the evaluation pipeline to use real-world voting data from [PrefLib](https://www.preflib.org/) for fair allocation experiments.

## Overview

PrefLib contains preference data from real elections, surveys, and collaborative filtering tasks. This integration converts ordinal rankings (preference orders) into cardinal valuations (numerical values) suitable for fair allocation problems.

## Directory Structure

```
preflib/
├── README.md                    # This file
├── convert_preflib.py          # Core conversion logic
├── generate_preflib_dataset.py # Dataset generation script
└── raw_datasets/               # PrefLib .soi files
    └── 00072_french-irv-2007/
        ├── 00072-00000001.soi
        ├── info.txt
        └── metadata.csv
```

## Quick Start

### 1. Generate a Dataset

```bash
cd eval_pipeline/preflib

# Generate 1000 matrices with 10 agents and 12 items using Borda count
uv run python generate_preflib_dataset.py \
  --dataset raw_datasets/00072_french-irv-2007 \
  --agents 10 \
  --items 12 \
  --num_matrices 1000

# Use linear conversion method instead
uv run python generate_preflib_dataset.py \
  --dataset raw_datasets/00072_french-irv-2007 \
  --agents 10 \
  --items 12 \
  --num_matrices 1000 \
  --method linear
```

### 2. Run Evaluations

The generated datasets are compatible with existing evaluation scripts:

```bash
cd eval_pipeline

# Evaluate with model
uv run python evaluation.py \
  datasets/borda-french-irv-2007/10_12_1000_dataset.npz \
  --eval_type model \
  --model_config sample_config.json

# Evaluate with baselines
uv run python evaluation.py \
  datasets/borda-french-irv-2007/10_12_1000_dataset.npz \
  --eval_type random

uv run python evaluation.py \
  datasets/borda-french-irv-2007/10_12_1000_dataset.npz \
  --eval_type rr
```

## Conversion Methods

**All methods are normalized to [0, 1] range for consistency with synthetic datasets.**

### Borda Count (`--method borda`)
- Normalized Borda count: `(n_items - rank) / n_items`
- Most preferred item: 1.0
- Least preferred ranked item: 1/n_items
- Unranked items: 0
- **Use case**: Standard voting theory with normalized scores

**Example**: For 12 items, ranking `[8, 1, 4]` becomes:
- Item 8: 1.0 (12/12)
- Item 1: 0.917 (11/12)
- Item 4: 0.833 (10/12)
- Items 2,3,5,6,7,9,10,11,12: 0

### Linear Decay (`--method linear`)
- Linear normalization: `1.0 - (rank / n_items)`
- Most preferred: 1.0
- Linearly decreasing with rank
- Unranked items: 0
- **Use case**: Alternative normalization with slightly different decay

**Example**: For 12 items, ranking `[8, 1, 4]` becomes:
- Item 8: 1.0 (1.0 - 0/12)
- Item 1: 0.917 (1.0 - 1/12)
- Item 4: 0.833 (1.0 - 2/12)
- Items 2,3,5,6,7,9,10,11,12: 0

**Note**: Both methods produce similar normalized values in [0, 1] range.

## Output Structure

Datasets are saved to `eval_pipeline/datasets/{method}-{dataset_name}/`:

```
datasets/
├── borda-french-irv-2007/
│   ├── 10_12_1000_dataset.npz              # Main dataset
│   ├── 10_12_1000_dataset_timing.csv        # Per-matrix timing
│   └── 10_12_1000_dataset_timing_stats.txt  # Summary statistics
└── linear-french-irv-2007/
    └── ...
```

**Dataset format** (`.npz` file):
- `matrices`: Valuation matrices `(num_matrices, n_agents, n_items)`
- `nash_welfare`: Optimal Nash welfare values `(num_matrices,)`
- `util_welfare`: Optimal utilitarian welfare values `(num_matrices,)`

## Data Filtering

### Incomplete Rankings
PrefLib data often contains **incomplete rankings** where voters only rank their top choices:
- `2: 12, 4` → 2 agents ranked only items 12 and 4
- `1: 8, 4, 1, 6` → 1 agent ranked 4 items

### Filtering Strategy
1. **Extract complete rankings**: Only use agents who ranked at least `n_items`
2. **Truncate to required length**: Take first `n_items` from longer rankings
3. **Sample unique sets**: Create multiple matrices by sampling different agent combinations

### Example
For `--agents 10 --items 12`:
- French IRV dataset has 893 voters total
- 367 voters ranked all 12 items
- Can create ~36 unique matrices without replacement
- With replacement, can create unlimited matrices (with some duplication)

## Command-Line Arguments

```bash
python generate_preflib_dataset.py [OPTIONS]

Required:
  --dataset PATH          Path to PrefLib dataset directory
  --num_matrices INT      Number of valuation matrices to generate

Optional:
  --agents INT            Number of agents per matrix (default: 10)
  --items INT             Number of items per matrix (default: 12)
  --method {borda,linear} Conversion method (default: borda)
  --seed INT              Random seed (default: 42)
  --output PATH           Custom output path (auto-generated if omitted)
```

## Adding New PrefLib Datasets

1. Download dataset from [PrefLib.org](https://www.preflib.org/)
2. Extract to `preflib/raw_datasets/{dataset_name}/`
3. Ensure directory contains `.soi` file(s)
4. Generate dataset using the script

Example datasets from PrefLib:
- **Elections**: Presidential, parliamentary, ranked-choice voting
- **Sports**: Formula 1 rankings, Skater preferences
- **Academic**: Paper reviews, course evaluations
- **Collaborative filtering**: Movie ratings, book preferences

## Compatibility with Evaluation Pipeline

Generated PrefLib datasets are **fully compatible** with the existing evaluation pipeline:

✅ Same `.npz` format as synthetic datasets
✅ Same keys: `matrices`, `nash_welfare`, `util_welfare`
✅ Pre-computed optimal welfare values
✅ Works with all evaluation scripts (`evaluation.py`, `summarize_results.py`, etc.)
✅ Works with all baselines (model, random, round-robin)

## Troubleshooting

### "Not enough complete rankings"
**Problem**: Not enough agents ranked all required items
**Solutions**:
- Reduce `--items` to use partial rankings
- Reduce `--agents` to need fewer agents per matrix
- Try a different PrefLib dataset with more complete data

### "Warning: Enabling sampling with replacement"
**Info**: More matrices requested than unique combinations available
**Impact**: Some agent combinations may be repeated across matrices
**Action**: Reduce `--num_matrices` for fully unique sets, or ignore if acceptable

### Slow generation
**Cause**: Optimal welfare computation (Gurobi) can be slow for difficult instances
**Solutions**:
- Start with smaller `--num_matrices` for testing
- Use datasets with simpler valuation patterns
- Check Gurobi license status

## Technical Details

### PrefLib .soi Format
- **SOI** = Strict Order - Incomplete
- Header metadata (# comments)
- Alternative names (item descriptions)
- Preference data: `count: item1, item2, ...`
- Item IDs are 1-indexed

### Valuation Conversion Algorithm
1. Parse `.soi` file → extract rankings
2. Filter for complete rankings (≥ n_items ranked)
3. For each matrix:
   - Sample n_agents rankings
   - Convert each ranking to valuation vector
   - Stack into (n_agents × n_items) matrix
4. Compute optimal Nash & utilitarian welfare
5. Save in standard format

### Dependencies
- `numpy`: Array operations
- `tqdm`: Progress bars
- `gurobipy`: Optimal welfare computation (requires license)
- Parent modules: `utils.max_utility`, `convert_preflib`

## Citation

If using PrefLib data in publications, cite:
- The original PrefLib dataset (see `info.txt` in each dataset directory)
- Nicholas Mattei and Toby Walsh. "PrefLib: A Library for Preferences." *ADT 2013*

## Further Reading

- [PrefLib Documentation](https://www.preflib.org/about/)
- [PrefLib Data Types](https://www.preflib.org/format/)
- Main project README for overall evaluation pipeline
