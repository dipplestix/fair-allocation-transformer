# Envy Transformer - Allocation Verifier Tool

A Python tool for verifying fairness properties of allocation problems using valuation and allocation matrices.

## Installation

```bash
uv sync
```

## Large-Scale Evaluation Pipeline

For evaluating allocation algorithms at scale with precomputed optimal welfare values.

### 1. Generate Dataset

```bash
uv run python generate_dataset.py --agents 10 --items 14 --num_matrices 100000 --output dataset.npz
```

**Arguments:**

- `--agents`: Number of agents (default: 10)
- `--items`: Number of items (default: 14)
- `--num_matrices`: Number of valuation matrices to generate (required)
- `--output`: Output .npz filename (required) Use the following format to use the rest of scripts:` {agents}_{items}_{number of valuation matrices}_dataset.npz`

**Output:** Compressed numpy archive containing:

- `matrices`: Valuation matrices (shape: `num_matrices × agents × items`)
- `nash_welfare`: Precomputed optimal Nash welfare values (shape: `num_matrices`)
- `util_welfare`: Precomputed optimal utilitarian welfare values (shape: `num_matrices`)

### 2. Run Evaluation

```bash
uv run python evaluation.py dataset.npz --batch_size 100 --eval_type random --output evaluation_results.csv
```

**Arguments:**

- `--data_file`: .npz file of dataset
- `--batch_size`: Number of allocations and inference to run in parallel, assumes the model can do parallel inference (default = 100)
- `--eval_type`: Inference allocations from random, rr (round robin), or model(required)
- `--output`: Output .csv filename (default uses the input dataset name to parse data)

**Process:** For each valuation matrix in the dataset:

1. Generate allocation using model (placeholder: `get_model_allocations()`)
2. Calculate all fairness metrics (EF, EF1, EFx)
3. Calculate welfare metrics (utility sum, Nash welfare)
4. Compute efficiency fractions using precomputed optimal values
5. Save comprehensive results to CSV

**Output CSV Columns:**

- `matrix_id`, `num_agents`, `num_items`
- `max_nash_welfare`, `max_util_welfare` (precomputed optimal values)
- `envy_free`, `ef1`, `efx` (fairness properties)
- `utility_sum`, `nash_welfare` (current allocation welfare)
- `fraction_util_welfare`, `fraction_nash_welfare` (efficiency ratios)
- 

### 3. Auto Run Evaluation

### 4. Get full summaries

  **Usage:**

```bash
# Verbose mode (default) - detailed breakdown
uv run python summarize_results.py --folder results/

# Non-verbose mode - aggregated summaries
uv run python summarize_results.py --no-verbose

# Specific inference types only
uv run python summarize_results.py --inference_types model random

# Custom folder
uv run python summarize_results.py --folder my_results/ --inference_types model
```

Output Format:

- Verbose: One section per (agents, items, inference) combination
- Non-verbose: One section per inference type, aggregated across all agent/item pairs
- Warnings for missing combinations

**Performance:** Processes 100k+ matrices efficiently with progress tracking and summary statistics.

### Integration with Model Inference

To integrate with your allocation model:

1. Replace the `get_model_allocations()` function in `evaluation.py`
2. Import your model inference functions
3. The function should take a valuation matrix and return an allocation matrix

```python
def get_model_allocations(valuation_matrix):
    # Replace with your model inference
    from your_model import predict_allocation
    return predict_allocation(valuation_matrix)
```
