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
# Single combination
uv run python generate_dataset.py --agents 10 --items 14 --num_matrices 100000

# Multiple combinations stored in datasets/
uv run python generate_dataset.py --agents 10 12 --items 14 16 --num_matrices 100000 --output datasets/

# Pairwise mode (10 agents/14 items and 12 agents/16 items)
uv run python generate_dataset.py --agents 10 12 --items 14 16 --num_matrices 100000 --pairwise
```

**Arguments:**

- `--agents`: One or more agent counts (default: 10)
- `--items`: One or more item counts (default: 14)
- `--num_matrices`: Number of valuation matrices to generate (required)
- `--output`: Output .npz filename when generating a single dataset, or directory when generating multiple datasets (optional). Default naming: `{agents}_{items}_{number of valuation matrices}_dataset.npz`
- `--seed`: Base random seed used to initialise NumPy (default: 10)
- `--pairwise`: Pair each entry from `--agents` with the corresponding entry from `--items` instead of generating every combination

**Output:** Compressed numpy archive containing:

- `matrices`: Valuation matrices (shape: `num_matrices × agents × items`)
- `nash_welfare`: Precomputed optimal Nash welfare values (shape: `num_matrices`)
- `util_welfare`: Precomputed optimal utilitarian welfare values (shape: `num_matrices`)
- Timing statistics for generating the dataset save to a csv file with the same name as root of the .npz
- Timing statistics summary saved to a txt file with the same name as root of the .npz

### 2. Run Evaluation

```bash
uv run python evaluation.py dataset.npz --batch_size 100 --eval_type random --output evaluation_results.csv
```

**Arguments:**

- `--data_file`: .npz file of dataset
- `--batch_size`: Number of allocations and inference to run in parallel, assumes the model can do parallel inference (default = 100)
- `--eval_type`: Inference allocations from random, rr (round robin), or model (default = random). If using model must provide a config file
- `--output`: Output .csv filename (default uses the input dataset name to parse data)
- `--model_config`: Config file if model is selected as eval_type

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
- `inference_time`

### 3. Auto Run Evaluation


```bash
uv run python auto_run_evals.py datasets/ --num_agents_list 10 --num_items_list 10 11 19 20 --num_matrices_list 1000 100000 --batch_size 100 --eval_type model --model_config sample_config.json
```

**Arguments:**

- `--data_folder`: Folder containing .npz dataset files
- `--num_agents_list`: List of agent counts to filter datasets
- `--num_items_list`: List of item counts to filter datasets
- `--num_matrices_list`: List of matrix counts to filter datasets
- `--batch_size`: Number of allocations and inference to run in parallel, assumes the model can do parallel inference (default = 100)
- `--eval_type`: Inference allocations from random, rr (round robin), or model (default = random)
- `--model_config`: Config file if model is selected as eval_type

**Model Config Example:**

```json
{
    "model_name": "model0",
    "model_weight_path": "../fatransformer/fatransformer_model_20_10-new.pt",
    "model_def_path": "../fatransformer/fatransformer.py",
    "model_class_def": "FATransformer",
    "n": 10,
    "m": 20,
    "d_model": 768,
    "num_heads": 12,
    "dropout": 0.0,
    "lr": 1e-4,
    "weight_decay": 1e-2,
    "steps": 20000,
    "batch_size": 512,
    "num_output_layers": 4,
    "initial_temperature": 1.0,
    "final_temperature": 0.01
}
```
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
