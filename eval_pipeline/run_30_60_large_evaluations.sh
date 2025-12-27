#!/bin/bash
# Run all evaluations for the 30x60 large model (d_model=256)

set -e

MODEL_PATH="checkpoints/residual_30_60_large/best_model.pt"

echo "=========================================="
echo "Running evaluations for 30x60 LARGE model"
echo "=========================================="

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model not found at $MODEL_PATH"
    echo "Training may not be complete yet."
    exit 1
fi

echo ""
echo "1. Evaluating on 50x50..."
echo "----------------------------------------"
uv run python eval_pipeline/evaluate_multi_model_large.py --n 50 --m 50

echo ""
echo "2. Evaluating on 100x100..."
echo "----------------------------------------"
uv run python eval_pipeline/evaluate_multi_model_large.py --n 100 --m 100

echo ""
echo "3. Evaluating on 50x100..."
echo "----------------------------------------"
uv run python eval_pipeline/evaluate_multi_model_large.py --n 50 --m 100

echo ""
echo "4. Running heatmap evaluation (10-30 range)..."
echo "----------------------------------------"
uv run python eval_pipeline/evaluate_heatmap_30_60_large.py

echo ""
echo "5. Generating heatmap plots..."
echo "----------------------------------------"
uv run python eval_pipeline/plot_heatmap_30_60_large.py

echo ""
echo "6. Model comparison heatmap (30-60 large vs 10-20)..."
echo "----------------------------------------"
uv run python eval_pipeline/evaluate_heatmap_model_comparison_large.py
uv run python eval_pipeline/plot_heatmap_model_comparison_large.py

echo ""
echo "=========================================="
echo "All evaluations complete!"
echo "=========================================="
