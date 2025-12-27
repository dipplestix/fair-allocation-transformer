#!/bin/bash
# Run all evaluations for the 30x60 trained model

set -e

echo "=========================================="
echo "Running evaluations for 30x60 model"
echo "=========================================="

# Check if model exists
if [ ! -f "checkpoints/residual_30_60/best_model.pt" ]; then
    echo "Error: Model not found at checkpoints/residual_30_60/best_model.pt"
    echo "Training may not be complete yet."
    exit 1
fi

echo ""
echo "1. Evaluating on 50x50..."
echo "----------------------------------------"
uv run python eval_pipeline/evaluate_multi_model.py --n 50 --m 50

echo ""
echo "2. Evaluating on 100x100..."
echo "----------------------------------------"
uv run python eval_pipeline/evaluate_multi_model.py --n 100 --m 100

echo ""
echo "3. Evaluating on 50x100..."
echo "----------------------------------------"
uv run python eval_pipeline/evaluate_multi_model.py --n 50 --m 100

echo ""
echo "4. Running heatmap evaluation (10-30 range)..."
echo "----------------------------------------"
uv run python eval_pipeline/evaluate_heatmap_30_60.py

echo ""
echo "5. Generating heatmap plots..."
echo "----------------------------------------"
uv run python eval_pipeline/plot_heatmap_30_60.py

echo ""
echo "=========================================="
echo "All evaluations complete!"
echo "=========================================="
echo "Results saved to:"
echo "  - results/heatmap_30_60_model.json"
echo "  - results/heatmap_30_60_vs_maxutil_utility.png"
echo "  - results/heatmap_30_60_vs_maxutil_nash.png"
echo "  - results/heatmap_30_60_vs_rr_utility.png"
echo "  - results/heatmap_30_60_vs_rr_nash.png"
