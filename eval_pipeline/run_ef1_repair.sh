#!/bin/bash
# Run EF1 Quick Repair evaluation on all datasets

echo "Running EF1 Quick Repair evaluations on all datasets..."

for m in {10..20}
do
    dataset="datasets/10_${m}_100000_dataset.npz"
    echo "Processing dataset: $dataset"

    uv run eval_pipeline/evaluation.py "$dataset" \
        --eval_type model_with_ef1_repair \
        --model_config eval_pipeline/best_model_config.json \
        --batch_size 100 \
        --ef1_repair_max_passes 10

    echo "Completed $dataset"
    echo "---"
done

echo "All EF1 Quick Repair evaluations complete!"
