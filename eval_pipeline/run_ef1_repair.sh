#!/bin/bash
# Run EF1 Quick Repair evaluation on all datasets

echo "Running EF1 Quick Repair evaluations on all datasets..."

for m in {10..17}
do
    dataset="datasets/movielens/10_${m}_1000_dataset.npz"
    echo "Processing dataset: $dataset"

    uv run evaluation.py "$dataset" \
        --eval_type model_with_ef1_repair \
        --model_config sample_config.json \
        --batch_size 100 \
        --ef1_repair_max_passes 10

    echo "Completed $dataset"
    echo "---"
done

echo "All EF1 Quick Repair evaluations complete!"
echo ""
echo "Results saved to: results/model/"
ls -lh results/model/*_with_ef1_repair.csv 2>/dev/null || echo "No results found yet"
