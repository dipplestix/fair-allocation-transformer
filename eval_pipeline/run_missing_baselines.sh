#!/bin/bash
# Run missing baseline evaluations for m=21-30

cd /home/dipplestix/Projects/fair-allocation-transformer

for m in 21 22 23 24 25 26 27 28 29 30; do
    dataset="datasets/10_${m}_100000_dataset.npz"

    if [ -f "$dataset" ]; then
        echo "=========================================="
        echo "Processing m=$m"
        echo "=========================================="

        # Random baseline
        if [ ! -f "results/evaluation_results_10_${m}_100000_random.csv" ]; then
            echo "Running Random baseline for m=$m..."
            uv run python eval_pipeline/evaluation.py "$dataset" --eval_type random --batch_size 100
        else
            echo "Random baseline already exists for m=$m"
        fi

        # RR baseline
        if [ ! -f "results/evaluation_results_10_${m}_100000_rr.csv" ]; then
            echo "Running RR baseline for m=$m..."
            uv run python eval_pipeline/evaluation.py "$dataset" --eval_type rr --batch_size 100
        else
            echo "RR baseline already exists for m=$m"
        fi
    else
        echo "Dataset not found: $dataset"
    fi
done

echo ""
echo "All baseline evaluations complete!"
