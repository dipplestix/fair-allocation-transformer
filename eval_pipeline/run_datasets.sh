#!/bin/bash

cd ~/Projects/fair-allocation-transformer || exit 1
mkdir -p logs

MAX_JOBS=3
running_jobs=0

for items in 15 21 22 23 24 25 26 27 28 29 30; do
    echo "Starting items=$items..."
    (
        uv run python eval_pipeline/generate_dataset.py \
            --agents 10 \
            --items "$items" \
            --num_matrices 100000 \
            > "logs/items_${items}.log" 2>&1
        echo "Finished items=$items."
    ) &

    ((running_jobs++))

    if ((running_jobs >= MAX_JOBS)); then
        wait -n  # wait for one to finish
        ((running_jobs--))
    fi
done

wait
echo "All runs completed."

