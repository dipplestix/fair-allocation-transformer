#!/bin/bash

MAX_JOBS=3
running_jobs=0

mkdir -p logs

for items in 7 8 9 10 11 12 13 14 15 16 17; do
    echo "Starting items=$items..."
    (
        uv run python generate_movielens_dataset.py \
            --csv raw_datasets/ratings.csv \
            --agents 7 \
            --items "$items" \
            --num_matrices 1000 \
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

