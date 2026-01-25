#!/bin/bash
# Run random baseline evaluation on datasets with 10-20 items

set -e  # Exit on error

# Color output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Starting random baseline evaluation on datasets (10-20 items)...${NC}"
echo ""

# Loop through datasets from 10_10 to 10_20
for m in {7..17}; do
    dataset="datasets/movielens/7_${m}_1000_dataset.npz"

    if [ ! -f "$dataset" ]; then
        echo -e "${BLUE}Skipping $dataset (not found)${NC}"
        continue
    fi

    echo -e "${GREEN}Evaluating (random): $dataset${NC}"

    # Run evaluation
    uv run evaluation.py "$dataset" \
        --eval_type random \
        --batch_size 100

    echo ""
    echo "---"
    echo ""
done

echo -e "${BLUE}Random baseline evaluations complete!${NC}"
echo ""
echo "Results saved to: results/random/"
ls -lh results/random/*_random.csv
