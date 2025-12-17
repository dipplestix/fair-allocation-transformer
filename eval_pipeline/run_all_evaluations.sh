#!/bin/bash
# Run evaluation on datasets with 10-20 items (model trained with m=20)

set -e  # Exit on error

# Color output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Starting evaluation on datasets (10-20 items)...${NC}"
echo ""

# Loop through datasets from 10_10 to 10_20
for m in {10..20}; do
    dataset="datasets/10_${m}_100000_dataset.npz"

    if [ ! -f "$dataset" ]; then
        echo -e "${BLUE}Skipping $dataset (not found)${NC}"
        continue
    fi

    echo -e "${GREEN}Evaluating: $dataset${NC}"

    # Run evaluation
    uv run eval_pipeline/evaluation.py "$dataset" \
        --eval_type model \
        --model_config eval_pipeline/best_model_config.json \
        --batch_size 100

    echo ""
    echo "---"
    echo ""
done

echo -e "${BLUE}All evaluations complete!${NC}"
echo ""
echo "Results saved to: results/"
ls -lh results/*.csv
