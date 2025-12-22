#!/bin/bash
# Run Model with Swaps evaluation on datasets with 10-20 items

set -e  # Exit on error

GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}Starting Model with Swaps evaluation (welfare_weight=0.5)...${NC}"
echo ""

for m in {10..20}; do
    dataset="datasets/10_${m}_100000_dataset.npz"

    if [ ! -f "$dataset" ]; then
        echo -e "${BLUE}Skipping $dataset (not found)${NC}"
        continue
    fi

    echo -e "${GREEN}Evaluating (Model+Swaps): $dataset${NC}"

    uv run eval_pipeline/evaluation.py "$dataset" \
        --eval_type model_with_swaps \
        --model_config eval_pipeline/best_model_config.json \
        --batch_size 100 \
        --swap_iterations 100 \
        --swap_welfare_weight 0.5

    echo ""
    echo "---"
    echo ""
done

echo -e "${BLUE}Model with Swaps evaluations complete!${NC}"
echo ""
echo "Results saved to: results/"
ls -lh results/*_with_swaps.csv
