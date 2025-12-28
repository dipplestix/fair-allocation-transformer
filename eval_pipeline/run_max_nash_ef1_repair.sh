#!/bin/bash
# Run max Nash welfare + EF1 repair evaluation on datasets with 10-20 items
# WARNING: This is slow as it uses Gurobi LP solver for each matrix

set -e  # Exit on error

# Color output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}Starting max Nash welfare + EF1 repair evaluation on datasets (10-20 items)...${NC}"
echo -e "${YELLOW}WARNING: This evaluation is slow (Gurobi LP per matrix). Expect several hours.${NC}"
echo ""

# Loop through datasets from 10_10 to 10_20
for m in {10..20}; do
    dataset="datasets/10_${m}_100000_dataset.npz"

    if [ ! -f "$dataset" ]; then
        echo -e "${BLUE}Skipping $dataset (not found)${NC}"
        continue
    fi

    echo -e "${GREEN}Evaluating (MaxNash+EF1): $dataset${NC}"

    # Run evaluation
    uv run eval_pipeline/evaluation.py "$dataset" \
        --eval_type max_nash_with_ef1_repair \
        --batch_size 100 \
        --ef1_repair_max_passes 10

    echo ""
    echo "---"
    echo ""
done

echo -e "${BLUE}Max Nash welfare + EF1 repair evaluations complete!${NC}"
echo ""
echo "Results saved to: results/max_nash/"
ls -lh results/max_nash/*_max_nash_with_ef1_repair.csv 2>/dev/null || echo "No results found yet"
