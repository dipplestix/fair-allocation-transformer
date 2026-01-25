#!/bin/bash
# Run C-RR (Welfare-Constrained Round Robin) baseline evaluation on datasets with 10-20 items
# WARNING: This is slow as it uses Gurobi LP for each item assignment

set -e  # Exit on error

# Color output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default to 1000 samples (can be overridden with --max_samples argument)
MAX_SAMPLES=${1:-1000}

echo -e "${BLUE}Starting C-RR baseline evaluation on datasets (10-20 items)...${NC}"
echo -e "${YELLOW}WARNING: This evaluation uses Gurobi LP per item. Using ${MAX_SAMPLES} samples.${NC}"
echo ""

# Loop through datasets from 10_10 to 10_17
for m in {7..17}; do
    dataset="datasets/movielens/7_${m}_1000_dataset.npz"

    if [ ! -f "$dataset" ]; then
        echo -e "${BLUE}Skipping $dataset (not found)${NC}"
        continue
    fi

    echo -e "${GREEN}Evaluating (C-RR): $dataset${NC}"

    # Run evaluation with limited samples
    uv run evaluation.py "$dataset" \
        --eval_type crr \
        --batch_size 100 \
        --max_samples $MAX_SAMPLES

    echo ""
    echo "---"
    echo ""
done

echo -e "${BLUE}C-RR baseline evaluations complete!${NC}"
echo ""
echo "Results saved to: results/crr/"
ls -lh results/crr/*_crr.csv 2>/dev/null || echo "No results found yet"
