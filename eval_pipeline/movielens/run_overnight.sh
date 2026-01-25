#!/bin/bash
#SBATCH --job-name=run_5_agent_gen
#SBATCH --output=run_5_agent_gen.out
#SBATCH --error=run_5_agent_gen.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=debug
#SBATCH --cpus-per-task=8
#SBATCH --account=engin1
#SBATCH --time=04:00:00
#SBATCH --mem=10GB

# Activate your venv
source /home/sagoyal/research/fair-allocation-transformer/.venv/bin/activate

cd /home/sagoyal/research/fair-allocation-transformer/eval_pipeline/movielens

# Already in the right directory, just run the command
./run_datasets.sh
