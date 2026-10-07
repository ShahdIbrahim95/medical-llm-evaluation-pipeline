#!/bin/bash

#SBATCH --job-name=hb_api
#SBATCH --output=logs/hb_api_%j.out
#SBATCH --error=logs/hb_api_%j.err
#SBATCH --partition=universe
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module load python/uv

echo "Starting HealthBench API job..."
/usr/bin/time -v uv run python -u main.py
echo "HealthBench API job finished."