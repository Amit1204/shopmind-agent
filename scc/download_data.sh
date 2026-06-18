#!/bin/bash
#SBATCH -J shopmind_download          # job name
#SBATCH -o logs/download_%j.out       # stdout log
#SBATCH -e logs/download_%j.err       # stderr log
#SBATCH -p shared                     # CPU partition (no GPU needed)
#SBATCH --mem=16G
#SBATCH -t 02:00:00                   # 2 hours max
#SBATCH -N 1
#SBATCH -n 4                          # 4 CPU cores
# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Download and parse Amazon Reviews 2023 dataset
# Submit: sbatch scc/download_data.sh
# ─────────────────────────────────────────────────────────────────────────────

set -e
cd $SLURM_SUBMIT_DIR

echo "=== Loading modules ==="
module load python3/3.11.4

echo "=== Activating venv ==="
source .venv-scc/bin/activate

echo "=== Downloading dataset ==="
python scripts/download_data.py

echo "=== Done! Check data/datasets/ for reviews.csv and products.csv ==="
