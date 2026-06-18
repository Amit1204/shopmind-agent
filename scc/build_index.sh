#!/bin/bash
#SBATCH -J shopmind_index             # job name
#SBATCH -o logs/build_index_%j.out
#SBATCH -e logs/build_index_%j.err
#SBATCH -p gpu                        # GPU partition (faster embeddings)
#SBATCH --gres=gpu:1                  # 1 GPU
#SBATCH --mem=32G
#SBATCH -t 04:00:00                   # 4 hours max
#SBATCH -N 1
#SBATCH -n 4
# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Build FAISS index + Chroma review store
# Run AFTER download_data.sh completes
# Submit: sbatch scc/build_index.sh
# ─────────────────────────────────────────────────────────────────────────────

set -e
cd $SLURM_SUBMIT_DIR

echo "=== Loading modules ==="
module load python3/3.11.4
module load cuda/11.8

echo "=== Activating venv ==="
source .venv-scc/bin/activate

echo "=== Building indexes ==="
python scripts/build_index.py

echo "=== Done! Check outputs/chroma_db/ and outputs/faiss_index/ ==="
