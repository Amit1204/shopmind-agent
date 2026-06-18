#!/bin/bash
#SBATCH -J shopmind_train             # job name
#SBATCH -o logs/train_%j.out
#SBATCH -e logs/train_%j.err
#SBATCH -p gpu                        # GPU partition
#SBATCH --gres=gpu:1                  # 1 GPU (V100 or A100)
#SBATCH --mem=32G
#SBATCH -t 06:00:00                   # 6 hours max
#SBATCH -N 1
#SBATCH -n 4
# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Train ABSA BERT + XGBoost fake review classifier
# Run AFTER download_data.sh completes
# Submit: sbatch scc/train.sh
# ─────────────────────────────────────────────────────────────────────────────

set -e
cd $SLURM_SUBMIT_DIR

echo "=== Loading modules ==="
module load python3/3.11.4
module load cuda/11.8

echo "=== Activating venv ==="
source .venv-scc/bin/activate

echo "=== GPU info ==="
nvidia-smi

echo "=== Training all models ==="
python scripts/train_models.py --model all

echo "=== Done! Check outputs/models/ ==="
echo "Files to copy to local:"
echo "  outputs/models/absa_model/"
echo "  outputs/models/fake_review_model.joblib"
