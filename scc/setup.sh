#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# setup.sh — Run ONCE on BU SCC to create your Python environment
# Usage: bash scc/setup.sh
# ─────────────────────────────────────────────────────────────────────────────

set -e

echo "=== Loading SCC modules ==="
module load python3/3.11.4
module load cuda/11.8

echo "=== Creating virtual environment ==="
python3 -m venv .venv-scc
source .venv-scc/bin/activate

echo "=== Upgrading pip ==="
pip install --upgrade pip

echo "=== Installing PyTorch with CUDA 11.8 ==="
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

echo "=== Installing remaining requirements ==="
pip install -r requirements-scc.txt

echo "=== Creating output directories ==="
mkdir -p outputs/models
mkdir -p outputs/chroma_db
mkdir -p outputs/faiss_index
mkdir -p data/datasets
mkdir -p logs

echo "=== Setup complete! ==="
echo "Activate with: source .venv-scc/bin/activate"
