#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# copy_outputs.sh — Copy trained models + indexes from SCC to your Mac
# Run this on your LOCAL Mac terminal (not on SCC)
#
# Usage:
#   bash scc/copy_outputs.sh your_bu_username
#
# Example:
#   bash scc/copy_outputs.sh sufiyan
# ─────────────────────────────────────────────────────────────────────────────

set -e

BU_USER=${1:?"Usage: bash scc/copy_outputs.sh <your_bu_username>"}
SCC_HOST="scc1.bu.edu"
# Change this to where you uploaded the project on SCC
SCC_PROJECT_DIR="/projectnb/your-lab/$BU_USER/shopmind-agent"

echo "=== Copying outputs from SCC to local ==="
echo "Source: $BU_USER@$SCC_HOST:$SCC_PROJECT_DIR/outputs/"
echo "Dest:   ./outputs/"

mkdir -p outputs/models outputs/chroma_db outputs/faiss_index

# Copy trained models
echo "--- Copying trained models ---"
scp -r "$BU_USER@$SCC_HOST:$SCC_PROJECT_DIR/outputs/models/" ./outputs/

# Copy Chroma DB (vector store for RAG)
echo "--- Copying Chroma DB ---"
scp -r "$BU_USER@$SCC_HOST:$SCC_PROJECT_DIR/outputs/chroma_db/" ./outputs/

# Copy FAISS index
echo "--- Copying FAISS index ---"
scp -r "$BU_USER@$SCC_HOST:$SCC_PROJECT_DIR/outputs/faiss_index/" ./outputs/

# Copy processed CSVs (optional — already have sample data locally)
echo "--- Copying processed data CSVs ---"
mkdir -p data/datasets
scp "$BU_USER@$SCC_HOST:$SCC_PROJECT_DIR/data/datasets/reviews.csv" ./data/datasets/
scp "$BU_USER@$SCC_HOST:$SCC_PROJECT_DIR/data/datasets/products.csv" ./data/datasets/

echo ""
echo "=== All outputs copied! ==="
echo "Your local outputs/ folder is now ready."
echo "Run the agent with: python main.py 'find me wireless headphones'"
