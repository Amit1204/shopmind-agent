"""
Step 2 — Build FAISS visual search index and populate Chroma review store.
Run after download_data.py:
  python scripts/build_index.py
"""
import os
import pandas as pd
from tqdm import tqdm
from services.vector_store import faiss_store, chroma_store
from services.embedding_service import embedding_service
from utils.logger import get_logger

logger = get_logger(__name__)

BATCH_SIZE = 512


def build_review_store(reviews_csv: str):
    """Chunk reviews and load into Chroma for RAG retrieval."""
    logger.info("Building Chroma review store...")
    df = pd.read_csv(reviews_csv)

    # Simple chunking: each review = one chunk (extend to sentence-level later)
    chunks, metadatas, ids = [], [], []
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Indexing reviews"):
        text = str(row.get("reviewText", "")).strip()
        if len(text) < 20:
            continue
        chunks.append(text[:1000])  # cap at 1000 chars
        metadatas.append({
            "product_id": str(row.get("product_id", "")),
            "review_id": f"rev_{i}",
            "rating": float(row.get("overall", 3.0)),
            "verified_purchase": int(row.get("verified_purchase", 1)),
        })
        ids.append(f"rev_{i}")

        # Batch add to Chroma
        if len(chunks) >= BATCH_SIZE:
            chroma_store.add_reviews(chunks, metadatas, ids)
            chunks, metadatas, ids = [], [], []

    if chunks:
        chroma_store.add_reviews(chunks, metadatas, ids)

    logger.info(f"Chroma store built ✓")


def build_visual_index(products_csv: str, images_dir: str = "./data/datasets/images"):
    """Build FAISS image index. Skip if no images directory."""
    if not os.path.exists(images_dir):
        logger.warning(f"No images dir at {images_dir} — skipping visual index")
        logger.info("To use visual search: download product images to data/datasets/images/<product_id>.jpg")
        return

    df = pd.read_csv(products_csv)
    image_paths, product_ids = [], []

    for _, row in df.iterrows():
        pid = str(row["product_id"])
        img_path = os.path.join(images_dir, f"{pid}.jpg")
        if os.path.exists(img_path):
            image_paths.append(img_path)
            product_ids.append(pid)

    if image_paths:
        from tools.visual_search import build_visual_index as _build
        _build(image_paths, product_ids)
    else:
        logger.warning("No images found — visual search will not work")


def main():
    reviews_csv = "./data/datasets/reviews.csv"
    products_csv = "./data/datasets/products.csv"

    if not os.path.exists(reviews_csv):
        logger.error("Run scripts/download_data.py first!")
        return

    build_review_store(reviews_csv)
    build_visual_index(products_csv)
    logger.info("All indexes built ✓  — ready to run the agent")


if __name__ == "__main__":
    main()
