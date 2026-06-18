"""
Tool 1: Visual Search
Given a product image, find the top-K visually similar products.
Uses CLIP embeddings + FAISS IVF index.
"""
from langchain_core.tools import tool
from PIL import Image
import numpy as np
from services.embedding_service import embedding_service
from services.vector_store import faiss_store
from utils.helpers import confidence_response
from utils.logger import get_logger

logger = get_logger(__name__)


@tool
def visual_search(image_path: str, top_k: int = 5) -> dict:
    """
    Find visually similar products given a product image path.
    Use this when the user uploads an image or asks to find something 'like this image'.

    Args:
        image_path: Path to query image file
        top_k: Number of similar products to return (default 5)

    Returns:
        dict with 'result' (list of products), 'confidence', 'source_count', 'evidence'
    """
    logger.info(f"Visual search: {image_path}, top_k={top_k}")

    try:
        image = Image.open(image_path).convert("RGB")
        query_embedding = embedding_service.embed_image(image)
        results = faiss_store.search(query_embedding, top_k=top_k)

        if not results:
            return confidence_response(
                result=[], confidence=0.0,
                source_count=0, evidence=[]
            )

        top_score = results[0]["score"] if results else 0.0
        confidence = min(float(top_score), 1.0)

        return confidence_response(
            result=results,
            confidence=confidence,
            source_count=len(results),
            evidence=[r["product_id"] for r in results]
        )

    except FileNotFoundError:
        logger.error(f"Image not found: {image_path}")
        return confidence_response(result=[], confidence=0.0)
    except Exception as e:
        logger.error(f"Visual search error: {e}")
        return confidence_response(result=[], confidence=0.0)


# ── Standalone: build FAISS index from product images ─────────────────────────
def build_visual_index(image_paths: list[str], product_ids: list[str]):
    """
    Run once during setup (scripts/build_index.py) to build the FAISS index.
    """
    embeddings = []
    valid_ids = []
    for path, pid in zip(image_paths, product_ids):
        try:
            img = Image.open(path).convert("RGB")
            emb = embedding_service.embed_image(img)
            embeddings.append(emb[0])
            valid_ids.append(pid)
        except Exception as e:
            logger.warning(f"Skipping {path}: {e}")

    if embeddings:
        faiss_store.build(np.array(embeddings), valid_ids)
        logger.info(f"Built visual index: {len(embeddings)} products")
