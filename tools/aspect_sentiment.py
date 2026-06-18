"""
Tool 3: Aspect-Based Sentiment Analysis (ABSA)
Returns sentiment scores per product attribute (battery, camera, price, etc.)
Uses fine-tuned BERT model (trained in ml/absa_trainer.py).
"""
from langchain_core.tools import tool
from transformers import pipeline
import numpy as np
from config.settings import settings
from utils.helpers import confidence_response
from utils.logger import get_logger

logger = get_logger(__name__)

SUPPORTED_ASPECTS = [
    "battery", "camera", "display", "performance",
    "build_quality", "price", "sound", "comfort",
    "delivery", "packaging"
]

# Lazy-loaded model
_absa_pipeline = None


def _get_absa_pipeline():
    global _absa_pipeline
    if _absa_pipeline is None:
        try:
            _absa_pipeline = pipeline(
                "text-classification",
                model=settings.absa_model_path,
                device=0  # GPU; use -1 for CPU
            )
            logger.info("ABSA model loaded")
        except Exception:
            # Fallback: zero-shot classification until model is trained
            logger.warning("ABSA model not found — using zero-shot fallback")
            _absa_pipeline = pipeline(
                "zero-shot-classification",
                model="cross-encoder/nli-MiniLM2-L6-H768"
            )
    return _absa_pipeline


@tool
def aspect_sentiment(product_id: str, aspects: list[str]) -> dict:
    """
    Get sentiment scores for specific aspects of a product.
    Scores range from -1.0 (very negative) to +1.0 (very positive).
    Use this to compare products on specific attributes like battery life or camera quality.

    Args:
        product_id: Product to analyze
        aspects: List of aspects e.g. ["battery", "camera", "price"]

    Returns:
        dict with per-aspect sentiment scores and overall confidence
    """
    logger.info(f"ABSA: product={product_id}, aspects={aspects}")

    # Validate aspects
    valid_aspects = [a for a in aspects if a.lower() in SUPPORTED_ASPECTS]
    if not valid_aspects:
        valid_aspects = aspects[:3]  # use as-is if not in standard list

    # Load review texts for this product from Chroma
    from services.vector_store import chroma_store
    chunks = chroma_store.query(
        " ".join(valid_aspects),
        product_id=product_id,
        top_k=30
    )

    if not chunks:
        return confidence_response(
            result={a: 0.0 for a in valid_aspects},
            confidence=0.0,
            source_count=0
        )

    pipe = _get_absa_pipeline()
    aspect_scores = {}

    for aspect in valid_aspects:
        relevant = [c["text"] for c in chunks if aspect.lower() in c["text"].lower()]
        if not relevant:
            relevant = [chunks[0]["text"]]  # fallback to top review

        # Score each relevant review chunk for this aspect
        scores = []
        for text in relevant[:10]:
            try:
                out = pipe(
                    text[:512],
                    candidate_labels=[f"positive {aspect}", f"negative {aspect}"],
                    hypothesis_template="This review expresses {} sentiment."
                )
                # Map to [-1, 1]
                pos_score = out["scores"][0] if "positive" in out["labels"][0] else out["scores"][1]
                scores.append(2 * pos_score - 1)
            except Exception:
                scores.append(0.0)

        aspect_scores[aspect] = round(float(np.mean(scores)), 3)

    overall_confidence = min(len(chunks) / 20, 1.0)

    return confidence_response(
        result=aspect_scores,
        confidence=overall_confidence,
        source_count=len(chunks),
        evidence=[c["metadata"].get("review_id", "") for c in chunks[:5]]
    )
