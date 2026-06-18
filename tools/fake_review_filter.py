"""
Tool 4: Fake Review Filter
Returns an authenticity score (0 = likely fake, 1 = likely genuine).
Uses XGBoost classifier trained on linguistic + behavioral features.
"""
from langchain_core.tools import tool
import joblib
import numpy as np
import re
from config.settings import settings
from utils.helpers import confidence_response
from utils.logger import get_logger

logger = get_logger(__name__)

_model = None


def _get_model():
    global _model
    if _model is None:
        try:
            _model = joblib.load(settings.fake_review_model_path)
            logger.info("Fake review model loaded")
        except FileNotFoundError:
            logger.warning("Fake review model not trained yet — using heuristic fallback")
    return _model


def _extract_features(reviews: list[dict]) -> np.ndarray:
    """
    Extract linguistic + behavioral features from reviews.
    These features are the input to the XGBoost classifier.
    """
    features = []
    for r in reviews:
        text = r.get("text", "")
        features.append([
            len(text),                                      # review length
            len(text.split()),                              # word count
            text.count("!"),                                # exclamation marks
            text.count("?"),                                # question marks
            len(re.findall(r'\b[A-Z]{2,}\b', text)),       # ALL CAPS words
            int(r.get("verified_purchase", 1)),             # verified purchase flag
            float(r.get("rating", 3.0)),                    # star rating
            abs(float(r.get("rating", 3.0)) - 3.0),        # rating extremity (1 or 5 = suspicious)
            float(r.get("helpful_votes", 0)),               # helpful votes
            float(r.get("reviewer_review_count", 1)),       # how many reviews by this reviewer
        ])
    return np.array(features, dtype=float)


@tool
def fake_review_filter(product_id: str) -> dict:
    """
    Assess the authenticity of reviews for a product.
    Returns an authenticity score between 0 (fake) and 1 (genuine).
    Always run this before trusting review-based recommendations.

    Args:
        product_id: Product whose reviews to assess

    Returns:
        dict with authenticity_score, fake_review_count, total_reviews
    """
    logger.info(f"Fake review filter: product={product_id}")

    # Load reviews from vector store metadata
    from services.vector_store import chroma_store
    chunks = chroma_store.query("review", product_id=product_id, top_k=50)

    if not chunks:
        return confidence_response(
            result={"authenticity_score": 1.0, "fake_count": 0, "total": 0},
            confidence=0.5,
            source_count=0
        )

    reviews = [c["metadata"] for c in chunks]
    features = _extract_features(reviews)
    model = _get_model()

    if model is not None:
        # Model predicts 0=fake, 1=genuine for each review
        predictions = model.predict(features)
        proba = model.predict_proba(features)[:, 1]  # P(genuine)
    else:
        # Heuristic fallback: flag very short or very extreme reviews
        predictions = np.ones(len(reviews))
        proba = np.ones(len(reviews))
        for i, r in enumerate(reviews):
            text = r.get("text", "")
            rating = float(r.get("rating", 3.0))
            if len(text) < 20 or abs(rating - 3.0) > 1.8:
                predictions[i] = 0
                proba[i] = 0.3

    fake_count = int((predictions == 0).sum())
    authenticity_score = float(np.mean(proba))

    return confidence_response(
        result={
            "authenticity_score": round(authenticity_score, 3),
            "fake_count": fake_count,
            "total_reviews": len(reviews),
            "fake_percentage": round(fake_count / len(reviews) * 100, 1)
        },
        confidence=min(len(reviews) / 20, 1.0),
        source_count=len(reviews)
    )
