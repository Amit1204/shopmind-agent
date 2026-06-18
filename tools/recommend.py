"""
Tool 7: Hybrid Recommender
Combines collaborative filtering + content similarity (text + image embeddings).
"""
from langchain_core.tools import tool
import numpy as np
import pandas as pd
from services.embedding_service import embedding_service
from config.settings import settings
from utils.helpers import confidence_response
from utils.logger import get_logger

logger = get_logger(__name__)

_user_item_matrix = None
_product_embeddings = None


def _load_recommender_data():
    global _user_item_matrix, _product_embeddings
    if _user_item_matrix is None:
        try:
            ratings_df = pd.read_csv("./data/reference/user_ratings.csv")
            _user_item_matrix = ratings_df.pivot_table(
                index="user_id", columns="product_id", values="rating", fill_value=0
            )
        except FileNotFoundError:
            logger.warning("user_ratings.csv not found")
            _user_item_matrix = pd.DataFrame()
    return _user_item_matrix


@tool
def recommend(user_id: str, context_products: list[str] = None, top_k: int = 5) -> dict:
    """
    Recommend products based on user history and current session context.
    Uses hybrid approach: collaborative filtering + content-based similarity.
    Use this to suggest products the user might like.

    Args:
        user_id: User identifier for personalization
        context_products: Product IDs from current session to base recommendations on
        top_k: Number of recommendations to return

    Returns:
        dict with recommended product IDs, scores, and explanation
    """
    logger.info(f"Recommend: user={user_id}, context={context_products}")

    matrix = _load_recommender_data()
    context_products = context_products or []

    collab_scores = {}
    content_scores = {}

    # Collaborative filtering: user-based
    if not matrix.empty and user_id in matrix.index:
        user_vec = matrix.loc[user_id].values
        similarities = matrix.dot(user_vec) / (
            np.linalg.norm(matrix.values, axis=1) * np.linalg.norm(user_vec) + 1e-9
        )
        top_users_idx = np.argsort(similarities)[::-1][1:11]
        for col in matrix.columns:
            collab_scores[col] = float(matrix.iloc[top_users_idx][col].mean())

    # Content-based: embed context products and find similar ones
    if context_products:
        try:
            products_df = pd.read_csv(settings.products_csv)
            ctx_df = products_df[products_df["product_id"].isin(context_products)]
            if not ctx_df.empty:
                ctx_texts = (ctx_df["title"].fillna("") + " " + ctx_df["description"].fillna("")).tolist()
                ctx_emb = embedding_service.embed_text(ctx_texts).mean(axis=0, keepdims=True)

                all_texts = (products_df["title"].fillna("") + " " + products_df["description"].fillna("")).tolist()
                all_embs = embedding_service.embed_text(all_texts)
                sims = (all_embs @ ctx_emb.T).flatten()

                for pid, score in zip(products_df["product_id"], sims):
                    if pid not in context_products:
                        content_scores[pid] = float(score)
        except FileNotFoundError:
            pass

    # Merge scores
    all_products = set(collab_scores) | set(content_scores)
    hybrid = {
        p: 0.5 * collab_scores.get(p, 0) + 0.5 * content_scores.get(p, 0)
        for p in all_products
    }

    top_products = sorted(hybrid, key=hybrid.get, reverse=True)[:top_k]
    results = [{"product_id": p, "score": round(hybrid[p], 4)} for p in top_products]

    return confidence_response(
        result=results,
        confidence=min(len(results) / top_k, 1.0),
        source_count=len(hybrid),
        evidence=top_products
    )
