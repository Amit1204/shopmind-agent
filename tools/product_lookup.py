"""
Tool 5: Product Lookup
Hybrid search (dense + BM25) over the product catalog with filter support.
"""
from langchain_core.tools import tool
import pandas as pd
import numpy as np
from rank_bm25 import BM25Okapi
from services.embedding_service import embedding_service
from config.settings import settings
from utils.helpers import confidence_response
from utils.logger import get_logger

logger = get_logger(__name__)

_products_df = None
_bm25_index = None


def _load_products():
    global _products_df, _bm25_index
    if _products_df is None:
        try:
            _products_df = pd.read_csv(settings.products_csv)
            corpus = (_products_df["title"].fillna("") + " " + _products_df["description"].fillna("")).tolist()
            tokenized = [doc.lower().split() for doc in corpus]
            _bm25_index = BM25Okapi(tokenized)
            logger.info(f"Product catalog loaded: {len(_products_df)} products")
        except FileNotFoundError:
            logger.warning("products.csv not found — run scripts/download_data.py first")
            _products_df = pd.DataFrame(columns=["product_id", "title", "price", "category", "brand"])
            _bm25_index = None
    return _products_df, _bm25_index


@tool
def product_lookup(query: str, filters: dict = None) -> dict:
    """
    Search the product catalog using natural language query with optional filters.
    Use this for any product discovery task.

    Supported filters: max_price, min_price, category, brand, min_rating

    Args:
        query: Natural language search query e.g. "wireless earphones good bass"
        filters: Optional dict e.g. {"max_price": 2000, "category": "electronics"}

    Returns:
        dict with list of matching products, confidence, source_count
    """
    logger.info(f"Product lookup: query={query}, filters={filters}")

    df, bm25 = _load_products()
    if df.empty:
        return confidence_response(result=[], confidence=0.0)

    filters = filters or {}

    # Apply hard filters first
    mask = pd.Series([True] * len(df), index=df.index)
    if "max_price" in filters:
        mask &= df["price"] <= filters["max_price"]
    if "min_price" in filters:
        mask &= df["price"] >= filters["min_price"]
    if "category" in filters:
        mask &= df["category"].str.lower() == filters["category"].lower()
    if "brand" in filters:
        mask &= df["brand"].str.lower() == filters["brand"].lower()
    if "min_rating" in filters:
        mask &= df.get("avg_rating", pd.Series(5.0, index=df.index)) >= filters["min_rating"]

    filtered_df = df[mask].reset_index(drop=True)

    if filtered_df.empty:
        return confidence_response(result=[], confidence=0.0, source_count=0)

    # BM25 ranking on filtered results
    if bm25 is not None:
        filtered_indices = filtered_df.index.tolist()
        corpus = (filtered_df["title"].fillna("") + " " + filtered_df["description"].fillna("")).tolist()
        local_bm25 = BM25Okapi([doc.lower().split() for doc in corpus])
        scores = local_bm25.get_scores(query.lower().split())
    else:
        scores = np.ones(len(filtered_df))

    # Dense re-ranking on top-50 BM25 results
    top_50_idx = np.argsort(scores)[::-1][:50]
    top_50_df = filtered_df.iloc[top_50_idx].reset_index(drop=True)
    top_50_texts = (top_50_df["title"].fillna("") + ". " + top_50_df["description"].fillna("")).tolist()

    query_emb = embedding_service.embed_text([query])
    doc_embs = embedding_service.embed_text(top_50_texts)
    dense_scores = (doc_embs @ query_emb.T).flatten()

    # Combine: 0.5 BM25 + 0.5 dense
    bm25_norm = scores[top_50_idx] / (scores[top_50_idx].max() + 1e-9)
    dense_norm = dense_scores / (dense_scores.max() + 1e-9)
    hybrid = 0.5 * bm25_norm + 0.5 * dense_norm

    final_idx = np.argsort(hybrid)[::-1][:10]
    results = top_50_df.iloc[final_idx][["product_id", "title", "price", "category", "brand"]].to_dict("records")

    return confidence_response(
        result=results,
        confidence=min(float(hybrid[final_idx[0]]), 1.0) if results else 0.0,
        source_count=len(results),
        evidence=[r["product_id"] for r in results]
    )
