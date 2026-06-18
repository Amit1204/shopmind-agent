"""
Hybrid Recommender — Collaborative Filtering + Content-Based.
Evaluated on NDCG@10 and Precision@10.
"""
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from utils.logger import get_logger

logger = get_logger(__name__)


class HybridRecommender:
    """
    Combines:
      - Matrix Factorisation (SVD) for collaborative filtering
      - Content similarity (BGE text embeddings) for cold-start
    """

    def __init__(self, n_factors: int = 50, alpha: float = 0.5):
        self.n_factors = n_factors
        self.alpha = alpha       # weight for CF vs content (0=all content, 1=all CF)
        self.svd = TruncatedSVD(n_components=n_factors, random_state=42)
        self.user_item_matrix = None
        self.product_embeddings = None
        self.product_ids = []
        self.user_ids = []

    def fit(self, ratings_df: pd.DataFrame, product_embeddings: np.ndarray, product_ids: list):
        """
        ratings_df: columns = [user_id, product_id, rating]
        product_embeddings: (N_products, embedding_dim) from BGE/CLIP
        product_ids: list of product_id strings matching embedding rows
        """
        logger.info("Fitting hybrid recommender...")

        # Build user-item matrix
        self.user_item_matrix = ratings_df.pivot_table(
            index="user_id", columns="product_id", values="rating", fill_value=0
        )
        self.user_ids = self.user_item_matrix.index.tolist()

        # SVD decomposition
        self.user_factors = self.svd.fit_transform(self.user_item_matrix.values)
        self.item_factors = self.svd.components_.T  # (n_items, n_factors)

        # Content similarity matrix
        self.product_embeddings = product_embeddings
        self.product_ids = product_ids
        self.content_sim = cosine_similarity(product_embeddings)

        logger.info(f"Recommender fitted: {len(self.user_ids)} users, {len(product_ids)} products")

    def recommend(self, user_id: str, context_products: list[str] = None,
                  exclude: list[str] = None, top_k: int = 10) -> list[dict]:
        """Recommend top_k products for a user."""
        exclude = set(exclude or [])

        # CF scores
        cf_scores = np.zeros(len(self.product_ids))
        if user_id in self.user_ids:
            user_idx = self.user_ids.index(user_id)
            cf_scores = self.user_factors[user_idx] @ self.item_factors.T

        # Content scores from session context
        content_scores = np.zeros(len(self.product_ids))
        if context_products:
            for pid in context_products:
                if pid in self.product_ids:
                    idx = self.product_ids.index(pid)
                    content_scores += self.content_sim[idx]
            content_scores /= len(context_products)

        # Normalize and combine
        cf_norm = cf_scores / (cf_scores.max() + 1e-9)
        content_norm = content_scores / (content_scores.max() + 1e-9)
        hybrid = self.alpha * cf_norm + (1 - self.alpha) * content_norm

        # Rank and filter
        ranked = np.argsort(hybrid)[::-1]
        results = []
        for idx in ranked:
            pid = self.product_ids[idx]
            if pid not in exclude and len(results) < top_k:
                results.append({"product_id": pid, "score": round(float(hybrid[idx]), 4)})

        return results

    def evaluate_ndcg(self, test_df: pd.DataFrame, k: int = 10) -> float:
        """Compute NDCG@k on test set."""
        ndcg_scores = []
        for user_id in test_df["user_id"].unique():
            user_test = test_df[test_df["user_id"] == user_id]
            relevant = set(user_test[user_test["rating"] >= 4]["product_id"].tolist())
            if not relevant:
                continue

            recs = self.recommend(user_id, top_k=k)
            rec_ids = [r["product_id"] for r in recs]

            dcg = sum(1 / np.log2(i + 2) for i, pid in enumerate(rec_ids) if pid in relevant)
            idcg = sum(1 / np.log2(i + 2) for i in range(min(len(relevant), k)))
            ndcg_scores.append(dcg / idcg if idcg > 0 else 0)

        ndcg = float(np.mean(ndcg_scores))
        logger.info(f"NDCG@{k}: {ndcg:.4f}")
        return ndcg
