"""
Manages FAISS index (for visual search) and Chroma (for review RAG).
"""
import os
import numpy as np
import faiss
import chromadb
from chromadb.config import Settings as ChromaSettings
from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class FAISSStore:
    """Product image vector index."""

    def __init__(self, dim: int = 512):
        self.dim = dim
        self.index_path = settings.faiss_index_path
        self.index = None
        self.id_map: list[str] = []  # maps FAISS int index → product_id

    def build(self, embeddings: np.ndarray, product_ids: list[str]):
        """Build and save FAISS IVF index from product image embeddings."""
        n = len(embeddings)
        nlist = min(int(np.sqrt(n)), 256)
        quantizer = faiss.IndexFlatIP(self.dim)
        self.index = faiss.IndexIVFFlat(quantizer, self.dim, nlist, faiss.METRIC_INNER_PRODUCT)
        self.index.train(embeddings.astype("float32"))
        self.index.add(embeddings.astype("float32"))
        self.id_map = product_ids

        os.makedirs(self.index_path, exist_ok=True)
        faiss.write_index(self.index, f"{self.index_path}/product_images.index")
        np.save(f"{self.index_path}/id_map.npy", np.array(product_ids))
        logger.info(f"FAISS index built: {n} products")

    def load(self):
        """Load existing FAISS index from disk."""
        self.index = faiss.read_index(f"{self.index_path}/product_images.index")
        self.id_map = np.load(f"{self.index_path}/id_map.npy", allow_pickle=True).tolist()
        self.index.nprobe = 10
        logger.info(f"FAISS index loaded: {self.index.ntotal} vectors")

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[dict]:
        """Return top_k similar products with scores."""
        if self.index is None:
            self.load()
        distances, indices = self.index.search(query_embedding.astype("float32"), top_k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1:
                results.append({"product_id": self.id_map[idx], "score": float(dist)})
        return results


class ChromaReviewStore:
    """Review chunk vector store for RAG."""

    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name="product_reviews",
            metadata={"hnsw:space": "cosine"}
        )

    def add_reviews(self, chunks: list[str], metadatas: list[dict], ids: list[str]):
        """Add review chunks to Chroma."""
        self.collection.add(documents=chunks, metadatas=metadatas, ids=ids)
        logger.info(f"Added {len(chunks)} review chunks to Chroma")

    def query(self, query_text: str, product_id: str = None, top_k: int = 10) -> list[dict]:
        """Dense retrieval with optional product_id filter."""
        where = {"product_id": product_id} if product_id else None
        results = self.collection.query(
            query_texts=[query_text],
            n_results=top_k,
            where=where
        )
        chunks = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            chunks.append({"text": doc, "metadata": meta, "score": 1 - dist})
        return chunks


# Global singletons
faiss_store = FAISSStore()
chroma_store = ChromaReviewStore()
