"""
Shared embedding service — one place to load and call embedding models.
Used by: visual_search, review_qa, product_lookup, recommend tools.
"""
import torch
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
import open_clip
from utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    _instance = None  # singleton

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def initialize(self):
        if self._initialized:
            return
        logger.info("Loading embedding models...")

        # Text embeddings — for reviews and product descriptions
        self.text_model = SentenceTransformer("BAAI/bge-large-en-v1.5")

        # Image + text embeddings — for visual search (CLIP)
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        self.clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
        self.clip_model.eval()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model = self.clip_model.to(self.device)

        self._initialized = True
        logger.info(f"Embedding models loaded on {self.device}")

    def embed_text(self, texts: list[str]) -> np.ndarray:
        """Embed text using BGE-large. Returns (N, 1024) array."""
        self.initialize()
        return self.text_model.encode(texts, normalize_embeddings=True)

    def embed_image(self, image: Image.Image) -> np.ndarray:
        """Embed image using CLIP. Returns (1, 512) array."""
        self.initialize()
        img_tensor = self.clip_preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.clip_model.encode_image(img_tensor)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy()

    def embed_text_clip(self, texts: list[str]) -> np.ndarray:
        """Embed text using CLIP (for cross-modal search). Returns (N, 512)."""
        self.initialize()
        tokens = self.clip_tokenizer(texts).to(self.device)
        with torch.no_grad():
            features = self.clip_model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy()


# Global singleton
embedding_service = EmbeddingService()
