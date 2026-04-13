"""
CogInfera — Embedding Model (BAAI/bge-small-en)
"""

import numpy as np
from sentence_transformers import SentenceTransformer
import config


class Embedder:
    """Wrapper around sentence-transformers for embedding text."""

    def __init__(self, model_name: str = config.EMBEDDING_MODEL):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of texts. Returns shape (n, dim)."""
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query. Returns shape (dim,)."""
        return self.model.encode(query, normalize_embeddings=True)
