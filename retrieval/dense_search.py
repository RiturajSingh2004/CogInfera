"""
CogInfera — Dense Vector Search (FAISS)
"""

import os
import json
import numpy as np
import faiss
import config


class DenseSearch:
    """FAISS-based dense vector search over chunk embeddings."""

    def __init__(self):
        self.index: faiss.IndexFlatIP | None = None
        self.chunks: list[dict] = []

    # ── Build ───────────────────────────────────────────────────────
    def build_index(self, embeddings: np.ndarray, chunks: list[dict]):
        """Build a FAISS inner-product index from embeddings."""
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # cosine sim (embeddings are normalized)
        self.index.add(embeddings.astype(np.float32))
        self.chunks = chunks

    # ── Save / Load ─────────────────────────────────────────────────
    def save(
        self,
        index_path: str = config.FAISS_INDEX_PATH,
        chunks_path: str = config.CHUNKS_PATH,
    ):
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        faiss.write_index(self.index, index_path)
        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, indent=2, ensure_ascii=False)

    def load(
        self,
        index_path: str = config.FAISS_INDEX_PATH,
        chunks_path: str = config.CHUNKS_PATH,
    ):
        self.index = faiss.read_index(index_path)
        with open(chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

    # ── Search ──────────────────────────────────────────────────────
    def search(self, query_embedding: np.ndarray, top_k: int = config.DENSE_TOP_K) -> list[dict]:
        """Return top-k chunks by cosine similarity."""
        if self.index is None or self.index.ntotal == 0:
            return []

        query_vec = query_embedding.reshape(1, -1).astype(np.float32)
        scores, indices = self.index.search(query_vec, min(top_k, self.index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            chunk = self.chunks[idx].copy()
            chunk["score"] = float(score)
            chunk["source"] = "dense"
            results.append(chunk)

        return results
