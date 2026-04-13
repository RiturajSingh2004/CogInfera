"""
CogInfera — Cross-Encoder Reranker
"""

from sentence_transformers import CrossEncoder
import config


class Reranker:
    """Rerank retrieved chunks using a cross-encoder model."""

    def __init__(self, model_name: str = config.RERANKER_MODEL):
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        chunks: list[dict],
        top_k: int = config.RERANK_TOP_K,
    ) -> list[dict]:
        """Rerank chunks by cross-encoder relevance score.

        Returns top_k chunks sorted by relevance (highest first).
        """
        if not chunks:
            return []

        pairs = [(query, chunk["text"]) for chunk in chunks]
        scores = self.model.predict(pairs)

        for chunk, score in zip(chunks, scores):
            chunk["rerank_score"] = float(score)

        ranked = sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)
        return ranked[:top_k]
