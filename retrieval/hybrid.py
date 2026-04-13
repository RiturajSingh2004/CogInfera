"""
CogInfera — Hybrid Retrieval Pipeline (Dense + Sparse + Reranker)
"""

import config
from ingestion.embedder import Embedder
from retrieval.dense_search import DenseSearch
from retrieval.sparse_search import SparseSearch
from retrieval.reranker import Reranker


class HybridRetriever:
    """Combines dense + sparse retrieval with Reciprocal Rank Fusion (RRF)
    followed by cross-encoder reranking."""

    def __init__(
        self,
        embedder: Embedder,
        dense: DenseSearch,
        sparse: SparseSearch,
        reranker: Reranker,
    ):
        self.embedder = embedder
        self.dense = dense
        self.sparse = sparse
        self.reranker = reranker

    def retrieve(
        self,
        query: str,
        dense_top_k: int = config.DENSE_TOP_K,
        sparse_top_k: int = config.SPARSE_TOP_K,
        final_top_k: int = config.RERANK_TOP_K,
    ) -> list[dict]:
        """Run the full hybrid retrieval pipeline."""
        # Stage 1 & 2: parallel retrieval
        query_emb = self.embedder.embed_query(query)
        dense_results = self.dense.search(query_emb, top_k=dense_top_k)
        sparse_results = self.sparse.search(query, top_k=sparse_top_k)

        # Merge using RRF
        merged = self._reciprocal_rank_fusion(dense_results, sparse_results)

        # Stage 3: Rerank merged candidates
        reranked = self.reranker.rerank(query, merged, top_k=final_top_k)
        return reranked

    # ── Reciprocal Rank Fusion ──────────────────────────────────────
    @staticmethod
    def _reciprocal_rank_fusion(
        *result_lists: list[dict],
        k: int = config.RRF_K,
    ) -> list[dict]:
        """Merge multiple ranked lists using RRF.

        Chunks are identified by their (document, position) tuple.
        """
        rrf_scores: dict[tuple, float] = {}
        chunk_map: dict[tuple, dict] = {}

        for results in result_lists:
            for rank, chunk in enumerate(results):
                key = (chunk["document"], chunk["position"])
                rrf_scores[key] = rrf_scores.get(key, 0) + 1.0 / (k + rank + 1)
                chunk_map[key] = chunk

        sorted_keys = sorted(rrf_scores, key=rrf_scores.get, reverse=True)
        merged = []
        for key in sorted_keys:
            c = chunk_map[key].copy()
            c["rrf_score"] = rrf_scores[key]
            merged.append(c)

        return merged
