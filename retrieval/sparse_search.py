"""
CogInfera — Sparse Keyword Search (BM25)
"""

import json
import os
import re
from rank_bm25 import BM25Okapi
import config


class SparseSearch:
    """BM25-based sparse keyword search over chunks."""

    def __init__(self):
        self.bm25: BM25Okapi | None = None
        self.chunks: list[dict] = []
        self.tokenized_corpus: list[list[str]] = []

    # ── Build ───────────────────────────────────────────────────────
    def build_index(self, chunks: list[dict]):
        """Build BM25 index from chunks."""
        self.chunks = chunks
        self.tokenized_corpus = [self._tokenize(c["text"]) for c in chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    # ── Save / Load (reuse chunks from dense search) ────────────────
    def load_from_chunks(self, chunks_path: str = config.CHUNKS_PATH):
        """Rebuild BM25 from saved chunks JSON."""
        with open(chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)
        self.tokenized_corpus = [self._tokenize(c["text"]) for c in self.chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    # ── Search ──────────────────────────────────────────────────────
    def search(self, query: str, top_k: int = config.SPARSE_TOP_K) -> list[dict]:
        """Return top-k chunks by BM25 score."""
        if self.bm25 is None:
            return []

        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        top_indices = scores.argsort()[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                break
            chunk = self.chunks[idx].copy()
            chunk["score"] = float(scores[idx])
            chunk["source"] = "sparse"
            results.append(chunk)

        return results

    # ── Tokenizer ───────────────────────────────────────────────────
    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())
