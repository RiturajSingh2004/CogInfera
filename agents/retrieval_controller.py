"""
CogInfera — Agentic Retrieval Controller

Plan → Retrieve → Evaluate → Refine → Re-Retrieve loop.
"""

from llm_client import LLMClient
from retrieval.hybrid import HybridRetriever
import config


EVALUATION_SYSTEM = """\
You are a retrieval quality evaluator for a RAG system.
Given a query and retrieved context chunks, evaluate whether the context is sufficient to answer the query.

Respond with valid JSON only:
{
  "sufficient": true | false,
  "coverage": "full" | "partial" | "none",
  "missing_aspects": ["aspect 1", "aspect 2"],
  "refined_query": "a refined/expanded query to retrieve missing information (only if not sufficient)"
}
"""


class RetrievalController:
    """Agentic loop: retrieve → evaluate → refine → re-retrieve."""

    def __init__(
        self,
        llm: LLMClient,
        retriever: HybridRetriever,
        max_iterations: int = config.MAX_RETRIEVAL_ITERATIONS,
    ):
        self.llm = llm
        self.retriever = retriever
        self.max_iterations = max_iterations

    def retrieve(self, query: str) -> list[dict]:
        """Run agentic retrieval loop for a single query.

        Returns the best set of chunks after iterative refinement.
        """
        all_chunks: dict[tuple, dict] = {}  # dedup by (doc, position)

        current_query = query
        for iteration in range(self.max_iterations):
            # Retrieve
            results = self.retriever.retrieve(current_query)

            # Accumulate (deduplicate)
            for chunk in results:
                key = (chunk["document"], chunk["position"])
                if key not in all_chunks:
                    all_chunks[key] = chunk

            # Evaluate sufficiency
            context_text = "\n---\n".join(
                c["text"] for c in list(all_chunks.values())[:15]
            )
            evaluation = self._evaluate(query, context_text)

            if evaluation.get("sufficient", True):
                break

            # Refine query for next iteration
            refined = evaluation.get("refined_query", "")
            if refined and refined != current_query:
                current_query = refined
            else:
                break

        # Return accumulated chunks sorted by rerank_score (if available)
        result_chunks = list(all_chunks.values())
        result_chunks.sort(
            key=lambda c: c.get("rerank_score", c.get("rrf_score", 0)),
            reverse=True,
        )
        return result_chunks

    def retrieve_multi(self, sub_queries: list[str]) -> list[dict]:
        """Retrieve for multiple sub-queries, merge and deduplicate."""
        all_chunks: dict[tuple, dict] = {}

        for sq in sub_queries:
            results = self.retrieve(sq)
            for chunk in results:
                key = (chunk["document"], chunk["position"])
                if key not in all_chunks:
                    all_chunks[key] = chunk

        result_chunks = list(all_chunks.values())
        result_chunks.sort(
            key=lambda c: c.get("rerank_score", c.get("rrf_score", 0)),
            reverse=True,
        )
        return result_chunks

    # ── Internal ────────────────────────────────────────────────────
    def _evaluate(self, query: str, context: str) -> dict:
        prompt = (
            f"Query: {query}\n\n"
            f"Retrieved Context:\n{context[:4000]}\n\n"
            "Evaluate if this context is sufficient to answer the query."
        )
        try:
            return self.llm.chat_json(
                [{"role": "user", "content": prompt}],
                system_prompt=EVALUATION_SYSTEM,
            )
        except ValueError:
            return {"sufficient": True}  # fail-safe: don't loop forever
