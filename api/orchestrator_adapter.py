"""
CogInfera — OrchestratorAdapter
Wraps the existing pipeline components to yield intermediate results
as JSON-serialisable dicts that the FastAPI SSE endpoint can stream.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from llm_client import LLMClient
from ingestion.embedder import Embedder
from retrieval.dense_search import DenseSearch
from retrieval.sparse_search import SparseSearch
from retrieval.reranker import Reranker
from retrieval.hybrid import HybridRetriever
from agents.query_planner import QueryPlanner
from agents.retrieval_controller import RetrievalController
from reasoning.context_compressor import ContextCompressor
from reasoning.graph_reasoner import GraphReasoner
from reasoning.multi_pass import MultiPassReasoner
from reasoning.self_rag import SelfRAGValidator
import config


class OrchestratorAdapter:
    """
    Streaming-capable orchestrator.
    `query_stream(question)` is a generator that yields stage dicts in order.
    """

    def __init__(self):
        self.llm        = LLMClient()
        self.embedder   = Embedder()
        self.dense      = DenseSearch()
        self.sparse     = SparseSearch()
        self.reranker   = Reranker()
        self.hybrid     = HybridRetriever(self.embedder, self.dense, self.sparse, self.reranker)
        self.planner    = QueryPlanner(self.llm)
        self.controller = RetrievalController(self.llm, self.hybrid)
        self.compressor = ContextCompressor(self.llm)
        self.grapher    = GraphReasoner(self.llm)
        self.multipass  = MultiPassReasoner(self.llm)
        self.validator  = SelfRAGValidator(self.llm)
        self._try_load()

    # ── Status ──────────────────────────────────────────────────────
    def status(self) -> dict:
        loaded = self.dense.index is not None and self.dense.index.ntotal > 0
        return {
            "loaded": loaded,
            "chunk_count": self.dense.index.ntotal if loaded else 0,
        }

    # ── Ingest ──────────────────────────────────────────────────────
    def ingest(self, pdf_path: str) -> dict:
        import numpy as np
        from ingestion.pdf_parser import parse_pdf
        from ingestion.chunker import SemanticChunker
        from ingestion.hierarchy import HierarchyBuilder

        chunker = SemanticChunker()
        pages   = parse_pdf(pdf_path)
        chunks  = chunker.chunk_pages(pages)
        texts   = [c["text"] for c in chunks]
        embs    = self.embedder.embed(texts).astype(np.float32)

        self.dense.build_index(embs, chunks)
        self.dense.save()
        self.sparse.build_index(chunks)

        hbuilder   = HierarchyBuilder(self.llm)
        hierarchy  = hbuilder.build(chunks)
        hbuilder.save(hierarchy)

        return {
            "pages":   len(pages),
            "chunks":  len(chunks),
            "summary": hierarchy.get("global_summary", ""),
        }

    # ── Streaming Query ─────────────────────────────────────────────
    def query_stream(self, question: str):
        """Generator yielding one dict per pipeline stage."""

        if not self.status()["loaded"]:
            yield {"stage": "error", "data": {"message": "No document indexed yet."}}
            return

        # Stage 1 — Planning
        yield {"stage": "planning", "status": "running", "data": {}}
        plan = self.planner.plan(question)
        yield {"stage": "planning", "status": "done", "data": plan}

        sub_queries = plan.get("sub_queries", [question])

        # Stage 2 — Retrieval
        yield {"stage": "retrieval", "status": "running", "data": {}}
        chunks = self.controller.retrieve_multi(sub_queries)
        top_preview = [
            {"text": c["text"][:300], "pages": f"{c.get('start_page','?')}-{c.get('end_page','?')}",
             "section": c.get("section", "?"), "score": round(c.get("rerank_score", 0), 4)}
            for c in chunks[:8]
        ]
        yield {"stage": "retrieval", "status": "done",
               "data": {"count": len(chunks), "top_chunks": top_preview}}

        # Stage 3 — Context Compression
        yield {"stage": "compression", "status": "running", "data": {}}
        compressed = self.compressor.compress(question, chunks[:15])
        yield {"stage": "compression", "status": "done",
               "data": {"compressed_context": compressed}}

        # Stage 4 — Graph Reasoning
        yield {"stage": "graph_reasoning", "status": "running", "data": {}}
        graph = self.grapher.reason(question, compressed)
        yield {"stage": "graph_reasoning", "status": "done", "data": graph}

        # Stage 5 — Multi-Pass Reasoning
        yield {"stage": "multi_pass", "status": "running", "data": {}}
        mp = self.multipass.reason(question, compressed, graph.get("graph_summary", ""))
        yield {"stage": "multi_pass", "status": "done", "data": mp}

        # Stage 6 — Self-RAG Validation
        yield {"stage": "validation", "status": "running", "data": {}}
        validation = self.validator.validate(question, compressed, mp["answer"])
        answer = mp["answer"]

        if validation.get("needs_retry"):
            refinement = (
                f"Issues: {validation.get('issues', [])}\n"
                f"Suggestion: {validation.get('suggestion', '')}\n\n"
                "Refine the answer to address these concerns."
            )
            answer = self.llm.chat(
                [{"role": "user", "content": (
                    f"Query: {question}\n\nContext:\n{compressed}\n\n"
                    f"Previous Answer:\n{answer}\n\n{refinement}"
                )}],
                system_prompt=(
                    "Improve the answer based on the feedback. "
                    "Stay grounded in context. Provide citations."
                ),
            )

        yield {"stage": "validation", "status": "done", "data": validation}

        # Stage 7 — Final Answer
        yield {"stage": "answer", "status": "done",
               "data": {"answer": answer, "question": question}}

    # ── Internal ────────────────────────────────────────────────────
    def _try_load(self):
        if os.path.isfile(config.FAISS_INDEX_PATH) and os.path.isfile(config.CHUNKS_PATH):
            try:
                self.dense.load()
                self.sparse.load_from_chunks()
            except Exception:
                pass
