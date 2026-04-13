"""
CogInfera — Main Pipeline Orchestrator

Wires all components together:
  Ingest PDF → Query → Decompose → Agentic Retrieve → Compress →
  Graph Reason → Multi-Pass Reason → Self-RAG Validate → Answer
"""

import json
import os
import numpy as np

import config
from llm_client import LLMClient
from ingestion.pdf_parser import parse_pdf
from ingestion.chunker import SemanticChunker
from ingestion.embedder import Embedder
from ingestion.hierarchy import HierarchyBuilder
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


class CogInfera:
    """End-to-end RAG system orchestrator."""

    def __init__(self):
        print("[CogInfera] Initializing components...")

        # Shared LLM client
        self.llm = LLMClient()

        # Ingestion
        self.chunker = SemanticChunker()
        self.embedder = Embedder()
        self.hierarchy_builder = HierarchyBuilder(self.llm)

        # Retrieval
        self.dense = DenseSearch()
        self.sparse = SparseSearch()
        self.reranker = Reranker()
        self.hybrid = HybridRetriever(
            self.embedder, self.dense, self.sparse, self.reranker
        )

        # Agents
        self.query_planner = QueryPlanner(self.llm)
        self.retrieval_controller = RetrievalController(
            self.llm, self.hybrid
        )

        # Reasoning
        self.compressor = ContextCompressor(self.llm)
        self.graph_reasoner = GraphReasoner(self.llm)
        self.multi_pass = MultiPassReasoner(self.llm)
        self.self_rag = SelfRAGValidator(self.llm)

        # Try loading existing index
        self._try_load_index()

        print("[CogInfera] Ready.")

    # ================================================================
    #  INGEST
    # ================================================================
    def ingest(self, pdf_path: str):
        """Ingest a PDF: parse → chunk → embed → build hierarchy → index."""
        print(f"\n[Ingest] Parsing PDF: {pdf_path}")
        pages = parse_pdf(pdf_path)
        print(f"  → {len(pages)} pages extracted")

        print("[Ingest] Chunking...")
        chunks = self.chunker.chunk_pages(pages)
        print(f"  → {len(chunks)} chunks created")

        print("[Ingest] Embedding chunks...")
        texts = [c["text"] for c in chunks]
        embeddings = self.embedder.embed(texts)

        print("[Ingest] Building FAISS index...")
        self.dense.build_index(embeddings.astype(np.float32), chunks)
        self.dense.save()

        print("[Ingest] Building BM25 index...")
        self.sparse.build_index(chunks)

        print("[Ingest] Building document hierarchy (LLM summaries)...")
        hierarchy = self.hierarchy_builder.build(chunks)
        self.hierarchy_builder.save(hierarchy)

        print(f"[Ingest] Done! Indexed {len(chunks)} chunks from {pdf_path}")
        print(f"  Global summary: {hierarchy['global_summary'][:200]}...")

    # ================================================================
    #  QUERY
    # ================================================================
    def query(self, question: str) -> dict:
        """Run the full query pipeline.

        Returns:
            {
                "question": str,
                "answer": str,
                "plan": dict,
                "validation": dict,
                "facts": str,
                "relationships": str,
            }
        """
        if self.dense.index is None or self.dense.index.ntotal == 0:
            return {
                "question": question,
                "answer": "No documents have been ingested yet. Please ingest a PDF first.",
                "plan": {},
                "validation": {},
            }

        # Step 1: Query Planning & Decomposition
        print(f"\n[Query] Planning: {question}")
        plan = self.query_planner.plan(question)
        sub_queries = plan["sub_queries"]
        print(f"  Complexity: {plan['complexity']}")
        print(f"  Sub-queries: {sub_queries}")

        # Step 2: Agentic Retrieval
        print("[Query] Agentic retrieval...")
        chunks = self.retrieval_controller.retrieve_multi(sub_queries)
        print(f"  → {len(chunks)} chunks retrieved")

        # Step 3: Context Compression
        print("[Query] Compressing context...")
        compressed = self.compressor.compress(question, chunks[:15])

        # Step 4: Dynamic Graph Reasoning
        print("[Query] Graph reasoning...")
        graph_result = self.graph_reasoner.reason(question, compressed)

        # Step 5: Multi-Pass Reasoning
        print("[Query] Multi-pass reasoning...")
        reasoning_result = self.multi_pass.reason(
            question, compressed, graph_result["graph_summary"]
        )
        answer = reasoning_result["answer"]

        # Step 6: Self-RAG Validation
        print("[Query] Self-RAG validation...")
        validation = self.self_rag.validate(question, compressed, answer)

        if validation.get("needs_retry", False):
            print("  ⚠ Validation flagged issues, refining answer...")
            # Re-synthesize with additional context from validation feedback
            refinement_prompt = (
                f"Issues found: {validation.get('issues', [])}\n"
                f"Suggestion: {validation.get('suggestion', '')}\n\n"
                f"Please refine your answer addressing these concerns."
            )
            answer = self.llm.chat(
                [
                    {"role": "user", "content": (
                        f"Query: {question}\n\n"
                        f"Context:\n{compressed}\n\n"
                        f"Previous Answer:\n{answer}\n\n"
                        f"{refinement_prompt}"
                    )},
                ],
                system_prompt=(
                    "You are an expert answer refiner. Improve the answer based on the "
                    "feedback. Stay grounded in the context. Provide citations."
                ),
            )

        result = {
            "question": question,
            "answer": answer,
            "plan": plan,
            "validation": validation,
            "facts": reasoning_result.get("facts", ""),
            "relationships": reasoning_result.get("relationships", ""),
        }

        print("\n[Query] Done!")
        return result

    # ── Helpers ─────────────────────────────────────────────────────
    def _try_load_index(self):
        """Try loading an existing index from disk."""
        if os.path.isfile(config.FAISS_INDEX_PATH) and os.path.isfile(config.CHUNKS_PATH):
            try:
                self.dense.load()
                self.sparse.load_from_chunks()
                print(f"[CogInfera] Loaded existing index ({self.dense.index.ntotal} chunks)")
            except Exception as e:
                print(f"[CogInfera] Could not load index: {e}")
