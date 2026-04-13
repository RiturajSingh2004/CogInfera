"""
CogInfera — Centralized Configuration
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── LLM (OpenRouter) ────────────────────────────────────────────────
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
LLM_MODEL = "qwen/qwen3.5-9b"
LLM_REASONING_ENABLED = True

# ── Embedding ───────────────────────────────────────────────────────
EMBEDDING_MODEL = "BAAI/bge-small-en"
EMBEDDING_DIMENSION = 384  # bge-small-en output dim

# ── Reranker ────────────────────────────────────────────────────────
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ── Chunking ────────────────────────────────────────────────────────
CHUNK_SIZE = 512          # target tokens per chunk
CHUNK_OVERLAP = 64        # overlap tokens between chunks

# ── Retrieval ───────────────────────────────────────────────────────
DENSE_TOP_K = 20          # candidates from FAISS
SPARSE_TOP_K = 20         # candidates from BM25
RERANK_TOP_K = 10         # final top-k after reranking
RRF_K = 60                # RRF constant

# ── Agentic Controller ─────────────────────────────────────────────
MAX_RETRIEVAL_ITERATIONS = 3

# ── Paths ───────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss.index")
HIERARCHY_PATH = os.path.join(DATA_DIR, "hierarchy.json")
CHUNKS_PATH = os.path.join(DATA_DIR, "chunks.json")
