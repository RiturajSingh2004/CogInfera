"""
CogInfera — LLM-Based Context Compression
"""

from llm_client import LLMClient


COMPRESSION_SYSTEM = """\
You are a context compression engine. Given a query and multiple retrieved text chunks,
compress them into a single dense context that contains ONLY the information relevant
to answering the query.

Rules:
- Preserve all facts, numbers, names, and relationships relevant to the query.
- Remove redundant or irrelevant information.
- Maintain source attribution (mention page numbers / sections when available).
- Keep the compressed context concise but complete.
- Do NOT add any information not present in the original chunks.
"""


class ContextCompressor:
    """Compress retrieved chunks into a dense, relevant context."""

    def __init__(self, llm: LLMClient | None = None):
        self.llm = llm or LLMClient()

    def compress(self, query: str, chunks: list[dict]) -> str:
        """Compress chunks into a focused context string."""
        if not chunks:
            return ""

        # Format chunks with metadata
        formatted_chunks = []
        for i, chunk in enumerate(chunks):
            header = (
                f"[Chunk {i+1} | Pages {chunk.get('start_page','?')}-{chunk.get('end_page','?')} "
                f"| Section: {chunk.get('section','?')}]"
            )
            formatted_chunks.append(f"{header}\n{chunk['text']}")

        context = "\n\n---\n\n".join(formatted_chunks)

        # Truncate if extremely long to stay within model limits
        context = context[:8000]

        compressed = self.llm.chat(
            [{"role": "user", "content": (
                f"Query: {query}\n\n"
                f"Retrieved Chunks:\n{context}\n\n"
                "Compress these chunks into a dense, relevant context for answering the query."
            )}],
            system_prompt=COMPRESSION_SYSTEM,
            reasoning=False,
        )

        return compressed
