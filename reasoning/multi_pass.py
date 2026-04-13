"""
CogInfera — Multi-Pass LLM Reasoning

Three-pass reasoning pipeline:
  1. Fact Extraction
  2. Relationship Mapping
  3. Answer Synthesis
"""

from llm_client import LLMClient


class MultiPassReasoner:
    """Three-pass reasoning over compressed context + graph insights."""

    def __init__(self, llm: LLMClient | None = None):
        self.llm = llm or LLMClient()

    def reason(
        self,
        query: str,
        compressed_context: str,
        graph_summary: str = "",
    ) -> dict:
        """Run 3-pass reasoning and return structured answer.

        Returns:
            {
                "facts": str,
                "relationships": str,
                "answer": str,
            }
        """
        # Pass 1: Fact Extraction
        facts = self._pass_fact_extraction(query, compressed_context)

        # Pass 2: Relationship Mapping
        relationships = self._pass_relationship_mapping(query, facts, graph_summary)

        # Pass 3: Answer Synthesis
        answer = self._pass_answer_synthesis(query, facts, relationships)

        return {
            "facts": facts,
            "relationships": relationships,
            "answer": answer,
        }

    # ── Pass 1 ──────────────────────────────────────────────────────
    def _pass_fact_extraction(self, query: str, context: str) -> str:
        return self.llm.chat(
            [{"role": "user", "content": (
                f"Query: {query}\n\n"
                f"Context:\n{context}\n\n"
                "Extract all facts from the context that are relevant to the query. "
                "List each fact as a bullet point. Include source references (page/section) when available."
            )}],
            system_prompt=(
                "You are a precise fact extractor. Extract ONLY facts present in the context. "
                "Do not infer or add information. Be thorough."
            ),
        )

    # ── Pass 2 ──────────────────────────────────────────────────────
    def _pass_relationship_mapping(
        self, query: str, facts: str, graph_summary: str
    ) -> str:
        graph_section = (
            f"\n\nGraph Relationships:\n{graph_summary}" if graph_summary else ""
        )
        return self.llm.chat(
            [{"role": "user", "content": (
                f"Query: {query}\n\n"
                f"Extracted Facts:\n{facts}"
                f"{graph_section}\n\n"
                "Map the relationships between the extracted facts. "
                "Identify causal links, dependencies, contradictions, and connections."
            )}],
            system_prompt=(
                "You are a relationship mapper. Analyze how facts connect to each other. "
                "Identify patterns, dependencies, and logical connections."
            ),
        )

    # ── Pass 3 ──────────────────────────────────────────────────────
    def _pass_answer_synthesis(
        self, query: str, facts: str, relationships: str
    ) -> str:
        return self.llm.chat(
            [{"role": "user", "content": (
                f"Query: {query}\n\n"
                f"Extracted Facts:\n{facts}\n\n"
                f"Relationship Map:\n{relationships}\n\n"
                "Synthesize a comprehensive, well-structured answer to the query. "
                "cite specific facts and relationships. "
                "If any part cannot be answered from the provided information, state that clearly."
            )}],
            system_prompt=(
                "You are an expert answer synthesizer for a RAG system. "
                "Answer ONLY from the provided facts and relationships. "
                "Provide citations. If information is missing, say 'NOT FOUND in document'. "
                "Be comprehensive yet concise."
            ),
        )
