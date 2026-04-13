"""
CogInfera — Query Planner & Decomposer
"""

from llm_client import LLMClient


DECOMPOSE_SYSTEM = """\
You are a query decomposition engine for a RAG system.
Given a user question, analyze its complexity and decompose it if needed.

Respond with valid JSON only:
{
  "complexity": "simple" | "multi_hop" | "comparative",
  "sub_queries": ["sub-query 1", "sub-query 2", ...],
  "reasoning": "brief explanation of your decomposition"
}

Rules:
- For simple factual questions, return just the original query in sub_queries.
- For multi-hop questions (requiring info from multiple places), break into independent sub-queries.
- For comparative questions, create sub-queries for each entity being compared.
- Maximum 4 sub-queries.
"""


class QueryPlanner:
    """Decomposes complex queries into sub-queries using LLM."""

    def __init__(self, llm: LLMClient | None = None):
        self.llm = llm or LLMClient()

    def plan(self, query: str) -> dict:
        """Analyze and decompose a query.

        Returns:
            {
                "complexity": str,
                "sub_queries": list[str],
                "reasoning": str,
                "original_query": str,
            }
        """
        result = self.llm.chat_json(
            [{"role": "user", "content": f"Decompose this query:\n\n{query}"}],
            system_prompt=DECOMPOSE_SYSTEM,
        )
        result["original_query"] = query

        # Ensure sub_queries is never empty
        if not result.get("sub_queries"):
            result["sub_queries"] = [query]

        return result
