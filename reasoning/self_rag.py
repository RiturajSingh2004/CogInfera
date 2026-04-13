"""
CogInfera — Self-RAG Validation Loop

Evaluates answer completeness/grounding and triggers re-retrieval if needed.
"""

from llm_client import LLMClient


VALIDATION_SYSTEM = """\
You are an answer quality validator for a RAG system.
Given the original query, the compressed context, and the generated answer, evaluate:

Respond with valid JSON only:
{
  "is_complete": true | false,
  "is_grounded": true | false,
  "confidence": 0.0 to 1.0,
  "issues": ["issue 1", "issue 2"],
  "missing_info": "description of missing information (if any)",
  "suggestion": "what additional retrieval/reasoning is needed (if any)"
}

Rules:
- is_complete: Does the answer fully address all parts of the query?
- is_grounded: Is every claim in the answer supported by the context?
- confidence: Overall confidence score.
- Be strict — if any claim lacks context support, mark is_grounded as false.
"""


class SelfRAGValidator:
    """Validate answers and trigger re-retrieval if quality is insufficient."""

    CONFIDENCE_THRESHOLD = 0.7

    def __init__(self, llm: LLMClient | None = None):
        self.llm = llm or LLMClient()

    def validate(
        self,
        query: str,
        context: str,
        answer: str,
    ) -> dict:
        """Evaluate answer quality.

        Returns:
            {
                "is_complete": bool,
                "is_grounded": bool,
                "confidence": float,
                "issues": list[str],
                "needs_retry": bool,
                "suggestion": str,
            }
        """
        try:
            result = self.llm.chat_json(
                [{"role": "user", "content": (
                    f"Query: {query}\n\n"
                    f"Context:\n{context[:4000]}\n\n"
                    f"Generated Answer:\n{answer}\n\n"
                    "Validate this answer."
                )}],
                system_prompt=VALIDATION_SYSTEM,
            )
        except ValueError:
            # If JSON parsing fails, assume it's ok
            return {
                "is_complete": True,
                "is_grounded": True,
                "confidence": 0.5,
                "issues": [],
                "needs_retry": False,
                "suggestion": "",
            }

        confidence = result.get("confidence", 0.5)
        is_complete = result.get("is_complete", True)
        is_grounded = result.get("is_grounded", True)

        result["needs_retry"] = (
            not is_complete
            or not is_grounded
            or confidence < self.CONFIDENCE_THRESHOLD
        )

        return result
