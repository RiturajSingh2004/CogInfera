"""
CogInfera — Hierarchical Document Representation

Document → Sections → Chunks, plus section summaries and a global summary.
"""

import json
import os
from llm_client import LLMClient
import config


class HierarchyBuilder:
    """Build a hierarchical representation from chunks and persist it."""

    def __init__(self, llm: LLMClient | None = None):
        self.llm = llm or LLMClient()

    def build(self, chunks: list[dict]) -> dict:
        """Build hierarchy dict from chunks.

        Returns:
            {
                "document": str,
                "global_summary": str,
                "sections": [
                    {
                        "title": str,
                        "summary": str,
                        "chunk_indices": [int, ...]
                    },
                    ...
                ]
            }
        """
        document = chunks[0]["document"] if chunks else "unknown"

        # Group chunks by section
        sections_map: dict[str, list[int]] = {}
        for i, chunk in enumerate(chunks):
            sec = chunk.get("section", "General")
            sections_map.setdefault(sec, []).append(i)

        # Generate section summaries
        sections = []
        section_summaries = []
        for title, indices in sections_map.items():
            combined = "\n\n".join(chunks[i]["text"] for i in indices)
            # Truncate to avoid huge prompts
            combined = combined[:3000]
            summary = self._summarize(combined, f"section '{title}'")
            sections.append({
                "title": title,
                "summary": summary,
                "chunk_indices": indices,
            })
            section_summaries.append(f"**{title}**: {summary}")

        # Generate global summary from section summaries
        all_section_text = "\n".join(section_summaries)
        global_summary = self._summarize(
            all_section_text, "the entire document"
        )

        hierarchy = {
            "document": document,
            "global_summary": global_summary,
            "sections": sections,
        }

        return hierarchy

    def save(self, hierarchy: dict, path: str = config.HIERARCHY_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(hierarchy, f, indent=2, ensure_ascii=False)

    def load(self, path: str = config.HIERARCHY_PATH) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    # ── Internal ────────────────────────────────────────────────────
    def _summarize(self, text: str, context_label: str) -> str:
        prompt = (
            f"Summarize the following content from {context_label} in 2-3 concise sentences. "
            "Focus on key facts and main ideas.\n\n"
            f"{text}"
        )
        try:
            return self.llm.chat(
                [{"role": "user", "content": prompt}],
                system_prompt="You are a precise summarizer. Respond with only the summary.",
                reasoning=False,
            )
        except Exception as e:
            print(f"[Warning] Failed to generate summary for {context_label}: {e}")
            return "Summary unavailable due to API error."
