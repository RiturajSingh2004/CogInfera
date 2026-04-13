"""
CogInfera — Dynamic Graph Reasoning

Builds on-demand entity-relationship graphs from context,
uses graph traversal for multi-hop reasoning.
"""

import networkx as nx
from llm_client import LLMClient


EXTRACTION_SYSTEM = """\
You are an entity-relationship extractor. Given a query and context, extract entities and their relationships.

Respond with valid JSON only:
{
  "entities": [
    {"name": "Entity Name", "type": "person|concept|organization|location|other", "description": "brief desc"}
  ],
  "relationships": [
    {"source": "Entity A", "target": "Entity B", "relation": "relationship description"}
  ]
}

Rules:
- Focus on entities and relationships relevant to the query.
- Keep entity names consistent (use the same name for the same entity).
- Extract at most 15 entities and 20 relationships.
"""


class GraphReasoner:
    """Build on-demand entity-relationship graphs and extract reasoning paths."""

    def __init__(self, llm: LLMClient | None = None):
        self.llm = llm or LLMClient()

    def reason(self, query: str, context: str) -> dict:
        """Build graph from context and extract relevant paths.

        Returns:
            {
                "entities": list[dict],
                "relationships": list[dict],
                "paths": list[str],
                "graph_summary": str,
            }
        """
        # Extract entities and relationships via LLM
        extraction = self._extract(query, context)

        entities = extraction.get("entities", [])
        relationships = extraction.get("relationships", [])

        # Build networkx graph
        G = nx.DiGraph()
        for ent in entities:
            G.add_node(ent["name"], **{k: v for k, v in ent.items() if k != "name"})

        for rel in relationships:
            G.add_edge(rel["source"], rel["target"], relation=rel["relation"])

        # Extract interesting paths (multi-hop connections)
        paths = self._extract_paths(G, entities)

        # Generate graph summary
        graph_summary = self._summarize_graph(paths, relationships)

        return {
            "entities": entities,
            "relationships": relationships,
            "paths": paths,
            "graph_summary": graph_summary,
        }

    # ── Internal ────────────────────────────────────────────────────
    def _extract(self, query: str, context: str) -> dict:
        try:
            return self.llm.chat_json(
                [{"role": "user", "content": (
                    f"Query: {query}\n\nContext:\n{context[:5000]}\n\n"
                    "Extract entities and relationships."
                )}],
                system_prompt=EXTRACTION_SYSTEM,
            )
        except ValueError:
            return {"entities": [], "relationships": []}

    def _extract_paths(self, G: nx.DiGraph, entities: list[dict]) -> list[str]:
        """Find short paths between entity pairs."""
        paths = []
        nodes = list(G.nodes)
        if len(nodes) < 2:
            return paths

        # Find paths between pairs of entities (up to 5 pairs)
        checked = 0
        for i, src in enumerate(nodes):
            for tgt in nodes[i + 1:]:
                if checked >= 5:
                    break
                try:
                    path = nx.shortest_path(G, src, tgt)
                    if 2 <= len(path) <= 4:
                        # Format path with relationships
                        parts = []
                        for j in range(len(path) - 1):
                            edge = G.edges[path[j], path[j + 1]]
                            parts.append(
                                f"{path[j]} --[{edge.get('relation', '?')}]--> {path[j+1]}"
                            )
                        paths.append(" → ".join(parts))
                        checked += 1
                except nx.NetworkXNoPath:
                    continue
            if checked >= 5:
                break

        return paths

    def _summarize_graph(self, paths: list[str], relationships: list[dict]) -> str:
        if not paths and not relationships:
            return "No significant relationships found."

        summary_parts = []
        if relationships:
            summary_parts.append("Key relationships:")
            for r in relationships[:8]:
                summary_parts.append(
                    f"  • {r['source']} → {r['relation']} → {r['target']}"
                )
        if paths:
            summary_parts.append("Multi-hop paths:")
            for p in paths:
                summary_parts.append(f"  • {p}")

        return "\n".join(summary_parts)
