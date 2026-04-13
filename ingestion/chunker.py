"""
CogInfera — Semantic / Structure-Aware Chunker
"""

import re
import config


class SemanticChunker:
    """Split document text into semantically coherent chunks.

    Strategy:
    1. Split by structural cues (headings, double-newlines).
    2. Merge small segments until they hit the target size.
    3. Add overlap between consecutive chunks.
    """

    HEADING_RE = re.compile(
        r"^(?:"
        r"(?:chapter|section|part)\s+\d+"         # "Chapter 1", "Section 3"
        r"|(?:\d+\.)+\s+"                          # "1.2 Title"
        r"|[A-Z][A-Z ]{3,}"                        # "ALL-CAPS HEADING"
        r")",
        re.IGNORECASE | re.MULTILINE,
    )

    def __init__(
        self,
        chunk_size: int = config.CHUNK_SIZE,
        chunk_overlap: int = config.CHUNK_OVERLAP,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # ── Public API ──────────────────────────────────────────────────
    def chunk_pages(self, pages: list[dict]) -> list[dict]:
        """Chunk a list of page dicts into semantic chunks.

        Returns list of dicts with keys:
            text, document, start_page, end_page, section, position
        """
        # Combine all page text with page markers
        segments = self._split_into_segments(pages)
        chunks = self._merge_segments(segments)
        return chunks

    # ── Internals ───────────────────────────────────────────────────
    def _split_into_segments(self, pages: list[dict]) -> list[dict]:
        """Split pages into small segments at structural boundaries."""
        segments = []
        current_section = "Introduction"

        for page in pages:
            paragraphs = re.split(r"\n{2,}", page["text"])
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                # Detect headings
                if self.HEADING_RE.match(para):
                    current_section = para.split("\n")[0].strip()[:120]

                segments.append({
                    "text": para,
                    "document": page["document"],
                    "page_num": page["page_num"],
                    "section": current_section,
                })

        return segments

    def _merge_segments(self, segments: list[dict]) -> list[dict]:
        """Merge small segments into chunks of ~chunk_size words."""
        chunks = []
        buffer_texts = []
        buffer_pages = []
        buffer_section = ""
        word_count = 0

        for seg in segments:
            seg_words = len(seg["text"].split())

            # If adding this segment exceeds target, flush buffer
            if word_count + seg_words > self.chunk_size and buffer_texts:
                chunks.append(self._make_chunk(
                    buffer_texts, buffer_pages, buffer_section,
                    len(chunks), segments[0]["document"],
                ))
                # Overlap: keep last portion of buffer
                overlap_texts, overlap_pages = self._get_overlap(
                    buffer_texts, buffer_pages
                )
                buffer_texts = overlap_texts
                buffer_pages = overlap_pages
                word_count = sum(len(t.split()) for t in buffer_texts)

            buffer_texts.append(seg["text"])
            buffer_pages.append(seg["page_num"])
            buffer_section = seg["section"]
            word_count += seg_words

        # Flush remaining
        if buffer_texts:
            chunks.append(self._make_chunk(
                buffer_texts, buffer_pages, buffer_section,
                len(chunks), segments[0]["document"] if segments else "",
            ))

        return chunks

    def _make_chunk(
        self, texts: list[str], pages: list[int],
        section: str, position: int, document: str,
    ) -> dict:
        return {
            "text": "\n\n".join(texts),
            "document": document,
            "start_page": min(pages),
            "end_page": max(pages),
            "section": section,
            "position": position,
        }

    def _get_overlap(
        self, texts: list[str], pages: list[int],
    ) -> tuple[list[str], list[int]]:
        """Return the tail of texts/pages that fits within overlap size."""
        overlap_texts = []
        overlap_pages = []
        count = 0
        for t, p in zip(reversed(texts), reversed(pages)):
            wc = len(t.split())
            if count + wc > self.chunk_overlap:
                break
            overlap_texts.insert(0, t)
            overlap_pages.insert(0, p)
            count += wc
        return overlap_texts, overlap_pages
