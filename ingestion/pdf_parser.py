"""
CogInfera — PDF Parser (PyMuPDF)
"""

import os
import fitz  # PyMuPDF


def parse_pdf(pdf_path: str) -> list[dict]:
    """Parse a PDF and return a list of page dicts.

    Each dict contains:
        - page_num (int): 1-based page number
        - text (str): extracted text
        - document (str): source filename
    """
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc_name = os.path.basename(pdf_path)
    pages = []

    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc):
            text = page.get_text("text")
            if text.strip():
                pages.append({
                    "page_num": i + 1,
                    "text": text,
                    "document": doc_name,
                })

    return pages
