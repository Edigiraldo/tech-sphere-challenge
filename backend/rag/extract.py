"""Text extraction from PDF documents.

Uses pdfplumber for reliable text extraction with page-level metadata.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

import pdfplumber

logger = logging.getLogger(__name__)


class ExtractionError(Exception):
    """Raised when text extraction from a document fails."""


def extract_pages(file_path: str | Path) -> Iterator[dict[str, object]]:
    """Extract text page-by-page from a PDF file.

    Yields:
        ``{"page_number": int, "text": str}`` for each page that has
        non-empty text. Pages with no extractable text are skipped silently.

    Raises:
        ExtractionError: If the file cannot be opened or read as a PDF.
    """
    path = Path(file_path)

    if not path.is_file():
        raise ExtractionError(f"File not found: {path}")

    try:
        with pdfplumber.open(path) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                text = (page.extract_text() or "").strip()
                if text:
                    yield {"page_number": page_number, "text": text}
    except Exception as exc:
        raise ExtractionError(
            f"Failed to extract text from {path.name}: {exc}"
        ) from exc
