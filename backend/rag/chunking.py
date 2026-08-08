"""Text chunking with configurable size and overlap.

Implements a simple fixed-size sliding-window chunker. Sentence-boundary
awareness can be added later as an enhancement (tracked in ARCHITECTURE.md
open decision D6).
"""

from __future__ import annotations

import logging
import uuid
from typing import Iterator, TypedDict

from backend.rag.config import RagConfig

logger = logging.getLogger(__name__)


class Chunk(TypedDict):
    """A single text chunk ready for embedding and storage."""

    chunk_id: str
    """Unique identifier for this chunk."""

    document_id: str
    """The document this chunk belongs to."""

    source_filename: str
    """Original filename of the source document."""

    chunk_index: int
    """Zero-based index of this chunk within the document."""

    page_number: int
    """Page number where the chunk's text originates (1-based)."""

    text: str
    """The chunk's text content."""


def _overlap_split(
    text: str, chunk_size: int, chunk_overlap: int
) -> list[str]:
    """Split *text* into overlapping chunks of approximately *chunk_size*
    characters with *chunk_overlap* overlap."""
    if chunk_overlap >= chunk_size:
        logger.warning(
            "chunk_overlap (%d) >= chunk_size (%d); clamping overlap to half the size",
            chunk_overlap,
            chunk_size,
        )
        chunk_overlap = chunk_size // 2

    chunks: list[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunks.append(text[start:end])
        if end == text_len:
            break
        start = end - chunk_overlap

    return chunks


def chunk_pages(
    pages: list[dict[str, object]],
    document_id: str,
    source_filename: str,
    config: RagConfig,
) -> Iterator[Chunk]:
    """Split extracted pages into overlapping text chunks.

    Each page is split independently with the chunk settings from *config*.
    Short pages are not merged — a page shorter than ``chunk_size`` produces
    a single chunk.

    Args:
        pages: List of ``{"page_number": int, "text": str}`` dicts from
            ``extract.extract_pages``.
        document_id: Stable identifier for the source document.
        source_filename: Original filename (for metadata).
        config: RAG configuration with chunk_size and chunk_overlap.

    Yields:
        ``Chunk`` dicts ready for embedding and storage.
    """
    chunk_index = 0

    for page in pages:
        page_number = int(page["page_number"])
        page_text = str(page["text"]).strip()

        if not page_text:
            continue

        splits = _overlap_split(page_text, config.chunk_size, config.chunk_overlap)

        for split_text in splits:
            chunk: Chunk = {
                "chunk_id": str(uuid.uuid4()),
                "document_id": document_id,
                "source_filename": source_filename,
                "chunk_index": chunk_index,
                "page_number": page_number,
                "text": split_text,
            }
            chunk_index += 1
            yield chunk
