"""Document ingestion pipeline: extract → chunk → embed → store.

Orchestrates the full ingestion workflow for a single document.
Includes density scanning of document text for injection-like patterns
(via ``backend.llm.injection.scan_document_density``) — density warnings
are logged but never reject legitimate clinical documents.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from backend.llm.injection import scan_document_density
from backend.persistence.chroma import ChromaStore
from backend.rag.chunking import Chunk, chunk_pages
from backend.rag.config import RagConfig
from backend.rag.extract import extract_pages
from backend.rag.store import add_chunks, init_store

logger = logging.getLogger(__name__)


def ingest_document(
    file_path: str | Path,
    document_id: str,
    config: RagConfig,
    store: ChromaStore | None = None,
) -> int:
    """Ingest a single document into the RAG vector store.

    Workflow:
        1. Extract text pages from the PDF.
        2. Split each page into overlapping chunks.
        3. Embed chunks via the ChromaDB embedding function (BGE-M3).
        4. Store in the ChromaDB ``clinical_knowledge`` collection.

    Args:
        file_path: Path to the PDF file.
        document_id: Stable identifier for this document (used for later
            deletion and source citation).
        config: RAG configuration.
        store: Optional pre-initialised ``ChromaStore``. If ``None``, a new
            store is initialised from *config*.

    Returns:
        Number of chunks ingested.

    Raises:
        ExtractionError: If the PDF cannot be read.
    """
    path = Path(file_path)
    source_filename = path.name

    if store is None:
        store = init_store(config)

    logger.info("Ingesting %r (document_id=%s) …", source_filename, document_id)

    # 1. Extract pages
    pages = list(extract_pages(path))
    if not pages:
        logger.warning("No text extracted from %r", source_filename)
        return 0

    # --- Density scan (warning only — never reject) ---
    all_text = "\n".join(str(p["text"]) for p in pages)
    density_result = scan_document_density(all_text, filename=source_filename)
    if density_result.warning:
        logger.warning(
            "Document %r density warning: %d/%d lines matched (%.2f%%). "
            "Document ingestion proceeds normally — this is NOT a rejection.",
            source_filename,
            density_result.match_count,
            density_result.total_lines,
            density_result.ratio * 100,
        )

    # 2. Chunk
    chunks = list(chunk_pages(pages, document_id, source_filename, config))
    if not chunks:
        logger.warning("No chunks produced for %r", source_filename)
        return 0

    # 3–4. Embed + store (ChromaDB handles embedding via the collection's
    # embedding function)
    chunk_ids = [c["chunk_id"] for c in chunks]
    texts = [c["text"] for c in chunks]
    ingested_at = datetime.now(timezone.utc).isoformat()

    metadatas = [
        {
            "document_id": c["document_id"],
            "source_filename": c["source_filename"],
            "chunk_index": c["chunk_index"],
            "page_number": c["page_number"],
            "ingested_at": ingested_at,
        }
        for c in chunks
    ]

    add_chunks(store, chunk_ids, texts, metadatas)

    logger.info(
        "Ingested %d chunks from %r (document_id=%s).",
        len(chunks),
        source_filename,
        document_id,
    )

    return len(chunks)
