"""RAG retrieval: embed query → similarity search → annotated chunks."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from backend.persistence.chroma import ChromaStore
from backend.rag.config import RagConfig
from backend.rag.store import init_store, query_similar

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """A single retrieved chunk with source traceability."""

    chunk_id: str
    document_id: str
    source_filename: str
    chunk_index: int
    page_number: int
    text: str
    similarity: float

    def citation(self) -> str:
        """Return a human-readable citation string."""
        return f"{self.source_filename} (p. {self.page_number})"


@dataclass
class RetrievalResult:
    """Result of a RAG retrieval query."""

    query: str
    chunks: list[RetrievedChunk] = field(default_factory=list)

    @property
    def has_results(self) -> bool:
        """``True`` if any chunks were retrieved above the threshold."""
        return len(self.chunks) > 0

    @property
    def sources(self) -> list[str]:
        """Deduplicated list of ``document_id`` values cited in results."""
        return sorted({c.document_id for c in self.chunks})


def retrieve(
    query: str,
    config: RagConfig,
    store: ChromaStore | None = None,
    top_k: int | None = None,
) -> RetrievalResult:
    """Retrieve the top-k most relevant chunks for *query*.

    Args:
        query: Natural-language query in Spanish or English.
        config: RAG configuration.
        store: Optional pre-initialised ``ChromaStore``.
        top_k: Override the default number of chunks to retrieve.

    Returns:
        A ``RetrievalResult`` containing the matching chunks (if any).
    """
    if store is None:
        store = init_store(config)

    k = top_k if top_k is not None else config.retrieval_top_k
    threshold = config.similarity_threshold

    results = query_similar(
        store, query, top_k=k, similarity_threshold=threshold
    )

    chunks = [
        RetrievedChunk(
            chunk_id=str(r.get("chunk_id", "")),
            document_id=str(r.get("document_id", "")),
            source_filename=str(r.get("source_filename", "")),
            chunk_index=int(r.get("chunk_index", 0)),
            page_number=int(r.get("page_number", 1)),
            text=str(r.get("text", "")),
            similarity=float(r.get("similarity", 0.0)),
        )
        for r in results
    ]

    logger.info(
        "Retrieved %d chunks for query %r from %d source(s).",
        len(chunks),
        query[:80],
        len({c.document_id for c in chunks}),
    )

    return RetrievalResult(query=query, chunks=chunks)
