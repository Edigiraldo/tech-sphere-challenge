"""ChromaDB collection management for the RAG module.

Thin operations that the ingestion and retrieval pipelines use to interact
with the vector store. All access goes through ``persistence.ChromaStore``.
"""

from __future__ import annotations

import logging
from typing import Optional

from backend.persistence.chroma import ChromaStore, get_chroma_store
from backend.rag.config import RagConfig
from backend.rag.embeddings import BgeM3EmbeddingFunction

logger = logging.getLogger(__name__)


def init_store(config: RagConfig) -> ChromaStore:
    """Initialise (or retrieve) the ChromaStore singleton for *config*."""
    ef = BgeM3EmbeddingFunction(config)
    return get_chroma_store(
        persist_directory=config.chroma_persist_dir,
        embedding_function=ef,
        collection_name=config.collection_name,
    )


def add_chunks(
    store: ChromaStore,
    chunk_ids: list[str],
    texts: list[str],
    metadatas: list[dict[str, object]],
) -> None:
    """Add a batch of text chunks to the ChromaDB collection.

    Args:
        store: An initialised ``ChromaStore``.
        chunk_ids: Unique IDs (one per chunk).
        texts: Raw text for each chunk (embedding is done by the store).
        metadatas: Metadata dicts with ``document_id``, ``source_filename``,
            ``chunk_index``, ``page_number``.
    """
    collection = store.get_or_create_collection()
    collection.add(
        ids=chunk_ids,
        documents=texts,
        metadatas=metadatas,
    )
    logger.info("Added %d chunks to collection.", len(chunk_ids))


def query_similar(
    store: ChromaStore,
    query_text: str,
    top_k: int = 5,
    similarity_threshold: float = 0.0,
) -> list[dict[str, object]]:
    """Query the collection for chunks similar to *query_text*.

    Args:
        store: An initialised ``ChromaStore``.
        query_text: Natural-language query text.
        top_k: Maximum number of chunks to retrieve.
        similarity_threshold: Minimum similarity (0.0–1.0). Chunks below this
            threshold are filtered out. A value of 0.0 disables filtering.

    Returns:
        List of result dicts with keys: ``chunk_id``, ``document_id``,
        ``source_filename``, ``chunk_index``, ``page_number``,
        ``text``, ``similarity``.
    """
    collection = store.get_or_create_collection()
    results = collection.query(
        query_texts=[query_text],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    # ChromaDB returns lists-of-lists
    ids = results.get("ids", [[]])[0]
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    output: list[dict[str, object]] = []
    for i, chunk_id in enumerate(ids):
        distance = distances[i] if i < len(distances) else 1.0
        similarity = 1.0 - distance

        if similarity_threshold > 0 and similarity < similarity_threshold:
            continue

        meta = metadatas[i] if i < len(metadatas) else {}
        output.append(
            {
                "chunk_id": chunk_id,
                "document_id": meta.get("document_id", ""),
                "source_filename": meta.get("source_filename", ""),
                "chunk_index": meta.get("chunk_index", 0),
                "page_number": meta.get("page_number", 1),
                "text": documents[i] if i < len(documents) else "",
                "similarity": round(similarity, 4),
            }
        )

    return output
