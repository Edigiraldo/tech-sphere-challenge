"""Persistence layer: ChromaDB access.

This package provides a clean boundary for the vector store (ChromaDB)
used by the ``rag/`` module. The RAG module owns the ChromaDB collection;
other modules must not write to it directly.

SQLite document metadata is owned by the ``documents/`` module and is
imported directly from ``backend.persistence.sqlite`` where needed.
"""

from backend.persistence.chroma import (
    ChromaStore,
    get_chroma_store,
)

__all__ = [
    "ChromaStore",
    "get_chroma_store",
]
