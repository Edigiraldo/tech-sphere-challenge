"""Persistence layer: SQLite and ChromaDB access.

This package provides a clean boundary for the vector store (ChromaDB) used
by the RAG module. The RAG module owns the ChromaDB collection; other modules
must not write to it directly.
"""

from backend.persistence.chroma import (
    ChromaStore,
    get_chroma_store,
)

__all__ = ["ChromaStore", "get_chroma_store"]
