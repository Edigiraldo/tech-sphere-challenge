"""Retrieval-Augmented Generation (RAG) module.

Owns document ingestion (extract, chunk, embed, store) and retrieval
(embed query, similarity search, return chunks with source metadata).
Does not own document lifecycle — that is the responsibility of the
``documents/`` module.
"""

from backend.rag.config import RagConfig
from backend.rag.ingestion import ingest_document
from backend.rag.retrieval import retrieve

__all__ = ["RagConfig", "ingest_document", "retrieve"]
