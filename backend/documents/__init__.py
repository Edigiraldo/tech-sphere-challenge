"""Document lifecycle module (``backend.documents``).

This module owns document upload, listing, processing status tracking, and
deletion. It orchestrates metadata in SQLite and delegates to ``rag/`` for
ingestion and ChromaDB chunk purging. It never calls ``rag/`` for retrieval.

Public interface:
    * ``Document``, ``DocumentStatus`` — domain models.
    * ``DocumentService`` — business logic (upload, list, delete); imported
      directly from ``backend.documents.service``.
    * ``reconcile`` — compare ChromaDB indexed IDs against SQLite registry.
    * ``clean_orphaned_chunks`` — delete orphaned ChromaDB chunks.
    * ``ReconciliationResult`` — structured reconciliation output.
"""

from backend.documents.models import Document, DocumentStatus
from backend.documents.reconciliation import (
    ReconciliationResult,
    clean_orphaned_chunks,
    reconcile,
)

__all__ = [
    "Document",
    "DocumentStatus",
    "ReconciliationResult",
    "clean_orphaned_chunks",
    "reconcile",
]

