"""Document lifecycle module (``backend.documents``).

This module owns document upload, listing, processing status tracking, and
deletion. It orchestrates metadata in SQLite and delegates to ``rag/`` for
ingestion and ChromaDB chunk purging. It never calls ``rag/`` for retrieval.

Public interface:
    * ``Document``, ``DocumentStatus`` — domain models.
    * ``DocumentService`` — business logic (upload, list, delete); imported
      directly from ``backend.documents.service``.
    Reconciliation helpers are imported from ``backend.documents.reconciliation``
    by the administration API to avoid a package-level SQLite import cycle.
"""

from backend.documents.models import Document, DocumentStatus
__all__ = [
    "Document",
    "DocumentStatus",
]

