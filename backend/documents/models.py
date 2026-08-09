"""Document domain models.

Pure data classes and enums — no persistence, RAG, or HTTP dependencies.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime


class DocumentStatus(str, enum.Enum):
    """Lifecycle status of a clinical document."""

    PENDING = "pending"
    """Document metadata created; ingestion not yet started."""

    PROCESSING = "processing"
    """Ingestion into the RAG vector store is in progress."""

    READY = "ready"
    """Ingestion completed successfully; chunks are available for retrieval."""

    FAILED = "failed"
    """Ingestion failed; see ``error_message`` for details."""

    DELETED = "deleted"
    """Document has been deleted and its indexed chunks have been purged."""


@dataclass
class Document:
    """Metadata for a single clinical document in the lifecycle system.

    The ``document_id`` is the stable identifier attached to every ChromaDB
    chunk so that deletion can purge all chunks belonging to this document.
    """

    document_id: str
    """Stable unique identifier (UUID hex string)."""

    filename: str
    """Original filename as provided by the uploader."""

    status: DocumentStatus
    """Current lifecycle status."""

    uploaded_at: datetime
    """UTC timestamp when the document was first received."""

    size_bytes: int
    """Size of the uploaded file in bytes."""

    content_hash: str | None = None
    """SHA-256 hex digest of the file content (for duplicate detection and
    idempotent ingestion).  ``None`` for legacy records created before the
    content-hash policy was introduced."""

    error_message: str | None = None
    """Human-readable error detail, set only when ``status == FAILED``."""
