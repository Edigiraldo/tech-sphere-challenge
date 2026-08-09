"""Document lifecycle business logic.

Orchestrates SQLite metadata and RAG ingestion/deletion. The ``DocumentService``
class is the single entry point for upload, list, and delete operations.

Duplicate uploads are detected via SHA-256 content hashing: when a file with
the same content is uploaded again and the existing registry record is active
(not ``DELETED``), the service returns the existing record without creating a
new one.  If the existing record has already been deleted, a new record is
created, preserving the audit trail of the previous record.
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

from backend.documents.models import Document, DocumentStatus
from backend.persistence.chroma import get_chroma_store
from backend.persistence.sqlite import (
    get_active_document_ids,
    get_all_documents,
    get_document_by_content_hash,
    get_document_by_id,
    init_sqlite,
    insert_document,
    update_document_status,
)
from backend.rag.config import RagConfig
from backend.rag.ingestion import ingest_document
from backend.rag.store import init_store

logger = logging.getLogger(__name__)


class DocumentService:
    """Manages the full document lifecycle.

    Responsibilities:
        * Upload: persist file → insert metadata → ingest into RAG.
        * List: query all document metadata from SQLite.
        * Delete: purge ChromaDB chunks → mark status as ``DELETED``.
    """

    def __init__(self, upload_dir: str | Path, db_path: str | Path) -> None:
        """Initialise the service with storage paths.

        Args:
            upload_dir: Directory where uploaded files are saved.
            db_path: Path to the SQLite database file.
        """
        self._upload_dir = Path(upload_dir)
        self._upload_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = Path(db_path)
        init_sqlite(self._db_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def upload(
        self, content: bytes, filename: str, config: RagConfig
    ) -> Document:
        """Upload and process a clinical document.

        1. Compute SHA-256 content hash for duplicate detection.
        2. If an active (non-deleted) record with the same hash exists,
           return it immediately — no new record or re-ingestion.
        3. Save the file to the upload directory.
        4. Insert metadata with ``PROCESSING`` status (including content_hash).
        5. Ingest into the RAG vector store via ``rag.ingestion``.
        6. Update status to ``READY`` (success) or ``FAILED`` (error).

        Duplicate policy: two uploads with identical content (matching
        SHA-256) that are both active share the same logical identity.
        The first non-deleted record is returned for all subsequent
        identical uploads.  If the original was deleted, a new record
        is created — deletion is unambiguous because the deleted record's
        ``document_id`` no longer matches any active duplicate query.

        Args:
            content: Raw file bytes.
            filename: Original filename (e.g. ``"guia.pdf"``).
            config: RAG configuration for ingestion.

        Returns:
            The ``Document`` metadata with its final status.
        """
        # Compute content hash for duplicate detection
        content_hash = hashlib.sha256(content).hexdigest()

        # Check for existing active record with same content
        existing = get_document_by_content_hash(content_hash)
        if existing is not None:
            logger.info(
                "Duplicate content detected (hash=%s) — returning existing "
                "document %s (%s).",
                content_hash[:12],
                existing.document_id,
                existing.filename,
            )
            return existing

        document_id = uuid.uuid4().hex

        # 1. Persist file — sanitize filename to prevent path traversal
        safe_filename = Path(filename).name
        file_path = self._upload_dir / f"{document_id}_{safe_filename}"
        file_path.write_bytes(content)

        # 2. Insert metadata
        now = datetime.now(timezone.utc)
        doc = Document(
            document_id=document_id,
            filename=filename,
            status=DocumentStatus.PROCESSING,
            uploaded_at=now,
            size_bytes=len(content),
            content_hash=content_hash,
        )
        insert_document(doc)
        logger.info(
            "Document %s (%s) metadata created (hash=%s) — starting ingestion.",
            document_id,
            filename,
            content_hash[:12],
        )

        # 3–4. Ingest + update status
        try:
            store = init_store(config)
            chunk_count = ingest_document(
                str(file_path), document_id, config, store
            )
            update_document_status(document_id, DocumentStatus.READY)
            doc.status = DocumentStatus.READY
            logger.info(
                "Document %s ingested: %d chunks, status=ready.",
                document_id,
                chunk_count,
            )
        except Exception as exc:
            error_msg = f"{type(exc).__name__}: {exc}"
            update_document_status(
                document_id, DocumentStatus.FAILED, error_msg
            )
            doc.status = DocumentStatus.FAILED
            doc.error_message = error_msg
            logger.error(
                "Document %s ingestion failed: %s", document_id, error_msg
            )

        return doc

    def list_all(self) -> list[Document]:
        """Return all document metadata rows, newest first."""
        return get_all_documents()

    def get_active_ids(self) -> set[str]:
        """Return the set of non-deleted ``document_id`` values.

        Used by the retrieval pipeline to filter out chunks from deleted
        or unregistered documents.
        """
        return get_active_document_ids()

    def delete(self, document_id: str, config: RagConfig) -> Document | None:
        """Delete a document: purge ChromaDB chunks and mark as ``DELETED``.

        Args:
            document_id: The document to delete.
            config: RAG configuration used to locate the ChromaDB store.

        Returns:
            The updated ``Document``, or ``None`` if *document_id* is not found.
        """
        doc = get_document_by_id(document_id)
        if doc is None:
            return None

        # 1. Purge ChromaDB chunks
        store = init_store(config)
        deleted_ids = store.delete_document_chunks(document_id)
        logger.info(
            "Deleted %d ChromaDB chunks for document %s.",
            len(deleted_ids),
            document_id,
        )

        # 2. Mark as deleted in SQLite
        update_document_status(document_id, DocumentStatus.DELETED)
        doc.status = DocumentStatus.DELETED

        return doc
