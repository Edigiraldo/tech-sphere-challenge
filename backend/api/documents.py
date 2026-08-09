"""Document lifecycle REST endpoints.

POST   /documents           — Upload a clinical PDF.
GET    /documents           — List all documents (optional status filter).
DELETE /documents/{id}      — Delete a document and purge its indexed chunks.
POST   /documents/reconcile — Validate and repair registry↔ChromaDB consistency.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

from backend.documents.models import Document, DocumentStatus
from backend.documents.reconciliation import (
    ReconciliationResult,
    clean_orphaned_chunks,
    reconcile,
)
from backend.documents.service import DocumentService
from backend.rag.config import RagConfig
from backend.rag.store import init_store

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

_rag_config: RagConfig | None = None
_service: DocumentService | None = None


def _get_rag_config() -> RagConfig:
    """Return the cached RAG configuration singleton."""
    global _rag_config
    if _rag_config is None:
        _rag_config = RagConfig()
    return _rag_config


def _get_service() -> DocumentService:
    """Return the cached DocumentService singleton."""
    global _service
    if _service is None:
        upload_dir = Path(os.getenv("DOCUMENTS_UPLOAD_DIR", "uploads"))
        db_path = Path(os.getenv("DOCUMENTS_DB_PATH", "data/documents.db"))
        _service = DocumentService(upload_dir=upload_dir, db_path=db_path)
    return _service


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

documents_router = APIRouter(prefix="/documents", tags=["documents"])


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class DocumentResponse(BaseModel):
    """Public representation of a document's metadata."""

    document_id: str = Field(..., description="Stable unique identifier")
    filename: str = Field(..., description="Original filename")
    status: str = Field(..., description="Lifecycle status")
    uploaded_at: str = Field(..., description="UTC ISO-8601 upload timestamp")
    size_bytes: int = Field(..., ge=0, description="File size in bytes")
    content_hash: Optional[str] = Field(
        None, description="SHA-256 hex digest of the file content"
    )
    error_message: Optional[str] = Field(
        None, description="Error detail (present only when status is 'failed')"
    )

    @classmethod
    def from_domain(cls, doc: Document) -> "DocumentResponse":
        """Build a response model from a domain ``Document``."""
        return cls(
            document_id=doc.document_id,
            filename=doc.filename,
            status=doc.status.value,
            uploaded_at=doc.uploaded_at.isoformat(),
            size_bytes=doc.size_bytes,
            content_hash=doc.content_hash,
            error_message=doc.error_message,
        )


class UploadResponse(BaseModel):
    """Response returned after a successful (or failed) document upload."""

    document_id: str
    filename: str
    status: str
    content_hash: Optional[str] = Field(
        None, description="SHA-256 hex digest of the file content"
    )
    message: str


class ListResponse(BaseModel):
    """Response returned when listing documents."""

    documents: list[DocumentResponse]
    total: int = Field(..., ge=0)


class DeleteResponse(BaseModel):
    """Response returned after a document deletion."""

    document_id: str
    status: str
    message: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@documents_router.post("", response_model=UploadResponse, status_code=201)
async def upload_document(
    file: UploadFile = File(...),
) -> UploadResponse:
    """Upload a clinical PDF document for RAG ingestion.

    The file is saved to disk, metadata is persisted in SQLite, and the
    document is ingested into ChromaDB via the RAG pipeline.  On success
    the status is ``ready``; on processing failure the status is ``failed``
    with an ``error_message``.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required.")

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400, detail="Only PDF files are accepted."
        )

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file.")

    service = _get_service()
    config = _get_rag_config()

    doc = service.upload(content, file.filename, config)

    if doc.status == DocumentStatus.READY:
        message = "Document uploaded and processed successfully."
    else:
        message = f"Document upload failed: {doc.error_message}"

    return UploadResponse(
        document_id=doc.document_id,
        filename=doc.filename,
        status=doc.status.value,
        content_hash=doc.content_hash,
        message=message,
    )


@documents_router.get("", response_model=ListResponse)
async def list_documents(
    status: Optional[str] = Query(
        None,
        description="Filter by status (pending, processing, ready, failed, deleted)",
    ),
) -> ListResponse:
    """List all document metadata, optionally filtered by status.

    Documents are returned newest-first.  *status* must be one of
    ``pending``, ``processing``, ``ready``, ``failed``, or ``deleted``.
    Returns a 400 error for any other value.
    """
    if status is not None:
        try:
            DocumentStatus(status)
        except ValueError:
            valid = ", ".join(sorted(s.value for s in DocumentStatus))
            raise HTTPException(
                status_code=400,
                detail=f"Invalid status '{status}'. Must be one of: {valid}.",
            )

    service = _get_service()
    docs = service.list_all()

    if status is not None:
        docs = [d for d in docs if d.status.value == status]

    return ListResponse(
        documents=[DocumentResponse.from_domain(d) for d in docs],
        total=len(docs),
    )


@documents_router.delete("/{document_id}", response_model=DeleteResponse)
async def delete_document(document_id: str) -> DeleteResponse:
    """Delete a document and purge all its indexed chunks from ChromaDB.

    The document's SQLite metadata row is marked with status ``deleted``.
    All ChromaDB chunks matching the ``document_id`` are removed so no
    orphaned chunks remain.
    """
    service = _get_service()
    config = _get_rag_config()

    doc = service.delete(document_id, config)

    if doc is None:
        raise HTTPException(
            status_code=404,
            detail=f"Document '{document_id}' not found.",
        )

    return DeleteResponse(
        document_id=doc.document_id,
        status=doc.status.value,
        message="Document deleted and indexed chunks purged.",
    )


# ---------------------------------------------------------------------------
# Reconciliation response models
# ---------------------------------------------------------------------------


class ReconcileResponse(BaseModel):
    """Response returned after a reconciliation run."""

    total_sqlite_docs: int = Field(
        ..., ge=0, description="Total SQLite document rows examined"
    )
    total_chroma_ids: int = Field(
        ..., ge=0, description="Unique document_id values in ChromaDB"
    )
    orphaned_chroma_ids: list[str] = Field(
        default_factory=list,
        description="Document IDs in ChromaDB but not registered in SQLite "
        "(or soft-deleted with lingering chunks)",
    )
    missing_chroma_ids: list[str] = Field(
        default_factory=list,
        description="SQLite documents with status 'ready'/'processing' that "
        "have no indexed chunks in ChromaDB",
    )
    is_clean: bool = Field(
        ..., description="True when no inconsistencies were detected"
    )
    cleaned_ids: Optional[list[str]] = Field(
        None,
        description="Document IDs whose orphaned chunks were cleaned "
        "(only present when clean=true query param is set)",
    )


# ---------------------------------------------------------------------------
# Reconciliation endpoint
# ---------------------------------------------------------------------------


@documents_router.post("/reconcile", response_model=ReconcileResponse)
async def reconcile_documents(
    clean: bool = Query(
        False,
        description=(
            "If true, delete orphaned ChromaDB chunks whose document_id "
            "is not registered or whose registry row is deleted."
        ),
    ),
) -> ReconcileResponse:
    """Validate consistency between the SQLite document registry and ChromaDB.

    Detects:

    - **Orphaned ChromaDB chunks**: chunks whose ``document_id`` is missing
      from the SQLite registry or whose registry row is ``deleted`` (chunks
      not purged on deletion).
    - **Missing ChromaDB entries**: SQLite documents with status ``ready``
      or ``processing`` that have zero indexed chunks.

    When ``clean=true``, orphaned chunks are also deleted from ChromaDB.
    """
    config = _get_rag_config()
    store = init_store(config)

    db_path = Path(os.getenv("DOCUMENTS_DB_PATH", "data/documents.db"))

    result: ReconciliationResult = reconcile(store, db_path)

    cleaned_ids: list[str] | None = None
    if clean:
        cleaned_set = clean_orphaned_chunks(store, db_path)
        cleaned_ids = sorted(cleaned_set)

    return ReconcileResponse(
        total_sqlite_docs=result.total_sqlite_docs,
        total_chroma_ids=result.total_chroma_ids,
        orphaned_chroma_ids=sorted(result.orphaned_chroma_ids),
        missing_chroma_ids=sorted(result.missing_chroma_ids),
        is_clean=result.is_clean,
        cleaned_ids=cleaned_ids,
    )
