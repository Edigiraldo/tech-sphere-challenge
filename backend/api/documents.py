"""Document lifecycle REST endpoints.

POST   /documents           — Upload a clinical PDF.
GET    /documents           — List all documents (optional status filter).
DELETE /documents/{id}      — Delete a document and purge its indexed chunks.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

from backend.documents.models import Document, DocumentStatus
from backend.documents.service import DocumentService
from backend.rag.config import RagConfig

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
            error_message=doc.error_message,
        )


class UploadResponse(BaseModel):
    """Response returned after a successful (or failed) document upload."""

    document_id: str
    filename: str
    status: str
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
