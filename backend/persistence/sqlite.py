"""SQLite access layer for document metadata and application data.

Provides a thin wrapper around SQLite for the ``documents`` table. Owned
by the ``documents/`` module; other modules must go through ``documents/``
for all document-metadata interactions.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from backend.documents.models import Document, DocumentStatus

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_db_path: Optional[Path] = None
"""Path to the SQLite database file. Set by ``init_sqlite``."""


def init_sqlite(db_path: str | Path) -> None:
    """Initialise (or re-initialise) the SQLite database at *db_path*.

    Creates the ``documents`` table if it does not exist. Safe to call
    multiple times — subsequent calls re-point the module to a new database
    (useful for tests that use a temp database).
    """
    global _db_path

    _db_path = Path(db_path)
    _db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = _get_conn()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                document_id  TEXT PRIMARY KEY,
                filename     TEXT    NOT NULL,
                status       TEXT    NOT NULL DEFAULT 'pending',
                uploaded_at  TEXT    NOT NULL,
                size_bytes   INTEGER NOT NULL DEFAULT 0,
                error_message TEXT
            )
            """
        )
        conn.commit()
        logger.info("SQLite database initialised at %s", _db_path)
    finally:
        conn.close()


def _reset_sqlite() -> None:
    """Reset the module-level database path (for test teardown).

    After calling this, ``init_sqlite`` must be called again before any
    other function in this module can be used.
    """
    global _db_path
    _db_path = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_conn() -> sqlite3.Connection:
    """Return a new connection to the configured SQLite database.

    Raises ``RuntimeError`` if ``init_sqlite`` has not been called.
    """
    if _db_path is None:
        raise RuntimeError(
            "SQLite database not initialised. Call init_sqlite() first."
        )
    conn = sqlite3.connect(str(_db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _row_to_document(row: sqlite3.Row) -> Document:
    """Convert a ``sqlite3.Row`` to a ``Document`` domain object."""
    return Document(
        document_id=row["document_id"],
        filename=row["filename"],
        status=DocumentStatus(row["status"]),
        uploaded_at=datetime.fromisoformat(row["uploaded_at"]),
        size_bytes=row["size_bytes"],
        error_message=row["error_message"],
    )


# ---------------------------------------------------------------------------
# Public CRUD operations
# ---------------------------------------------------------------------------


def insert_document(doc: Document) -> None:
    """Insert a new document metadata row.

    Args:
        doc: The document to insert. ``document_id`` must be unique.
    """
    conn = _get_conn()
    try:
        conn.execute(
            """INSERT INTO documents
               (document_id, filename, status, uploaded_at, size_bytes, error_message)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                doc.document_id,
                doc.filename,
                doc.status.value,
                doc.uploaded_at.isoformat(),
                doc.size_bytes,
                doc.error_message,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def update_document_status(
    document_id: str,
    status: DocumentStatus,
    error_message: str | None = None,
) -> None:
    """Update the status (and optionally error_message) for a document.

    Args:
        document_id: The document to update.
        status: New lifecycle status.
        error_message: Human-readable error detail (meaningful only for
            ``FAILED`` status). Pass ``None`` to leave unchanged.
    """
    conn = _get_conn()
    try:
        if error_message is not None:
            conn.execute(
                "UPDATE documents SET status = ?, error_message = ? "
                "WHERE document_id = ?",
                (status.value, error_message, document_id),
            )
        else:
            conn.execute(
                "UPDATE documents SET status = ? WHERE document_id = ?",
                (status.value, document_id),
            )
        conn.commit()
    finally:
        conn.close()


def get_all_documents() -> list[Document]:
    """Return all document metadata rows, newest first."""
    conn = _get_conn()
    try:
        rows = conn.execute(
            "SELECT * FROM documents ORDER BY uploaded_at DESC"
        ).fetchall()
        return [_row_to_document(r) for r in rows]
    finally:
        conn.close()


def get_document_by_id(document_id: str) -> Document | None:
    """Return the document with *document_id*, or ``None`` if not found."""
    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT * FROM documents WHERE document_id = ?", (document_id,)
        ).fetchone()
        return _row_to_document(row) if row is not None else None
    finally:
        conn.close()
