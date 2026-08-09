"""Document registry reconciliation and validation.

Compares ChromaDB indexed document IDs against the SQLite document registry
to detect:

- **Orphaned ChromaDB chunks**: chunks whose ``document_id`` has no
  corresponding SQLite row (or whose row is ``DELETED`` but chunks were not
  purged).
- **Missing ChromaDB entries**: SQLite documents with status ``READY`` that
  have no indexed chunks.

This module is called explicitly — it never runs automatically at startup.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from backend.documents.models import DocumentStatus
from backend.persistence.chroma import ChromaStore
from backend.persistence.sqlite import (
    get_all_documents,
    init_sqlite,
)

logger = logging.getLogger(__name__)


@dataclass
class ReconciliationResult:
    """Result of a reconciliation pass between SQLite registry and ChromaDB.

    Attributes:
        orphaned_chroma_ids: ``document_id`` values that exist in ChromaDB
            but are either missing from the SQLite registry or have status
            ``DELETED`` (chunks not purged on delete).
        missing_chroma_ids: ``document_id`` values in the SQLite registry
            with status ``READY`` (or ``PROCESSING``) that have zero indexed
            chunks in ChromaDB.
        total_sqlite_docs: Total number of SQLite document rows examined.
        total_chroma_ids: Number of unique ``document_id`` values in ChromaDB.
    """

    orphaned_chroma_ids: set[str] = field(default_factory=set)
    missing_chroma_ids: set[str] = field(default_factory=set)
    total_sqlite_docs: int = 0
    total_chroma_ids: int = 0

    @property
    def is_clean(self) -> bool:
        """``True`` when no orphaned or missing IDs are detected."""
        return not self.orphaned_chroma_ids and not self.missing_chroma_ids

    @property
    def orphaned_count(self) -> int:
        """Number of orphaned ChromaDB document IDs."""
        return len(self.orphaned_chroma_ids)

    @property
    def missing_count(self) -> int:
        """Number of SQLite documents missing from ChromaDB."""
        return len(self.missing_chroma_ids)


def reconcile(
    store: ChromaStore,
    db_path: str | Path,
) -> ReconciliationResult:
    """Compare ChromaDB indexed IDs against the SQLite document registry.

    Args:
        store: An initialised ``ChromaStore``.
        db_path: Path to the SQLite database file.

    Returns:
        A ``ReconciliationResult`` describing any inconsistencies found.
    """
    # Ensure SQLite is initialised for the given path
    init_sqlite(db_path)

    # Collect all SQLite documents
    all_docs = get_all_documents()
    sqlite_ids: dict[str, DocumentStatus] = {
        d.document_id: d.status for d in all_docs
    }

    # Collect all ChromaDB document IDs
    chroma_ids = store.get_all_document_ids()

    # Orphaned: in ChromaDB but not in SQLite, OR in SQLite but DELETED
    orphaned: set[str] = set()
    for cid in chroma_ids:
        if cid not in sqlite_ids:
            orphaned.add(cid)
        elif sqlite_ids[cid] == DocumentStatus.DELETED:
            orphaned.add(cid)

    # Missing: in SQLite with READY or PROCESSING status but not in ChromaDB
    missing: set[str] = set()
    active_statuses = {DocumentStatus.READY, DocumentStatus.PROCESSING}
    for sid, status in sqlite_ids.items():
        if status in active_statuses and sid not in chroma_ids:
            missing.add(sid)

    logger.info(
        "Reconciliation complete: %d SQLite docs, %d ChromaDB IDs — "
        "%d orphaned, %d missing.",
        len(sqlite_ids),
        len(chroma_ids),
        len(orphaned),
        len(missing),
    )

    return ReconciliationResult(
        orphaned_chroma_ids=orphaned,
        missing_chroma_ids=missing,
        total_sqlite_docs=len(sqlite_ids),
        total_chroma_ids=len(chroma_ids),
    )


def clean_orphaned_chunks(store: ChromaStore, db_path: str | Path) -> set[str]:
    """Delete all orphaned ChromaDB chunks detected by reconciliation.

    Orphaned chunks are those whose ``document_id`` has no matching SQLite
    registry entry or whose entry is ``DELETED`` (chunks not purged).

    Args:
        store: An initialised ``ChromaStore``.
        db_path: Path to the SQLite database file.

    Returns:
        The set of ``document_id`` values that were cleaned.
    """
    result = reconcile(store, db_path)
    cleaned: set[str] = set()

    for cid in result.orphaned_chroma_ids:
        deleted = store.delete_document_chunks(cid)
        cleaned.add(cid)
        logger.info(
            "Cleaned %d orphaned chunks for document_id=%r.",
            len(deleted),
            cid,
        )

    return cleaned
