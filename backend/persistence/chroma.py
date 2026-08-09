"""ChromaDB access layer for the RAG module.

Provides a thin wrapper around ChromaDB's PersistentClient for collection
management and document operations. Owned by the `rag/` module; other
modules must go through `rag/` for all ChromaDB interactions.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.api import Collection
from chromadb.api.types import EmbeddingFunction

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public wrapper
# ---------------------------------------------------------------------------


class ChromaStore:
    """Manages a single ChromaDB persistent collection for clinical knowledge.

    The store uses ChromaDB's built-in embedding function support so that
    documents can be added with raw text and the embedding function handles
    vectorization transparently.

    Public interface:
        get_or_create_collection() -> Collection
        delete_document_chunks(document_id: str) -> list[str]
        count() -> int
    """

    def __init__(
        self,
        persist_directory: str | Path,
        embedding_function: EmbeddingFunction,
        collection_name: str = "clinical_knowledge",
    ) -> None:
        """Initialise the ChromaDB client and prepare the collection.

        Args:
            persist_directory: Filesystem path for ChromaDB's persistent storage.
            embedding_function: A ChromaDB-compatible embedding function (must
                implement ``__call__``).
            collection_name: Name of the ChromaDB collection to manage.
        """
        self._persist_directory = Path(persist_directory)
        self._collection_name = collection_name
        self._embedding_function = embedding_function
        self._client: Optional[chromadb.PersistentClient] = None
        self._collection: Optional[Collection] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_or_create_collection(self) -> Collection:
        """Return (creating if necessary) the configured ChromaDB collection."""
        if self._collection is not None:
            return self._collection

        client = self._get_client()
        self._collection = client.get_or_create_collection(
            name=self._collection_name,
            embedding_function=self._embedding_function,
            metadata={
                "hnsw:space": "cosine",
                "description": "Clinical knowledge chunks for RAG",
            },
        )
        logger.info(
            "ChromaDB collection %r ready (%d documents)",
            self._collection_name,
            self._collection.count(),
        )
        return self._collection

    def delete_document_chunks(self, document_id: str) -> list[str]:
        """Remove all chunks belonging to *document_id* from the collection.

        Returns:
            The list of deleted chunk IDs (may be empty).
        """
        collection = self.get_or_create_collection()
        # Find all chunks with the given document_id in metadata
        results = collection.get(where={"document_id": document_id})
        chunk_ids = results.get("ids", [])

        if chunk_ids:
            collection.delete(ids=chunk_ids)
            logger.info(
                "Deleted %d chunks for document_id=%r", len(chunk_ids), document_id
            )

        return chunk_ids

    def count(self) -> int:
        """Return the number of documents currently in the collection."""
        return self.get_or_create_collection().count()

    def get_all_document_ids(self) -> set[str]:
        """Return the set of all unique ``document_id`` values in the collection.

        Used by reconciliation to detect orphaned ChromaDB chunks whose
        ``document_id`` has no corresponding SQLite registry entry.
        """
        collection = self.get_or_create_collection()
        if collection.count() == 0:
            return set()

        # ChromaDB get() with no filter returns all items.  We only need
        # the metadata.document_id field(s), so we fetch metadatas only.
        result = collection.get(include=["metadatas"])
        metadatas = result.get("metadatas", [])
        if not metadatas:
            return set()

        return {
            meta["document_id"]
            for meta in metadatas
            if isinstance(meta, dict) and "document_id" in meta
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_client(self) -> chromadb.PersistentClient:
        if self._client is None:
            self._persist_directory.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=str(self._persist_directory),
                settings=chromadb.Settings(anonymized_telemetry=False),
            )
        return self._client


# ---------------------------------------------------------------------------
# Module-level singleton factory
# ---------------------------------------------------------------------------

_store: Optional[ChromaStore] = None


def get_chroma_store(
    persist_directory: str | Path | None = None,
    embedding_function: EmbeddingFunction | None = None,
    collection_name: str = "clinical_knowledge",
) -> ChromaStore:
    """Return the module-level ChromaStore singleton, creating it on first call.

    All arguments are only used on the first invocation; subsequent calls
    ignore them.
    """
    global _store  # noqa: PLW0603

    if _store is None:
        if persist_directory is None:
            raise ValueError("persist_directory is required on first call")
        if embedding_function is None:
            raise ValueError("embedding_function is required on first call")
        _store = ChromaStore(
            persist_directory=persist_directory,
            embedding_function=embedding_function,
            collection_name=collection_name,
        )

    return _store
