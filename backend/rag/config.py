"""RAG configuration — all tunables in one place.

Values are read from environment variables with sensible defaults so that
no code changes are needed for deployment tuning.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class RagConfig:
    """Immutable configuration for the RAG pipeline.

    All values can be overridden via environment variables.
    """

    # -- Embedding model ---------------------------------------------------
    embedding_model: str = field(
        default_factory=lambda: os.getenv(
            "RAG_EMBEDDING_MODEL", "BAAI/bge-m3"
        )
    )
    """HuggingFace model ID for the embedding model (BGE-M3, 1024-dim)."""

    # -- Storage ------------------------------------------------------------
    chroma_persist_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("RAG_CHROMA_DIR", "chroma_data")
        )
    )
    """Filesystem directory for ChromaDB persistent storage."""

    collection_name: str = field(
        default_factory=lambda: os.getenv(
            "RAG_COLLECTION_NAME", "clinical_knowledge"
        )
    )
    """ChromaDB collection name."""

    # -- Chunking -----------------------------------------------------------
    chunk_size: int = field(
        default_factory=lambda: int(os.getenv("RAG_CHUNK_SIZE", "800"))
    )
    """Target character count per chunk."""

    chunk_overlap: int = field(
        default_factory=lambda: int(os.getenv("RAG_CHUNK_OVERLAP", "150"))
    )
    """Character overlap between consecutive chunks."""

    # -- Retrieval ----------------------------------------------------------
    retrieval_top_k: int = field(
        default_factory=lambda: int(os.getenv("RAG_TOP_K", "5"))
    )
    """Default number of chunks to retrieve per query."""

    similarity_threshold: float = field(
        default_factory=lambda: float(
            os.getenv("RAG_SIMILARITY_THRESHOLD", "0.25")
        )
    )
    """Minimum similarity score for a chunk to be included in results.

    Chunks below this threshold are filtered out. The default of 0.25
    provides a reasonable balance between recall and precision for
    clinical BGE-M3 embeddings.  Values below 0.10 are treated as
    practically irrelevant.
    """

    min_chunks_for_answer: int = field(
        default_factory=lambda: int(
            os.getenv("RAG_MIN_CHUNKS", "2")
        )
    )
    """Minimum number of chunks above threshold required before the LLM
    is invoked.  When fewer chunks pass the similarity threshold, the
    response falls back to ``insufficient_knowledge`` without calling
    the LLM, preventing weak/unreliable retrieval from reaching the
    model.
    """

    min_avg_similarity: float = field(
        default_factory=lambda: float(
            os.getenv("RAG_MIN_AVG_SIMILARITY", "0.30")
        )
    )
    """Minimum average similarity across the retrieved chunks required
    before the LLM is invoked.  This prevents a single borderline chunk
    from reaching the model while all others are far below threshold.
    Set to 0.0 to disable this check.
    """
