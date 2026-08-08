"""Embedding adapter — wraps sentence-transformers for BGE-M3.

Provides a ChromaDB-compatible ``EmbeddingFunction`` so the vector store can
auto-embed documents and queries.
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from sentence_transformers import SentenceTransformer

from backend.rag.config import RagConfig

logger = logging.getLogger(__name__)


@runtime_checkable
class Embedder(Protocol):
    """Protocol for an embedder that returns numpy arrays."""

    def encode(
        self, sentences: list[str], **kwargs: object
    ) -> "list[list[float]]": ...


class BgeM3EmbeddingFunction(EmbeddingFunction):
    """ChromaDB-compatible embedding function using BGE-M3 via sentence-transformers.

    See https://huggingface.co/BAAI/bge-m3 — 1024-dimensional embeddings
    with multilingual support (critical for Spanish content).
    """

    def __init__(self, config: RagConfig) -> None:
        model_name = config.embedding_model
        logger.info("Loading embedding model %r …", model_name)
        # trust_remote_code=True is required for the known BAAI/bge-m3 model
        # (HuggingFace custom modelling code). The identifier is pinned via
        # RagConfig.embedding_model and never supplied from untrusted input.
        self._model: Embedder = SentenceTransformer(
            model_name, trust_remote_code=True
        )
        logger.info("Embedding model %r ready.", model_name)

    def __call__(self, input: Documents) -> Embeddings:  # noqa: D105
        # ChromaDB passes a list of strings
        embeddings: "list[list[float]]" = self._model.encode(  # type: ignore[assignment]
            input, normalize_embeddings=True
        ).tolist()
        return embeddings
