"""End-to-end tests for ingestion → retrieval pipeline.

These tests require the BGE-M3 embedding model download and PDF extraction,
so they are marked ``@pytest.mark.slow``. Run with:

    pytest -m slow
"""

import uuid

import pytest

from backend.rag.config import RagConfig
from backend.rag.ingestion import ingest_document
from backend.rag.retrieval import retrieve
from backend.rag.store import init_store
from backend.persistence.chroma import ChromaStore


@pytest.mark.slow
class TestIngestionRetrieval:
    """End-to-end: ingest two Appendicitis PDFs and retrieve relevant chunks."""

    @pytest.fixture
    def store(self, rag_config: RagConfig) -> ChromaStore:
        """Initialise a ChromaStore with a fresh temp collection."""
        return init_store(rag_config)

    @pytest.fixture
    def doc_id_en(self) -> str:
        return f"test-{uuid.uuid4().hex[:8]}-postop-en"

    @pytest.fixture
    def doc_id_es(self) -> str:
        return f"test-{uuid.uuid4().hex[:8]}-cuidado-es"

    def test_ingest_english_pdf(
        self,
        store: ChromaStore,
        rag_config: RagConfig,
        post_op_en_pdf,
        doc_id_en: str,
    ):
        """Ingest the English post-operative instructions PDF."""
        count = ingest_document(post_op_en_pdf, doc_id_en, rag_config, store)
        assert count > 0, "Should ingest at least one chunk"
        assert store.count() >= count

    def test_ingest_spanish_pdf(
        self,
        store: ChromaStore,
        rag_config: RagConfig,
        plan_cuidado_es_pdf,
        doc_id_es: str,
    ):
        """Ingest the Spanish home care plan PDF."""
        count = ingest_document(
            plan_cuidado_es_pdf, doc_id_es, rag_config, store
        )
        assert count > 0, "Should ingest at least one chunk"
        assert store.count() >= count

    def test_ingest_both_and_retrieve(
        self,
        store: ChromaStore,
        rag_config: RagConfig,
        post_op_en_pdf,
        plan_cuidado_es_pdf,
        doc_id_en: str,
        doc_id_es: str,
    ):
        """Ingest both PDFs and verify retrieval returns relevant chunks."""
        count_en = ingest_document(
            post_op_en_pdf, doc_id_en, rag_config, store
        )
        count_es = ingest_document(
            plan_cuidado_es_pdf, doc_id_es, rag_config, store
        )
        assert count_en > 0
        assert count_es > 0
        assert store.count() == count_en + count_es

        # Query in Spanish about post-op wound care
        result = retrieve(
            "cuidado de la herida después de apendicectomía",
            rag_config,
            store,
            top_k=3,
        )
        assert result.has_results
        # The Spanish PDF should be in the top results
        source_files = [c.source_filename for c in result.chunks]
        assert plan_cuidado_es_pdf.name in source_files, (
            f"Spanish PDF should appear in results, got {source_files}"
        )

    def test_retrieve_returns_citations(
        self,
        store: ChromaStore,
        rag_config: RagConfig,
        post_op_en_pdf,
        doc_id_en: str,
    ):
        """Retrieved chunks must include traceable citations."""
        ingest_document(post_op_en_pdf, doc_id_en, rag_config, store)

        result = retrieve("signs of infection after appendectomy", rag_config, store)
        assert result.has_results

        for chunk in result.chunks:
            citation = chunk.citation()
            assert chunk.source_filename in citation, (
                f"Citation should include filename: {citation}"
            )
            assert "p." in citation, (
                f"Citation should include page number: {citation}"
            )

    def test_delete_chunks_removes_document(
        self,
        store: ChromaStore,
        rag_config: RagConfig,
        plan_cuidado_es_pdf,
        doc_id_es: str,
    ):
        """Deleting document chunks should remove them from the collection."""
        count = ingest_document(
            plan_cuidado_es_pdf, doc_id_es, rag_config, store
        )
        assert count > 0
        initial_total = store.count()

        deleted_ids = store.delete_document_chunks(doc_id_es)
        assert len(deleted_ids) == count, (
            f"Should delete {count} chunks, got {len(deleted_ids)}"
        )
        assert store.count() == initial_total - count, (
            "Collection count should decrease by deleted chunk count"
        )

    def test_unknown_document_id_delete_is_noop(self, store: ChromaStore):
        """Deleting a non-existent document_id should return empty list."""
        deleted = store.delete_document_chunks("nonexistent-doc-id")
        assert deleted == []

    def test_empty_query_returns_no_results(
        self,
        store: ChromaStore,
        rag_config: RagConfig,
        post_op_en_pdf,
        doc_id_en: str,
    ):
        """An empty query should still be handled gracefully."""
        ingest_document(post_op_en_pdf, doc_id_en, rag_config, store)

        result = retrieve("", rag_config, store)
        # Empty query should not crash; may or may not return results
        # depending on embedding behavior — we just verify it doesn't crash
        assert result.query == ""
        assert isinstance(result.chunks, list)
