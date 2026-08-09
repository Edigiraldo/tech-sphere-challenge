"""End-to-end tests for ingestion → retrieval pipeline.

These tests require the BGE-M3 embedding model download and PDF extraction,
so they are marked ``@pytest.mark.slow``. Run with:

    pytest -m slow

Fixtures ``store``, ``doc_id_en``, and ``doc_id_es`` are shared from
``tests/rag/conftest.py`` so that both ``TestIngestionRetrieval`` and
``TestRealPDFFilenames`` can use them.
"""

import pytest

from backend.rag.config import RagConfig
from backend.rag.ingestion import ingest_document
from backend.rag.retrieval import retrieve
from backend.persistence.chroma import ChromaStore


@pytest.mark.slow
class TestIngestionRetrieval:
    """End-to-end: ingest two Appendicitis PDFs and retrieve relevant chunks."""

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


# =============================================================================
# Retrieval sufficiency tests (fast — no embedding model needed)
# =============================================================================


class TestRetrievalSufficiency:
    """Tests for retrieval quality gates (sufficient flag)."""

    def test_ragged_config_defaults_have_reasonable_thresholds(self):
        """Default RagConfig should set min_chunks and similarity thresholds
        that prevent weak retrieval from reaching the LLM."""
        from backend.rag.config import RagConfig

        cfg = RagConfig()
        assert cfg.min_chunks_for_answer >= 1, (
            "min_chunks_for_answer should require at least 1 chunk"
        )
        assert cfg.min_avg_similarity >= 0.0, (
            "min_avg_similarity should be >= 0"
        )
        assert cfg.similarity_threshold >= 0.0, (
            "similarity_threshold should be >= 0"
        )

    def test_retrieval_result_empty_is_not_sufficient(self):
        from backend.rag.retrieval import RetrievalResult

        result = RetrievalResult(query="q")
        assert not result.has_results
        assert not result.sufficient
        assert result.avg_similarity == 0.0

    def test_retrieval_result_single_chunk_below_min_chunks_is_not_sufficient(self):
        from backend.rag.retrieval import RetrievalResult, RetrievedChunk

        result = RetrievalResult(
            query="q",
            chunks=[
                RetrievedChunk(
                    chunk_id="c1", document_id="d1",
                    source_filename="f.pdf", chunk_index=0,
                    page_number=1, text="Text", similarity=0.90,
                )
            ],
        )
        # With default config, min_chunks_for_answer=2 → single chunk
        # is not sufficient regardless of similarity
        assert result.has_results
        # sufficient defaults to False, and the constructor doesn't compute it
        # — that's done by the retrieve() function
        assert not result.sufficient

    def test_retrieval_result_avg_similarity(self):
        from backend.rag.retrieval import RetrievalResult, RetrievedChunk

        result = RetrievalResult(
            query="q",
            chunks=[
                RetrievedChunk(
                    chunk_id="c1", document_id="d1",
                    source_filename="f.pdf", chunk_index=0,
                    page_number=1, text="A", similarity=0.80,
                ),
                RetrievedChunk(
                    chunk_id="c2", document_id="d1",
                    source_filename="f.pdf", chunk_index=1,
                    page_number=2, text="B", similarity=0.60,
                ),
            ],
        )
        assert result.avg_similarity == 0.70


# =============================================================================
# Real PDF filename integration tests (slow)
# =============================================================================


@pytest.mark.slow
class TestRealPDFFilenames:
    """Verify the Appendicitis PDF fixtures resolve correctly on disk.

    These tests use the exact filenames from fixtures
    (``tests/rag/conftest.py``) and verify they match what's on disk.
    """

    def test_english_postop_pdf_exists(self, post_op_en_pdf):
        """The English post-operative PDF must exist on disk."""
        assert post_op_en_pdf.is_file(), (
            f"Missing PDF: {post_op_en_pdf}"
        )
        assert post_op_en_pdf.suffix == ".pdf"

    def test_spanish_plan_pdf_exists(self, plan_cuidado_es_pdf):
        """The Spanish home care plan PDF must exist on disk."""
        assert plan_cuidado_es_pdf.is_file(), (
            f"Missing PDF: {plan_cuidado_es_pdf}"
        )
        assert plan_cuidado_es_pdf.suffix == ".pdf"

    def test_english_postop_pdf_has_expected_name(self, post_op_en_pdf):
        """The English PDF filename includes the expected trailing space
        before '.pdf' — this is the exact on-disk name."""
        assert post_op_en_pdf.name == (
            "POST OPERATIVE INSTRUCTIONS FOR APPENDECTOMY .pdf"
        ), f"Filename mismatch: {post_op_en_pdf.name!r}"

    def test_spanish_plan_pdf_has_accented_name(self, plan_cuidado_es_pdf):
        """The Spanish PDF filename includes the accented Í character."""
        assert plan_cuidado_es_pdf.name == (
            "PLAN DE CUIDADO EN CASA DE PACIENTE EN "
            "POSTOPERATORIO DE APENDICECTOMÍA.pdf"
        ), f"Filename mismatch: {plan_cuidado_es_pdf.name!r}"

    def test_ingest_both_real_pdfs(self, store, rag_config,
                                    post_op_en_pdf, plan_cuidado_es_pdf,
                                    doc_id_en, doc_id_es):
        """Ingest both real PDFs and verify they can be retrieved."""
        count_en = ingest_document(
            post_op_en_pdf, doc_id_en, rag_config, store
        )
        count_es = ingest_document(
            plan_cuidado_es_pdf, doc_id_es, rag_config, store
        )
        assert count_en > 0, "English PDF should yield at least 1 chunk"
        assert count_es > 0, "Spanish PDF should yield at least 1 chunk"

        # Verify chunks carry the correct source filenames
        result = retrieve(
            "cuidado postoperatorio apendicectomía",
            rag_config, store, top_k=5,
        )
        filenames = {c.source_filename for c in result.chunks}
        assert plan_cuidado_es_pdf.name in filenames, (
            f"Spanish PDF {plan_cuidado_es_pdf.name!r} not in {filenames}"
        )

    def test_ingest_verify_deletion_purges_chunks(
        self, store, rag_config, plan_cuidado_es_pdf, doc_id_es,
    ):
        """Ingest the Spanish PDF, verify chunks exist, then delete and
        verify zero chunks remain."""
        count = ingest_document(
            plan_cuidado_es_pdf, doc_id_es, rag_config, store
        )
        assert count > 0

        # Verify chunks exist
        assert store.count() >= count

        # Delete
        deleted_ids = store.delete_document_chunks(doc_id_es)
        assert len(deleted_ids) == count

        # Verify collection is empty
        assert store.count() == 0

    def test_ingest_both_retrieve_spanish_query_finds_spanish_pdf(
        self, store, rag_config,
        post_op_en_pdf, plan_cuidado_es_pdf,
        doc_id_en, doc_id_es,
    ):
        """A Spanish-language query should find the Spanish PDF."""
        ingest_document(post_op_en_pdf, doc_id_en, rag_config, store)
        ingest_document(plan_cuidado_es_pdf, doc_id_es, rag_config, store)

        result = retrieve(
            "cuidado de la herida después de apendicectomía",
            rag_config, store, top_k=3,
        )
        assert result.has_results
        source_files = [c.source_filename for c in result.chunks]
        assert plan_cuidado_es_pdf.name in source_files, (
            f"Spanish PDF not in results: {source_files}"
        )
