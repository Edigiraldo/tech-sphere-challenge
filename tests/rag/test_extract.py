"""Tests for RAG text extraction from PDF files."""

import pytest

from backend.rag.extract import ExtractionError, extract_pages


class TestExtractPages:
    """Tests for pdfplumber-based text extraction."""

    def test_raises_on_missing_file(self):
        """Extracting a non-existent file should raise ExtractionError."""
        with pytest.raises(ExtractionError, match="File not found"):
            list(extract_pages("/nonexistent/file.pdf"))

    def test_raises_on_non_pdf(self, tmp_path):
        """Extracting a non-PDF file should raise ExtractionError."""
        txt_file = tmp_path / "not_a_pdf.txt"
        txt_file.write_text("Some text content")
        with pytest.raises(ExtractionError, match="Failed to extract"):
            list(extract_pages(txt_file))

    @pytest.mark.slow
    def test_extract_english_postop_pdf(self, post_op_en_pdf):
        """Extract text from the English post-operative instructions PDF."""
        pages = list(extract_pages(post_op_en_pdf))
        assert len(pages) > 0, "Should extract at least one page"
        for page in pages:
            assert "page_number" in page
            assert "text" in page
            assert isinstance(page["text"], str)
            assert len(page["text"].strip()) > 0, (
                f"Page {page['page_number']} should have non-empty text"
            )

    @pytest.mark.slow
    def test_extract_spanish_plan_pdf(self, plan_cuidado_es_pdf):
        """Extract text from the Spanish home care plan PDF."""
        pages = list(extract_pages(plan_cuidado_es_pdf))
        assert len(pages) > 0, "Should extract at least one page"
        all_text = " ".join(str(p["text"]) for p in pages)
        # Should contain Spanish clinical terms
        assert any(
            word in all_text.lower()
            for word in ["apendicectomía", "herida", "cuidado", "postoperatorio"]
        ), "Should contain Spanish post-op terms"

    @pytest.mark.slow
    def test_pages_have_correct_structure(self, plan_cuidado_es_pdf):
        """Pages should have the expected dict structure."""
        pages = list(extract_pages(plan_cuidado_es_pdf))
        for page in pages:
            assert isinstance(page["page_number"], int)
            assert page["page_number"] >= 1
            assert isinstance(page["text"], str)
