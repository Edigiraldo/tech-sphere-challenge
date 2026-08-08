"""Tests for the chunking module.

All tests are fast (pure Python, no model or PDF needed).
"""

import uuid

from backend.rag.chunking import Chunk, chunk_pages, _overlap_split
from backend.rag.config import RagConfig


class TestOverlapSplit:
    """Unit tests for the internal _overlap_split helper."""

    def test_empty_text_produces_no_chunks(self):
        chunks = _overlap_split("", chunk_size=100, chunk_overlap=20)
        assert chunks == []

    def test_text_shorter_than_chunk_size(self):
        text = "Short text."
        chunks = _overlap_split(text, chunk_size=100, chunk_overlap=20)
        assert chunks == [text]

    def test_text_exactly_chunk_size(self):
        text = "A" * 100
        chunks = _overlap_split(text, chunk_size=100, chunk_overlap=20)
        assert chunks == [text]

    def test_overlap_preserves_context(self):
        """Verify that overlapping chunks share boundary text."""
        text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        chunk_size = 10
        chunk_overlap = 5
        chunks = _overlap_split(text, chunk_size, chunk_overlap)

        assert len(chunks) >= 2, "Should produce multiple overlapping chunks"
        # Last 5 chars of chunk[0] should equal first 5 chars of chunk[1]
        assert chunks[0][-5:] == chunks[1][:5]

    def test_overlap_clamps_when_too_large(self):
        """When overlap >= chunk_size, it's clamped to half."""
        text = "A" * 300
        chunks = _overlap_split(text, chunk_size=10, chunk_overlap=15)
        # Should still work without crashing
        assert len(chunks) > 0
        # Each chunk should be <= chunk_size
        for c in chunks:
            assert len(c) <= 10

    def test_large_text_produces_multiple_chunks(self):
        text = "X" * 1000
        chunks = _overlap_split(text, chunk_size=200, chunk_overlap=50)
        assert len(chunks) >= 4


class TestChunkPages:
    """Tests for the public chunk_pages function."""

    def config(self):
        return RagConfig(chunk_size=100, chunk_overlap=20)

    def test_empty_pages_produce_no_chunks(self):
        chunks = list(chunk_pages([], "doc-1", "test.pdf", self.config()))
        assert chunks == []

    def test_single_short_page_produces_one_chunk(self):
        pages = [{"page_number": 1, "text": "Hello world"}]
        chunks = list(chunk_pages(pages, "doc-1", "test.pdf", self.config()))
        assert len(chunks) == 1
        assert chunks[0]["document_id"] == "doc-1"
        assert chunks[0]["source_filename"] == "test.pdf"
        assert chunks[0]["page_number"] == 1
        assert chunks[0]["chunk_index"] == 0
        assert chunks[0]["text"] == "Hello world"

    def test_multiple_pages_produce_sequential_chunk_indices(self):
        pages = [
            {"page_number": 1, "text": "Page one content."},
            {"page_number": 2, "text": "Page two content."},
        ]
        chunks = list(chunk_pages(pages, "doc-2", "multi.pdf", self.config()))
        indices = [c["chunk_index"] for c in chunks]
        assert indices == list(range(len(chunks))), (
            "Chunk indices should be sequential starting at 0"
        )

    def test_page_with_empty_text_is_skipped(self):
        pages = [
            {"page_number": 1, "text": "  "},
            {"page_number": 2, "text": "Actual content here."},
        ]
        chunks = list(chunk_pages(pages, "doc-3", "skip.pdf", self.config()))
        assert len(chunks) == 1
        assert chunks[0]["page_number"] == 2

    def test_chunk_metadata_is_complete(self, sample_spanish_text):
        """Each chunk must carry all required metadata fields."""
        pages = [{"page_number": 3, "text": sample_spanish_text}]
        chunks = list(
            chunk_pages(pages, "doc-spanish", "cuidado.pdf", self.config())
        )

        for chunk in chunks:
            # Verify chunk_id is a valid UUID string
            uuid.UUID(chunk["chunk_id"])
            assert chunk["document_id"] == "doc-spanish"
            assert chunk["source_filename"] == "cuidado.pdf"
            assert chunk["page_number"] == 3
            assert isinstance(chunk["chunk_index"], int)
            assert chunk["chunk_index"] >= 0
            assert len(chunk["text"]) > 0

    def test_spanish_text_is_preserved(self, sample_spanish_text):
        """Chunking should preserve Spanish characters intact."""
        pages = [{"page_number": 1, "text": sample_spanish_text}]
        chunks = list(
            chunk_pages(pages, "doc-es", "spanish.pdf", self.config())
        )
        assert len(chunks) >= 1
        all_chunk_text = " ".join(c["text"] for c in chunks)
        # Key Spanish clinical terms must survive
        assert "apendicectomía" in all_chunk_text
        assert "herida" in all_chunk_text
        assert "fiebre" in all_chunk_text
