"""Tests for the document lifecycle endpoints and service.

Covers:
- POST /documents (upload success, content-hash duplicate detection,
  invalid file, processing failure)
- GET  /documents (listing, filtering, content-hash in response)
- DELETE /documents/{id} (success, not-found)
- POST /documents/reconcile (registry↔ChromaDB consistency)
- Retrieval availability after ingestion
- Complete indexed-chunk deletion (no orphaned ChromaDB chunks)
- Registry-filtered retrieval (excludes deleted/unregistered document IDs)
- Content-hash duplicate detection and idempotent corpus ingestion
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

from backend.main import app
from backend.persistence.chroma import ChromaStore
from backend.persistence.sqlite import _reset_sqlite
from backend.rag.config import RagConfig
from backend.rag.ingestion import ingest_document
from backend.rag.retrieval import retrieve
from backend.rag.store import init_store


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture(autouse=True)
def _reset_document_service():
    """Reset the DocumentService singleton between tests.

    Also resets the SQLite module so each test gets a fresh database.
    """
    import backend.api.documents as api_docs

    original = api_docs._service
    api_docs._service = None
    _reset_sqlite()
    yield
    api_docs._service = original


@pytest.fixture(autouse=True)
def _reset_chroma_store():
    """Reset the ChromaStore singleton between tests.

    This is needed because tests in ``tests/`` (not ``tests/rag/``)
    don't inherit the autouse fixture from ``tests/rag/conftest.py``.
    """
    import backend.persistence.chroma as chroma_mod

    original = chroma_mod._store
    chroma_mod._store = None
    yield
    chroma_mod._store = original


@pytest.fixture(autouse=True)
def _reset_rag_config_singletons():
    """Reset the cached RAG/Llm config singletons in the API layer between tests.

    This ensures that when a test patches ``os.environ`` to set RAG env vars,
    the patched values are picked up by a fresh ``RagConfig()`` call rather
    than reusing a previously cached instance.
    """
    import backend.api.rag as api_rag
    import backend.api.documents as api_docs

    orig_rag = api_rag._rag_config
    orig_llm = api_rag._llm_config
    orig_docs_rag = api_docs._rag_config
    api_rag._rag_config = None
    api_rag._llm_config = None
    api_docs._rag_config = None
    yield
    api_rag._rag_config = orig_rag
    api_rag._llm_config = orig_llm
    api_docs._rag_config = orig_docs_rag


@pytest.fixture
def temp_upload_dir():
    """A temporary directory for uploaded files."""
    with tempfile.TemporaryDirectory(prefix="doc_uploads_") as d:
        yield Path(d)


@pytest.fixture
def temp_db_path():
    """A temporary file path for the SQLite database."""
    fd, path = tempfile.mkstemp(suffix=".db", prefix="test_docs_")
    os.close(fd)
    yield Path(path)
    # Cleanup
    try:
        os.unlink(path)
    except OSError:
        pass


@pytest.fixture
def document_service(temp_upload_dir, temp_db_path, request):
    """A DocumentService backed by temporary directories.

    Uses monkeypatch to set env vars so the API-level singleton picks up
    the temp paths.
    """
    from backend.documents.service import DocumentService

    monkeypatch = request.getfixturevalue("monkeypatch")
    monkeypatch.setenv("DOCUMENTS_UPLOAD_DIR", str(temp_upload_dir))
    monkeypatch.setenv("DOCUMENTS_DB_PATH", str(temp_db_path))
    return DocumentService(
        upload_dir=temp_upload_dir, db_path=temp_db_path
    )


@pytest.fixture
def rag_config():
    """A RAG config pointing at a temp ChromaDB directory."""
    return RagConfig(
        chroma_persist_dir=Path(tempfile.mkdtemp(prefix="chroma_docs_")),
        chunk_size=400,
        chunk_overlap=80,
        retrieval_top_k=3,
        collection_name="test_documents_collection",
    )


@pytest.fixture
def test_pdf_dir():
    """Return the path to the Appendicitis test PDF directory."""
    return Path(__file__).parent.parent / "dataset" / "textos" / "Appendicitis"


@pytest.fixture
def test_pdf(test_pdf_dir):
    """Path to the Spanish home-care PDF (valid test document)."""
    return (
        test_pdf_dir
        / "PLAN DE CUIDADO EN CASA DE PACIENTE EN POSTOPERATORIO DE APENDICECTOMÍA.pdf"
    )


# ======================================================================
# Fast API-level tests (no embedding model needed)
# ======================================================================


class TestUploadValidation:
    """Fast tests for upload request validation."""

    @pytest.mark.asyncio
    async def test_reject_non_pdf_extension(self, temp_upload_dir, temp_db_path):
        """Uploading a file without .pdf extension returns 400."""
        import backend.api.documents as api_docs

        api_docs._service = None
        with patch.dict(
            os.environ,
            {
                "DOCUMENTS_UPLOAD_DIR": str(temp_upload_dir),
                "DOCUMENTS_DB_PATH": str(temp_db_path),
            },
        ):
            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                response = await client.post(
                    "/documents",
                    files={"file": ("test.txt", b"hello", "text/plain")},
                )

        assert response.status_code == 400
        data = response.json()
        assert "PDF" in data["detail"]

    @pytest.mark.asyncio
    async def test_reject_empty_file(self, temp_upload_dir, temp_db_path):
        """Uploading an empty file returns 400."""
        import backend.api.documents as api_docs

        api_docs._service = None
        with patch.dict(
            os.environ,
            {
                "DOCUMENTS_UPLOAD_DIR": str(temp_upload_dir),
                "DOCUMENTS_DB_PATH": str(temp_db_path),
            },
        ):
            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                response = await client.post(
                    "/documents",
                    files={"file": ("empty.pdf", b"", "application/pdf")},
                )

        assert response.status_code == 400
        data = response.json()
        assert "Empty" in data["detail"]

    @pytest.mark.asyncio
    async def test_reject_no_filename(self, temp_upload_dir, temp_db_path):
        """Uploading without a filename is rejected (FastAPI/pydantic returns 422)."""
        import backend.api.documents as api_docs

        api_docs._service = None
        with patch.dict(
            os.environ,
            {
                "DOCUMENTS_UPLOAD_DIR": str(temp_upload_dir),
                "DOCUMENTS_DB_PATH": str(temp_db_path),
            },
        ):
            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                response = await client.post(
                    "/documents",
                    files={"file": ("", b"content", "application/pdf")},
                )

        assert response.status_code in (400, 422)


    @pytest.mark.asyncio
    async def test_sanitize_path_traversal_filename(
        self, temp_upload_dir, temp_db_path
    ):
        """A filename with path traversal characters must be sanitized
        so the file is stored only under the configured upload directory."""
        import backend.api.documents as api_docs

        api_docs._service = None
        chroma_dir = tempfile.mkdtemp(prefix="chroma_docs_")
        with patch.dict(
            os.environ,
            {
                "DOCUMENTS_UPLOAD_DIR": str(temp_upload_dir),
                "DOCUMENTS_DB_PATH": str(temp_db_path),
                "RAG_CHROMA_DIR": chroma_dir,
                "RAG_COLLECTION_NAME": "test_sanitize",
            },
        ):
            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                response = await client.post(
                    "/documents",
                    files={
                        "file": (
                            "../../../etc/malicious.pdf",
                            b"%PDF-1.0\n%%EOF",
                            "application/pdf",
                        )
                    },
                )

        # Upload is accepted (201) even if ingestion fails — the file
        # should be saved under the upload directory, not escaped.
        assert response.status_code == 201, (
            f"Expected 201, got {response.status_code}: {response.text}"
        )
        data = response.json()
        # The original filename is preserved in the Document model for display
        assert data["filename"] == "../../../etc/malicious.pdf"
        # Only the basename should appear in any file stored on disk
        on_disk = list(temp_upload_dir.iterdir())
        for p in on_disk:
            assert ".." not in p.name, (
                f"Path traversal in stored file: {p.name}"
            )
            assert p.name.startswith(data["document_id"] + "_")
            assert p.name.endswith("malicious.pdf")


class TestProcessingFailure:
    """Tests for ingestion failure handling."""

    @pytest.mark.asyncio
    async def test_non_pdf_disguised_as_pdf_results_in_failed(
        self, temp_upload_dir, temp_db_path
    ):
        """A text file renamed to .pdf should fail during extraction/ingestion
        and the document status should be 'failed'."""
        import backend.api.documents as api_docs

        chroma_dir = tempfile.mkdtemp(prefix="chroma_docs_")
        api_docs._service = None
        with patch.dict(
            os.environ,
            {
                "DOCUMENTS_UPLOAD_DIR": str(temp_upload_dir),
                "DOCUMENTS_DB_PATH": str(temp_db_path),
                "RAG_CHROMA_DIR": chroma_dir,
                "RAG_COLLECTION_NAME": "test_failed_docs",
            },
        ):
            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                response = await client.post(
                    "/documents",
                    files={
                        "file": (
                            "not_a_pdf.pdf",
                            b"This is not a PDF file.",
                            "application/pdf",
                        )
                    },
                )

        # The upload itself is accepted (201), but processing fails
        assert response.status_code == 201
        data = response.json()
        assert data["status"] == "failed"
        assert "ExtractionError" in data["message"], (
            f"Expected 'ExtractionError' in message, got: {data.get('message')}"
        )


class TestListDocuments:
    """Tests for GET /documents listing."""

    @pytest.mark.asyncio
    async def test_list_empty(self, temp_upload_dir, temp_db_path):
        """Listing when no documents exist returns empty list."""
        import backend.api.documents as api_docs

        api_docs._service = None
        with patch.dict(
            os.environ,
            {
                "DOCUMENTS_UPLOAD_DIR": str(temp_upload_dir),
                "DOCUMENTS_DB_PATH": str(temp_db_path),
            },
        ):
            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                response = await client.get("/documents")

        assert response.status_code == 200
        data = response.json()
        assert data["documents"] == []
        assert data["total"] == 0

    @pytest.mark.asyncio
    async def test_invalid_status_returns_400(
        self, temp_upload_dir, temp_db_path
    ):
        """A status query parameter that is not a valid DocumentStatus
        must return 400 with a helpful error message listing valid values."""
        import backend.api.documents as api_docs

        api_docs._service = None
        with patch.dict(
            os.environ,
            {
                "DOCUMENTS_UPLOAD_DIR": str(temp_upload_dir),
                "DOCUMENTS_DB_PATH": str(temp_db_path),
            },
        ):
            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                response = await client.get(
                    "/documents", params={"status": "archived"}
                )

        assert response.status_code == 400
        data = response.json()
        assert "archived" in data["detail"]
        assert "deleted" in data["detail"]
        assert "ready" in data["detail"]

    @pytest.mark.asyncio
    async def test_list_includes_failed_documents(
        self, temp_upload_dir, temp_db_path
    ):
        """After a failed upload, the document appears in the list."""
        import backend.api.documents as api_docs

        api_docs._service = None
        with patch.dict(
            os.environ,
            {
                "DOCUMENTS_UPLOAD_DIR": str(temp_upload_dir),
                "DOCUMENTS_DB_PATH": str(temp_db_path),
            },
        ):
            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                # Upload a document that will fail
                await client.post(
                    "/documents",
                    files={
                        "file": (
                            "bad.pdf",
                            b"not a pdf",
                            "application/pdf",
                        )
                    },
                )

                # List all documents
                response = await client.get("/documents")

        assert response.status_code == 200
        data = response.json()
        assert data["total"] >= 1
        statuses = [d["status"] for d in data["documents"]]
        assert "failed" in statuses


class TestDeleteDocument:
    """Tests for DELETE /documents/{document_id}."""

    @pytest.mark.asyncio
    async def test_delete_nonexistent_returns_404(
        self, temp_upload_dir, temp_db_path
    ):
        """Deleting a document that doesn't exist returns 404."""
        import backend.api.documents as api_docs

        api_docs._service = None
        with patch.dict(
            os.environ,
            {
                "DOCUMENTS_UPLOAD_DIR": str(temp_upload_dir),
                "DOCUMENTS_DB_PATH": str(temp_db_path),
            },
        ):
            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                response = await client.delete(
                    "/documents/nonexistent-id"
                )

        assert response.status_code == 404


# ======================================================================
# Slow tests (require BGE-M3 embedding model)
# ======================================================================


@pytest.mark.slow
class TestDocumentLifecycleE2E:
    """End-to-end tests: upload → list → retrieve → delete → verify purge."""

    @pytest.mark.asyncio
    async def test_upload_real_pdf_succeeds(
        self, temp_upload_dir, temp_db_path, test_pdf, rag_config
    ):
        """Upload a real PDF and verify status becomes 'ready'."""
        import backend.api.documents as api_docs

        api_docs._service = None
        with patch.dict(
            os.environ,
            {
                "DOCUMENTS_UPLOAD_DIR": str(temp_upload_dir),
                "DOCUMENTS_DB_PATH": str(temp_db_path),
                "RAG_CHROMA_DIR": str(rag_config.chroma_persist_dir),
                "RAG_COLLECTION_NAME": rag_config.collection_name,
            },
        ):
            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                with open(test_pdf, "rb") as f:
                    response = await client.post(
                        "/documents",
                        files={
                            "file": (
                                test_pdf.name,
                                f.read(),
                                "application/pdf",
                            )
                        },
                    )

        assert response.status_code == 201
        data = response.json()
        assert data["status"] == "ready", (
            f"Expected status 'ready', got '{data['status']}'. "
            f"Message: {data.get('message', 'none')}"
        )
        assert data["document_id"]
        assert data["filename"] == test_pdf.name

    @pytest.mark.asyncio
    async def test_list_after_upload(
        self, temp_upload_dir, temp_db_path, test_pdf, rag_config
    ):
        """Upload a PDF and verify it appears in the list."""
        import backend.api.documents as api_docs

        api_docs._service = None
        with patch.dict(
            os.environ,
            {
                "DOCUMENTS_UPLOAD_DIR": str(temp_upload_dir),
                "DOCUMENTS_DB_PATH": str(temp_db_path),
                "RAG_CHROMA_DIR": str(rag_config.chroma_persist_dir),
                "RAG_COLLECTION_NAME": rag_config.collection_name,
            },
        ):
            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                # Upload
                with open(test_pdf, "rb") as f:
                    upload_resp = await client.post(
                        "/documents",
                        files={
                            "file": (
                                test_pdf.name,
                                f.read(),
                                "application/pdf",
                            )
                        },
                    )
                doc_id = upload_resp.json()["document_id"]

                # List
                list_resp = await client.get("/documents")

        assert list_resp.status_code == 200
        data = list_resp.json()
        assert data["total"] >= 1
        doc = next(
            d for d in data["documents"] if d["document_id"] == doc_id
        )
        assert doc["filename"] == test_pdf.name
        assert doc["status"] == "ready"
        assert doc["size_bytes"] > 0

    @pytest.mark.asyncio
    async def test_filter_by_status(
        self, temp_upload_dir, temp_db_path, test_pdf, rag_config
    ):
        """The status query parameter filters results correctly."""
        import backend.api.documents as api_docs

        api_docs._service = None
        with patch.dict(
            os.environ,
            {
                "DOCUMENTS_UPLOAD_DIR": str(temp_upload_dir),
                "DOCUMENTS_DB_PATH": str(temp_db_path),
                "RAG_CHROMA_DIR": str(rag_config.chroma_persist_dir),
                "RAG_COLLECTION_NAME": rag_config.collection_name,
            },
        ):
            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                # Upload a real PDF (status = ready)
                with open(test_pdf, "rb") as f:
                    await client.post(
                        "/documents",
                        files={
                            "file": (
                                test_pdf.name,
                                f.read(),
                                "application/pdf",
                            )
                        },
                    )

                # Filter by 'deleted' — should be empty
                resp_deleted = await client.get(
                    "/documents", params={"status": "deleted"}
                )
                # Filter by 'ready' — should have the document
                resp_ready = await client.get(
                    "/documents", params={"status": "ready"}
                )

        assert resp_deleted.status_code == 200
        assert resp_deleted.json()["total"] == 0

        assert resp_ready.status_code == 200
        assert resp_ready.json()["total"] >= 1

    @pytest.mark.asyncio
    async def test_retrieval_availability_after_upload(
        self, temp_upload_dir, temp_db_path, test_pdf, rag_config
    ):
        """After uploading a document, its chunks must be retrievable."""
        import backend.api.documents as api_docs

        api_docs._service = None
        with patch.dict(
            os.environ,
            {
                "DOCUMENTS_UPLOAD_DIR": str(temp_upload_dir),
                "DOCUMENTS_DB_PATH": str(temp_db_path),
                "RAG_CHROMA_DIR": str(rag_config.chroma_persist_dir),
                "RAG_COLLECTION_NAME": rag_config.collection_name,
                "RAG_CHUNK_SIZE": str(rag_config.chunk_size),
                "RAG_CHUNK_OVERLAP": str(rag_config.chunk_overlap),
            },
        ):
            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                with open(test_pdf, "rb") as f:
                    upload_resp = await client.post(
                        "/documents",
                        files={
                            "file": (
                                test_pdf.name,
                                f.read(),
                                "application/pdf",
                            )
                        },
                    )

        assert upload_resp.status_code == 201
        doc_id = upload_resp.json()["document_id"]

        # Now query RAG directly to verify the chunks are available
        store = init_store(rag_config)
        result = retrieve(
            "cuidado de la herida después de apendicectomía",
            rag_config,
            store,
            top_k=3,
        )
        assert result.has_results, (
            "Expected at least one retrieved chunk after uploading a PDF"
        )
        # The chunks should belong to our document
        doc_ids = {c.document_id for c in result.chunks}
        assert doc_id in doc_ids, (
            f"Expected chunks from document {doc_id}, got {doc_ids}"
        )

    @pytest.mark.asyncio
    async def test_complete_indexed_chunk_deletion(
        self, temp_upload_dir, temp_db_path, test_pdf, rag_config
    ):
        """Deleting a document must:
        - Mark its status as 'deleted' in SQLite.
        - Remove all ChromaDB chunks for that document_id.
        - Leave other documents' chunks intact (if applicable).
        """
        import backend.api.documents as api_docs

        api_docs._service = None
        with patch.dict(
            os.environ,
            {
                "DOCUMENTS_UPLOAD_DIR": str(temp_upload_dir),
                "DOCUMENTS_DB_PATH": str(temp_db_path),
                "RAG_CHROMA_DIR": str(rag_config.chroma_persist_dir),
                "RAG_COLLECTION_NAME": rag_config.collection_name,
                "RAG_CHUNK_SIZE": str(rag_config.chunk_size),
                "RAG_CHUNK_OVERLAP": str(rag_config.chunk_overlap),
            },
        ):
            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                # Upload
                with open(test_pdf, "rb") as f:
                    upload_resp = await client.post(
                        "/documents",
                        files={
                            "file": (
                                test_pdf.name,
                                f.read(),
                                "application/pdf",
                            )
                        },
                    )
                doc_id = upload_resp.json()["document_id"]

                # Verify chunks exist in ChromaDB
                store = init_store(rag_config)
                count_before = store.count()
                assert count_before > 0, (
                    "Expected at least one chunk after ingestion"
                )

                # Delete
                delete_resp = await client.delete(f"/documents/{doc_id}")
                assert delete_resp.status_code == 200
                assert delete_resp.json()["status"] == "deleted"

                # Verify chunks are gone
                count_after = store.count()
                assert count_after == 0, (
                    f"Expected 0 chunks after deletion, got {count_after}"
                )

                # Verify SQLite status is 'deleted'
                list_resp = await client.get(
                    "/documents", params={"status": "deleted"}
                )
                assert list_resp.status_code == 200
                deleted_docs = list_resp.json()["documents"]
                assert len(deleted_docs) == 1
                assert deleted_docs[0]["document_id"] == doc_id
                assert deleted_docs[0]["status"] == "deleted"

                # Verify no chunks can be retrieved for the deleted document
                result = retrieve(
                    "cuidado de la herida",
                    rag_config,
                    store,
                    top_k=5,
                )
                doc_ids = {c.document_id for c in result.chunks}
                assert doc_id not in doc_ids, (
                    f"Deleted document {doc_id} still has retrievable chunks"
                )

    @pytest.mark.asyncio
    async def test_delete_then_retrieve_finds_nothing(
        self, temp_upload_dir, temp_db_path, test_pdf, rag_config
    ):
        """After deletion, retrieval should find no chunks from the
        deleted document, even for highly relevant queries."""
        import backend.api.documents as api_docs

        api_docs._service = None
        with patch.dict(
            os.environ,
            {
                "DOCUMENTS_UPLOAD_DIR": str(temp_upload_dir),
                "DOCUMENTS_DB_PATH": str(temp_db_path),
                "RAG_CHROMA_DIR": str(rag_config.chroma_persist_dir),
                "RAG_COLLECTION_NAME": rag_config.collection_name,
            },
        ):
            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                with open(test_pdf, "rb") as f:
                    upload_resp = await client.post(
                        "/documents",
                        files={
                            "file": (
                                test_pdf.name,
                                f.read(),
                                "application/pdf",
                            )
                        },
                    )
                doc_id = upload_resp.json()["document_id"]

                # Delete
                await client.delete(f"/documents/{doc_id}")

        # Retrieval should now return nothing
        store = init_store(rag_config)
        result = retrieve(
            "apendicectomía cuidado postoperatorio herida",
            rag_config,
            store,
            top_k=10,
        )
        # All chunks are gone from the collection
        assert not result.has_results, (
            "Expected no results after deleting the only document"
        )

    @pytest.mark.asyncio
    async def test_duplicate_pdf_upload_isolated_deletion(
        self, temp_upload_dir, temp_db_path, test_pdf, test_pdf_dir, rag_config
    ):
        """Upload two different PDFs and verify that deleting one does not
        affect the other's chunks or retrievability.

        Content-hash duplicate detection makes identical-content uploads
        idempotent (same document_id), so isolation is tested with two
        PDFs that have different content: deleting one must preserve the
        other's chunks.
        """
        import backend.api.documents as api_docs

        # Second PDF — English Appendicitis post-op instructions
        second_pdf = (
            test_pdf_dir
            / "POST OPERATIVE INSTRUCTIONS FOR APPENDECTOMY .pdf"
        )

        api_docs._service = None
        with patch.dict(
            os.environ,
            {
                "DOCUMENTS_UPLOAD_DIR": str(temp_upload_dir),
                "DOCUMENTS_DB_PATH": str(temp_db_path),
                "RAG_CHROMA_DIR": str(rag_config.chroma_persist_dir),
                "RAG_COLLECTION_NAME": rag_config.collection_name,
            },
        ):
            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                # Upload first PDF (Spanish home care plan)
                with open(test_pdf, "rb") as f:
                    resp1 = await client.post(
                        "/documents",
                        files={
                            "file": (test_pdf.name, f.read(), "application/pdf")
                        },
                    )
                assert resp1.status_code == 201
                doc1_id = resp1.json()["document_id"]
                assert resp1.json()["status"] == "ready"

                # Upload second PDF (English post-op instructions)
                with open(second_pdf, "rb") as f:
                    resp2 = await client.post(
                        "/documents",
                        files={
                            "file": (second_pdf.name, f.read(), "application/pdf")
                        },
                    )
                assert resp2.status_code == 201
                doc2_id = resp2.json()["document_id"]
                assert resp2.json()["status"] == "ready"

                # Different PDFs must have different IDs
                assert doc1_id != doc2_id, (
                    "Different PDFs must get distinct document IDs"
                )

                # Verify both are listed
                list_resp = await client.get(
                    "/documents", params={"status": "ready"}
                )
                ready_ids = {
                    d["document_id"] for d in list_resp.json()["documents"]
                }
                assert doc1_id in ready_ids
                assert doc2_id in ready_ids

                # Total chunk count should have chunks from both PDFs
                store = init_store(rag_config)
                total_before = store.count()
                assert total_before >= 2, (
                    "Expected at least 2 chunks (one from each document)"
                )

                # Delete first document
                delete_resp = await client.delete(f"/documents/{doc1_id}")
                assert delete_resp.status_code == 200

                # Verify doc1 is deleted, doc2 is still ready
                list_after = await client.get("/documents")
                docs_after = list_after.json()["documents"]
                doc1_status = next(
                    d["status"] for d in docs_after
                    if d["document_id"] == doc1_id
                )
                doc2_status = next(
                    d["status"] for d in docs_after
                    if d["document_id"] == doc2_id
                )
                assert doc1_status == "deleted"
                assert doc2_status == "ready"

                # Verify chunks from doc2 are still retrievable
                result = retrieve(
                    "appendectomy wound care",
                    rag_config,
                    store,
                    top_k=5,
                )
                doc_ids = {c.document_id for c in result.chunks}
                assert doc1_id not in doc_ids, (
                    "Deleted document chunks must not appear in retrieval"
                )
                assert doc2_id in doc_ids, (
                    "Second document's chunks must still be retrievable"
                )


# ======================================================================
# Content-hash duplicate detection tests
# ======================================================================


class TestContentHashDuplicateDetection:
    """Tests for SHA-256 content-hash duplicate upload detection."""

    def test_document_model_accepts_content_hash(self):
        """The Document dataclass must accept a content_hash field."""
        from backend.documents.models import Document, DocumentStatus
        from datetime import datetime, timezone

        doc = Document(
            document_id="abc123",
            filename="test.pdf",
            status=DocumentStatus.READY,
            uploaded_at=datetime.now(timezone.utc),
            size_bytes=100,
            content_hash="abc123def456",
        )
        assert doc.content_hash == "abc123def456"

    def test_document_model_content_hash_defaults_to_none(self):
        """content_hash should default to None for backward compatibility."""
        from backend.documents.models import Document, DocumentStatus
        from datetime import datetime, timezone

        doc = Document(
            document_id="abc123",
            filename="test.pdf",
            status=DocumentStatus.READY,
            uploaded_at=datetime.now(timezone.utc),
            size_bytes=100,
        )
        assert doc.content_hash is None

    def test_insert_and_retrieve_content_hash(self, temp_db_path):
        """Inserting a document with content_hash and retrieving it
        must preserve the hash value."""
        from backend.documents.models import Document, DocumentStatus
        from backend.persistence.sqlite import (
            get_document_by_id,
            init_sqlite,
            insert_document,
        )
        from datetime import datetime, timezone

        init_sqlite(temp_db_path)

        doc = Document(
            document_id="hash-test-1",
            filename="test.pdf",
            status=DocumentStatus.READY,
            uploaded_at=datetime.now(timezone.utc),
            size_bytes=42,
            content_hash="abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
        )
        insert_document(doc)

        retrieved = get_document_by_id("hash-test-1")
        assert retrieved is not None
        assert retrieved.content_hash == (
            "abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890"
        )

    def test_get_by_content_hash_finds_active(self, temp_db_path):
        """get_document_by_content_hash must find an active document
        matching the hash."""
        from backend.documents.models import Document, DocumentStatus
        from backend.persistence.sqlite import (
            get_document_by_content_hash,
            init_sqlite,
            insert_document,
        )
        from datetime import datetime, timezone

        init_sqlite(temp_db_path)

        doc = Document(
            document_id="ch-1",
            filename="a.pdf",
            status=DocumentStatus.READY,
            uploaded_at=datetime.now(timezone.utc),
            size_bytes=10,
            content_hash="hash-abc",
        )
        insert_document(doc)

        found = get_document_by_content_hash("hash-abc")
        assert found is not None
        assert found.document_id == "ch-1"

    def test_get_by_content_hash_ignores_deleted(self, temp_db_path):
        """get_document_by_content_hash must NOT return a deleted document."""
        from backend.documents.models import Document, DocumentStatus
        from backend.persistence.sqlite import (
            get_document_by_content_hash,
            init_sqlite,
            insert_document,
            update_document_status,
        )
        from datetime import datetime, timezone

        init_sqlite(temp_db_path)

        doc = Document(
            document_id="ch-2",
            filename="b.pdf",
            status=DocumentStatus.READY,
            uploaded_at=datetime.now(timezone.utc),
            size_bytes=10,
            content_hash="hash-deleted",
        )
        insert_document(doc)
        update_document_status("ch-2", DocumentStatus.DELETED)

        found = get_document_by_content_hash("hash-deleted")
        assert found is None, (
            "get_document_by_content_hash should not return deleted documents"
        )

    def test_get_by_content_hash_ignores_failed(self, temp_db_path):
        """get_document_by_content_hash must NOT return a failed document.
        
        FAILED documents have no ingested chunks, so re-uploading the same
        content must create a new ingestion record instead of blocking on
        the stale failed entry.
        """
        from backend.documents.models import Document, DocumentStatus
        from backend.persistence.sqlite import (
            get_document_by_content_hash,
            init_sqlite,
            insert_document,
            update_document_status,
        )
        from datetime import datetime, timezone

        init_sqlite(temp_db_path)

        doc = Document(
            document_id="ch-failed",
            filename="failed.pdf",
            status=DocumentStatus.READY,
            uploaded_at=datetime.now(timezone.utc),
            size_bytes=10,
            content_hash="hash-failed",
        )
        insert_document(doc)
        update_document_status("ch-failed", DocumentStatus.FAILED)

        found = get_document_by_content_hash("hash-failed")
        assert found is None, (
            "get_document_by_content_hash should not return failed documents"
        )

    def test_get_by_content_hash_nonexistent(self, temp_db_path):
        """get_document_by_content_hash returns None for unknown hash."""
        from backend.persistence.sqlite import (
            get_document_by_content_hash,
            init_sqlite,
        )

        init_sqlite(temp_db_path)
        found = get_document_by_content_hash("nonexistent-hash")
        assert found is None

    def test_service_upload_returns_existing_on_duplicate(
        self, temp_upload_dir, temp_db_path, rag_config
    ):
        """Uploading the same content twice must return the first document
        on the second attempt (idempotent upload).

        Uses a valid minimal PDF so ingestion succeeds (READY), which is
        the prerequisite for content-hash duplicate detection — FAILED
        documents are intentionally excluded from the duplicate lookup.
        """
        from backend.documents.service import DocumentService

        service = DocumentService(
            upload_dir=temp_upload_dir, db_path=temp_db_path,
        )

        # Create a minimal valid PDF that pdfplumber can actually extract text from
        content = (
            b"%PDF-1.4\n"
            b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
            b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
            b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]"
            b"/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj\n"
            b"4 0 obj<</Length 24>>stream\n"
            b"BT /F1 12 Tf 72 720 Td (Hello) Tj ET\n"
            b"endstream\nendobj\n"
            b"5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n"
            b"xref\n0 6\n0000000000 65535 f \n0000000009 00000 n \n"
            b"0000000058 00000 n \n0000000115 00000 n \n"
            b"0000000266 00000 n \n0000000360 00000 n \n"
            b"trailer<</Size 6/Root 1 0 R>>\nstartxref\n433\n%%EOF"
        )

        doc1 = service.upload(content, "manual.pdf", rag_config)
        assert doc1.status.value == "ready", (
            f"Expected READY for valid PDF, got {doc1.status.value}: "
            f"{doc1.error_message}"
        )
        assert doc1.content_hash is not None

        doc2 = service.upload(content, "manual.pdf", rag_config)
        # The second upload must return the existing record
        assert doc2.document_id == doc1.document_id, (
            f"Duplicate upload should return existing ({doc1.document_id}), "
            f"got {doc2.document_id}"
        )
        assert doc2.content_hash == doc1.content_hash

    def test_service_upload_new_after_failed(
        self, temp_upload_dir, temp_db_path, rag_config
    ):
        """After a document fails ingestion, re-uploading the same content
        must create a new record (the failed record is ignored by
        content-hash lookup, allowing retry)."""
        from backend.documents.service import DocumentService
        from backend.persistence.sqlite import (
            get_document_by_content_hash,
            get_document_by_id,
            update_document_status,
        )
        from backend.documents.models import DocumentStatus

        service = DocumentService(
            upload_dir=temp_upload_dir, db_path=temp_db_path,
        )

        # Use content that will fail extraction (not a real PDF)
        content = b"this is not a pdf at all"
        import hashlib

        content_hash = hashlib.sha256(content).hexdigest()

        doc1 = service.upload(content, "bad.pdf", rag_config)
        assert doc1.status == DocumentStatus.FAILED, (
            f"Expected FAILED, got {doc1.status.value}"
        )
        assert doc1.content_hash == content_hash

        # Verify content-hash lookup excludes the failed record
        found = get_document_by_content_hash(content_hash)
        assert found is None, (
            "get_document_by_content_hash must not return a FAILED document"
        )

        # Re-upload same content — must create a NEW record
        doc2 = service.upload(content, "bad.pdf", rag_config)
        assert doc2.document_id != doc1.document_id, (
            "Re-upload after FAILED must create a new document_id"
        )
        assert doc2.content_hash == content_hash
        assert doc2.status == DocumentStatus.FAILED, (
            f"Still expected FAILED for bad content, got {doc2.status.value}"
        )

        # Verify the old record is still FAILED (audit trail preserved)
        old_doc = get_document_by_id(doc1.document_id)
        assert old_doc is not None
        assert old_doc.status == DocumentStatus.FAILED

        # Verify list_all has both records
        all_docs = service.list_all()
        doc_ids = {d.document_id for d in all_docs}
        assert doc1.document_id in doc_ids
        assert doc2.document_id in doc_ids
        # Both have the same hash — content-hash lookup returns None
        # because they're both FAILED, confirming re-upload always
        # produces a fresh record for failed content
        assert get_document_by_content_hash(content_hash) is None

    def test_service_upload_new_after_delete(
        self, temp_upload_dir, temp_db_path, rag_config
    ):
        """After deleting a document, re-uploading the same content must
        create a new record (the deleted record is ignored by content-hash
        lookup)."""
        from backend.documents.service import DocumentService

        service = DocumentService(
            upload_dir=temp_upload_dir, db_path=temp_db_path,
        )

        content = b"%PDF-1.4\nunique content for reupload test\n%%EOF"

        doc1 = service.upload(content, "reup.pdf", rag_config)
        assert doc1.content_hash is not None

        # Delete it
        deleted = service.delete(doc1.document_id, rag_config)
        assert deleted is not None
        assert deleted.status.value == "deleted"

        # Re-upload same content
        doc2 = service.upload(content, "reup.pdf", rag_config)
        # Must be a NEW document_id since the old one is deleted
        assert doc2.document_id != doc1.document_id, (
            "Re-upload after delete must create a new document_id"
        )
        assert doc2.content_hash == doc1.content_hash
        assert doc2.status.value in ("ready", "failed")

    @pytest.mark.asyncio
    async def test_api_upload_returns_content_hash_in_response(
        self, temp_upload_dir, temp_db_path
    ):
        """The upload API response must include the content_hash field."""
        import backend.api.documents as api_docs

        api_docs._service = None
        with patch.dict(
            os.environ,
            {
                "DOCUMENTS_UPLOAD_DIR": str(temp_upload_dir),
                "DOCUMENTS_DB_PATH": str(temp_db_path),
            },
        ):
            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                response = await client.post(
                    "/documents",
                    files={
                        "file": (
                            "test.pdf",
                            b"%PDF-1.4\n%%EOF",
                            "application/pdf",
                        )
                    },
                )

        assert response.status_code == 201
        data = response.json()
        assert "content_hash" in data, (
            f"UploadResponse must include content_hash, got keys: "
            f"{list(data.keys())}"
        )
        assert data["content_hash"] is not None

    @pytest.mark.asyncio
    async def test_list_documents_includes_content_hash(
        self, temp_upload_dir, temp_db_path
    ):
        """Listing documents must include content_hash in each response item."""
        import backend.api.documents as api_docs

        api_docs._service = None
        with patch.dict(
            os.environ,
            {
                "DOCUMENTS_UPLOAD_DIR": str(temp_upload_dir),
                "DOCUMENTS_DB_PATH": str(temp_db_path),
            },
        ):
            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                # Upload
                await client.post(
                    "/documents",
                    files={
                        "file": (
                            "list_test.pdf",
                            b"%PDF-1.4\n%%EOF",
                            "application/pdf",
                        )
                    },
                )

                # List
                response = await client.get("/documents")

        assert response.status_code == 200
        data = response.json()
        assert data["total"] >= 1
        for doc in data["documents"]:
            assert "content_hash" in doc, (
                f"DocumentResponse must include content_hash, got keys: "
                f"{list(doc.keys())}"
            )


# ======================================================================
# Reconciliation tests
# ======================================================================


class TestReconciliation:
    """Tests for ChromaDB↔SQLite registry reconciliation."""

    def test_reconcile_empty_both_is_clean(self, temp_db_path, rag_config):
        """When both SQLite and ChromaDB are empty, reconciliation is clean."""
        from backend.documents.reconciliation import reconcile
        from backend.rag.store import init_store

        store = init_store(rag_config)

        result = reconcile(store, temp_db_path)
        assert result.is_clean
        assert result.orphaned_count == 0
        assert result.missing_count == 0

    def test_reconcile_detects_orphaned_chroma_id(self, temp_db_path, rag_config):
        """When a document_id exists in ChromaDB with no SQLite row,
        reconciliation must detect it as orphaned."""
        from backend.documents.reconciliation import reconcile
        from backend.persistence.chroma import ChromaStore
        from backend.rag.store import init_store

        store = init_store(rag_config)

        # Manually add chunks with a document_id not in SQLite
        collection = store.get_or_create_collection()
        collection.add(
            ids=["orphan-chunk-1"],
            documents=["some text for orphaned document"],
            metadatas=[{"document_id": "orphan-doc-id"}],
        )

        result = reconcile(store, temp_db_path)
        assert not result.is_clean
        assert "orphan-doc-id" in result.orphaned_chroma_ids
        assert result.orphaned_count == 1

    def test_reconcile_detects_deleted_with_lingering_chunks(
        self, temp_upload_dir, temp_db_path, rag_config
    ):
        """When a SQLite document is DELETED but ChromaDB chunks remain,
        reconciliation must flag it as orphaned."""
        from datetime import datetime, timezone

        from backend.documents.models import Document, DocumentStatus
        from backend.documents.reconciliation import reconcile
        from backend.persistence.sqlite import (
            init_sqlite,
            insert_document,
            update_document_status,
        )
        from backend.rag.store import init_store

        init_sqlite(temp_db_path)

        doc_id = "deleted-lingering-doc"
        doc = Document(
            document_id=doc_id,
            filename="lingering.pdf",
            status=DocumentStatus.READY,
            uploaded_at=datetime.now(timezone.utc),
            size_bytes=100,
            content_hash="hash-lingering",
        )
        insert_document(doc)

        # Manually add chunks to ChromaDB
        store = init_store(rag_config)
        collection = store.get_or_create_collection()
        collection.add(
            ids=["linger-chunk-1", "linger-chunk-2"],
            documents=["text about post-op care", "more care instructions"],
            metadatas=[
                {
                    "document_id": doc_id,
                    "source_filename": "lingering.pdf",
                    "chunk_index": 0,
                    "page_number": 1,
                    "ingested_at": "2025-01-01T00:00:00+00:00",
                },
                {
                    "document_id": doc_id,
                    "source_filename": "lingering.pdf",
                    "chunk_index": 1,
                    "page_number": 2,
                    "ingested_at": "2025-01-01T00:00:00+00:00",
                },
            ],
        )
        assert store.count() == 2

        # Soft-delete in SQLite but DON'T purge ChromaDB chunks
        update_document_status(doc_id, DocumentStatus.DELETED)

        # Reconcile — should find orphaned
        result = reconcile(store, temp_db_path)
        assert not result.is_clean
        assert doc_id in result.orphaned_chroma_ids, (
            f"Expected {doc_id} in orphaned set, got "
            f"{result.orphaned_chroma_ids}"
        )

    def test_reconcile_detects_missing_chroma_entries(
        self, temp_db_path, rag_config
    ):
        """A document in SQLite with READY status but no ChromaDB chunks
        must be detected as missing."""
        from datetime import datetime, timezone

        from backend.documents.models import Document, DocumentStatus
        from backend.documents.reconciliation import reconcile
        from backend.persistence.sqlite import init_sqlite, insert_document
        from backend.rag.store import init_store

        init_sqlite(temp_db_path)

        doc = Document(
            document_id="missing-chroma-doc",
            filename="missing.pdf",
            status=DocumentStatus.READY,
            uploaded_at=datetime.now(timezone.utc),
            size_bytes=100,
            content_hash="hash-missing",
        )
        insert_document(doc)

        store = init_store(rag_config)
        result = reconcile(store, temp_db_path)
        assert not result.is_clean
        assert "missing-chroma-doc" in result.missing_chroma_ids

    def test_clean_orphaned_chunks_removes_them(self, temp_db_path, rag_config):
        """clean_orphaned_chunks must delete orphaned entries from ChromaDB."""
        from backend.documents.reconciliation import (
            clean_orphaned_chunks,
            reconcile,
        )
        from backend.rag.store import init_store

        store = init_store(rag_config)

        # Add orphaned chunks
        collection = store.get_or_create_collection()
        collection.add(
            ids=["orphan-to-clean-1", "orphan-to-clean-2"],
            documents=["text a", "text b"],
            metadatas=[
                {"document_id": "orphan-clean-id"},
                {"document_id": "orphan-clean-id"},
            ],
        )
        assert store.count() == 2

        cleaned = clean_orphaned_chunks(store, temp_db_path)
        assert "orphan-clean-id" in cleaned
        assert store.count() == 0

    @pytest.mark.asyncio
    async def test_reconcile_api_endpoint_returns_structure(
        self, temp_upload_dir, temp_db_path, rag_config
    ):
        """POST /documents/reconcile must return a valid ReconcileResponse."""
        import backend.api.documents as api_docs

        api_docs._service = None
        chroma_dir = str(rag_config.chroma_persist_dir)
        with patch.dict(
            os.environ,
            {
                "DOCUMENTS_UPLOAD_DIR": str(temp_upload_dir),
                "DOCUMENTS_DB_PATH": str(temp_db_path),
                "RAG_CHROMA_DIR": chroma_dir,
                "RAG_COLLECTION_NAME": rag_config.collection_name,
            },
        ):
            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                response = await client.post("/documents/reconcile")

        assert response.status_code == 200
        data = response.json()
        assert "total_sqlite_docs" in data
        assert "total_chroma_ids" in data
        assert "orphaned_chroma_ids" in data
        assert "missing_chroma_ids" in data
        assert "is_clean" in data
        assert isinstance(data["orphaned_chroma_ids"], list)
        assert isinstance(data["missing_chroma_ids"], list)

    @pytest.mark.asyncio
    async def test_reconcile_api_with_clean_flag(
        self, temp_upload_dir, temp_db_path, rag_config
    ):
        """POST /documents/reconcile?clean=true must clean orphans."""
        import backend.api.documents as api_docs

        api_docs._service = None
        chroma_dir = str(rag_config.chroma_persist_dir)
        with patch.dict(
            os.environ,
            {
                "DOCUMENTS_UPLOAD_DIR": str(temp_upload_dir),
                "DOCUMENTS_DB_PATH": str(temp_db_path),
                "RAG_CHROMA_DIR": chroma_dir,
                "RAG_COLLECTION_NAME": rag_config.collection_name,
            },
        ):
            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                # First add orphaned chunks manually
                from backend.rag.store import init_store
                store = init_store(rag_config)
                collection = store.get_or_create_collection()
                collection.add(
                    ids=["api-orphan-1"],
                    documents=["orphaned text"],
                    metadatas=[{"document_id": "api-orphan-doc"}],
                )

                response = await client.post(
                    "/documents/reconcile", params={"clean": True}
                )

        assert response.status_code == 200
        data = response.json()
        assert "api-orphan-doc" in data["orphaned_chroma_ids"]
        assert data["cleaned_ids"] is not None
        assert "api-orphan-doc" in data["cleaned_ids"]


# ======================================================================
# Retrieval filtering by registry tests
# ======================================================================


class TestRetrievalRegistryFiltering:
    """Tests that retrieval excludes deleted/unregistered document IDs."""

    def test_retrieve_filters_by_valid_document_ids(
        self, temp_db_path, rag_config
    ):
        """When valid_document_ids is provided, chunks outside the set
        must be excluded from results."""
        from backend.rag.retrieval import retrieve
        from backend.rag.store import init_store

        store = init_store(rag_config)

        # Add chunks with two different document IDs
        collection = store.get_or_create_collection()
        collection.add(
            ids=["valid-chunk-1", "orphaned-chunk-1"],
            documents=[
                "cuidado de la herida postoperatoria",
                "texto de documento huérfano",
            ],
            metadatas=[
                {
                    "document_id": "valid-doc",
                    "source_filename": "valid.pdf",
                    "chunk_index": 0,
                    "page_number": 1,
                    "ingested_at": "2025-01-01T00:00:00+00:00",
                },
                {
                    "document_id": "orphan-doc",
                    "source_filename": "orphan.pdf",
                    "chunk_index": 0,
                    "page_number": 1,
                    "ingested_at": "2025-01-01T00:00:00+00:00",
                },
            ],
        )

        # Retrieve with only "valid-doc" in the allowed set
        result = retrieve(
            "cuidado de la herida",
            rag_config,
            store,
            top_k=5,
            valid_document_ids={"valid-doc"},
        )

        doc_ids = {c.document_id for c in result.chunks}
        assert "valid-doc" in doc_ids, (
            "Valid document must appear in filtered results"
        )
        assert "orphan-doc" not in doc_ids, (
            "Orphaned/unregistered document must NOT appear"
        )

    def test_retrieve_without_valid_ids_is_backward_compatible(
        self, temp_db_path, rag_config
    ):
        """When valid_document_ids is None, all chunks are returned
        (backward-compatible behavior)."""
        from backend.rag.retrieval import retrieve
        from backend.rag.store import init_store

        store = init_store(rag_config)

        collection = store.get_or_create_collection()
        collection.add(
            ids=["bc-chunk-1", "bc-chunk-2"],
            documents=["text a", "text b"],
            metadatas=[
                {
                    "document_id": "doc-a",
                    "source_filename": "a.pdf",
                    "chunk_index": 0,
                    "page_number": 1,
                    "ingested_at": "2025-01-01T00:00:00+00:00",
                },
                {
                    "document_id": "doc-b",
                    "source_filename": "b.pdf",
                    "chunk_index": 0,
                    "page_number": 1,
                    "ingested_at": "2025-01-01T00:00:00+00:00",
                },
            ],
        )

        result = retrieve("text", rag_config, store, top_k=5)
        # Without filter, both should be present
        doc_ids = {c.document_id for c in result.chunks}
        assert len(doc_ids) == 2

    def test_get_active_document_ids_excludes_deleted(
        self, temp_db_path
    ):
        """get_active_document_ids must not include deleted document IDs."""
        from backend.documents.models import Document, DocumentStatus
        from backend.persistence.sqlite import (
            get_active_document_ids,
            init_sqlite,
            insert_document,
            update_document_status,
        )
        from datetime import datetime, timezone

        init_sqlite(temp_db_path)

        doc1 = Document(
            document_id="active-1",
            filename="a.pdf",
            status=DocumentStatus.READY,
            uploaded_at=datetime.now(timezone.utc),
            size_bytes=10,
        )
        doc2 = Document(
            document_id="deleted-1",
            filename="b.pdf",
            status=DocumentStatus.READY,
            uploaded_at=datetime.now(timezone.utc),
            size_bytes=10,
        )
        insert_document(doc1)
        insert_document(doc2)
        update_document_status("deleted-1", DocumentStatus.DELETED)

        active_ids = get_active_document_ids()
        assert "active-1" in active_ids
        assert "deleted-1" not in active_ids

    def test_get_active_document_ids_includes_processing(
        self, temp_db_path
    ):
        """get_active_document_ids must include processing documents
        (they are not deleted, just not yet ready)."""
        from backend.documents.models import Document, DocumentStatus
        from backend.persistence.sqlite import (
            get_active_document_ids,
            init_sqlite,
            insert_document,
        )
        from datetime import datetime, timezone

        init_sqlite(temp_db_path)

        doc = Document(
            document_id="processing-1",
            filename="p.pdf",
            status=DocumentStatus.PROCESSING,
            uploaded_at=datetime.now(timezone.utc),
            size_bytes=10,
        )
        insert_document(doc)

        active_ids = get_active_document_ids()
        assert "processing-1" in active_ids

    def test_retrieve_filters_out_deleted_document_chunks(
        self, temp_upload_dir, temp_db_path, rag_config
    ):
        """After uploading a document through the service and then deleting
        it, retrieval with valid_document_ids must exclude its chunks."""
        from backend.documents.service import DocumentService
        from backend.rag.retrieval import retrieve
        from backend.rag.store import init_store

        service = DocumentService(
            upload_dir=temp_upload_dir, db_path=temp_db_path,
        )

        content = b"%PDF-1.4\nsome pdf content for registry filter test\n%%EOF"

        # Upload
        doc = service.upload(content, "filter_test.pdf", rag_config)
        assert doc.content_hash is not None
        doc_id = doc.document_id

        # Retrieve — should find chunks
        store = init_store(rag_config)
        from backend.persistence.sqlite import get_active_document_ids

        valid_ids = get_active_document_ids()
        assert doc_id in valid_ids

        result1 = retrieve(
            "pdf content filter",
            rag_config,
            store,
            top_k=5,
            valid_document_ids=valid_ids,
        )

        # May or may not have results (depends on embedding match with synthetic
        # text), but if it does, doc_id should be in sources
        if result1.has_results:
            doc_ids_1 = {c.document_id for c in result1.chunks}
            assert doc_id in doc_ids_1, (
                "Active document must appear in retrieval"
            )

        # Delete
        service.delete(doc_id, rag_config)

        # Retrieve again — must not include deleted doc
        valid_ids_after = get_active_document_ids()
        assert doc_id not in valid_ids_after

        result2 = retrieve(
            "pdf content filter",
            rag_config,
            store,
            top_k=5,
            valid_document_ids=valid_ids_after,
        )
        doc_ids_2 = {c.document_id for c in result2.chunks}
        assert doc_id not in doc_ids_2, (
            f"Deleted document {doc_id} must not appear in retrieval"
        )


# ======================================================================
# Idempotent corpus ingestion tests
# ======================================================================


class TestCorpusIngestionIdempotency:
    """Tests for explicit, idempotent corpus ingestion."""

    @pytest.mark.slow
    def test_ingest_same_pdf_twice_produces_one_registry_record(
        self, temp_upload_dir, temp_db_path, test_pdf, rag_config
    ):
        """Uploading the same PDF content twice through the service must
        produce exactly one registry record (the second call returns the
        existing one)."""
        from backend.documents.service import DocumentService

        service = DocumentService(
            upload_dir=temp_upload_dir, db_path=temp_db_path,
        )

        content = test_pdf.read_bytes()

        doc1 = service.upload(content, test_pdf.name, rag_config)
        doc2 = service.upload(content, test_pdf.name, rag_config)

        # Second upload must return the existing record
        assert doc2.document_id == doc1.document_id
        assert doc2.content_hash == doc1.content_hash
        assert doc2.status.value in ("ready", "failed")

        # Only one record in the list
        all_docs = service.list_all()
        active_docs = [
            d for d in all_docs if d.document_id == doc1.document_id
        ]
        assert len(active_docs) == 1, (
            f"Expected 1 record, got {len(active_docs)}: {active_docs}"
        )

    @pytest.mark.slow
    def test_ingest_same_pdf_after_delete_creates_new_record(
        self, temp_upload_dir, temp_db_path, test_pdf, rag_config
    ):
        """After deleting a corpus document, re-ingesting the same PDF
        must create a new record with a different document_id."""
        from backend.documents.service import DocumentService

        service = DocumentService(
            upload_dir=temp_upload_dir, db_path=temp_db_path,
        )

        content = test_pdf.read_bytes()

        doc1 = service.upload(content, test_pdf.name, rag_config)
        doc1_id = doc1.document_id
        assert doc1.status.value in ("ready", "failed")

        # Delete
        service.delete(doc1_id, rag_config)

        # Re-ingest
        doc2 = service.upload(content, test_pdf.name, rag_config)
        doc2_id = doc2.document_id

        assert doc2_id != doc1_id, (
            "Re-ingestion after delete must produce a new document_id"
        )
        assert doc2.content_hash == doc1.content_hash
        assert doc2.status.value in ("ready", "failed")

        # Verify old is deleted, new is active
        all_docs = service.list_all()
        doc1_status = next(
            d.status.value for d in all_docs if d.document_id == doc1_id
        )
        doc2_status = next(
            d.status.value for d in all_docs if d.document_id == doc2_id
        )
        assert doc1_status == "deleted"
        assert doc2_status in ("ready", "failed")
