"""Tests for the document lifecycle endpoints and service.

Covers:
- POST /documents (upload success, invalid file, processing failure)
- GET  /documents (listing, filtering)
- DELETE /documents/{id} (success, not-found)
- Retrieval availability after ingestion
- Complete indexed-chunk deletion (no orphaned ChromaDB chunks)
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
