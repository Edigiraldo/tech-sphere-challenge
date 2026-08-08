"""Tests for the POST /rag/query endpoint.

All tests mock both the RAG retrieve (ChromaDB) and LLM adapter so they
execute quickly without external dependencies.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from backend.api.rag import Citation, RagQueryRequest, RagQueryResponse
from backend.main import app


# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------


class TestRagQueryRequest:
    def test_valid_request(self):
        req = RagQueryRequest(query="¿Cómo cuidar mi herida?")
        assert req.query == "¿Cómo cuidar mi herida?"

    def test_empty_query_rejected(self):
        with pytest.raises(ValueError):
            RagQueryRequest(query="")

    def test_query_too_long_rejected(self):
        with pytest.raises(ValueError):
            RagQueryRequest(query="x" * 2001)


# ---------------------------------------------------------------------------
# Response model
# ---------------------------------------------------------------------------


class TestCitationModel:
    def test_citation_construction(self):
        c = Citation(
            chunk_id="c1",
            document_id="doc-1",
            source_filename="guia.pdf",
            page_number=3,
            excerpt="Mantener la herida limpia...",
        )
        assert c.chunk_id == "c1"
        assert c.document_id == "doc-1"
        assert c.source_filename == "guia.pdf"
        assert c.page_number == 3
        assert c.excerpt == "Mantener la herida limpia..."

    def test_citation_page_number_ge_one(self):
        """page_number must be >= 1 (1-based)."""
        with pytest.raises(ValueError):
            Citation(
                chunk_id="c1",
                document_id="doc-1",
                source_filename="f.pdf",
                page_number=0,
            )

    def test_citation_page_number_negative_rejected(self):
        with pytest.raises(ValueError):
            Citation(
                chunk_id="c1",
                document_id="doc-1",
                source_filename="f.pdf",
                page_number=-1,
            )


# ---------------------------------------------------------------------------
# Endpoint — mocked RAG + LLM
# ---------------------------------------------------------------------------


class TestRagQueryEndpoint:
    """Integration-style tests that mock the backend dependencies."""

    @pytest.fixture
    def mock_retrieve(self):
        """Return a MagicMock suitable for ``retrieve`` calls."""
        return MagicMock()

    @pytest.fixture
    def mock_generate(self):
        """Return a MagicMock suitable for ``generate_rag_answer`` calls."""
        return MagicMock()

    # -- Insufficient knowledge (no RAG results) -----------------------------

    @pytest.mark.asyncio
    async def test_no_rag_results_returns_insufficient_knowledge(self):
        """When no RAG chunks are found, no LLM call is made and the
        response has insufficient_knowledge=True."""
        with patch(
            "backend.api.rag._context_chunks_from_retrieval",
            return_value=[],
        ):
            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                response = await client.post(
                    "/rag/query",
                    json={"query": "¿Cómo cuido mi herida?"},
                )

        assert response.status_code == 200
        data = response.json()
        assert data["insufficient_knowledge"] is True
        assert "No tengo suficiente información" in data["answer"]
        assert data["citations"] == []

    # -- Successful RAG + LLM ------------------------------------------------

    @pytest.mark.asyncio
    async def test_successful_rag_answer(self):
        """When RAG chunks are retrieved and the LLM produces a valid answer,
        the endpoint returns the answer with citations."""
        context_chunks = [
            {
                "chunk_id": "c1",
                "document_id": "doc-1",
                "source_filename": "guia.pdf",
                "page_number": 3,
                "text": "Mantener la herida limpia.",
            }
        ]

        mock_result = MagicMock()
        mock_result.answer = (
            "Debe mantener la herida limpia y seca, cambiando "
            "el apósito diariamente."
        )
        mock_result.citations = []
        mock_result.insufficient_knowledge = False
        mock_result.model = "llama-3.1-70b-versatile"
        mock_result.validation_warnings = []

        with patch(
            "backend.api.rag._context_chunks_from_retrieval",
            return_value=context_chunks,
        ):
            with patch(
                "backend.api.rag.generate_rag_answer",
                return_value=mock_result,
            ):
                transport = ASGITransport(app=app)
                async with AsyncClient(
                    transport=transport, base_url="http://test"
                ) as client:
                    response = await client.post(
                        "/rag/query",
                        json={"query": "¿Cómo cuido mi herida?"},
                    )

        assert response.status_code == 200
        data = response.json()
        assert data["insufficient_knowledge"] is False
        assert "herida limpia" in data["answer"]
        assert data["model"] == "llama-3.1-70b-versatile"

    @pytest.mark.asyncio
    async def test_answer_with_citations(self):
        """When the LLM produces citations, they are returned in the
        response."""
        from backend.llm.adapter import RagCitation

        context_chunks = [
            {
                "chunk_id": "c1",
                "document_id": "doc-1",
                "source_filename": "guia.pdf",
                "page_number": 3,
                "text": "Texto de la guía.",
            }
        ]

        mock_result = MagicMock()
        mock_result.answer = "Respuesta basada en la guía."
        mock_result.citations = [
            RagCitation(
                chunk_id="c1",
                document_id="doc-1",
                source_filename="guia.pdf",
                page_number=3,
                excerpt="Texto de la guía.",
            )
        ]
        mock_result.insufficient_knowledge = False
        mock_result.model = "llama-3.1-70b-versatile"
        mock_result.validation_warnings = []

        with patch(
            "backend.api.rag._context_chunks_from_retrieval",
            return_value=context_chunks,
        ):
            with patch(
                "backend.api.rag.generate_rag_answer",
                return_value=mock_result,
            ):
                transport = ASGITransport(app=app)
                async with AsyncClient(
                    transport=transport, base_url="http://test"
                ) as client:
                    response = await client.post(
                        "/rag/query",
                        json={"query": "¿Qué dice la guía?"},
                    )

        assert response.status_code == 200
        data = response.json()
        assert len(data["citations"]) == 1
        assert data["citations"][0]["chunk_id"] == "c1"
        assert data["citations"][0]["document_id"] == "doc-1"
        assert data["citations"][0]["source_filename"] == "guia.pdf"
        assert data["citations"][0]["page_number"] == 3

    @pytest.mark.asyncio
    async def test_llm_insufficient_knowledge(self):
        """When the LLM returns insufficient_knowledge=True, the endpoint
        propagates it."""
        context_chunks = [
            {
                "chunk_id": "c1",
                "document_id": "doc-1",
                "source_filename": "guia.pdf",
                "page_number": 3,
                "text": "Texto irrelevante.",
            }
        ]

        mock_result = MagicMock()
        mock_result.answer = "No tengo información sobre ese tema."
        mock_result.citations = []
        mock_result.insufficient_knowledge = True
        mock_result.model = "llama-3.1-70b-versatile"
        mock_result.validation_warnings = []

        with patch(
            "backend.api.rag._context_chunks_from_retrieval",
            return_value=context_chunks,
        ):
            with patch(
                "backend.api.rag.generate_rag_answer",
                return_value=mock_result,
            ):
                transport = ASGITransport(app=app)
                async with AsyncClient(
                    transport=transport, base_url="http://test"
                ) as client:
                    response = await client.post(
                        "/rag/query",
                        json={"query": "¿Cuál es la dosis?"},
                    )

        assert response.status_code == 200
        data = response.json()
        assert data["insufficient_knowledge"] is True

    # -- Error handling ------------------------------------------------------

    @pytest.mark.asyncio
    async def test_llm_runtime_error_returns_502(self):
        """When the LLM raises RuntimeError, the endpoint returns 502."""
        context_chunks = [
            {
                "chunk_id": "c1",
                "document_id": "doc-1",
                "source_filename": "guia.pdf",
                "page_number": 3,
                "text": "Texto.",
            }
        ]

        with patch(
            "backend.api.rag._context_chunks_from_retrieval",
            return_value=context_chunks,
        ):
            with patch(
                "backend.api.rag.generate_rag_answer",
                side_effect=RuntimeError("API key missing"),
            ):
                transport = ASGITransport(app=app)
                async with AsyncClient(
                    transport=transport, base_url="http://test"
                ) as client:
                    response = await client.post(
                        "/rag/query",
                        json={"query": "pregunta"},
                    )

        assert response.status_code == 502
        data = response.json()
        assert "detail" in data
        assert "no está disponible" in data["detail"]

    # -- Request validation --------------------------------------------------

    @pytest.mark.asyncio
    async def test_empty_query_returns_422(self):
        """FastAPI should reject empty queries at the model level."""
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            response = await client.post(
                "/rag/query",
                json={"query": ""},
            )

        assert response.status_code == 422

    # -- Response schema -----------------------------------------------------

    @pytest.mark.asyncio
    async def test_response_conforms_to_model(self):
        """The response JSON should deserialise into RagQueryResponse."""
        context_chunks = [
            {
                "chunk_id": "c1",
                "document_id": "doc-1",
                "source_filename": "guia.pdf",
                "page_number": 3,
                "text": "Texto.",
            }
        ]

        mock_result = MagicMock()
        mock_result.answer = "Respuesta."
        mock_result.citations = []
        mock_result.insufficient_knowledge = False
        mock_result.model = "llama-3.1-70b-versatile"
        mock_result.validation_warnings = []

        with patch(
            "backend.api.rag._context_chunks_from_retrieval",
            return_value=context_chunks,
        ):
            with patch(
                "backend.api.rag.generate_rag_answer",
                return_value=mock_result,
            ):
                transport = ASGITransport(app=app)
                async with AsyncClient(
                    transport=transport, base_url="http://test"
                ) as client:
                    response = await client.post(
                        "/rag/query",
                        json={"query": "pregunta"},
                    )

        assert response.status_code == 200
        # Verify we can construct the response model
        parsed = RagQueryResponse(**response.json())
        assert parsed.query == "pregunta"
        assert parsed.answer == "Respuesta."
        assert parsed.insufficient_knowledge is False
        assert parsed.model == "llama-3.1-70b-versatile"
