"""POST /rag/query — RAG-backed clinical answer endpoint.

Accepts a Spanish clinical question, retrieves relevant chunks from
ChromaDB, generates a validated answer via the permitted LLM (Gemini
1.5 Flash), and returns the answer with traceable source citations.
When the RAG store has no matching documents the endpoint returns
``insufficient_knowledge: true`` without calling the LLM.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.llm.adapter import generate_rag_answer, RagCitation as LlmCitation
from backend.llm.config import LlmConfig
from backend.rag.config import RagConfig
from backend.rag.retrieval import retrieve

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton configs (cached to avoid re-instantiation per request)
# ---------------------------------------------------------------------------

_rag_config: RagConfig | None = None
_llm_config: LlmConfig | None = None


def _get_rag_config() -> RagConfig:
    """Return the cached RAG configuration singleton."""
    global _rag_config
    if _rag_config is None:
        _rag_config = RagConfig()
    return _rag_config


def _get_llm_config() -> LlmConfig:
    """Return the cached LLM configuration singleton."""
    global _llm_config
    if _llm_config is None:
        _llm_config = LlmConfig()
    return _llm_config

# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

rag_router = APIRouter(prefix="/rag", tags=["rag"])

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class Citation(BaseModel):
    """A single traceable source citation returned to the client."""

    chunk_id: str = Field(..., description="Unique chunk identifier")
    document_id: str = Field(..., description="Stable document identifier")
    source_filename: str = Field(
        ..., description="Original PDF filename"
    )
    page_number: int = Field(
        ..., ge=1, description="1-based page number in the source PDF"
    )
    excerpt: str = Field(
        "",
        description="First 200 characters of the cited text "
        "(for traceability)",
    )


class RagQueryRequest(BaseModel):
    """Request body for the RAG query endpoint."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Clinical question in Spanish",
        examples=["¿Cómo debo cuidar mi herida después de la cirugía?"],
    )


class RagQueryResponse(BaseModel):
    """Response body for the RAG query endpoint."""

    query: str = Field(..., description="Original query (echoed)")
    answer: str = Field(..., description="Validated answer in Spanish")
    citations: list[Citation] = Field(
        default_factory=list,
        description="Traceable source citations",
    )
    insufficient_knowledge: bool = Field(
        ...,
        description=(
            "True when the RAG store has no matching documents or the "
            "LLM cannot produce a source-grounded answer"
        ),
    )
    model: str = Field(
        ..., description="Language model used to generate the answer"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _context_chunks_from_retrieval(
    query: str,
    rag_config: RagConfig,
) -> list[dict[str, Any]]:
    """Retrieve RAG chunks and normalise them into the shape expected by
    the LLM adapter."""
    result = retrieve(query, config=rag_config)

    return [
        {
            "chunk_id": c.chunk_id,
            "document_id": c.document_id,
            "source_filename": c.source_filename,
            "page_number": c.page_number,
            "text": c.text,
        }
        for c in result.chunks
    ]


def _llm_citation_to_api(c: LlmCitation) -> Citation:
    """Convert an internal ``RagCitation`` to the API response model."""
    return Citation(
        chunk_id=c.chunk_id,
        document_id=c.document_id,
        source_filename=c.source_filename,
        page_number=c.page_number,
        excerpt=c.excerpt,
    )


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@rag_router.post("/query", response_model=RagQueryResponse)
async def rag_query(body: RagQueryRequest) -> RagQueryResponse:
    """Answer a clinical question using RAG-grounded generation.

    Flow:
    1. Retrieve relevant chunks from the ChromaDB collection.
    2. If no chunks match → ``insufficient_knowledge: true`` (no LLM call).
    3. Assemble context, call the permitted LLM (Gemini 1.5 Flash) with a
       Spanish system prompt that restricts answers to the provided sources.
    4. Validate the LLM output for Spanish language, citation integrity, and
       clinical safety.
    5. Return the validated answer with traceable citations.

    The endpoint never invents clinical claims, diagnoses, or medication
    doses not grounded in a retrieved source.
    """
    query = body.query.strip()
    logger.info("RAG query received: %r", query[:120])

    # 1. Retrieve
    rag_config = _get_rag_config()
    context_chunks = _context_chunks_from_retrieval(query, rag_config)

    # 2. No results → insufficient_knowledge (no LLM call)
    if not context_chunks:
        logger.info("No RAG results for query — returning insufficient_knowledge.")
        return RagQueryResponse(
            query=query,
            answer=(
                "No tengo suficiente información para responder a su "
                "pregunta. Por favor, consulte a su médico tratante para "
                "obtener orientación específica sobre su caso."
            ),
            citations=[],
            insufficient_knowledge=True,
            model="none (no RAG results)",
        )

    # 3. Generate via LLM
    llm_config = _get_llm_config()
    try:
        result = generate_rag_answer(query, context_chunks, llm_config)
    except RuntimeError as exc:
        logger.error("LLM generation failed: %s", exc)
        raise HTTPException(
            status_code=502,
            detail="El servicio de lenguaje no está disponible. "
            "Intente de nuevo más tarde.",
        ) from exc

    # 4. Build response
    citations = [_llm_citation_to_api(c) for c in result.citations]

    # Log validation warnings if present
    if result.validation_warnings:
        logger.warning(
            "RAG query validation warnings: %s",
            "; ".join(result.validation_warnings),
        )

    return RagQueryResponse(
        query=query,
        answer=result.answer,
        citations=citations,
        insufficient_knowledge=result.insufficient_knowledge,
        model=result.model,
    )
