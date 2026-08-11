"""Read-only summary REST endpoint.

GET  /calls/{call_id}/summary  — Return the structured summary for a completed call.

This endpoint is **read-only**: it reads the pre-generated summary from SQLite
(via ``get_summary_for_call``).  It never generates new summaries — that
responsibility belongs to ``backend/api/calls.py`` → ``_persist_call_summary``.

The response model exposes every section the standalone summary page and the
inline call-completion view need: patient demographics, procedure, symptoms,
escalation decision, traceable sources, and recommended next steps.
"""

from __future__ import annotations

import json
import logging

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.persistence.sqlite import get_summary_for_call

logger = logging.getLogger(__name__)

summaries_router = APIRouter(prefix="/calls", tags=["summaries"])


# ---------------------------------------------------------------------------
# Response model
# ---------------------------------------------------------------------------


class SourceItem(BaseModel):
    """A single traceable source citation in the summary response."""

    document_id: str = Field(..., description="Unique document identifier")
    source_filename: str = Field(..., description="Original PDF filename")
    page_number: int = Field(..., ge=0, description="Page number (0 if unknown)")


class SummaryResponse(BaseModel):
    """Read-only structured summary for a completed call.

    Every field is populated from the persisted ``SummaryRecord`` in SQLite.
    Source citations are deserialised from the ``sources_json`` column.
    """

    call_id: str = Field(..., description="Call this summary belongs to")
    summary_id: str = Field(..., description="Unique summary identifier")
    created_at: str = Field(..., description="UTC ISO-8601 timestamp when generated")

    patient_summary: str = Field(
        ..., description="Spanish text: patient name, age, city, EPS"
    )
    procedure_summary: str = Field(
        ..., description="Spanish text: procedure name, post-operative day"
    )
    symptoms_summary: str = Field(
        ..., description="Aggregated Spanish text of patient responses per domain"
    )
    decision_summary: str = Field(
        ..., description="Spanish text: escalation decision, severity, rationale"
    )
    next_steps: str = Field(
        ..., description="Spanish text: recommended next steps"
    )

    sources: list[SourceItem] = Field(
        default_factory=list,
        description="Traceable source citations (may be empty)",
    )


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@summaries_router.get(
    "/{call_id}/summary",
    response_model=SummaryResponse,
    summary="Get the structured summary for a completed call",
)
def get_call_summary(call_id: str) -> SummaryResponse:
    """Return the pre-generated structured summary for *call_id*.

    Returns a ``SummaryResponse`` built from the persisted ``SummaryRecord``
    in SQLite.  This is a **read-only** operation — the summary was already
    generated and persisted when the call ended.

    Returns ``404`` when:
    - The call has no summary (call never ended or still in progress).
    - The call does not exist.
    """
    record = get_summary_for_call(call_id)
    if record is None:
        raise HTTPException(
            status_code=404,
            detail=f"No summary found for call '{call_id}'. "
            "The call may not have ended yet or does not exist.",
        )

    # Deserialise sources from the JSON column.
    sources: list[SourceItem] = []
    try:
        raw = json.loads(record.sources_json)
        if isinstance(raw, list):
            for entry in raw:
                if isinstance(entry, list) and len(entry) >= 2:
                    sources.append(
                        SourceItem(
                            document_id=str(entry[0]) if entry[0] else "",
                            source_filename=str(entry[1]) if entry[1] else "",
                            page_number=int(entry[2]) if len(entry) > 2 and entry[2] else 0,
                        )
                    )
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        logger.warning(
            "Failed to parse sources_json for summary %s (call %s): %s",
            record.summary_id,
            call_id,
            exc,
        )

    return SummaryResponse(
        call_id=record.call_id,
        summary_id=record.summary_id,
        created_at=record.created_at.isoformat(),
        patient_summary=record.patient_summary,
        procedure_summary=record.procedure_summary,
        symptoms_summary=record.symptoms_summary,
        decision_summary=record.decision_summary,
        next_steps=record.next_steps,
        sources=sources,
    )
