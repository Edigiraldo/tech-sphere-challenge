"""Read-only metrics endpoints.

GET  /metrics/summary           — Global aggregate across all ended calls.
GET  /metrics/calls             — Per-call aggregates for all ended calls.
GET  /metrics/calls/{call_id}  — Per-turn detail for a single ended call.

All endpoints are read-only.  The module-level ``metrics_collector`` singleton
is an ``InMemoryMetricsCollector`` shared with other API modules for
instrumentation.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.metrics.collector import InMemoryMetricsCollector
from backend.metrics.cost import estimate_cost
from backend.metrics.models import CallMetrics, TurnMetrics

# ---------------------------------------------------------------------------
# Module-level singleton (shared for instrumentation by other API modules)
# ---------------------------------------------------------------------------

metrics_collector = InMemoryMetricsCollector()

# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

metrics_router = APIRouter(prefix="/metrics", tags=["metrics"])


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class MetricsSummaryResponse(BaseModel):
    """Global aggregate across all ended calls."""

    call_count: int = Field(
        ..., ge=0, description="Number of ended calls"
    )
    total_turns: int = Field(
        ..., ge=0, description="Total turns across all ended calls"
    )
    total_input_tokens: Optional[int] = Field(
        None, description="Total input tokens (None when no data)"
    )
    total_output_tokens: Optional[int] = Field(
        None, description="Total output tokens (None when no data)"
    )
    total_rag_queries: int = Field(
        ..., ge=0, description="Total RAG retrieval queries"
    )
    total_model_calls: int = Field(
        ..., ge=0, description="Total language-model invocations"
    )
    total_estimated_cost_usd: Optional[float] = Field(
        None, description="Total estimated USD cost (None when no data)"
    )
    latency_p50_ms: Optional[float] = Field(
        None, description="P50 end-to-end latency in milliseconds"
    )
    latency_p95_ms: Optional[float] = Field(
        None, description="P95 end-to-end latency in milliseconds"
    )
    tts_p50_ms: Optional[float] = Field(
        None, description="P50 TTS duration in milliseconds"
    )
    tts_p95_ms: Optional[float] = Field(
        None, description="P95 TTS duration in milliseconds"
    )
    stt_p50_ms: Optional[float] = Field(
        None, description="P50 STT duration in milliseconds"
    )
    stt_p95_ms: Optional[float] = Field(
        None, description="P95 STT duration in milliseconds"
    )
    llm_p50_ms: Optional[float] = Field(
        None, description="P50 LLM duration in milliseconds"
    )
    llm_p95_ms: Optional[float] = Field(
        None, description="P95 LLM duration in milliseconds"
    )


class CallMetricsItem(BaseModel):
    """Per-call aggregate returned in the calls list."""

    call_id: str = Field(..., description="Unique call identifier")
    patient_id: str = Field(..., description="Patient identifier")
    turn_count: int = Field(..., ge=0, description="Number of turns")
    total_latency_ms: float = Field(
        ..., ge=0, description="Sum of per-turn latencies"
    )
    total_input_tokens: Optional[int] = Field(
        None, description="Total input tokens (None when no data)"
    )
    total_output_tokens: Optional[int] = Field(
        None, description="Total output tokens (None when no data)"
    )
    total_rag_queries: int = Field(
        ..., ge=0, description="Total RAG retrieval queries"
    )
    model_calls: int = Field(
        ..., ge=0, description="Turns where LLM was invoked"
    )
    estimated_cost_usd: Optional[float] = Field(
        None, description="Estimated USD cost (None when no data)"
    )


class TurnMetricsItem(BaseModel):
    """Per-turn observation returned in the call detail."""

    call_id: str = Field(..., description="Unique call identifier")
    turn_index: int = Field(
        ..., ge=0, description="Zero-based turn index"
    )
    total_latency_ms: float = Field(
        ..., ge=0, description="End-to-end latency in milliseconds"
    )
    model: str = Field(
        ..., min_length=1, description="Language model identifier"
    )
    rag_queries: int = Field(
        ..., ge=0, description="RAG retrieval queries this turn"
    )
    timestamp: str = Field(
        ..., description="UTC ISO-8601 timestamp of the turn"
    )
    tts_duration_ms: Optional[float] = Field(
        None, description="TTS synthesis duration (None when absent)"
    )
    stt_duration_ms: Optional[float] = Field(
        None, description="STT transcription duration (None when absent)"
    )
    llm_duration_ms: Optional[float] = Field(
        None, description="LLM inference duration (None when absent)"
    )
    input_tokens: Optional[int] = Field(
        None, description="Input tokens consumed (None when absent)"
    )
    output_tokens: Optional[int] = Field(
        None, description="Output tokens consumed (None when absent)"
    )
    estimated_cost_usd: Optional[float] = Field(
        None, description="Estimated USD cost (None when token data absent)"
    )

    @classmethod
    def from_turn_metrics(cls, tm: TurnMetrics) -> "TurnMetricsItem":
        """Build a response item from a domain ``TurnMetrics``."""
        cost: Optional[float] = None
        if (
            tm.input_tokens is not None
            and tm.output_tokens is not None
        ):
            cost = estimate_cost(
                input_tokens=tm.input_tokens,
                output_tokens=tm.output_tokens,
                input_cost_per_million=0.0,
                output_cost_per_million=0.0,
            )
        return cls(
            call_id=tm.call_id,
            turn_index=tm.turn_index,
            total_latency_ms=tm.total_latency_ms,
            model=tm.model,
            rag_queries=tm.rag_queries,
            timestamp=tm.timestamp.isoformat(),
            tts_duration_ms=tm.tts_duration_ms,
            stt_duration_ms=tm.stt_duration_ms,
            llm_duration_ms=tm.llm_duration_ms,
            input_tokens=tm.input_tokens,
            output_tokens=tm.output_tokens,
            estimated_cost_usd=cost,
        )


class CallsListResponse(BaseModel):
    """List of per-call aggregates for ended calls."""

    calls: list[CallMetricsItem] = Field(
        default_factory=list, description="Per-call aggregates"
    )


class CallDetailResponse(BaseModel):
    """Full detail for a single ended call, including per-turn breakdown."""

    call_id: str = Field(..., description="Unique call identifier")
    patient_id: str = Field(..., description="Patient identifier")
    turn_count: int = Field(..., ge=0, description="Number of turns")
    total_latency_ms: float = Field(
        ..., ge=0, description="Sum of per-turn latencies"
    )
    total_input_tokens: Optional[int] = Field(
        None, description="Total input tokens (None when no data)"
    )
    total_output_tokens: Optional[int] = Field(
        None, description="Total output tokens (None when no data)"
    )
    total_rag_queries: int = Field(
        ..., ge=0, description="Total RAG retrieval queries"
    )
    model_calls: int = Field(
        ..., ge=0, description="Turns where LLM was invoked"
    )
    estimated_cost_usd: Optional[float] = Field(
        None, description="Estimated USD cost (None when no data)"
    )
    turns: list[TurnMetricsItem] = Field(
        default_factory=list, description="Per-turn observations"
    )


# ---------------------------------------------------------------------------
# Helper: convert a domain CallMetrics → CallMetricsItem
# ---------------------------------------------------------------------------


def _call_metrics_to_item(cm: CallMetrics) -> CallMetricsItem:
    """Convert a domain ``CallMetrics`` to a Pydantic response item."""
    return CallMetricsItem(
        call_id=cm.call_id,
        patient_id=cm.patient_id,
        turn_count=cm.turn_count,
        total_latency_ms=cm.total_latency_ms,
        total_input_tokens=cm.total_input_tokens,
        total_output_tokens=cm.total_output_tokens,
        total_rag_queries=cm.total_rag_queries,
        model_calls=cm.model_calls,
        estimated_cost_usd=cm.estimated_cost_usd,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@metrics_router.get(
    "/summary", response_model=MetricsSummaryResponse
)
async def get_metrics_summary() -> MetricsSummaryResponse:
    """Return a global aggregate across all ended calls.

    Only calls that have been explicitly ended (via ``end_call()``) and
    have at least one recorded turn contribute to the summary.  Calls
    still in progress are excluded.
    """
    summary = metrics_collector.get_summary()
    return MetricsSummaryResponse(
        call_count=summary.call_count,
        total_turns=summary.total_turns,
        total_input_tokens=summary.total_input_tokens,
        total_output_tokens=summary.total_output_tokens,
        total_rag_queries=summary.total_rag_queries,
        total_model_calls=summary.total_model_calls,
        total_estimated_cost_usd=summary.total_estimated_cost_usd,
        latency_p50_ms=summary.latency_p50_ms,
        latency_p95_ms=summary.latency_p95_ms,
        tts_p50_ms=summary.tts_p50_ms,
        tts_p95_ms=summary.tts_p95_ms,
        stt_p50_ms=summary.stt_p50_ms,
        stt_p95_ms=summary.stt_p95_ms,
        llm_p50_ms=summary.llm_p50_ms,
        llm_p95_ms=summary.llm_p95_ms,
    )


@metrics_router.get("/calls", response_model=CallsListResponse)
async def list_call_metrics() -> CallsListResponse:
    """Return per-call aggregates for all ended calls."""
    call_metrics = metrics_collector.get_all_call_metrics()
    return CallsListResponse(
        calls=[_call_metrics_to_item(cm) for cm in call_metrics],
    )


@metrics_router.get(
    "/calls/{call_id}", response_model=CallDetailResponse
)
async def get_call_detail(call_id: str) -> CallDetailResponse:
    """Return full detail for a single ended call, including per-turn
    observations.

    Raises ``404`` if *call_id* has no metrics (never started or never
    ended with at least one turn recorded).
    """
    # Get the per-call aggregate (validates the call exists and is ended).
    call_metrics = metrics_collector.get_call_metrics(call_id)
    if call_metrics is None:
        raise HTTPException(
            status_code=404,
            detail=f"No metrics found for call '{call_id}'. "
            "The call may not have been started, not yet ended, "
            "or have no recorded turns.",
        )

    # Get the raw turns (for per-turn detail).  This uses the internal
    # access method that returns TurnMetrics for ended calls only.
    turns = metrics_collector.get_call_turns(call_id)
    turn_items = [TurnMetricsItem.from_turn_metrics(t) for t in turns]

    return CallDetailResponse(
        call_id=call_metrics.call_id,
        patient_id=call_metrics.patient_id,
        turn_count=call_metrics.turn_count,
        total_latency_ms=call_metrics.total_latency_ms,
        total_input_tokens=call_metrics.total_input_tokens,
        total_output_tokens=call_metrics.total_output_tokens,
        total_rag_queries=call_metrics.total_rag_queries,
        model_calls=call_metrics.model_calls,
        estimated_cost_usd=call_metrics.estimated_cost_usd,
        turns=turn_items,
    )
