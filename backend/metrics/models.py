"""Typed frozen data models for metrics instrumentation.

Defines ``TurnMetrics`` (per-turn observations), ``CallMetrics``
(aggregated per-call statistics), and ``MetricsSummary`` (global
aggregate across all calls).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# TurnMetrics — per-turn observation
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TurnMetrics:
    """A single conversation-turn observation recorded by the orchestrator.

    All fields are immutable.  Optional component durations and optional
    token counts let callers report only what is available; the aggregation
    layer handles missing values correctly.

    Attributes
    ----------
    call_id : str
        Non-empty identifier for the call this turn belongs to.
    turn_index : int
        Zero-based index of the turn within the call (must be >= 0).
    tts_duration_ms : float | None
        Optional duration of the text-to-speech synthesis for this turn.
    stt_duration_ms : float | None
        Optional duration of the speech-to-text transcription.
    llm_duration_ms : float | None
        Optional duration of the language-model inference.
    total_latency_ms : float
        End-to-end latency for the turn in milliseconds (>= 0).
    input_tokens : int | None
        Optional number of input (prompt) tokens consumed.
    output_tokens : int | None
        Optional number of output (completion) tokens consumed.
    model : str
        Identifier of the language model used (e.g. ``"llama-3.1-70b-versatile"``).
    rag_queries : int
        Number of RAG retrieval queries executed during this turn (>= 0).
    timestamp : datetime
        UTC timestamp when the turn was recorded.
    """

    call_id: str
    turn_index: int
    total_latency_ms: float
    model: str
    rag_queries: int
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    # -- optional component durations ---------------------------------------
    tts_duration_ms: float | None = None
    stt_duration_ms: float | None = None
    llm_duration_ms: float | None = None

    # -- optional token counts ----------------------------------------------
    input_tokens: int | None = None
    output_tokens: int | None = None

    def __post_init__(self) -> None:
        if not self.call_id.strip():
            raise ValueError("call_id must be non-empty")
        if self.turn_index < 0:
            raise ValueError(
                f"turn_index must be >= 0, got {self.turn_index}"
            )
        if self.total_latency_ms < 0:
            raise ValueError(
                f"total_latency_ms must be >= 0, got {self.total_latency_ms}"
            )
        if self.rag_queries < 0:
            raise ValueError(
                f"rag_queries must be >= 0, got {self.rag_queries}"
            )
        if self.tts_duration_ms is not None and self.tts_duration_ms < 0:
            raise ValueError(
                f"tts_duration_ms must be >= 0 if provided, "
                f"got {self.tts_duration_ms}"
            )
        if self.stt_duration_ms is not None and self.stt_duration_ms < 0:
            raise ValueError(
                f"stt_duration_ms must be >= 0 if provided, "
                f"got {self.stt_duration_ms}"
            )
        if self.llm_duration_ms is not None and self.llm_duration_ms < 0:
            raise ValueError(
                f"llm_duration_ms must be >= 0 if provided, "
                f"got {self.llm_duration_ms}"
            )


# ---------------------------------------------------------------------------
# CallMetrics — per-call aggregation
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CallMetrics:
    """Aggregated metrics for a single call.

    Built from a sequence of ``TurnMetrics`` via ``from_turns()``.

    Token aggregates are ``None`` when **all** turns report ``None``
    for that token side; otherwise absent values are treated as zero.
    These semantics prevent a single missing observation from zeroing
    the aggregate while still distinguishing "no data at all".

    ``estimated_cost_usd`` is ``None`` when either ``total_input_tokens``
    or ``total_output_tokens`` is ``None`` (i.e., insufficient data to
    compute cost).

    Attributes
    ----------
    call_id : str
    patient_id : str
    turn_count : int
    total_latency_ms : float
    total_input_tokens : int | None
    total_output_tokens : int | None
    total_rag_queries : int
    model_calls : int
        Number of turns where ``llm_duration_ms`` was provided.
    estimated_cost_usd : float | None
    """

    call_id: str
    patient_id: str
    turn_count: int
    total_latency_ms: float
    total_input_tokens: int | None
    total_output_tokens: int | None
    total_rag_queries: int
    model_calls: int
    estimated_cost_usd: float | None

    @staticmethod
    def from_turns(
        turns: list[TurnMetrics],
        *,
        patient_id: str,
        input_cost_per_million: float = 0.0,
        output_cost_per_million: float = 0.0,
    ) -> CallMetrics:
        """Aggregate a sequence of ``TurnMetrics`` into ``CallMetrics``.

        Token aggregation rules
        -----------------------
        - If **every** turn has ``input_tokens is None``, the aggregate
          is ``None``.  Otherwise, absent values are treated as 0.
        - Same rule for output tokens.

        ``model_calls`` counts turns where ``llm_duration_ms is not None``.

        ``estimated_cost_usd`` is ``None`` when either aggregate token
        side is ``None``.  Cost config values default to 0 so that callers
        who do not need cost can omit them.
        """
        if not turns:
            raise ValueError("at least one turn is required")

        # All turns must share the same call_id — the collector enforces
        # this, but we still validate defensively.
        call_id = turns[0].call_id
        if any(t.call_id != call_id for t in turns):
            raise ValueError("all turns must have the same call_id")

        turn_count = len(turns)

        # -- numeric sums (always present) ----------------------------------
        total_latency_ms = sum(t.total_latency_ms for t in turns)
        total_rag_queries = sum(t.rag_queries for t in turns)

        # -- model calls (turns where llm_duration_ms is provided) ----------
        model_calls = sum(1 for t in turns if t.llm_duration_ms is not None)

        # -- token aggregation with "all-None → None" semantics -------------
        any_input = any(t.input_tokens is not None for t in turns)
        any_output = any(t.output_tokens is not None for t in turns)

        if any_input:
            total_input_tokens = sum(
                t.input_tokens if t.input_tokens is not None else 0
                for t in turns
            )
        else:
            total_input_tokens = None

        if any_output:
            total_output_tokens = sum(
                t.output_tokens if t.output_tokens is not None else 0
                for t in turns
            )
        else:
            total_output_tokens = None

        # -- cost (None when either token side is missing data) -------------
        if total_input_tokens is not None and total_output_tokens is not None:
            from backend.metrics.cost import estimate_cost

            estimated_cost_usd = estimate_cost(
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                input_cost_per_million=input_cost_per_million,
                output_cost_per_million=output_cost_per_million,
            )
        else:
            estimated_cost_usd = None

        return CallMetrics(
            call_id=call_id,
            patient_id=patient_id,
            turn_count=turn_count,
            total_latency_ms=total_latency_ms,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            total_rag_queries=total_rag_queries,
            model_calls=model_calls,
            estimated_cost_usd=estimated_cost_usd,
        )


# ---------------------------------------------------------------------------
# MetricsSummary — global aggregation
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MetricsSummary:
    """Global aggregate of all completed calls.

    Percentile fields are ``None`` when there are no calls or no data
    points for the relevant component.

    Attributes
    ----------
    call_count : int
    total_turns : int
    total_input_tokens : int | None
    total_output_tokens : int | None
    total_rag_queries : int
    total_model_calls : int
    total_estimated_cost_usd : float | None
    latency_p50_ms : float | None
    latency_p95_ms : float | None
    tts_p50_ms : float | None
    tts_p95_ms : float | None
    stt_p50_ms : float | None
    stt_p95_ms : float | None
    llm_p50_ms : float | None
    llm_p95_ms : float | None
    """

    call_count: int
    total_turns: int
    total_input_tokens: int | None
    total_output_tokens: int | None
    total_rag_queries: int
    total_model_calls: int
    total_estimated_cost_usd: float | None

    latency_p50_ms: float | None
    latency_p95_ms: float | None

    tts_p50_ms: float | None
    tts_p95_ms: float | None

    stt_p50_ms: float | None
    stt_p95_ms: float | None

    llm_p50_ms: float | None
    llm_p95_ms: float | None
