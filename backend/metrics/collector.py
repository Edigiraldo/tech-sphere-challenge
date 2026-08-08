"""Metrics collector — Protocol and thread-safe in-memory implementation.

Defines the ``MetricsCollector`` Protocol (the public contract for all
collector implementations) and ``InMemoryMetricsCollector``, a
thread-safe, stdlib-only collector that stores turn metrics in memory
and returns immutable aggregate snapshots.

``get_summary()`` computes P50/P95 latency and component-duration
percentiles from the raw per-turn observations stored by the collector.
"""

from __future__ import annotations

import threading
from typing import Protocol

from backend.metrics.models import CallMetrics, MetricsSummary, TurnMetrics
from backend.metrics.percentiles import percentile


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class MetricsCollector(Protocol):
    """Public interface for metrics collection.

    Implementations must be safe to call from multiple threads
    concurrently.
    """

    def start_call(self, call_id: str, patient_id: str) -> None:
        """Register the start of a new call."""
        ...

    def record_turn(self, metrics: TurnMetrics) -> None:
        """Record a single conversation turn."""
        ...

    def end_call(self, call_id: str) -> None:
        """Mark a call as completed."""
        ...

    def get_call_metrics(self, call_id: str) -> CallMetrics | None:
        """Return aggregated metrics for *call_id*, or ``None``."""
        ...

    def get_summary(self) -> MetricsSummary:
        """Return a global aggregate across all completed calls."""
        ...

    def get_all_call_metrics(self) -> list[CallMetrics]:
        """Return per-call aggregates for all ended calls."""
        ...

    def get_call_turns(self, call_id: str) -> list[TurnMetrics]:
        """Return raw per-turn observations for an ended call."""
        ...





# ---------------------------------------------------------------------------
# In-memory implementation
# ---------------------------------------------------------------------------


class InMemoryMetricsCollector:
    """Thread-safe, in-memory ``MetricsCollector`` implementation.

    Stores per-call turn lists in a dictionary protected by a re-entrant
    lock.  The collector supports the full lifecycle — ``start_call``,
    ``record_turn``, ``end_call`` — and provides both per-call and global
    immutable snapshots.

    ``get_call_metrics()`` delegates to ``CallMetrics.from_turns()``
    which treats optional token fields with "all-None → None" semantics.

    ``get_summary()`` computes P50/P95 percentiles from the raw per-turn
    latencies and optional component durations stored for every ended call.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()

        # _turns: call_id → list[TurnMetrics]  (populated by record_turn)
        self._turns: dict[str, list[TurnMetrics]] = {}

        # _patients: call_id → patient_id  (set by start_call)
        self._patients: dict[str, str] = {}

        # _ended: set of call_ids that have been ended
        self._ended: set[str] = set()

    # -- lifecycle ----------------------------------------------------------

    def reset(self) -> None:
        """Clear all internal state (useful for test isolation)."""
        with self._lock:
            self._turns.clear()
            self._patients.clear()
            self._ended.clear()

    def start_call(self, call_id: str, patient_id: str) -> None:
        """Register a new call.

        Raises ``ValueError`` if *call_id* is already active.
        """
        if not call_id.strip():
            raise ValueError("call_id must be non-empty")
        if not patient_id.strip():
            raise ValueError("patient_id must be non-empty")

        with self._lock:
            if call_id in self._turns:
                raise ValueError(
                    f"call_id {call_id!r} is already active"
                )
            self._turns[call_id] = []
            self._patients[call_id] = patient_id

    def record_turn(self, metrics: TurnMetrics) -> None:
        """Record a turn observation.

        Raises ``ValueError`` if the *call_id* has not been started via
        ``start_call()``.
        """
        with self._lock:
            if metrics.call_id not in self._turns:
                raise ValueError(
                    f"call_id {metrics.call_id!r} has not been started"
                )
            self._turns[metrics.call_id].append(metrics)

    def end_call(self, call_id: str) -> None:
        """Mark a call as ended.

        Raises ``ValueError`` if *call_id* has not been started.
        No-op (succeeds silently) if already ended.
        """
        with self._lock:
            if call_id not in self._turns:
                raise ValueError(
                    f"call_id {call_id!r} has not been started"
                )
            self._ended.add(call_id)

    # -- queries ------------------------------------------------------------

    def get_call_metrics(self, call_id: str) -> CallMetrics | None:
        """Return aggregated metrics for *call_id*, or ``None``.

        Only calls that have been explicitly ended and have at least one
        recorded turn are included.
        """
        with self._lock:
            if call_id not in self._ended:
                return None
            turns = self._turns.get(call_id)
            if turns is None or len(turns) == 0:
                return None
            patient_id = self._patients[call_id]
            # Take a copy so the aggregation runs outside the lock.
            turns_copy = list(turns)

        return CallMetrics.from_turns(turns_copy, patient_id=patient_id)

    def get_all_call_metrics(self) -> list[CallMetrics]:
        """Return per-call aggregates for all ended calls.

        Only calls that have been explicitly ended and have at least one
        recorded turn are included.  Results are sorted by *call_id*.

        The method snapshots the raw data under the lock and builds
        ``CallMetrics`` aggregates outside the lock, consistent with
        ``get_summary()``.
        """
        with self._lock:
            snapshots: list[tuple[str, str, list[TurnMetrics]]] = []
            for call_id in sorted(self._ended):
                turns = self._turns.get(call_id)
                if turns is None or len(turns) == 0:
                    continue
                patient_id = self._patients[call_id]
                turns_copy = list(turns)
                snapshots.append((call_id, patient_id, turns_copy))

        return [
            CallMetrics.from_turns(turns, patient_id=patient_id)
            for call_id, patient_id, turns in snapshots
        ]

    def get_call_turns(self, call_id: str) -> list[TurnMetrics]:
        """Return raw per-turn observations for *call_id*.

        Only ended calls with at least one recorded turn return data.
        Returns an empty list when *call_id* has not been started, has
        not been ended, or has no turns.
        """
        with self._lock:
            if call_id not in self._ended:
                return []
            turns = self._turns.get(call_id)
            if turns is None or len(turns) == 0:
                return []
            return list(turns)

    def get_summary(self) -> MetricsSummary:
        """Return a global aggregate across all ended calls with
        per-turn P50/P95 percentiles.

        Only calls that have been explicitly ended and have at least one
        recorded turn contribute to the summary.
        """
        # Snapshot ended-call data under the lock, then build the
        # summary outside the lock for safety.
        with self._lock:
            snapshots: list[tuple[str, list[TurnMetrics]]] = []
            for call_id in sorted(self._ended):
                turns = self._turns.get(call_id)
                if turns is None or len(turns) == 0:
                    continue
                snapshots.append((call_id, list(turns)))

            patients_snapshot = dict(self._patients)

        # -- Build CallMetrics for each ended call --------------------------
        call_metrics_list: list[CallMetrics] = []
        for call_id, turns in snapshots:
            patient_id = patients_snapshot[call_id]
            call_metrics_list.append(
                CallMetrics.from_turns(turns, patient_id=patient_id)
            )

        # -- Gather all per-turn observations for percentiles ---------------
        all_total_latencies: list[float] = []
        all_tts_durations: list[float] = []
        all_stt_durations: list[float] = []
        all_llm_durations: list[float] = []

        for _, turns in snapshots:
            for t in turns:
                all_total_latencies.append(t.total_latency_ms)
                if t.tts_duration_ms is not None:
                    all_tts_durations.append(t.tts_duration_ms)
                if t.stt_duration_ms is not None:
                    all_stt_durations.append(t.stt_duration_ms)
                if t.llm_duration_ms is not None:
                    all_llm_durations.append(t.llm_duration_ms)

        # -- Aggregate numeric fields ---------------------------------------
        call_count = len(call_metrics_list)
        total_turns = sum(c.turn_count for c in call_metrics_list)
        total_rag_queries = sum(c.total_rag_queries for c in call_metrics_list)
        total_model_calls = sum(c.model_calls for c in call_metrics_list)

        # Token aggregation (all-None → None)
        any_input = any(
            c.total_input_tokens is not None for c in call_metrics_list
        )
        any_output = any(
            c.total_output_tokens is not None for c in call_metrics_list
        )

        total_input_tokens: int | None
        total_output_tokens: int | None

        if any_input:
            total_input_tokens = sum(
                c.total_input_tokens if c.total_input_tokens is not None else 0
                for c in call_metrics_list
            )
        else:
            total_input_tokens = None

        if any_output:
            total_output_tokens = sum(
                c.total_output_tokens if c.total_output_tokens is not None else 0
                for c in call_metrics_list
            )
        else:
            total_output_tokens = None

        # Cost aggregation
        any_cost = any(
            c.estimated_cost_usd is not None for c in call_metrics_list
        )
        if any_cost:
            total_estimated_cost_usd = sum(
                c.estimated_cost_usd
                if c.estimated_cost_usd is not None
                else 0.0
                for c in call_metrics_list
            )
        else:
            total_estimated_cost_usd = None

        return MetricsSummary(
            call_count=call_count,
            total_turns=total_turns,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            total_rag_queries=total_rag_queries,
            total_model_calls=total_model_calls,
            total_estimated_cost_usd=total_estimated_cost_usd,
            latency_p50_ms=percentile(all_total_latencies, 50),
            latency_p95_ms=percentile(all_total_latencies, 95),
            tts_p50_ms=percentile(all_tts_durations, 50),
            tts_p95_ms=percentile(all_tts_durations, 95),
            stt_p50_ms=percentile(all_stt_durations, 50),
            stt_p95_ms=percentile(all_stt_durations, 95),
            llm_p50_ms=percentile(all_llm_durations, 50),
            llm_p95_ms=percentile(all_llm_durations, 95),
        )
