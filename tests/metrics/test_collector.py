"""Tests for ``backend.metrics.collector`` — InMemoryMetricsCollector
lifecycle, thread safety, queries, and aggregation."""

from __future__ import annotations

import threading
import time

import pytest

from backend.metrics.collector import InMemoryMetricsCollector
from backend.metrics.models import CallMetrics, MetricsSummary, TurnMetrics


def _turn(
    call_id: str = "c1",
    turn_index: int = 0,
    total_latency_ms: float = 100.0,
    model: str = "test-model",
    rag_queries: int = 1,
    tts_duration_ms: float | None = None,
    stt_duration_ms: float | None = None,
    llm_duration_ms: float | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
) -> TurnMetrics:
    return TurnMetrics(
        call_id=call_id,
        turn_index=turn_index,
        total_latency_ms=total_latency_ms,
        model=model,
        rag_queries=rag_queries,
        tts_duration_ms=tts_duration_ms,
        stt_duration_ms=stt_duration_ms,
        llm_duration_ms=llm_duration_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    """start_call / record_turn / end_call lifecycle."""

    def test_start_and_record(self):
        c = InMemoryMetricsCollector()
        c.start_call("c1", "p1")
        c.record_turn(_turn("c1"))
        c.end_call("c1")

        metrics = c.get_call_metrics("c1")
        assert metrics is not None
        assert metrics.call_id == "c1"
        assert metrics.patient_id == "p1"
        assert metrics.turn_count == 1

    def test_start_already_active_raises(self):
        c = InMemoryMetricsCollector()
        c.start_call("c1", "p1")
        with pytest.raises(ValueError, match="already active"):
            c.start_call("c1", "p2")

    def test_record_before_start_raises(self):
        c = InMemoryMetricsCollector()
        with pytest.raises(ValueError, match="has not been started"):
            c.record_turn(_turn("c1"))

    def test_end_before_start_raises(self):
        c = InMemoryMetricsCollector()
        with pytest.raises(ValueError, match="has not been started"):
            c.end_call("c1")

    def test_end_twice_is_noop(self):
        c = InMemoryMetricsCollector()
        c.start_call("c1", "p1")
        c.record_turn(_turn("c1"))
        c.end_call("c1")
        # Second end_call should succeed silently.
        c.end_call("c1")
        metrics = c.get_call_metrics("c1")
        assert metrics is not None

    def test_get_call_metrics_before_end_returns_none(self):
        c = InMemoryMetricsCollector()
        c.start_call("c1", "p1")
        c.record_turn(_turn("c1"))
        assert c.get_call_metrics("c1") is None

    def test_get_call_metrics_nonexistent_returns_none(self):
        c = InMemoryMetricsCollector()
        assert c.get_call_metrics("nonexistent") is None

    def test_empty_call_id_raises(self):
        c = InMemoryMetricsCollector()
        with pytest.raises(ValueError, match="call_id must be non-empty"):
            c.start_call("   ", "p1")

    def test_empty_patient_id_raises(self):
        c = InMemoryMetricsCollector()
        with pytest.raises(ValueError, match="patient_id must be non-empty"):
            c.start_call("c1", "  ")


# ---------------------------------------------------------------------------
# Queries and aggregation
# ---------------------------------------------------------------------------


class TestQueries:
    """get_call_metrics and get_summary correctness."""

    def test_get_call_metrics_aggregates_turns(self):
        c = InMemoryMetricsCollector()
        c.start_call("c1", "p1")
        c.record_turn(
            _turn("c1", turn_index=0, total_latency_ms=100, rag_queries=2,
                  input_tokens=50, output_tokens=25, llm_duration_ms=500)
        )
        c.record_turn(
            _turn("c1", turn_index=1, total_latency_ms=200, rag_queries=3,
                  input_tokens=100, output_tokens=50, llm_duration_ms=600)
        )
        c.end_call("c1")

        cm = c.get_call_metrics("c1")
        assert cm is not None
        assert cm.turn_count == 2
        assert cm.total_latency_ms == 300.0
        assert cm.total_rag_queries == 5
        assert cm.model_calls == 2
        assert cm.total_input_tokens == 150
        assert cm.total_output_tokens == 75

    def test_get_call_metrics_returns_immutable_snapshot(self):
        """Prevent mutation of internal state through returned snapshot."""
        c = InMemoryMetricsCollector()
        c.start_call("c1", "p1")
        c.record_turn(
            _turn("c1", total_latency_ms=100, llm_duration_ms=500,
                  input_tokens=50, output_tokens=25)
        )
        c.end_call("c1")

        cm = c.get_call_metrics("c1")
        assert cm is not None
        # Frozen dataclass — assignment should raise.
        with pytest.raises(Exception):
            cm.turn_count = 99  # type: ignore[misc]

    def test_get_summary_omits_unended_calls(self):
        c = InMemoryMetricsCollector()
        c.start_call("c1", "p1")
        c.record_turn(
            _turn("c1", total_latency_ms=100, llm_duration_ms=500,
                  input_tokens=50, output_tokens=25)
        )
        c.end_call("c1")

        # Start a second call but don't end it.
        c.start_call("c2", "p2")
        c.record_turn(
            _turn("c2", total_latency_ms=200, llm_duration_ms=600,
                  input_tokens=200, output_tokens=100)
        )

        summary = c.get_summary()
        # Only c1 should be counted.
        assert summary.call_count == 1
        assert summary.total_turns == 1

    def test_get_summary_multiple_calls(self):
        c = InMemoryMetricsCollector()

        c.start_call("c1", "p1")
        c.record_turn(
            _turn("c1", turn_index=0, total_latency_ms=100, rag_queries=2,
                  input_tokens=50, output_tokens=25, llm_duration_ms=500)
        )
        c.end_call("c1")

        c.start_call("c2", "p2")
        c.record_turn(
            _turn("c2", turn_index=0, total_latency_ms=200, rag_queries=1,
                  input_tokens=100, output_tokens=50, llm_duration_ms=600)
        )
        c.record_turn(
            _turn("c2", turn_index=1, total_latency_ms=300, rag_queries=3,
                  input_tokens=150, output_tokens=75, llm_duration_ms=700)
        )
        c.end_call("c2")

        summary = c.get_summary()
        assert summary.call_count == 2
        assert summary.total_turns == 3
        assert summary.total_rag_queries == 6
        assert summary.total_model_calls == 3
        assert summary.total_input_tokens == 300
        assert summary.total_output_tokens == 150

    def test_get_summary_percentiles(self):
        c = InMemoryMetricsCollector()

        c.start_call("c1", "p1")
        for i, lat in enumerate([100.0, 200.0, 300.0, 400.0, 500.0]):
            c.record_turn(
                _turn("c1", turn_index=i, total_latency_ms=lat,
                      llm_duration_ms=lat / 2, tts_duration_ms=lat / 10,
                      stt_duration_ms=lat / 5)
            )
        c.end_call("c1")

        summary = c.get_summary()
        # P50 of [100, 200, 300, 400, 500] = 300
        assert summary.latency_p50_ms == 300.0
        # P95: index = 0.95 * 4 = 3.8 → 400*0.2 + 500*0.8 = 80 + 400 = 480
        assert summary.latency_p95_ms == pytest.approx(480.0)
        assert summary.llm_p50_ms == 150.0
        assert summary.tts_p50_ms == 30.0
        assert summary.stt_p50_ms == 60.0

    def test_get_summary_empty_when_no_ended_calls(self):
        c = InMemoryMetricsCollector()
        summary = c.get_summary()
        assert summary.call_count == 0
        assert summary.total_turns == 0
        assert summary.total_input_tokens is None
        assert summary.total_output_tokens is None
        assert summary.total_estimated_cost_usd is None
        assert summary.latency_p50_ms is None

    def test_get_summary_all_none_tokens(self):
        c = InMemoryMetricsCollector()
        c.start_call("c1", "p1")
        c.record_turn(
            _turn("c1", total_latency_ms=100, llm_duration_ms=500,
                  input_tokens=None, output_tokens=None)
        )
        c.end_call("c1")

        summary = c.get_summary()
        assert summary.total_input_tokens is None
        assert summary.total_output_tokens is None
        assert summary.total_estimated_cost_usd is None

    def test_get_summary_mixed_token_presence(self):
        c = InMemoryMetricsCollector()

        # Call 1: has token data
        c.start_call("c1", "p1")
        c.record_turn(
            _turn("c1", total_latency_ms=100, llm_duration_ms=500,
                  input_tokens=50, output_tokens=25)
        )
        c.end_call("c1")

        # Call 2: no token data
        c.start_call("c2", "p2")
        c.record_turn(
            _turn("c2", total_latency_ms=200, llm_duration_ms=600,
                  input_tokens=None, output_tokens=None)
        )
        c.end_call("c2")

        summary = c.get_summary()
        # At least one call has tokens, so total is not None
        assert summary.total_input_tokens == 50
        assert summary.total_output_tokens == 25


# ---------------------------------------------------------------------------
# Missing optional values
# ---------------------------------------------------------------------------


class TestMissingOptionalValues:
    """Behavior when optional fields are absent."""

    def test_call_with_no_llm_duration(self):
        c = InMemoryMetricsCollector()
        c.start_call("c1", "p1")
        c.record_turn(
            _turn("c1", total_latency_ms=100, llm_duration_ms=None)
        )
        c.end_call("c1")

        cm = c.get_call_metrics("c1")
        assert cm is not None
        assert cm.model_calls == 0

    def test_component_percentiles_with_mixed_data(self):
        """Some turns have component durations, some don't."""
        c = InMemoryMetricsCollector()
        c.start_call("c1", "p1")
        c.record_turn(
            _turn("c1", turn_index=0, total_latency_ms=100,
                  llm_duration_ms=500, tts_duration_ms=50)
        )
        c.record_turn(
            _turn("c1", turn_index=1, total_latency_ms=200,
                  llm_duration_ms=None, tts_duration_ms=None)
        )
        c.record_turn(
            _turn("c1", turn_index=2, total_latency_ms=300,
                  llm_duration_ms=700, tts_duration_ms=70)
        )
        c.end_call("c1")

        summary = c.get_summary()
        # LLM percentiles: [500, 700]
        assert summary.llm_p50_ms == pytest.approx(600.0)
        # TTS: [50, 70]
        assert summary.tts_p50_ms == pytest.approx(60.0)

    def test_component_percentiles_none_when_no_data(self):
        c = InMemoryMetricsCollector()
        c.start_call("c1", "p1")
        c.record_turn(
            _turn("c1", total_latency_ms=100,
                  llm_duration_ms=None, tts_duration_ms=None,
                  stt_duration_ms=None)
        )
        c.end_call("c1")

        summary = c.get_summary()
        assert summary.llm_p50_ms is None
        assert summary.tts_p50_ms is None
        assert summary.stt_p50_ms is None


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    """InMemoryMetricsCollector must be safe for concurrent access."""

    def test_concurrent_recording(self):
        c = InMemoryMetricsCollector()
        threads_count = 10
        turns_per_thread = 100
        errors: list[Exception] = []

        c.start_call("c1", "p1")

        def record_turns(start_idx: int) -> None:
            try:
                for i in range(turns_per_thread):
                    idx = start_idx + i
                    c.record_turn(
                        _turn("c1", turn_index=idx,
                              total_latency_ms=float(idx))
                    )
            except Exception as exc:
                errors.append(exc)

        threads: list[threading.Thread] = []
        for t in range(threads_count):
            thread = threading.Thread(
                target=record_turns, args=(t * turns_per_thread,)
            )
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        assert len(errors) == 0

        c.end_call("c1")
        cm = c.get_call_metrics("c1")
        assert cm is not None
        assert cm.turn_count == threads_count * turns_per_thread

    def test_concurrent_start_and_record(self):
        c = InMemoryMetricsCollector()
        errors: list[Exception] = []

        c.start_call("shared", "p1")

        def worker(idx: int) -> None:
            try:
                for _ in range(50):
                    c.record_turn(
                        _turn("shared", turn_index=idx,
                              total_latency_ms=10.0)
                    )
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,))
                   for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

        c.end_call("shared")
        cm = c.get_call_metrics("shared")
        assert cm is not None
        assert cm.turn_count == 8 * 50

    def test_get_summary_during_recording_returns_consistent_snapshot(self):
        c = InMemoryMetricsCollector()
        c.start_call("c1", "p1")
        c.record_turn(
            _turn("c1", turn_index=0, total_latency_ms=100,
                  llm_duration_ms=500)
        )
        c.end_call("c1")

        # Start a second call but keep recording during summary read.
        c.start_call("c2", "p2")
        c.record_turn(
            _turn("c2", turn_index=0, total_latency_ms=200,
                  llm_duration_ms=600)
        )

        # Summary should only see ended calls (c1), even though c2 is active.
        summary = c.get_summary()
        assert summary.call_count == 1
        assert summary.total_turns == 1


# ---------------------------------------------------------------------------
# Defensive recording
# ---------------------------------------------------------------------------


class TestDefensiveRecording:
    """Edge-case and invalid-input scenarios."""

    def test_record_after_end_still_records(self):
        """end_call marks the call as ended but doesn't block further recording.
        This is intentional — the lifecycle just marks the call for
        query inclusion; it doesn't prevent additional turns from being
        appended."""
        c = InMemoryMetricsCollector()
        c.start_call("c1", "p1")
        c.record_turn(_turn("c1", turn_index=0))
        c.end_call("c1")
        c.record_turn(_turn("c1", turn_index=1))
        # Both turns should appear.
        cm = c.get_call_metrics("c1")
        assert cm is not None
        assert cm.turn_count == 2

    def test_call_with_only_optional_fields(self):
        c = InMemoryMetricsCollector()
        c.start_call("c1", "p1")
        c.record_turn(
            _turn("c1", total_latency_ms=50, model="m", rag_queries=0,
                  input_tokens=None, output_tokens=None,
                  llm_duration_ms=None, tts_duration_ms=None,
                  stt_duration_ms=None)
        )
        c.end_call("c1")
        cm = c.get_call_metrics("c1")
        assert cm is not None
        assert cm.total_input_tokens is None
        assert cm.total_output_tokens is None
        assert cm.model_calls == 0
        assert cm.estimated_cost_usd is None
