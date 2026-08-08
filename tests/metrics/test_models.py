"""Tests for ``backend.metrics.models`` — TurnMetrics, CallMetrics,
MetricsSummary construction and validation."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from backend.metrics.models import CallMetrics, MetricsSummary, TurnMetrics


class TestTurnMetrics:
    """TurnMetrics construction and validation."""

    # -- valid construction -------------------------------------------------

    def test_minimal_construction(self):
        t = TurnMetrics(
            call_id="c1",
            turn_index=0,
            total_latency_ms=100.0,
            model="llama-3.1-70b-versatile",
            rag_queries=1,
        )
        assert t.call_id == "c1"
        assert t.turn_index == 0
        assert t.total_latency_ms == 100.0
        assert t.model == "llama-3.1-70b-versatile"
        assert t.rag_queries == 1
        assert isinstance(t.timestamp, datetime)
        assert t.tts_duration_ms is None
        assert t.stt_duration_ms is None
        assert t.llm_duration_ms is None
        assert t.input_tokens is None
        assert t.output_tokens is None

    def test_full_construction(self):
        ts = datetime(2026, 8, 8, 12, 0, 0, tzinfo=timezone.utc)
        t = TurnMetrics(
            call_id="c1",
            turn_index=2,
            tts_duration_ms=50.0,
            stt_duration_ms=200.0,
            llm_duration_ms=500.0,
            total_latency_ms=800.0,
            input_tokens=100,
            output_tokens=50,
            model="llama-3.1-70b-versatile",
            rag_queries=3,
            timestamp=ts,
        )
        assert t.call_id == "c1"
        assert t.turn_index == 2
        assert t.tts_duration_ms == 50.0
        assert t.stt_duration_ms == 200.0
        assert t.llm_duration_ms == 500.0
        assert t.total_latency_ms == 800.0
        assert t.input_tokens == 100
        assert t.output_tokens == 50
        assert t.rag_queries == 3
        assert t.timestamp == ts

    def test_default_timestamp_is_utc(self):
        t = TurnMetrics(
            call_id="c1",
            turn_index=0,
            total_latency_ms=100.0,
            model="m",
            rag_queries=0,
        )
        assert t.timestamp.tzinfo is not None
        assert t.timestamp.tzinfo == timezone.utc

    def test_turn_index_zero_allowed(self):
        t = TurnMetrics(
            call_id="c1",
            turn_index=0,
            total_latency_ms=1.0,
            model="m",
            rag_queries=0,
        )
        assert t.turn_index == 0

    def test_rag_queries_zero_allowed(self):
        t = TurnMetrics(
            call_id="c1",
            turn_index=0,
            total_latency_ms=1.0,
            model="m",
            rag_queries=0,
        )
        assert t.rag_queries == 0

    def test_component_durations_none_allowed(self):
        t = TurnMetrics(
            call_id="c1",
            turn_index=0,
            total_latency_ms=100.0,
            model="m",
            rag_queries=1,
            tts_duration_ms=None,
            stt_duration_ms=None,
            llm_duration_ms=None,
        )
        assert t.tts_duration_ms is None

    def test_component_durations_zero_allowed(self):
        t = TurnMetrics(
            call_id="c1",
            turn_index=0,
            tts_duration_ms=0.0,
            stt_duration_ms=0.0,
            llm_duration_ms=0.0,
            total_latency_ms=100.0,
            model="m",
            rag_queries=1,
        )
        assert t.tts_duration_ms == 0.0

    # -- validation ---------------------------------------------------------

    def test_empty_call_id_raises(self):
        with pytest.raises(ValueError, match="call_id must be non-empty"):
            TurnMetrics(
                call_id="   ",
                turn_index=0,
                total_latency_ms=100.0,
                model="m",
                rag_queries=0,
            )

    def test_negative_turn_index_raises(self):
        with pytest.raises(ValueError, match="turn_index must be >= 0"):
            TurnMetrics(
                call_id="c1",
                turn_index=-1,
                total_latency_ms=100.0,
                model="m",
                rag_queries=0,
            )

    def test_negative_latency_raises(self):
        with pytest.raises(ValueError, match="total_latency_ms must be >= 0"):
            TurnMetrics(
                call_id="c1",
                turn_index=0,
                total_latency_ms=-0.1,
                model="m",
                rag_queries=0,
            )

    def test_negative_rag_queries_raises(self):
        with pytest.raises(ValueError, match="rag_queries must be >= 0"):
            TurnMetrics(
                call_id="c1",
                turn_index=0,
                total_latency_ms=100.0,
                model="m",
                rag_queries=-1,
            )

    def test_negative_tts_duration_raises(self):
        with pytest.raises(ValueError, match="tts_duration_ms must be >= 0"):
            TurnMetrics(
                call_id="c1",
                turn_index=0,
                tts_duration_ms=-1.0,
                total_latency_ms=100.0,
                model="m",
                rag_queries=0,
            )

    def test_negative_stt_duration_raises(self):
        with pytest.raises(ValueError, match="stt_duration_ms must be >= 0"):
            TurnMetrics(
                call_id="c1",
                turn_index=0,
                stt_duration_ms=-0.1,
                total_latency_ms=100.0,
                model="m",
                rag_queries=0,
            )

    def test_negative_llm_duration_raises(self):
        with pytest.raises(ValueError, match="llm_duration_ms must be >= 0"):
            TurnMetrics(
                call_id="c1",
                turn_index=0,
                llm_duration_ms=-1.0,
                total_latency_ms=100.0,
                model="m",
                rag_queries=0,
            )

    # -- immutability -------------------------------------------------------

    def test_frozen(self):
        t = TurnMetrics(
            call_id="c1",
            turn_index=0,
            total_latency_ms=100.0,
            model="m",
            rag_queries=0,
        )
        with pytest.raises(Exception):
            t.call_id = "c2"  # type: ignore[misc]


class TestCallMetrics:
    """CallMetrics construction and from_turns aggregation."""

    def _turn(
        self,
        call_id: str = "c1",
        turn_index: int = 0,
        llm_duration_ms: float | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        total_latency_ms: float = 100.0,
        rag_queries: int = 1,
        tts_duration_ms: float | None = None,
        stt_duration_ms: float | None = None,
    ) -> TurnMetrics:
        return TurnMetrics(
            call_id=call_id,
            turn_index=turn_index,
            total_latency_ms=total_latency_ms,
            model="test-model",
            rag_queries=rag_queries,
            llm_duration_ms=llm_duration_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            tts_duration_ms=tts_duration_ms,
            stt_duration_ms=stt_duration_ms,
        )

    def test_single_turn_basic_aggregation(self):
        cm = CallMetrics.from_turns(
            [self._turn(turn_index=0, total_latency_ms=200, rag_queries=2)],
            patient_id="p1",
        )
        assert cm.call_id == "c1"
        assert cm.patient_id == "p1"
        assert cm.turn_count == 1
        assert cm.total_latency_ms == 200.0
        assert cm.total_rag_queries == 2
        assert cm.model_calls == 0  # llm_duration_ms is None

    def test_multiple_turns_aggregation(self):
        turns = [
            self._turn(turn_index=0, total_latency_ms=100, rag_queries=1),
            self._turn(turn_index=1, total_latency_ms=200, rag_queries=2),
            self._turn(turn_index=2, total_latency_ms=300, rag_queries=3),
        ]
        cm = CallMetrics.from_turns(turns, patient_id="p1")
        assert cm.turn_count == 3
        assert cm.total_latency_ms == 600.0
        assert cm.total_rag_queries == 6

    def test_model_calls_count(self):
        turns = [
            self._turn(turn_index=0, llm_duration_ms=500.0),
            self._turn(turn_index=1, llm_duration_ms=None),
            self._turn(turn_index=2, llm_duration_ms=300.0),
        ]
        cm = CallMetrics.from_turns(turns, patient_id="p1")
        assert cm.model_calls == 2

    def test_tokens_all_present(self):
        turns = [
            self._turn(turn_index=0, input_tokens=100, output_tokens=50),
            self._turn(turn_index=1, input_tokens=200, output_tokens=75),
        ]
        cm = CallMetrics.from_turns(turns, patient_id="p1")
        assert cm.total_input_tokens == 300
        assert cm.total_output_tokens == 125

    def test_tokens_all_none(self):
        turns = [
            self._turn(turn_index=0, input_tokens=None, output_tokens=None),
            self._turn(turn_index=1, input_tokens=None, output_tokens=None),
        ]
        cm = CallMetrics.from_turns(turns, patient_id="p1")
        assert cm.total_input_tokens is None
        assert cm.total_output_tokens is None

    def test_tokens_some_none_treated_as_zero(self):
        """When at least one turn has a token count, Nones become zero."""
        turns = [
            self._turn(turn_index=0, input_tokens=100, output_tokens=None),
            self._turn(turn_index=1, input_tokens=None, output_tokens=50),
        ]
        cm = CallMetrics.from_turns(turns, patient_id="p1")
        # input: one turn has 100, the other None→0 → total 100
        assert cm.total_input_tokens == 100
        # output: one turn has 50, the other None→0 → total 50
        assert cm.total_output_tokens == 50

    def test_cost_none_when_missing_token_side(self):
        """If either token side is all-None, cost is None."""
        turns = [
            self._turn(turn_index=0, input_tokens=100, output_tokens=None),
        ]
        cm = CallMetrics.from_turns(
            turns,
            patient_id="p1",
            input_cost_per_million=0.50,
            output_cost_per_million=1.50,
        )
        # input present but output all-None → cost None
        assert cm.total_input_tokens == 100
        assert cm.total_output_tokens is None
        assert cm.estimated_cost_usd is None

    def test_cost_when_both_token_sides_present(self):
        turns = [
            self._turn(turn_index=0, input_tokens=1000, output_tokens=500),
        ]
        cm = CallMetrics.from_turns(
            turns,
            patient_id="p1",
            input_cost_per_million=0.50,
            output_cost_per_million=1.50,
        )
        expected = (1000 / 1e6) * 0.50 + (500 / 1e6) * 1.50
        assert cm.estimated_cost_usd == pytest.approx(expected)

    def test_empty_turns_raises(self):
        with pytest.raises(ValueError, match="at least one turn"):
            CallMetrics.from_turns([], patient_id="p1")

    def test_mismatched_call_ids_raises(self):
        turns = [
            self._turn(call_id="c1", turn_index=0),
            self._turn(call_id="c2", turn_index=1),
        ]
        with pytest.raises(ValueError, match="same call_id"):
            CallMetrics.from_turns(turns, patient_id="p1")

    def test_frozen(self):
        cm = CallMetrics(
            call_id="c1",
            patient_id="p1",
            turn_count=1,
            total_latency_ms=100.0,
            total_input_tokens=10,
            total_output_tokens=5,
            total_rag_queries=2,
            model_calls=1,
            estimated_cost_usd=0.0001,
        )
        with pytest.raises(Exception):
            cm.turn_count = 2  # type: ignore[misc]


class TestMetricsSummary:
    """MetricsSummary construction."""

    def test_all_fields_default(self):
        ms = MetricsSummary(
            call_count=0,
            total_turns=0,
            total_input_tokens=None,
            total_output_tokens=None,
            total_rag_queries=0,
            total_model_calls=0,
            total_estimated_cost_usd=None,
            latency_p50_ms=None,
            latency_p95_ms=None,
            tts_p50_ms=None,
            tts_p95_ms=None,
            stt_p50_ms=None,
            stt_p95_ms=None,
            llm_p50_ms=None,
            llm_p95_ms=None,
        )
        assert ms.call_count == 0

    def test_with_percentiles(self):
        ms = MetricsSummary(
            call_count=5,
            total_turns=20,
            total_input_tokens=1000,
            total_output_tokens=500,
            total_rag_queries=10,
            total_model_calls=15,
            total_estimated_cost_usd=0.0025,
            latency_p50_ms=200.0,
            latency_p95_ms=800.0,
            tts_p50_ms=50.0,
            tts_p95_ms=150.0,
            stt_p50_ms=300.0,
            stt_p95_ms=900.0,
            llm_p50_ms=600.0,
            llm_p95_ms=1200.0,
        )
        assert ms.latency_p50_ms == 200.0
        assert ms.llm_p95_ms == 1200.0

    def test_frozen(self):
        ms = MetricsSummary(
            call_count=1,
            total_turns=1,
            total_input_tokens=10,
            total_output_tokens=5,
            total_rag_queries=1,
            total_model_calls=1,
            total_estimated_cost_usd=0.00001,
            latency_p50_ms=100.0,
            latency_p95_ms=100.0,
            tts_p50_ms=None,
            tts_p95_ms=None,
            stt_p50_ms=None,
            stt_p95_ms=None,
            llm_p50_ms=None,
            llm_p95_ms=None,
        )
        with pytest.raises(Exception):
            ms.call_count = 2  # type: ignore[misc]
