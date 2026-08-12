"""Tests for the read-only metrics REST endpoints.

GET  /metrics/summary           — Global aggregate (empty + populated).
GET  /metrics/calls             — Per-call list (empty + populated).
GET  /metrics/calls/{call_id}  — Call detail (found, 404, per-turn breakdown).

All tests exercise the ``InMemoryMetricsCollector`` singleton via the
FastAPI application.  STT/TTS are not needed — the collector is populated
directly.
"""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from backend.api.metrics import metrics_collector
from backend.main import app
from backend.metrics.models import TurnMetrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_turn(
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


def _populate_call(call_id: str, patient_id: str, turn_count: int = 3) -> None:
    """Start, record turns, and end a call in the collector."""
    metrics_collector.start_call(call_id, patient_id)
    for i in range(turn_count):
        metrics_collector.record_turn(
            _make_turn(
                call_id=call_id,
                turn_index=i,
                total_latency_ms=100.0 + i * 50.0,
                tts_duration_ms=80.0 + i,
                stt_duration_ms=120.0 + i,
                llm_duration_ms=200.0 + i,
                input_tokens=500 + i * 100,
                output_tokens=200 + i * 50,
                rag_queries=1,
            )
        )
    metrics_collector.end_call(call_id)


def _start_but_not_end(call_id: str, patient_id: str) -> None:
    """Start a call and record a turn, but do NOT end it."""
    metrics_collector.start_call(call_id, patient_id)
    metrics_collector.record_turn(_make_turn(call_id=call_id))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_collector():
    """Reset the collector singleton before every test so tests are isolated."""
    from backend.api import metrics as metrics_mod
    from backend.api import calls as calls_mod

    metrics_mod.metrics_collector.reset()
    calls_mod._call_turn_index.clear()

    yield

    metrics_mod.metrics_collector.reset()
    calls_mod._call_turn_index.clear()


# ---------------------------------------------------------------------------
# GET /metrics/summary — empty state
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_summary_empty():
    """When no calls have been ended, all counts are zero and percentiles null."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/metrics/summary")

    assert resp.status_code == 200
    data = resp.json()
    assert data["call_count"] == 0
    assert data["total_turns"] == 0
    assert data["total_input_tokens"] is None
    assert data["total_output_tokens"] is None
    assert data["total_rag_queries"] == 0
    assert data["total_model_calls"] == 0
    assert data["total_estimated_cost_usd"] is None
    assert data["latency_p50_ms"] is None
    assert data["latency_p95_ms"] is None
    assert data["tts_p50_ms"] is None
    assert data["tts_p95_ms"] is None
    assert data["stt_p50_ms"] is None
    assert data["stt_p95_ms"] is None
    assert data["llm_p50_ms"] is None
    assert data["llm_p95_ms"] is None


# ---------------------------------------------------------------------------
# GET /metrics/summary — populated
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_summary_with_data():
    """A call with 3 turns produces correct aggregated summary."""
    _populate_call("c-abc", "p-123", turn_count=3)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/metrics/summary")

    assert resp.status_code == 200
    data = resp.json()
    assert data["call_count"] == 1
    assert data["total_turns"] == 3
    assert data["total_input_tokens"] == 500 + 600 + 700  # 1800
    assert data["total_output_tokens"] == 200 + 250 + 300  # 750
    assert data["total_rag_queries"] == 3
    assert data["total_model_calls"] == 3
    assert data["total_estimated_cost_usd"] is not None  # 0.0 (zero-cost rates)
    assert data["total_estimated_cost_usd"] == 0.0
    assert data["latency_p50_ms"] is not None
    assert data["latency_p95_ms"] is not None
    assert data["tts_p50_ms"] is not None
    assert data["tts_p95_ms"] is not None
    assert data["stt_p50_ms"] is not None
    assert data["stt_p95_ms"] is not None
    assert data["llm_p50_ms"] is not None
    assert data["llm_p95_ms"] is not None


@pytest.mark.asyncio
async def test_summary_excludes_in_progress_calls():
    """Calls that are started but not ended are excluded from the summary."""
    _start_but_not_end("c-in-progress", "p-456")

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/metrics/summary")

    assert resp.status_code == 200
    data = resp.json()
    assert data["call_count"] == 0


@pytest.mark.asyncio
async def test_summary_multiple_calls():
    """Two ended calls produce correct aggregated counts and percentiles."""
    _populate_call("c-a", "p-a", turn_count=2)
    _populate_call("c-b", "p-b", turn_count=4)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/metrics/summary")

    assert resp.status_code == 200
    data = resp.json()
    assert data["call_count"] == 2
    assert data["total_turns"] == 6
    assert data["total_rag_queries"] == 6
    assert data["total_model_calls"] == 6


# ---------------------------------------------------------------------------
# GET /metrics/summary — optional values serialise as null
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_summary_optional_values_null():
    """When no optional component or token data is provided, fields are null."""
    metrics_collector.start_call("c-null", "p-null")
    metrics_collector.record_turn(
        _make_turn(
            call_id="c-null",
            turn_index=0,
            total_latency_ms=50.0,
            model="test",
            rag_queries=0,
            # No optional durations or tokens
        )
    )
    metrics_collector.end_call("c-null")

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/metrics/summary")

    assert resp.status_code == 200
    data = resp.json()
    assert data["call_count"] == 1
    assert data["total_input_tokens"] is None
    assert data["total_output_tokens"] is None
    assert data["total_estimated_cost_usd"] is None
    assert data["latency_p50_ms"] == 50.0
    assert data["latency_p95_ms"] == 50.0
    # Component percentiles should be None (no data points)
    assert data["tts_p50_ms"] is None
    assert data["tts_p95_ms"] is None
    assert data["stt_p50_ms"] is None
    assert data["stt_p95_ms"] is None
    assert data["llm_p50_ms"] is None
    assert data["llm_p95_ms"] is None


# ---------------------------------------------------------------------------
# GET /metrics/calls — empty + populated
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_calls_list_empty():
    """When no calls have been ended, the calls list is empty."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/metrics/calls")

    assert resp.status_code == 200
    data = resp.json()
    assert data["calls"] == []


@pytest.mark.asyncio
async def test_calls_list_with_data():
    """An ended call appears in the calls list."""
    _populate_call("c-abc", "p-123", turn_count=3)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/metrics/calls")

    assert resp.status_code == 200
    data = resp.json()
    assert len(data["calls"]) == 1
    call = data["calls"][0]
    assert call["call_id"] == "c-abc"
    assert call["patient_id"] == "p-123"
    assert call["turn_count"] == 3
    assert call["total_latency_ms"] == 100.0 + 150.0 + 200.0  # 450
    assert call["total_input_tokens"] == 1800
    assert call["total_output_tokens"] == 750
    assert call["total_rag_queries"] == 3
    assert call["model_calls"] == 3


@pytest.mark.asyncio
async def test_calls_list_excludes_in_progress():
    """In-progress calls are excluded from the calls list."""
    _start_but_not_end("c-progress", "p-789")

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/metrics/calls")

    assert resp.status_code == 200
    data = resp.json()
    assert data["calls"] == []


# ---------------------------------------------------------------------------
# GET /metrics/calls/{call_id} — detail
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_call_detail_found():
    """Call detail returns per-call aggregate and per-turn breakdown."""
    _populate_call("c-detail", "p-detail", turn_count=2)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/metrics/calls/c-detail")

    assert resp.status_code == 200
    data = resp.json()
    assert data["call_id"] == "c-detail"
    assert data["patient_id"] == "p-detail"
    assert data["turn_count"] == 2
    assert len(data["turns"]) == 2

    # First turn
    t0 = data["turns"][0]
    assert t0["turn_index"] == 0
    assert t0["total_latency_ms"] == 100.0
    assert t0["tts_duration_ms"] == 80.0
    assert t0["stt_duration_ms"] == 120.0
    assert t0["llm_duration_ms"] == 200.0
    assert t0["input_tokens"] == 500
    assert t0["output_tokens"] == 200
    assert t0["estimated_cost_usd"] is not None
    assert t0["rag_queries"] == 1
    assert "timestamp" in t0
    assert t0["model"] == "test-model"

    # Second turn
    t1 = data["turns"][1]
    assert t1["turn_index"] == 1


@pytest.mark.asyncio
async def test_call_detail_not_found():
    """Non-existent call_id returns 404."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/metrics/calls/nonexistent")

    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_call_detail_in_progress_returns_404():
    """A call that is started but not ended returns 404."""
    _start_but_not_end("c-pending", "p-pending")

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/metrics/calls/c-pending")

    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /metrics/calls/{call_id} — optional values in turns
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_call_detail_optional_fields_null():
    """Turn fields with no optional data serialize as null."""
    metrics_collector.start_call("c-opt", "p-opt")
    metrics_collector.record_turn(
        _make_turn(
            call_id="c-opt",
            turn_index=0,
            total_latency_ms=75.0,
            model="test",
            rag_queries=0,
        )
    )
    metrics_collector.end_call("c-opt")

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/metrics/calls/c-opt")

    assert resp.status_code == 200
    data = resp.json()
    t0 = data["turns"][0]
    assert t0["tts_duration_ms"] is None
    assert t0["stt_duration_ms"] is None
    assert t0["llm_duration_ms"] is None
    assert t0["input_tokens"] is None
    assert t0["output_tokens"] is None
    assert t0["estimated_cost_usd"] is None


# ---------------------------------------------------------------------------
# Route regression — metrics routes do not interfere with existing routes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_still_works_with_metrics_router():
    """GET /health returns 200 after metrics router is registered."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_metrics_html_served():
    """GET /metrics returns the metrics HTML page."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/metrics")

    assert resp.status_code == 200
    content_type = resp.headers.get("content-type", "")
    assert "text/html" in content_type
    body = resp.text
    assert "Métricas del Sistema" in body
    assert "Resumen Global" in body
    assert "Llamadas Completadas" in body


@pytest.mark.asyncio
async def test_metrics_js_served():
    """GET /static/metrics.js returns JavaScript."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/static/metrics.js")

    assert resp.status_code == 200
    content_type = resp.headers.get("content-type", "")
    assert "javascript" in content_type or "text/javascript" in content_type
    body = resp.text
    assert "Métricas del Sistema" in body or "Metrics Dashboard" in body
    assert "fetchJSON" in body


# ---------------------------------------------------------------------------
# Percentile correctness via API
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_summary_percentiles_correct():
    """Percentiles computed via the API match expected values."""
    # Record turns with known latencies: 100, 200, 300, 400
    metrics_collector.start_call("c-pct", "p-pct")
    for i, lat in enumerate([100.0, 200.0, 300.0, 400.0]):
        metrics_collector.record_turn(
            _make_turn(
                call_id="c-pct",
                turn_index=i,
                total_latency_ms=lat,
                tts_duration_ms=lat,
                llm_duration_ms=lat + 10,
                stt_duration_ms=lat - 10,
            )
        )
    metrics_collector.end_call("c-pct")

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/metrics/summary")

    assert resp.status_code == 200
    data = resp.json()
    # P50 of [100, 200, 300, 400] = 250
    assert data["latency_p50_ms"] == 250.0
    # P95 of [100, 200, 300, 400] → index = 0.95 * 3 = 2.85 → between 300 and 400
    assert data["latency_p95_ms"] == pytest.approx(385.0, rel=1e-6)
    assert data["tts_p50_ms"] == 250.0
    assert data["tts_p95_ms"] == pytest.approx(385.0, rel=1e-6)
    # stt = lat - 10 → [90, 190, 290, 390]
    assert data["stt_p50_ms"] == 240.0
    assert data["stt_p95_ms"] == pytest.approx(375.0, rel=1e-6)
    # llm = lat + 10 → [110, 210, 310, 410]
    assert data["llm_p50_ms"] == 260.0
    assert data["llm_p95_ms"] == pytest.approx(395.0, rel=1e-6)


# ---------------------------------------------------------------------------
# Multiple calls sorted by call_id
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_calls_list_sorted_by_call_id():
    """Multiple ended calls are returned sorted by call_id."""
    _populate_call("c-ccc", "p-c", turn_count=1)
    _populate_call("c-aaa", "p-a", turn_count=1)
    _populate_call("c-bbb", "p-b", turn_count=1)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/metrics/calls")

    assert resp.status_code == 200
    data = resp.json()
    call_ids = [c["call_id"] for c in data["calls"]]
    assert call_ids == ["c-aaa", "c-bbb", "c-ccc"]


# ---------------------------------------------------------------------------
# Token and component-duration fields in turn detail
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_call_detail_includes_token_fields_when_populated():
    """When a turn has non-None token counts, they appear in the
    per-turn detail response."""
    metrics_collector.start_call("c-tokens", "p-tokens")
    metrics_collector.record_turn(
        _make_turn(
            call_id="c-tokens",
            turn_index=0,
            total_latency_ms=200.0,
            model="test-model",
            rag_queries=1,
            tts_duration_ms=80.0,
            stt_duration_ms=120.0,
            llm_duration_ms=500.0,
            input_tokens=1024,
            output_tokens=256,
        )
    )
    metrics_collector.end_call("c-tokens")

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/metrics/calls/c-tokens")

    assert resp.status_code == 200
    data = resp.json()
    t0 = data["turns"][0]
    assert t0["turn_index"] == 0
    assert t0["tts_duration_ms"] == 80.0
    assert t0["stt_duration_ms"] == 120.0
    assert t0["llm_duration_ms"] == 500.0
    assert t0["input_tokens"] == 1024
    assert t0["output_tokens"] == 256
    assert t0["rag_queries"] == 1
    assert t0["model"] == "test-model"
    # estimated_cost_usd is computed when both token fields are present.
    assert t0["estimated_cost_usd"] is not None
    assert t0["estimated_cost_usd"] == 0.0  # default zero-cost rates


@pytest.mark.asyncio
async def test_summary_includes_model_calls():
    """The summary correctly counts model_calls (turns with llm_duration_ms)."""
    metrics_collector.start_call("c-mc", "p-mc")
    # Turn with LLM invocation
    metrics_collector.record_turn(
        _make_turn(
            call_id="c-mc",
            turn_index=0,
            total_latency_ms=100.0,
            model="test",
            rag_queries=1,
            llm_duration_ms=300.0,
            input_tokens=500,
            output_tokens=200,
        )
    )
    # Turn without LLM invocation
    metrics_collector.record_turn(
        _make_turn(
            call_id="c-mc",
            turn_index=1,
            total_latency_ms=50.0,
            model="test",
            rag_queries=0,
            # No llm_duration_ms
        )
    )
    metrics_collector.end_call("c-mc")

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/metrics/summary")

    assert resp.status_code == 200
    data = resp.json()
    assert data["call_count"] == 1
    assert data["total_turns"] == 2
    assert data["total_rag_queries"] == 1
    assert data["total_model_calls"] == 1


# ---------------------------------------------------------------------------
# Regression: final turn recorded before call marked ended
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_final_turn_visible_immediately_after_end_call():
    """When a call is ended, all turns including the final one are
    immediately visible via get_call_metrics and get_call_turns — the
    final turn is never dropped regardless of ordering."""
    metrics_collector.start_call("c-final", "p-final")

    # Record turns 0, 1, 2 (including the "final" turn 2 last)
    for idx in range(3):
        metrics_collector.record_turn(
            _make_turn(
                call_id="c-final",
                turn_index=idx,
                total_latency_ms=100.0 + idx * 50,
                tts_duration_ms=80.0 + idx,
                stt_duration_ms=100.0 + idx,
                llm_duration_ms=200.0 + idx,
                input_tokens=500,
                output_tokens=200,
                rag_queries=1,
            )
        )

    # Mark ended AFTER recording all turns (the fixed ordering)
    metrics_collector.end_call("c-final")

    # Immediately query — all 3 turns must be present.
    call_metrics = metrics_collector.get_call_metrics("c-final")
    assert call_metrics is not None
    assert call_metrics.turn_count == 3

    turns = metrics_collector.get_call_turns("c-final")
    assert len(turns) == 3
    assert [t.turn_index for t in turns] == [0, 1, 2]

    # Summary must count the call and its 3 turns.
    summary = metrics_collector.get_summary()
    assert summary.call_count == 1
    assert summary.total_turns == 3


@pytest.mark.asyncio
async def test_call_not_visible_until_ended():
    """A call with recorded turns that has not been ended is NOT visible
    via the public metrics queries (get_call_metrics returns None,
    get_call_turns returns [], summary excludes it)."""
    metrics_collector.start_call("c-active", "p-active")
    metrics_collector.record_turn(
        _make_turn(call_id="c-active", turn_index=0, total_latency_ms=50.0)
    )

    # Not yet ended — queries must exclude it.
    assert metrics_collector.get_call_metrics("c-active") is None
    assert metrics_collector.get_call_turns("c-active") == []
    summary = metrics_collector.get_summary()
    assert summary.call_count == 0


@pytest.mark.asyncio
async def test_record_turn_after_end_call_still_appends():
    """record_turn succeeds even after end_call has been called (the
    collector's _turns dict is not cleared by end_call).  This is a
    design-regression test documenting the current behaviour; it must
    not regress because it would break intermediate teardown sequences."""
    metrics_collector.start_call("c-post-end", "p-post-end")
    metrics_collector.record_turn(
        _make_turn(call_id="c-post-end", turn_index=0, total_latency_ms=50.0)
    )
    metrics_collector.end_call("c-post-end")

    # Record another turn after end_call — must succeed.
    metrics_collector.record_turn(
        _make_turn(call_id="c-post-end", turn_index=1, total_latency_ms=60.0)
    )

    # Both turns are visible because the call is in _ended.
    call_metrics = metrics_collector.get_call_metrics("c-post-end")
    assert call_metrics is not None
    assert call_metrics.turn_count == 2

    turns = metrics_collector.get_call_turns("c-post-end")
    assert len(turns) == 2


@pytest.mark.asyncio
async def test_sequential_final_turn_index_edges():
    """The final turn index in a multi-turn call is the last sequential
    index in the range (len-1), not 0 and not duplicated."""
    metrics_collector.start_call("c-seq", "p-seq")
    for idx in range(5):
        metrics_collector.record_turn(
            _make_turn(
                call_id="c-seq",
                turn_index=idx,
                total_latency_ms=100.0,
                tts_duration_ms=50.0,
                stt_duration_ms=50.0,
                llm_duration_ms=100.0,
            )
        )
    metrics_collector.end_call("c-seq")

    call_metrics = metrics_collector.get_call_metrics("c-seq")
    assert call_metrics is not None
    assert call_metrics.turn_count == 5

    turns = metrics_collector.get_call_turns("c-seq")
    indices = [t.turn_index for t in turns]
    assert indices == [0, 1, 2, 3, 4]
    assert len(set(indices)) == 5  # no duplicates


@pytest.mark.asyncio
async def test_summary_includes_final_turn_data():
    """When a call ends, the global summary correctly aggregates data
    from all turns including the final one."""
    metrics_collector.start_call("c-sum-final", "p-sum-final")
    for idx in range(4):
        metrics_collector.record_turn(
            _make_turn(
                call_id="c-sum-final",
                turn_index=idx,
                total_latency_ms=100.0,
                model="test",
                rag_queries=1,
                tts_duration_ms=80.0,
                stt_duration_ms=120.0,
                llm_duration_ms=200.0,
                input_tokens=500,
                output_tokens=200,
            )
        )
    metrics_collector.end_call("c-sum-final")

    summary = metrics_collector.get_summary()
    assert summary.call_count == 1
    assert summary.total_turns == 4
    assert summary.total_rag_queries == 4
    assert summary.total_model_calls == 4
    assert summary.total_input_tokens == 2000   # 4 * 500
    assert summary.total_output_tokens == 800   # 4 * 200
    # Percentiles must be present (4 data points → P50 = average of middle two)
    assert summary.latency_p50_ms is not None
    assert summary.latency_p95_ms is not None
    assert summary.tts_p50_ms is not None
    assert summary.stt_p50_ms is not None
    assert summary.llm_p50_ms is not None


# ---------------------------------------------------------------------------
# Regression: restart survival (SQLite persistence + reconstruction)
# ---------------------------------------------------------------------------


@pytest.fixture
def _sqlite_metrics_setup(tmp_path):
    """Initialise SQLite on a temp database and reset the collector."""
    from backend.persistence import sqlite as sqlite_mod

    db_path = tmp_path / "metrics_test.db"
    sqlite_mod.init_sqlite(db_path)
    metrics_collector.reset()
    yield
    metrics_collector.reset()
    sqlite_mod._reset_sqlite()


def _persist_ended_call(
    call_id: str,
    patient_id: str,
    turns: list[tuple],
) -> None:
    """Helper: persist a completed call with turn metrics to SQLite.

    Each element of *turns* is a tuple:
    (turn_index, total_latency_ms, tts_duration_ms, stt_duration_ms,
     llm_duration_ms, input_tokens, output_tokens, rag_queries)
    """
    from datetime import datetime, timezone

    from backend.persistence.sqlite import (
        insert_call_metrics,
        insert_turn_metrics_row,
        update_call_metrics_ended,
    )

    insert_call_metrics(call_id, patient_id)

    ts = datetime.now(timezone.utc).isoformat()
    for (
        turn_index,
        total_latency_ms,
        tts_dur,
        stt_dur,
        llm_dur,
        input_tok,
        output_tok,
        rag_q,
    ) in turns:
        insert_turn_metrics_row(
            call_id=call_id,
            turn_index=turn_index,
            total_latency_ms=total_latency_ms,
            model="test-model",
            rag_queries=rag_q,
            timestamp=ts,
            tts_duration_ms=tts_dur,
            stt_duration_ms=stt_dur,
            llm_duration_ms=llm_dur,
            input_tokens=input_tok,
            output_tokens=output_tok,
        )

    update_call_metrics_ended(call_id)


@pytest.mark.asyncio
async def test_single_call_survives_restart_via_sqlite(
    _sqlite_metrics_setup, tmp_path
):
    """A completed call persisted to SQLite is visible through all three
    metrics endpoints after the in-memory collector is reset (simulating
    a restart)."""
    from backend.metrics.collector import load_metrics_from_sqlite

    call_id = "restart-c1"
    patient_id = "restart-p1"

    _persist_ended_call(
        call_id,
        patient_id,
        [
            (0, 100.0, 80.0, 120.0, 200.0, 500, 200, 1),
            (1, 150.0, 90.0, 130.0, 250.0, 600, 250, 1),
            (2, 200.0, 100.0, 140.0, 300.0, 700, 300, 0),
        ],
    )

    # Simulate restart: reset collector and load from SQLite
    metrics_collector.reset()
    n = load_metrics_from_sqlite(metrics_collector)
    assert n == 1

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # GET /metrics/summary
        resp = await client.get("/metrics/summary")
        assert resp.status_code == 200
        summary = resp.json()
        assert summary["call_count"] == 1
        assert summary["total_turns"] == 3
        assert summary["total_rag_queries"] == 2
        assert summary["total_model_calls"] == 3
        assert summary["total_input_tokens"] == 1800  # 500+600+700
        assert summary["total_output_tokens"] == 750  # 200+250+300
        assert summary["latency_p50_ms"] is not None

        # GET /metrics/calls
        resp = await client.get("/metrics/calls")
        assert resp.status_code == 200
        calls = resp.json()["calls"]
        assert len(calls) == 1
        assert calls[0]["call_id"] == call_id
        assert calls[0]["patient_id"] == patient_id
        assert calls[0]["turn_count"] == 3

        # GET /metrics/calls/{call_id}
        resp = await client.get(f"/metrics/calls/{call_id}")
        assert resp.status_code == 200
        detail = resp.json()
        assert detail["call_id"] == call_id
        assert detail["patient_id"] == patient_id
        assert detail["turn_count"] == 3
        assert len(detail["turns"]) == 3
        indices = [t["turn_index"] for t in detail["turns"]]
        assert indices == [0, 1, 2]
        # Verify token fields are present in turn detail
        t0 = detail["turns"][0]
        assert t0["input_tokens"] == 500
        assert t0["output_tokens"] == 200
        assert t0["tts_duration_ms"] == 80.0
        assert t0["stt_duration_ms"] == 120.0
        assert t0["llm_duration_ms"] == 200.0
        assert t0["model"] == "test-model"


@pytest.mark.asyncio
async def test_multiple_calls_survive_restart_via_sqlite(
    _sqlite_metrics_setup, tmp_path
):
    """Multiple completed calls persisted to SQLite are all visible
    after reconstruction."""
    from backend.metrics.collector import load_metrics_from_sqlite

    _persist_ended_call(
        "multi-1", "pat-1",
        [(0, 100.0, 80.0, 120.0, 200.0, 100, 50, 1)],
    )
    _persist_ended_call(
        "multi-2", "pat-2",
        [(0, 110.0, 85.0, 125.0, 210.0, 200, 60, 0),
         (1, 160.0, 95.0, 135.0, 260.0, 300, 80, 1)],
    )
    _persist_ended_call(
        "multi-3", "pat-3",
        [(0, 120.0, 90.0, 130.0, 220.0, 400, 100, 1),
         (1, 170.0, 100.0, 140.0, 270.0, 500, 120, 0),
         (2, 220.0, 110.0, 150.0, 320.0, 600, 140, 1)],
    )

    metrics_collector.reset()
    n = load_metrics_from_sqlite(metrics_collector)
    assert n == 3

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Summary
        resp = await client.get("/metrics/summary")
        assert resp.status_code == 200
        s = resp.json()
        assert s["call_count"] == 3
        assert s["total_turns"] == 6  # 1+2+3

        # Calls list
        resp = await client.get("/metrics/calls")
        calls = resp.json()["calls"]
        assert len(calls) == 3
        call_ids = {c["call_id"] for c in calls}
        assert call_ids == {"multi-1", "multi-2", "multi-3"}
        # Verify turn_count per call
        tc_by_id = {c["call_id"]: c["turn_count"] for c in calls}
        assert tc_by_id["multi-1"] == 1
        assert tc_by_id["multi-2"] == 2
        assert tc_by_id["multi-3"] == 3

        # Per-call detail for multi-2 (most interesting with 2 turns)
        resp = await client.get("/metrics/calls/multi-2")
        assert resp.status_code == 200
        d = resp.json()
        assert d["patient_id"] == "pat-2"
        assert len(d["turns"]) == 2
        assert d["turns"][0]["input_tokens"] == 200
        assert d["turns"][1]["input_tokens"] == 300


@pytest.mark.asyncio
async def test_not_ended_call_not_loaded_via_sqlite(
    _sqlite_metrics_setup, tmp_path
):
    """A calls_metrics row with ended=0 must NOT be loaded by
    reconstruction (call still in progress)."""
    from backend.metrics.collector import load_metrics_from_sqlite
    from backend.persistence.sqlite import (
        insert_call_metrics,
        insert_turn_metrics_row,
    )
    from datetime import datetime, timezone

    call_id = "in-progress-sql"
    patient_id = "pat-progress"
    insert_call_metrics(call_id, patient_id)
    insert_turn_metrics_row(
        call_id=call_id,
        turn_index=0,
        total_latency_ms=100.0,
        model="test",
        rag_queries=0,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
    # NOT calling update_call_metrics_ended — call is in progress

    metrics_collector.reset()
    n = load_metrics_from_sqlite(metrics_collector)
    assert n == 0

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/metrics/calls")
        assert resp.json()["calls"] == []


@pytest.mark.asyncio
async def test_live_and_reconstructed_calls_visible_together(
    _sqlite_metrics_setup, tmp_path
):
    """When the collector has both live in-memory calls and
    SQLite-reconstructed calls, all are visible together."""
    from backend.metrics.collector import load_metrics_from_sqlite

    # First, persist a "historical" call to SQLite
    _persist_ended_call(
        "historical-1", "hp-1",
        [(0, 100.0, 80.0, 120.0, 200.0, 300, 150, 0)],
    )

    # Load the historical call
    metrics_collector.reset()
    n = load_metrics_from_sqlite(metrics_collector)
    assert n == 1

    # Now add a "live" call directly to the in-memory collector
    metrics_collector.start_call("live-1", "lp-1")
    metrics_collector.record_turn(
        _make_turn(
            call_id="live-1",
            turn_index=0,
            total_latency_ms=200.0,
            model="test",
            rag_queries=1,
            tts_duration_ms=90.0,
            stt_duration_ms=130.0,
            llm_duration_ms=300.0,
            input_tokens=400,
            output_tokens=200,
        )
    )
    metrics_collector.end_call("live-1")

    # Both calls should be visible
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/metrics/calls")
        calls = resp.json()["calls"]
        assert len(calls) == 2
        call_ids = {c["call_id"] for c in calls}
        assert call_ids == {"historical-1", "live-1"}

        resp = await client.get("/metrics/summary")
        s = resp.json()
        assert s["call_count"] == 2
        assert s["total_turns"] == 2

        # Verify each call's detail is accessible
        for cid in ("historical-1", "live-1"):
            resp = await client.get(f"/metrics/calls/{cid}")
            assert resp.status_code == 200


@pytest.mark.asyncio
async def test_idempotent_reconstruction_does_not_duplicate(
    _sqlite_metrics_setup, tmp_path
):
    """Calling load_metrics_from_sqlite twice does not duplicate calls."""
    from backend.metrics.collector import load_metrics_from_sqlite

    _persist_ended_call(
        "idem-1", "ip-1",
        [(0, 100.0, 80.0, 120.0, 200.0, 100, 50, 0)],
    )

    # First load
    metrics_collector.reset()
    n1 = load_metrics_from_sqlite(metrics_collector)
    assert n1 == 1

    # Second load must not double-count
    n2 = load_metrics_from_sqlite(metrics_collector)
    assert n2 == 0  # already loaded, start_call raises ValueError

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/metrics/calls")
        calls = resp.json()["calls"]
        assert len(calls) == 1


@pytest.mark.asyncio
async def test_empty_sqlite_loads_zero_calls(
    _sqlite_metrics_setup, tmp_path
):
    """When SQLite has no ended calls, load_metrics_from_sqlite returns 0
    and endpoints remain empty."""
    from backend.metrics.collector import load_metrics_from_sqlite

    metrics_collector.reset()
    n = load_metrics_from_sqlite(metrics_collector)
    assert n == 0

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/metrics/summary")
        assert resp.json()["call_count"] == 0
