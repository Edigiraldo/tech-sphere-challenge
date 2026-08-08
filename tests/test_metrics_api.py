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
