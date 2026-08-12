"""Tests for the read-only summary API endpoint (``GET /calls/{call_id}/summary``).

These tests verify:

- ``GET /calls/{call_id}/summary`` returns ``SummaryResponse`` for a completed call.
- Returns ``404`` when no summary exists (call never ended or does not exist).
- The response contains all required fields: patient, procedure, symptoms,
  decision, sources, next_steps.
- Source citations are correctly deserialised from the ``sources_json`` column.
- The endpoint is read-only — it never generates summaries, only reads them.
"""

from __future__ import annotations

import base64
import datetime
import json as _json
import uuid
from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from backend.persistence.sqlite import (
    SummaryRecord,
    insert_call,
    insert_summary,
    _reset_sqlite,
    init_sqlite,
    CallRecord,
)
from backend.main import app

# ---------------------------------------------------------------------------
# Test constants
# ---------------------------------------------------------------------------

_MOCK_AUDIO_BYTES = b"\x00\x01\x02" * 100
_MOCK_AUDIO_B64 = base64.b64encode(_MOCK_AUDIO_BYTES).decode("ascii")
_MOCK_WAV_BYTES = b"RIFF....WAVE...."


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_call_record(
    call_id: str,
    paciente_id: str = "P001",
    state: str = "ENDED",
    escalated: bool = False,
) -> CallRecord:
    now = datetime.datetime.now(datetime.timezone.utc)
    return CallRecord(
        call_id=call_id,
        paciente_id=paciente_id,
        nombre_completo="Test Patient",
        procedimiento="Apendicectomía",
        dia_postop=3,
        eps="EPS Test",
        state=state,
        started_at=now,
        ended_at=now,
        total_turns=10,
        escalated=escalated,
    )


def _make_summary_record(
    call_id: str,
    patient_text: str = "Paciente: Test Patient.",
    procedure_text: str = "Procedimiento: Apendicectomía laparoscópica.",
    symptoms_text: str = "Dolor: sin dolor.\nFiebre: sin fiebre.",
    decision_text: str = "Decisión: No se requirió escalamiento.",
    next_steps_text: str = "Próximos pasos: continuar seguimiento.",
    sources: list | None = None,
) -> SummaryRecord:
    if sources is None:
        sources = [
            ["doc-001", "guia_apendicectomia.pdf", 3],
        ]
    return SummaryRecord(
        summary_id=uuid.uuid4().hex,
        call_id=call_id,
        created_at=datetime.datetime.now(datetime.timezone.utc),
        patient_summary=patient_text,
        procedure_summary=procedure_text,
        symptoms_summary=symptoms_text,
        decision_summary=decision_text,
        sources_json=_json.dumps(sources, ensure_ascii=False),
        next_steps=next_steps_text,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db(tmp_path):
    """Set up a temporary SQLite database for the test."""
    _reset_sqlite()
    db_path = tmp_path / "test_summary_api.db"
    init_sqlite(db_path)
    yield
    _reset_sqlite()


# ---------------------------------------------------------------------------
# GET /calls/{call_id}/summary — success cases
# ---------------------------------------------------------------------------


class TestGetSummarySuccess:
    """Happy-path tests for the summary endpoint."""

    @pytest.mark.asyncio
    async def test_returns_full_summary_for_completed_call(self, db):
        """A completed call with a persisted summary returns all expected fields."""
        call_id = "call-summary-001"
        insert_call(_make_call_record(call_id, state="ENDED"))
        insert_summary(_make_summary_record(call_id))

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get(f"/calls/{call_id}/summary")

        assert response.status_code == 200
        data = response.json()

        # Structural fields
        assert data["call_id"] == call_id
        assert "summary_id" in data
        assert len(data["summary_id"]) > 0
        assert "created_at" in data

        # Content fields
        assert "patient_summary" in data
        assert "Paciente" in data["patient_summary"]
        assert "procedure_summary" in data
        assert "Apendicectomía" in data["procedure_summary"]
        assert "symptoms_summary" in data
        assert "decision_summary" in data
        assert "next_steps" in data

        # Sources
        assert "sources" in data
        assert isinstance(data["sources"], list)
        assert len(data["sources"]) == 1
        src = data["sources"][0]
        assert "document_id" in src
        assert src["document_id"] == "doc-001"
        assert "source_filename" in src
        assert src["source_filename"] == "guia_apendicectomia.pdf"
        assert "page_number" in src
        assert src["page_number"] == 3

    @pytest.mark.asyncio
    async def test_summary_with_multiple_sources(self, db):
        """Multiple source citations are all returned."""
        call_id = "call-summary-multi"
        insert_call(_make_call_record(call_id))
        sources = [
            ["doc-001", "guia_apendicectomia.pdf", 3],
            ["doc-002", "manejo_dolor.pdf", 12],
            ["doc-003", "cuidados_herida.pdf", 7],
        ]
        insert_summary(_make_summary_record(call_id, sources=sources))

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get(f"/calls/{call_id}/summary")

        assert response.status_code == 200
        data = response.json()
        assert len(data["sources"]) == 3
        filenames = [s["source_filename"] for s in data["sources"]]
        assert "guia_apendicectomia.pdf" in filenames
        assert "manejo_dolor.pdf" in filenames
        assert "cuidados_herida.pdf" in filenames

    @pytest.mark.asyncio
    async def test_summary_with_empty_sources(self, db):
        """A summary with no sources returns an empty sources list."""
        call_id = "call-summary-empty-src"
        insert_call(_make_call_record(call_id))
        insert_summary(_make_summary_record(call_id, sources=[]))

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get(f"/calls/{call_id}/summary")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data["sources"], list)
        assert len(data["sources"]) == 0

    @pytest.mark.asyncio
    async def test_escalation_summary_preserves_red_severity(self, db):
        """RED escalation text is preserved verbatim."""
        call_id = "call-red-esc"
        insert_call(_make_call_record(call_id, escalated=True))
        insert_summary(
            _make_summary_record(
                call_id,
                decision_text="ESCALAMIENTO INMEDIATO (ROJO). Razón: dolor severo.",
            )
        )

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get(f"/calls/{call_id}/summary")

        assert response.status_code == 200
        data = response.json()
        assert "ROJO" in data["decision_summary"]
        assert "ESCALAMIENTO INMEDIATO" in data["decision_summary"]

    @pytest.mark.asyncio
    async def test_created_at_is_iso_format(self, db):
        """The created_at field is a valid ISO-8601 string."""
        call_id = "call-iso-check"
        insert_call(_make_call_record(call_id))
        insert_summary(_make_summary_record(call_id))

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get(f"/calls/{call_id}/summary")

        assert response.status_code == 200
        data = response.json()
        # Should parse as ISO datetime
        from datetime import datetime as _dt
        parsed = _dt.fromisoformat(data["created_at"])
        assert parsed.tzinfo is not None  # timezone-aware


# ---------------------------------------------------------------------------
# GET /calls/{call_id}/summary — error cases
# ---------------------------------------------------------------------------


class TestGetSummaryErrors:
    """Error-path tests for the summary endpoint."""

    @pytest.mark.asyncio
    async def test_nonexistent_call_returns_404(self, db):
        """A call_id that does not exist returns 404."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/calls/nonexistent-xyz/summary")

        assert response.status_code == 404
        data = response.json()
        assert "detail" in data

    @pytest.mark.asyncio
    async def test_call_exists_but_no_summary_returns_404(self, db):
        """A call that was created but never completed has no summary."""
        call_id = "call-no-summary"
        insert_call(_make_call_record(call_id, state="QUESTIONS"))  # in progress

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get(f"/calls/{call_id}/summary")

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_malformed_call_id_returns_404(self, db):
        """Any call_id without a summary returns 404."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/calls/../../etc/passwd/summary")

        assert response.status_code == 404


# ---------------------------------------------------------------------------
# Read-only guarantee
# ---------------------------------------------------------------------------


class TestSummaryReadOnly:
    """Verify the summary endpoint is strictly read-only."""

    @pytest.mark.asyncio
    async def test_get_is_safe_and_idempotent(self, db):
        """The endpoint is idempotent — two GETs return the same data."""
        call_id = "call-idempotent"
        insert_call(_make_call_record(call_id))
        insert_summary(_make_summary_record(call_id))

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            r1 = await client.get(f"/calls/{call_id}/summary")
            r2 = await client.get(f"/calls/{call_id}/summary")

        assert r1.status_code == 200
        assert r2.status_code == 200
        d1 = r1.json()
        d2 = r2.json()
        assert d1["call_id"] == d2["call_id"]
        assert d1["patient_summary"] == d2["patient_summary"]
        assert d1["decision_summary"] == d2["decision_summary"]

    @pytest.mark.asyncio
    async def test_other_methods_not_allowed(self, db):
        """POST, PUT, DELETE on the summary endpoint return 405."""
        call_id = "call-methods"
        insert_call(_make_call_record(call_id))
        insert_summary(_make_summary_record(call_id))

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            post = await client.post(f"/calls/{call_id}/summary", json={})
            put = await client.put(f"/calls/{call_id}/summary", json={})
            delete = await client.delete(f"/calls/{call_id}/summary")

        assert post.status_code == 405
        assert put.status_code == 405
        assert delete.status_code == 405


# ---------------------------------------------------------------------------
# XSS safety: content is served as text, not executed
# ---------------------------------------------------------------------------


class TestSummaryXssSafety:
    """Verify persisted text is served safely — HTML is not injected in JSON."""

    @pytest.mark.asyncio
    async def test_html_in_persisted_text_is_preserved_as_string(self, db):
        """HTML-like content in summary fields is returned as plain strings.
        The frontend must use textContent / createTextNode to render safely."""
        call_id = "call-xss"
        insert_call(_make_call_record(call_id))
        insert_summary(
            _make_summary_record(
                call_id,
                patient_text="Paciente: <script>alert('xss')</script> Test.",
            )
        )

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get(f"/calls/{call_id}/summary")

        assert response.status_code == 200
        data = response.json()
        # The content is a plain JSON string — safe for frontend rendering
        # via textContent / createTextNode.
        assert "<script>" in data["patient_summary"]
        assert isinstance(data["patient_summary"], str)


# ---------------------------------------------------------------------------
# Summary endpoint integrated into full call flow
# ---------------------------------------------------------------------------


class TestSummaryAfterFullCall:
    """End-to-end: create call → complete it → verify summary endpoint."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        """Set up SQLite for the full-flow test."""
        _reset_sqlite()
        db_path = tmp_path / "test_full_flow.db"
        init_sqlite(db_path)

        # Also need STT/TTS mocks
        from backend.api.calls import _call_escalations, _call_turn_index, _call_citations, _call_consecutive_yellows
        from backend.api.metrics import metrics_collector

        metrics_collector.reset()
        _call_turn_index.clear()
        _call_escalations.clear()
        _call_citations.clear()
        _call_consecutive_yellows.clear()

        # Build mocks
        async def _mock_stt(audio_data: bytes):
            from backend.voice.models import TranscriptionResult
            return TranscriptionResult(
                text="Sí, claro, todo bien, sin dolor, sin fiebre, herida limpia, tengo buen apetito, duermo bien, camino sin problema",
                language="es",
                duration_seconds=1.5,
                model="whisper-large-v3",
            )

        mock_tts = MagicMock()
        from backend.voice.tts.protocol import TTSResult
        mock_tts.configure_mock(
            **{
                "synthesize.return_value": TTSResult(
                    audio_bytes=_MOCK_WAV_BYTES,
                    sample_rate=24000,
                    duration_ms=100.0,
                    text="mock",
                    voice="ef_dora",
                )
            }
        )

        with patch("backend.api.calls._stt", _mock_stt), patch(
            "backend.api.calls._tts", mock_tts
        ), patch(
            "backend.api.calls.RagConfig", return_value=None
        ), patch(
            "backend.api.calls.LlmConfig", return_value=None
        ), patch(
            "backend.api.calls._get_patients", return_value={}
        ):
            yield

        metrics_collector.reset()
        _call_turn_index.clear()
        _call_escalations.clear()
        _call_citations.clear()
        _call_consecutive_yellows.clear()
        _reset_sqlite()

    @pytest.mark.asyncio
    async def test_summary_available_after_full_call(self):
        """After a full call completes, the summary endpoint returns data."""
        from backend.api.call_store import call_store as global_store

        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            # Create call
            resp = await client.post(
                "/calls",
                json={
                    "patient_id": "P-full-flow",
                    "dia_postop": 2,
                    "procedimiento": "Hernioplastia inguinal",
                    "nombre_completo": "Test Patient",
                    "eps": "EPS",
                },
            )
            assert resp.status_code == 201
            call_id = resp.json()["call_id"]

            # Walk through all turns until ENDED
            for _ in range(10):
                r = await client.post(
                    f"/calls/{call_id}/turn",
                    json={"audio_base64": _MOCK_AUDIO_B64},
                )
                if r.status_code != 200:
                    break
                if r.json().get("call_ended"):
                    break

            # Small delay to let async summary persistence complete
            import asyncio
            await asyncio.sleep(0.1)

            # Now the summary endpoint should return data
            summary_resp = await client.get(f"/calls/{call_id}/summary")
            assert summary_resp.status_code == 200, (
                f"Summary endpoint failed: {summary_resp.text}"
            )
            data = summary_resp.json()
            assert data["call_id"] == call_id
            assert len(data["patient_summary"]) > 0
            assert len(data["procedure_summary"]) > 0
            assert len(data["symptoms_summary"]) > 0
            assert len(data["decision_summary"]) > 0
            assert len(data["next_steps"]) > 0
            assert isinstance(data["sources"], list)

            # Clean up orchestrator
            await global_store.remove(call_id)
