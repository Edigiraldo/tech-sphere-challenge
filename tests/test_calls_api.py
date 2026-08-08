"""Tests for the voice call REST endpoints.

All tests mock STT and TTS dependencies so they execute quickly without
external services.  The ``ConversationOrchestrator`` uses real
``RagConfig`` / ``LlmConfig`` (constructed from env vars) with built-in
safe fallbacks when the underlying providers are unavailable.
"""

from __future__ import annotations

import base64
import datetime
from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from backend.api.calls import (
    CitationResponse,
    CreateCallRequest,
    CreateCallResponse,
    EscalationInfo,
    TurnRequest,
    TurnResponse,
    _decode_base64_audio,
)
from backend.api.call_store import CallStore, call_store as global_store
from backend.conversation.state import State
from backend.voice.models import TranscriptionResult
from backend.voice.tts.protocol import TTSResult
from backend.main import app

# ---------------------------------------------------------------------------
# Reusable test constants
# ---------------------------------------------------------------------------

_MOCK_AUDIO_BYTES = b"\x00\x01\x02" * 100
_MOCK_AUDIO_B64 = base64.b64encode(_MOCK_AUDIO_BYTES).decode("ascii")
_MOCK_WAV_BYTES = b"RIFF....WAVE...."
_MOCK_WAV_B64 = base64.b64encode(_MOCK_WAV_BYTES).decode("ascii")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_stt():
    """Return a mock async STT function that returns a known transcription."""
    async def _transcribe(audio_data: bytes) -> TranscriptionResult:
        return TranscriptionResult(
            text="Sí, acepto continuar con la llamada.",
            language="es",
            duration_seconds=1.5,
            model="whisper-large-v3",
        )
    return _transcribe


@pytest.fixture
def mock_tts():
    """Return a mock TTS adapter whose ``synthesize`` returns known WAV bytes."""
    tts = MagicMock()
    tts.configure_mock(
        **{
            "synthesize.return_value": TTSResult(
                audio_bytes=_MOCK_WAV_BYTES,
                sample_rate=24000,
                duration_ms=100.0,
                text="mock greeting",
                voice="ef_dora",
            )
        }
    )
    return tts


@pytest.fixture(autouse=True)
def setup_voice_mocks(mock_stt, mock_tts):
    """Inject mock STT/TTS into the calls module and reset shared
    module-level state (metrics collector, turn-index counter, patient
    cache) before each test so tests are isolated.

    Uses ``unittest.mock.patch`` on the module-level globals so every
    endpoint call goes through the mocks.

    Also patches ``RagConfig`` / ``LlmConfig`` to return ``None`` so that
    tests do not trigger model downloads or external API connections, and
    ``_get_patients`` to return an empty dict so tests do not read XLSX
    files.  Tests that need to verify config or patient wiring must opt
    out via their own ``patch`` calls.
    """
    from backend.api.calls import _call_escalations, _call_turn_index, _call_citations
    from backend.api.metrics import metrics_collector

    metrics_collector.reset()
    _call_turn_index.clear()
    _call_escalations.clear()
    _call_citations.clear()

    with patch("backend.api.calls._stt", mock_stt), patch(
        "backend.api.calls._tts", mock_tts
    ), patch(
        "backend.api.calls.RagConfig",
        return_value=None,
    ), patch(
        "backend.api.calls.LlmConfig",
        return_value=None,
    ), patch(
        "backend.api.calls._get_patients",
        return_value={},
    ):
        yield

    metrics_collector.reset()
    _call_turn_index.clear()
    _call_escalations.clear()
    _call_citations.clear()


# ---------------------------------------------------------------------------
# CallStore
# ---------------------------------------------------------------------------


class TestCallStore:
    """Unit tests for the thread-safe in-memory call store."""

    @pytest.fixture
    def store(self):
        return CallStore()

    @pytest.mark.asyncio
    async def test_put_and_get(self, store):
        orch = MagicMock()
        orch.call_context.patient_context.call_id = "call-1"
        await store.put("call-1", orch)
        assert await store.get("call-1") is orch

    @pytest.mark.asyncio
    async def test_get_missing_returns_none(self, store):
        assert await store.get("nonexistent") is None

    @pytest.mark.asyncio
    async def test_exists(self, store):
        orch = MagicMock()
        await store.put("call-1", orch)
        assert await store.exists("call-1") is True
        assert await store.exists("missing") is False

    @pytest.mark.asyncio
    async def test_remove(self, store):
        orch = MagicMock()
        await store.put("call-1", orch)
        await store.remove("call-1")
        assert await store.get("call-1") is None

    @pytest.mark.asyncio
    async def test_remove_missing_no_error(self, store):
        await store.remove("nonexistent")  # no-op, no error

    @pytest.mark.asyncio
    async def test_concurrent_put_get(self, store):
        """Multiple puts and gets under concurrency are safe."""
        import asyncio

        async def put_and_get(idx: int) -> bool:
            orch = MagicMock()
            cid = f"call-{idx}"
            await store.put(cid, orch)
            retrieved = await store.get(cid)
            return retrieved is orch

        tasks = [put_and_get(i) for i in range(10)]
        results = await asyncio.gather(*tasks)
        assert all(results)


# ---------------------------------------------------------------------------
# Request / Response model validation
# ---------------------------------------------------------------------------


class TestCreateCallRequest:
    def test_valid(self):
        req = CreateCallRequest(
            patient_id="pac-1",
            dia_postop=3,
            procedimiento="Apendicectomía",
            nombre_completo="María García",
            eps="Compensar EPS",
        )
        assert req.patient_id == "pac-1"
        assert req.dia_postop == 3

    def test_empty_patient_id_rejected(self):
        with pytest.raises(ValueError):
            CreateCallRequest(
                patient_id="",
                dia_postop=3,
                procedimiento="Test",
                nombre_completo="Test",
            )

    def test_negative_dia_postop_rejected(self):
        with pytest.raises(ValueError):
            CreateCallRequest(
                patient_id="pac-1",
                dia_postop=-1,
                procedimiento="Test",
                nombre_completo="Test",
            )

    def test_empty_procedimiento_rejected(self):
        with pytest.raises(ValueError):
            CreateCallRequest(
                patient_id="pac-1",
                dia_postop=3,
                procedimiento="",
                nombre_completo="Test",
            )

    def test_empty_nombre_completo_rejected(self):
        with pytest.raises(ValueError):
            CreateCallRequest(
                patient_id="pac-1",
                dia_postop=3,
                procedimiento="Test",
                nombre_completo="",
            )

    def test_default_eps(self):
        req = CreateCallRequest(
            patient_id="pac-1",
            dia_postop=0,
            procedimiento="Test",
            nombre_completo="Test",
        )
        assert req.eps == "EPS"


class TestTurnRequest:
    def test_valid(self):
        req = TurnRequest(audio_base64="dGVzdA==")
        assert req.audio_base64 == "dGVzdA=="

    def test_empty_rejected(self):
        with pytest.raises(ValueError):
            TurnRequest(audio_base64="")


class TestEscalationInfo:
    def test_from_minimal_result(self):
        from backend.decision import EscalationResult, Severity

        result = EscalationResult(
            severity=Severity.GREEN,
            should_escalate=False,
            reason="El paciente reporta evolución favorable.",
            next_action="Continuar con el seguimiento.",
            domain="dolor",
            source="rule",
        )
        info = EscalationInfo.from_result(result)
        assert info.severity == "GREEN"
        assert not info.should_escalate
        assert info.domain == "dolor"

    def test_from_red_result(self):
        from backend.decision import EscalationResult, Severity

        result = EscalationResult(
            severity=Severity.RED,
            should_escalate=True,
            reason="Señal de alerta crítica.",
            next_action="Transferir al médico.",
            domain="herida",
            source="rule",
        )
        info = EscalationInfo.from_result(result)
        assert info.severity == "RED"
        assert info.should_escalate
        assert info.next_action == "Transferir al médico."


class TestCitationResponse:
    def test_construction(self):
        c = CitationResponse(
            chunk_id="c1",
            document_id="doc-1",
            source_filename="guia.pdf",
            page_number=3,
        )
        assert c.chunk_id == "c1"
        assert c.document_id == "doc-1"
        assert c.source_filename == "guia.pdf"
        assert c.page_number == 3

    def test_page_number_ge_one(self):
        with pytest.raises(ValueError):
            CitationResponse(
                chunk_id="c1",
                document_id="doc-1",
                source_filename="f.pdf",
                page_number=0,
            )


class TestBase64Decoder:
    def test_valid_base64(self):
        data = b"\x00\x01\x02"
        encoded = base64.b64encode(data).decode("ascii")
        result = _decode_base64_audio(encoded)
        assert result == data

    def test_empty_after_decode(self):
        with pytest.raises(Exception):
            _decode_base64_audio(
                base64.b64encode(b"").decode("ascii")
            )

    def test_invalid_base64_raises_http(self):
        from fastapi import HTTPException

        with pytest.raises(HTTPException, match="valid base64"):
            _decode_base64_audio("!!!not-valid-base64!!!")

    def test_invalid_base64_does_not_silence_other_exceptions(self):
        """Only ``binascii.Error`` and ``ValueError`` are caught;
        arbitrary exceptions (e.g. ``TypeError``) propagate."""
        with pytest.raises(TypeError):
            _decode_base64_audio(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# POST /calls — endpoint tests
# ---------------------------------------------------------------------------


class TestCreateCallEndpoint:
    """Full integration-style tests for POST /calls."""

    @pytest.mark.asyncio
    async def test_create_call_returns_201_with_audio(self, mock_tts):
        """A valid request creates a call, stores it, and returns
        base64-encoded WAV audio of the agent greeting."""
        # Ensure TTS returns valid WAV
        mock_tts.synthesize.return_value = TTSResult(
            audio_bytes=_MOCK_WAV_BYTES,
            sample_rate=24000,
            duration_ms=120.0,
            text="greeting",
            voice="ef_dora",
        )

        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            response = await client.post(
                "/calls",
                json={
                    "patient_id": "pac-test-001",
                    "dia_postop": 3,
                    "procedimiento": "Apendicectomía",
                    "nombre_completo": "María García",
                    "eps": "Compensar EPS",
                },
            )

        assert response.status_code == 201
        data = response.json()

        # Model conformance
        parsed = CreateCallResponse(**data)

        assert parsed.call_id
        assert len(parsed.audio_base64) > 0
        assert parsed.state == State.GREETING.value
        assert parsed.requires_response is True
        assert parsed.question_index is None
        assert parsed.total_questions == 6
        assert parsed.call_ended is False

        # Verify base64 decodes to expected WAV bytes
        decoded = base64.b64decode(parsed.audio_base64)
        assert decoded == _MOCK_WAV_BYTES

        # Verify TTS was called with a Spanish greeting
        mock_tts.synthesize.assert_called_once()
        call_args = mock_tts.synthesize.call_args
        text_arg = call_args[0][0]
        assert "Buenos días" in text_arg
        assert "María García" in text_arg

    @pytest.mark.asyncio
    async def test_create_call_patient_id_in_agent_message(self):
        """The agent greeting includes the patient's name."""
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            response = await client.post(
                "/calls",
                json={
                    "patient_id": "pac-42",
                    "dia_postop": 1,
                    "procedimiento": "Colecistectomía",
                    "nombre_completo": "Carlos López",
                    "eps": "Nueva EPS",
                },
            )

        assert response.status_code == 201
        data = response.json()
        assert data["state"] == "GREETING"
        assert data["total_questions"] == 6

    @pytest.mark.asyncio
    async def test_create_call_validation_errors(self):
        """Invalid request bodies return 422."""
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            # Missing required fields
            resp = await client.post("/calls", json={})
            assert resp.status_code == 422

            # Negative dia_postop
            resp = await client.post(
                "/calls",
                json={
                    "patient_id": "p",
                    "dia_postop": -1,
                    "procedimiento": "T",
                    "nombre_completo": "N",
                },
            )
            assert resp.status_code == 422

            # Empty patient_id
            resp = await client.post(
                "/calls",
                json={
                    "patient_id": "",
                    "dia_postop": 0,
                    "procedimiento": "T",
                    "nombre_completo": "N",
                },
            )
            assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_create_call_stores_in_global_store(self):
        """After creating a call, the global call_store contains it."""
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            response = await client.post(
                "/calls",
                json={
                    "patient_id": "pac-store",
                    "dia_postop": 2,
                    "procedimiento": "Apendicectomía",
                    "nombre_completo": "Store Test",
                    "eps": "EPS",
                },
            )

        call_id = response.json()["call_id"]
        assert await global_store.exists(call_id)

        # Clean up
        await global_store.remove(call_id)


# ---------------------------------------------------------------------------
# POST /calls/{call_id}/turn — endpoint tests
# ---------------------------------------------------------------------------


class TestTurnEndpoint:
    """Integration-style tests for POST /calls/{call_id}/turn."""

    @pytest.fixture
    async def call_id(self):
        """Create a call and return its ID."""
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            response = await client.post(
                "/calls",
                json={
                    "patient_id": "pac-turn",
                    "dia_postop": 3,
                    "procedimiento": "Apendicectomía",
                    "nombre_completo": "Turn Test",
                    "eps": "EPS",
                },
            )
        assert response.status_code == 201
        cid = response.json()["call_id"]
        yield cid
        # Cleanup
        await global_store.remove(cid)

    @pytest.mark.asyncio
    async def test_turn_returns_200_with_audio(self, call_id):
        """A valid turn request transcribes, processes, synthesises, and
        returns the agent response as base64 WAV."""
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            response = await client.post(
                f"/calls/{call_id}/turn",
                json={"audio_base64": _MOCK_AUDIO_B64},
            )

        assert response.status_code == 200
        data = response.json()

        parsed = TurnResponse(**data)
        assert parsed.call_id == call_id
        assert len(parsed.audio_base64) > 0
        assert len(parsed.transcription) > 0
        assert parsed.state in (
            State.CONSENT.value,
            State.QUESTIONS.value,
            State.CLOSING.value,
            State.ENDED.value,
        )

        # Audio should decode to expected WAV
        decoded = base64.b64decode(parsed.audio_base64)
        assert decoded == _MOCK_WAV_BYTES

    @pytest.mark.asyncio
    async def test_turn_unknown_call_id_returns_404(self):
        """A turn request for a non-existent call returns 404."""
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            response = await client.post(
                "/calls/nonexistent-id/turn",
                json={"audio_base64": _MOCK_AUDIO_B64},
            )

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_turn_invalid_base64_returns_400(self, call_id):
        """Invalid base64 in the audio field returns 400."""
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            response = await client.post(
                f"/calls/{call_id}/turn",
                json={"audio_base64": "!!!invalid!!!"},
            )

        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_turn_empty_audio_base64_returns_422(self, call_id):
        """Empty audio_base64 field is rejected at the model level."""
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            response = await client.post(
                f"/calls/{call_id}/turn",
                json={"audio_base64": ""},
            )

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_turn_empty_transcription_returns_400(self, call_id):
        """When STT returns empty text, the endpoint returns 400."""
        empty_stt = MagicMock()

        async def _empty_transcribe(audio_data: bytes) -> TranscriptionResult:
            return TranscriptionResult(
                text="   ",
                language="es",
                duration_seconds=0.1,
                model="whisper-large-v3",
            )

        empty_stt.side_effect = _empty_transcribe

        with patch("backend.api.calls._stt", empty_stt):
            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                response = await client.post(
                    f"/calls/{call_id}/turn",
                    json={"audio_base64": _MOCK_AUDIO_B64},
                )

        assert response.status_code == 400
        assert "empty" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_full_call_flow_greeting_to_ended(self, call_id):
        """Walk through a complete call from greeting to ENDED."""
        transport = ASGITransport(app=app)

        # The mock STT always transcribes to "Sí, acepto continuar con la llamada."
        # Step 1: greeting response (patient says hello) → consent
        async with AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            resp1 = await client.post(
                f"/calls/{call_id}/turn",
                json={"audio_base64": _MOCK_AUDIO_B64},
            )
        assert resp1.status_code == 200
        d1 = resp1.json()
        assert d1["state"] == State.CONSENT.value
        assert "continuar" in d1["transcription"].lower()

        # Step 2: consent response (patient says yes) → QUESTIONS
        async with AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            resp2 = await client.post(
                f"/calls/{call_id}/turn",
                json={"audio_base64": _MOCK_AUDIO_B64},
            )
        assert resp2.status_code == 200
        d2 = resp2.json()
        assert d2["state"] == State.QUESTIONS.value
        assert d2["question_index"] == 0
        # No escalation yet (first question just asked, no answer)
        assert d2["escalation"] is None

        # Steps 3-8: answer all 6 questions → CLOSING → ENDED
        for i in range(6):
            async with AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                resp = await client.post(
                    f"/calls/{call_id}/turn",
                    json={"audio_base64": _MOCK_AUDIO_B64},
                )
            assert resp.status_code == 200
            d = resp.json()

            if i < 5:
                # Still in QUESTIONS, next question asked
                assert d["question_index"] == i + 1
                # Escalation should be present (patient answered question i)
                assert d["escalation"] is not None
                assert d["escalation"]["severity"] in ("GREEN", "YELLOW", "RED")
            elif i == 5:
                # Last question answered → CLOSING with question_index=_NUM_QUESTIONS
                assert d["state"] in (State.CLOSING.value, State.QUESTIONS.value)
                # Escalation must now be present for mobility answer (was a bug)
                assert d["escalation"] is not None
                assert d["escalation"]["severity"] in ("GREEN", "YELLOW", "RED")
                assert d["escalation"]["domain"] == "movilidad"

        # Step 9: closing response → ENDED
        async with AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            resp_end = await client.post(
                f"/calls/{call_id}/turn",
                json={"audio_base64": _MOCK_AUDIO_B64},
            )
        assert resp_end.status_code == 200
        d_end = resp_end.json()
        assert d_end["state"] == State.ENDED.value
        assert d_end["call_ended"] is True
        assert d_end["requires_response"] is False

        # Orchestrator should be removed from store after call ends
        assert not await global_store.exists(call_id)

        # Further turns on ended call return 404
        async with AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            resp_after = await client.post(
                f"/calls/{call_id}/turn",
                json={"audio_base64": _MOCK_AUDIO_B64},
            )
        assert resp_after.status_code == 404

    @pytest.mark.asyncio
    async def test_turn_indices_sequential_across_ended_call(self, call_id):
        """After walking through a full multi-turn call, the metrics
        collector reports sequential turn indices (0, 1, 2, …) including
        for the final turn (regression: the final turn was reported as
        index 0 because ``_call_turn_index`` was popped before the index
        was read).
        """
        transport = ASGITransport(app=app)

        # Walk the full call: greeting → consent → 6 questions → closing → ENDED
        # That is 2 (greeting + consent) + 6 (question answers) + 1 (closing) = 9 turns
        for _ in range(9):
            async with AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                resp = await client.post(
                    f"/calls/{call_id}/turn",
                    json={"audio_base64": _MOCK_AUDIO_B64},
                )
            assert resp.status_code == 200

        # The call should be ended now.
        async with AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            detail_resp = await client.get(f"/metrics/calls/{call_id}")

        assert detail_resp.status_code == 200
        detail = detail_resp.json()

        # All 9 turns must be recorded.
        assert detail["turn_count"] == 9
        assert len(detail["turns"]) == 9

        # Turn indices must be exactly 0, 1, 2, …, 8 in order.
        indices = [t["turn_index"] for t in detail["turns"]]
        assert indices == list(range(9)), (
            f"Expected sequential indices 0..8, got {indices}"
        )


# ---------------------------------------------------------------------------
# STT / TTS error handling
# ---------------------------------------------------------------------------


class TestProviderErrors:
    """Tests for STT and TTS failure paths."""

    @pytest.fixture
    async def call_id(self):
        """Create a call for use in error tests."""
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            response = await client.post(
                "/calls",
                json={
                    "patient_id": "pac-error",
                    "dia_postop": 3,
                    "procedimiento": "Apendicectomía",
                    "nombre_completo": "Error Test",
                    "eps": "EPS",
                },
            )
        cid = response.json()["call_id"]
        yield cid
        await global_store.remove(cid)

    @pytest.mark.asyncio
    async def test_stt_provider_error_returns_502(self, call_id):
        """When the STT provider raises SttProviderError, the endpoint
        returns 502."""
        from backend.voice.models import SttProviderError

        async def _failing_stt(audio_data: bytes) -> TranscriptionResult:
            raise SttProviderError(
                "Límite de tasa de Groq excedido."
            )

        with patch("backend.api.calls._stt", _failing_stt):
            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                response = await client.post(
                    f"/calls/{call_id}/turn",
                    json={"audio_base64": _MOCK_AUDIO_B64},
                )

        assert response.status_code == 502
        assert "transcripción" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_stt_config_error_returns_502(self, call_id):
        """When the STT provider raises SttConfigError, the endpoint
        returns 502."""
        from backend.voice.models import SttConfigError

        async def _failing_stt(audio_data: bytes) -> TranscriptionResult:
            raise SttConfigError("GROQ_API_KEY no configurada.")

        with patch("backend.api.calls._stt", _failing_stt):
            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                response = await client.post(
                    f"/calls/{call_id}/turn",
                    json={"audio_base64": _MOCK_AUDIO_B64},
                )

        assert response.status_code == 502

    @pytest.mark.asyncio
    async def test_tts_synthesis_error_returns_502(self, call_id):
        """When the TTS adapter raises TTSSynthesisError, the endpoint
        returns 502."""
        from backend.voice.tts.protocol import TTSSynthesisError

        with patch("backend.api.calls._tts") as mock_tts_local:
            mock_tts_local.synthesize.side_effect = TTSSynthesisError(
                "Kokoro synthesis failed",
                provider="kokoro",
            )

            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                response = await client.post(
                    f"/calls/{call_id}/turn",
                    json={"audio_base64": _MOCK_AUDIO_B64},
                )

        assert response.status_code == 502
        assert "síntesis" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_stt_not_configured_returns_500(self, call_id):
        """When _stt is None, the endpoint returns 500."""
        with patch("backend.api.calls._stt", None):
            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                response = await client.post(
                    f"/calls/{call_id}/turn",
                    json={"audio_base64": _MOCK_AUDIO_B64},
                )

        assert response.status_code == 500
        assert "STT" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_tts_not_configured_on_create_call(self):
        """When _tts is None during call creation, the endpoint returns 500."""
        with patch("backend.api.calls._tts", None):
            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                response = await client.post(
                    "/calls",
                    json={
                        "patient_id": "pac-err",
                        "dia_postop": 1,
                        "procedimiento": "Test",
                        "nombre_completo": "Test",
                    },
                )

        assert response.status_code == 500
        assert "TTS" in response.json()["detail"]


# ---------------------------------------------------------------------------
# Escalation coverage
# ---------------------------------------------------------------------------


class TestEscalationInEndpoint:
    """Verify escalation classification is called and returned correctly."""

    @pytest.fixture
    async def call_id_after_consent(self):
        """Create a call and advance through greeting + consent so the
        next turn is a question answer."""
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            # Create call
            resp = await client.post(
                "/calls",
                json={
                    "patient_id": "pac-esc",
                    "dia_postop": 3,
                    "procedimiento": "Apendicectomía",
                    "nombre_completo": "Escalation Test",
                    "eps": "EPS",
                },
            )
            cid = resp.json()["call_id"]

            # Greeting → Consent
            await client.post(
                f"/calls/{cid}/turn",
                json={"audio_base64": _MOCK_AUDIO_B64},
            )
            # Consent → Questions (first question asked, q_index=0)
            await client.post(
                f"/calls/{cid}/turn",
                json={"audio_base64": _MOCK_AUDIO_B64},
            )

        yield cid
        await global_store.remove(cid)

    @pytest.mark.asyncio
    async def test_question_answer_triggers_escalation(self, call_id_after_consent):
        """Answering the first question (pain) triggers escalation
        classification and returns an EscalationInfo."""
        # Override STT with a pain-related response
        async def _pain_stt(audio_data: bytes) -> TranscriptionResult:
            return TranscriptionResult(
                text="Me duele mucho, como un 9 de 10.",
                language="es",
                duration_seconds=2.0,
                model="whisper-large-v3",
            )

        with patch("backend.api.calls._stt", _pain_stt):
            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                response = await client.post(
                    f"/calls/{call_id_after_consent}/turn",
                    json={"audio_base64": _MOCK_AUDIO_B64},
                )

        assert response.status_code == 200
        data = response.json()

        # question_index should be 1 (second question just asked)
        assert data["question_index"] == 1

        # Escalation for pain report of 9 → RED
        esc = data["escalation"]
        assert esc is not None
        assert esc["domain"] == "dolor"
        assert esc["severity"] == "RED"
        assert esc["should_escalate"] is True

    @pytest.mark.asyncio
    async def test_benign_answer_triggers_green_escalation(self, call_id_after_consent):
        """A benign pain report gets GREEN escalation."""
        async def _benign_stt(audio_data: bytes) -> TranscriptionResult:
            return TranscriptionResult(
                text="Muy bien, casi nada de dolor, un 1.",
                language="es",
                duration_seconds=1.0,
                model="whisper-large-v3",
            )

        with patch("backend.api.calls._stt", _benign_stt):
            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                response = await client.post(
                    f"/calls/{call_id_after_consent}/turn",
                    json={"audio_base64": _MOCK_AUDIO_B64},
                )

        assert response.status_code == 200
        esc = response.json()["escalation"]
        assert esc is not None
        assert esc["domain"] == "dolor"
        assert esc["severity"] == "GREEN"
        assert esc["should_escalate"] is False

    @pytest.mark.asyncio
    async def test_no_escalation_when_call_ended(self, call_id_after_consent):
        """When call ends, no escalation is generated (question_index is
        None during CLOSING)."""
        # Walk through all questions quickly to reach ENDED
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            # Answer all 6 questions + closing → ENDED
            for _ in range(7):
                resp = await client.post(
                    f"/calls/{call_id_after_consent}/turn",
                    json={"audio_base64": _MOCK_AUDIO_B64},
                )
                if resp.status_code != 200:
                    break

            last = resp.json()

        # Last turn should be ENDED with no escalation
        assert last["state"] == State.ENDED.value
        assert last["call_ended"] is True

    @pytest.mark.asyncio
    async def test_mobility_answer_triggers_escalation(self):
        """Answering the last follow-up question (mobility) triggers
        escalation classification with domain=movilidad.

        This was previously a bug: the CLOSING turn omitted
        question_index so the escalation layer never classified the
        mobility answer.  After the fix, question_index=_NUM_QUESTIONS
        (6) is passed, which maps to domain_idx=5 → movilidad.
        """
        transport = ASGITransport(app=app)

        # Advance through the first 5 questions (pain → sleep) with the
        # default mock STT.  Each question answer is a turn after the
        # question is asked, so we need 8 total turns (greeting +
        # consent + 6 question answers = 2 + 6 = 8 posts).
        async with AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            # Create call
            resp = await client.post(
                "/calls",
                json={
                    "patient_id": "pac-mob",
                    "dia_postop": 5,
                    "procedimiento": "Apendicectomía",
                    "nombre_completo": "Mobility Test",
                    "eps": "EPS",
                },
            )
            cid = resp.json()["call_id"]

            # Turns 1-7: greeting → consent → pain → fever → wound →
            # appetite → sleep (mobility is asked after sleep answer).
            for _ in range(7):
                resp = await client.post(
                    f"/calls/{cid}/turn",
                    json={"audio_base64": _MOCK_AUDIO_B64},
                )
                assert resp.status_code == 200

            # Verify we are now at mobility question (q_index=5)
            d = resp.json()
            assert d["question_index"] == 5  # mobility just asked

            # Turn 8: answer mobility with a red-flag response
            async def _mobility_red_stt(audio_data: bytes) -> TranscriptionResult:
                return TranscriptionResult(
                    text="No puedo caminar, me mareo mucho y casi me caigo.",
                    language="es",
                    duration_seconds=2.5,
                    model="whisper-large-v3",
                )

            with patch("backend.api.calls._stt", _mobility_red_stt):
                resp_mob = await client.post(
                    f"/calls/{cid}/turn",
                    json={"audio_base64": _MOCK_AUDIO_B64},
                )

        assert resp_mob.status_code == 200
        d_mob = resp_mob.json()

        # After the mobility answer the orchestrator transitions to
        # CLOSING and passes question_index=6.
        esc = d_mob["escalation"]
        assert esc is not None, (
            "Mobility answer must be escalated — question_index was "
            "previously None, causing the escalation layer to skip "
            "classification for the last question."
        )
        assert esc["domain"] == "movilidad"
        assert esc["severity"] in ("YELLOW", "RED")
        assert esc["should_escalate"] is True

        # Clean up
        await global_store.remove(cid)


# ---------------------------------------------------------------------------
# Patient transcription
# ---------------------------------------------------------------------------


class TestPatientTranscription:
    """Verify the ``patient_transcription`` field in TurnResponse."""

    @pytest.fixture
    async def call_id(self):
        """Create a call and advance through greeting so the first turn
        is the consent request."""
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            response = await client.post(
                "/calls",
                json={
                    "patient_id": "pac-pt",
                    "dia_postop": 3,
                    "procedimiento": "Apendicectomía",
                    "nombre_completo": "Patient Transcript",
                    "eps": "EPS",
                },
            )
        cid = response.json()["call_id"]
        yield cid
        await global_store.remove(cid)

    @pytest.mark.asyncio
    async def test_turn_response_includes_patient_transcription(self, call_id):
        """The TurnResponse includes a ``patient_transcription`` field
        containing the STT output for the patient's speech."""
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            response = await client.post(
                f"/calls/{call_id}/turn",
                json={"audio_base64": _MOCK_AUDIO_B64},
            )

        assert response.status_code == 200
        data = response.json()

        # Backward-compatible: the field must be present in the model.
        parsed = TurnResponse(**data)

        # The mock STT always returns "Sí, acepto continuar con la llamada."
        assert parsed.patient_transcription is not None
        assert "acepto" in parsed.patient_transcription.lower()
        assert parsed.patient_transcription == (
            "Sí, acepto continuar con la llamada."
        )

    @pytest.mark.asyncio
    async def test_patient_transcription_is_none_for_create_call(self):
        """CreateCallResponse does NOT include patient_transcription
        (the field only exists on TurnResponse)."""
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            response = await client.post(
                "/calls",
                json={
                    "patient_id": "pac-pt2",
                    "dia_postop": 1,
                    "procedimiento": "Test",
                    "nombre_completo": "Test",
                },
            )

        assert response.status_code == 201
        data = response.json()

        # CreateCallResponse must validate (no patient_transcription field
        # expected).
        parsed = CreateCallResponse(**data)
        assert parsed.call_id
        assert "patient_transcription" not in data


# ---------------------------------------------------------------------------
# Patient lookup from dataset
# ---------------------------------------------------------------------------


class TestPatientLookup:
    """Verify the real patient loader integration in create_call."""

    @pytest.mark.asyncio
    async def test_lookup_uses_dataset_patient_when_found(self):
        """When a patient_id exists in the dataset, the full patient
        profile is used (real demographics)."""
        from backend.data.models import Patient as DataPatient

        real_patient = DataPatient(
            paciente_id="P001",
            bundle_id="bundle-001",
            synthea_runtime="synthea_v3",
            modulo_synthea="appendicitis",
            procedimiento="Apendicectomía laparoscópica",
            fecha_cirugia=datetime.date(2026, 7, 1),
            edad=45,
            genero="F",
            comorbilidades=["hipertension"],
            complicacion_encounter=False,
            nombre_completo="María Elena Gómez",
            direccion="Calle 123",
            ciudad="Bogotá",
            departamento="Cundinamarca",
            documento_cc="123456789",
            eps="Compensar EPS",
            source_country="CO",
            adapted_country="CO",
            adaptation_fields=["nombre_completo"],
        )

        with patch(
            "backend.api.calls._get_patients",
            return_value={"P001": real_patient},
        ):
            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                response = await client.post(
                    "/calls",
                    json={
                        "patient_id": "P001",
                        "dia_postop": 3,
                        "procedimiento": "Apendicectomía laparoscópica",
                        "nombre_completo": "María Elena Gómez",
                        "eps": "Compensar EPS",
                    },
                )

        assert response.status_code == 201
        data = response.json()
        assert data["state"] == "GREETING"

    @pytest.mark.asyncio
    async def test_lookup_fallback_when_patient_not_found(self):
        """When a patient_id is NOT in the dataset, the request-body
        fields are used as a fallback (no crash)."""
        with patch(
            "backend.api.calls._get_patients",
            return_value={"P001": MagicMock()},
        ):
            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                response = await client.post(
                    "/calls",
                    json={
                        "patient_id": "NONEXISTENT",
                        "dia_postop": 2,
                        "procedimiento": "Colecistectomía",
                        "nombre_completo": "Juan Pérez",
                        "eps": "Nueva EPS",
                    },
                )

        assert response.status_code == 201
        data = response.json()
        assert data["state"] == "GREETING"

    @pytest.mark.asyncio
    async def test_lookup_handles_loader_exception_gracefully(self):
        """When the dataset loader throws, the endpoint falls back to
        the request-body fields without crashing."""
        def _failing_loader():
            raise RuntimeError("Dataset XLSX not found")

        with patch(
            "backend.api.calls._get_patients",
            side_effect=_failing_loader,
        ):
            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                response = await client.post(
                    "/calls",
                    json={
                        "patient_id": "P099",
                        "dia_postop": 1,
                        "procedimiento": "Test",
                        "nombre_completo": "Fallback User",
                        "eps": "EPS",
                    },
                )

        assert response.status_code == 201
        data = response.json()
        assert data["state"] == "GREETING"


# ---------------------------------------------------------------------------
# RagConfig / LlmConfig wiring
# ---------------------------------------------------------------------------


class TestOrchestratorConfigWiring:
    """Verify that the orchestrator is constructed with configs.

    The autouse ``setup_voice_mocks`` fixture patches ``RagConfig`` and
    ``LlmConfig`` to return ``None``.  These tests override those patches
    to verify the wiring paths.
    """

    @pytest.mark.asyncio
    async def test_orchestrator_receives_configs_when_available(self):
        """When RagConfig/LlmConfig return real objects, the orchestrator
        stores them (not None)."""
        mock_rag = MagicMock()
        mock_llm = MagicMock()

        transport = ASGITransport(app=app)

        with patch(
            "backend.api.calls.RagConfig", return_value=mock_rag
        ), patch(
            "backend.api.calls.LlmConfig", return_value=mock_llm
        ):
            async with AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                response = await client.post(
                    "/calls",
                    json={
                        "patient_id": "pac-cfg",
                        "dia_postop": 3,
                        "procedimiento": "Apendicectomía",
                        "nombre_completo": "Config Test",
                        "eps": "EPS",
                    },
                )

        assert response.status_code == 201
        call_id = response.json()["call_id"]

        # Verify the orchestrator stored in call_store has the configs.
        orch = await global_store.get(call_id)
        assert orch is not None
        assert orch._rag_config is mock_rag, (
            "Orchestrator must store the RagConfig return value"
        )
        assert orch._llm_config is mock_llm, (
            "Orchestrator must store the LlmConfig return value"
        )

        await global_store.remove(call_id)

    @pytest.mark.asyncio
    async def test_orchestrator_receives_none_when_config_fails(self):
        """When RagConfig/LlmConfig constructors raise, the orchestrator
        receives None (graceful degradation), and the call still works."""
        def _failing_config():
            raise RuntimeError("Config unavailable")

        transport = ASGITransport(app=app)

        with patch(
            "backend.api.calls.RagConfig",
            side_effect=_failing_config,
        ), patch(
            "backend.api.calls.LlmConfig",
            side_effect=_failing_config,
        ):
            async with AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                response = await client.post(
                    "/calls",
                    json={
                        "patient_id": "pac-cfg-fail",
                        "dia_postop": 1,
                        "procedimiento": "Test",
                        "nombre_completo": "Fail Test",
                        "eps": "EPS",
                    },
                )

        # The endpoint should still succeed (201)
        assert response.status_code == 201
        call_id = response.json()["call_id"]

        orch = await global_store.get(call_id)
        assert orch is not None
        assert orch._rag_config is None, (
            "Orchestrator should receive None when RagConfig fails"
        )
        assert orch._llm_config is None, (
            "Orchestrator should receive None when LlmConfig fails"
        )

        await global_store.remove(call_id)


# ---------------------------------------------------------------------------
# TurnMetrics integration — STT/TTS/LLM duration, token counts, RAG queries
# ---------------------------------------------------------------------------


class TestTurnMetricsIntegration:
    """Verify that TurnMetrics are populated with component durations
    and per-turn metadata after voice endpoints process a turn."""

    @pytest.mark.asyncio
    async def test_stt_and_tts_durations_recorded(self):
        """After a turn, the metrics collector records non-zero STT and TTS
        durations."""
        transport = ASGITransport(app=app)

        async with AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            # Create call
            resp = await client.post(
                "/calls",
                json={
                    "patient_id": "pac-metrics-1",
                    "dia_postop": 3,
                    "procedimiento": "Apendicectomía",
                    "nombre_completo": "Metrics Test",
                    "eps": "EPS",
                },
            )
            assert resp.status_code == 201
            call_id = resp.json()["call_id"]

            # Process a turn (greeting response → consent)
            await client.post(
                f"/calls/{call_id}/turn",
                json={"audio_base64": _MOCK_AUDIO_B64},
            )

        # Verify metrics were recorded via the collector's internal state.
        from backend.api.metrics import metrics_collector

        turns = metrics_collector.get_call_turns(call_id)
        # The call hasn't been ended yet, so get_call_turns returns [].
        # Use the raw internal lookup for testing.
        with metrics_collector._lock:
            raw_turns = metrics_collector._turns.get(call_id, [])

        assert len(raw_turns) >= 1
        t = raw_turns[0]
        assert t.stt_duration_ms is not None
        assert t.stt_duration_ms >= 0
        assert t.tts_duration_ms is not None
        assert t.tts_duration_ms >= 0

    @pytest.mark.asyncio
    async def test_rag_queries_zero_when_rag_disabled(self):
        """When RAG is not configured (default test fixture), rag_queries
        is 0 for every turn."""
        transport = ASGITransport(app=app)

        async with AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            resp = await client.post(
                "/calls",
                json={
                    "patient_id": "pac-metrics-2",
                    "dia_postop": 3,
                    "procedimiento": "Apendicectomía",
                    "nombre_completo": "RAG Zero Test",
                    "eps": "EPS",
                },
            )
            call_id = resp.json()["call_id"]

            # Process all turns to end the call
            for _ in range(9):
                r = await client.post(
                    f"/calls/{call_id}/turn",
                    json={"audio_base64": _MOCK_AUDIO_B64},
                )
                if r.status_code != 200:
                    break

        # Verify via the metrics endpoint
        async with AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            detail_resp = await client.get(f"/metrics/calls/{call_id}")

        assert detail_resp.status_code == 200
        detail = detail_resp.json()
        for t in detail["turns"]:
            assert t["rag_queries"] == 0, (
                f"rag_queries should be 0 when RAG is disabled, "
                f"got {t['rag_queries']} at turn {t['turn_index']}"
            )

    @pytest.mark.asyncio
    async def test_model_field_in_metrics(self):
        """The model field in TurnMetrics reflects the LLM config model
        name (or the default when no LLM is configured)."""
        transport = ASGITransport(app=app)

        async with AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            resp = await client.post(
                "/calls",
                json={
                    "patient_id": "pac-metrics-3",
                    "dia_postop": 3,
                    "procedimiento": "Apendicectomía",
                    "nombre_completo": "Model Test",
                    "eps": "EPS",
                },
            )
            call_id = resp.json()["call_id"]

            # Process all turns to end the call
            for _ in range(9):
                r = await client.post(
                    f"/calls/{call_id}/turn",
                    json={"audio_base64": _MOCK_AUDIO_B64},
                )
                if r.status_code != 200:
                    break

        async with AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            detail_resp = await client.get(f"/metrics/calls/{call_id}")

        assert detail_resp.status_code == 200
        detail = detail_resp.json()
        for t in detail["turns"]:
            assert t["model"] == "llama-3.1-70b-versatile", (
                f"model should be the default model name, "
                f"got {t['model']} at turn {t['turn_index']}"
            )

    @pytest.mark.asyncio
    async def test_turn_metrics_include_optional_fields_in_response(self):
        """After a full call, the metrics detail endpoint returns
        the optional component-duration and token fields (even when
        they are None due to no LLM being invoked)."""
        transport = ASGITransport(app=app)

        async with AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            resp = await client.post(
                "/calls",
                json={
                    "patient_id": "pac-metrics-4",
                    "dia_postop": 3,
                    "procedimiento": "Apendicectomía",
                    "nombre_completo": "Optional Fields Test",
                    "eps": "EPS",
                },
            )
            call_id = resp.json()["call_id"]

            for _ in range(9):
                await client.post(
                    f"/calls/{call_id}/turn",
                    json={"audio_base64": _MOCK_AUDIO_B64},
                )

        async with AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            detail_resp = await client.get(f"/metrics/calls/{call_id}")

        assert detail_resp.status_code == 200
        detail = detail_resp.json()
        assert detail["turn_count"] >= 1

        t0 = detail["turns"][0]
        # Optional component-duration fields should exist (even if None).
        assert "tts_duration_ms" in t0
        assert "stt_duration_ms" in t0
        assert "llm_duration_ms" in t0
        assert "input_tokens" in t0
        assert "output_tokens" in t0

        # When no LLM was invoked, llm_duration_ms and tokens are None.
        assert t0["llm_duration_ms"] is None
        assert t0["input_tokens"] is None
        assert t0["output_tokens"] is None

        # STT and TTS durations should be non-None (recorded).
        assert t0["stt_duration_ms"] is not None
        assert t0["stt_duration_ms"] >= 0
        assert t0["tts_duration_ms"] is not None
        assert t0["tts_duration_ms"] >= 0

    @pytest.mark.asyncio
    async def test_metrics_include_rag_queries_and_model_calls(self):
        """The metrics summary endpoint correctly aggregates rag_queries
        and model_calls from turns (zero when no RAG/LLM configured)."""
        transport = ASGITransport(app=app)

        async with AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            resp = await client.post(
                "/calls",
                json={
                    "patient_id": "pac-metrics-5",
                    "dia_postop": 3,
                    "procedimiento": "Apendicectomía",
                    "nombre_completo": "Aggregation Test",
                    "eps": "EPS",
                },
            )
            call_id = resp.json()["call_id"]

            for _ in range(9):
                await client.post(
                    f"/calls/{call_id}/turn",
                    json={"audio_base64": _MOCK_AUDIO_B64},
                )

        async with AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            summary_resp = await client.get("/metrics/summary")

        assert summary_resp.status_code == 200
        summary = summary_resp.json()
        assert summary["call_count"] == 1
        assert summary["total_turns"] >= 1
        # Without RAG, rag_queries should be 0 for all turns.
        assert summary["total_rag_queries"] == 0
        # Without LLM invocation, model_calls should be 0.
        assert summary["total_model_calls"] == 0


# ---------------------------------------------------------------------------
# Voice persistence integration tests
# ---------------------------------------------------------------------------


class TestVoicePersistence:
    """Verify that voice call data is persisted to SQLite.

    These tests initialise a temporary SQLite database and walk through
    call creation, turns, and completion, then assert that the expected
    records exist in the database tables.
    """

    @pytest.fixture(autouse=True)
    def _persistence_setup(self, tmp_path):
        """Initialise SQLite with a temp database so persistence calls
        succeed in every test in this class."""
        from backend.persistence.sqlite import _reset_sqlite, init_sqlite

        _reset_sqlite()
        db_path = tmp_path / "test_voice.db"
        init_sqlite(db_path)
        yield
        _reset_sqlite()

    # -- call creation persistence -------------------------------------------

    @pytest.mark.asyncio
    async def test_call_creation_persists_call_record(self):
        """After POST /calls, a CallRecord is inserted into SQLite."""
        from backend.persistence.sqlite import get_call_by_id

        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            response = await client.post(
                "/calls",
                json={
                    "patient_id": "pac-persist-1",
                    "dia_postop": 4,
                    "procedimiento": "Colecistectomía",
                    "nombre_completo": "Persistencia Test",
                    "eps": "Sura EPS",
                },
            )

        assert response.status_code == 201
        call_id = response.json()["call_id"]

        record = get_call_by_id(call_id)
        assert record is not None, (
            "CallRecord must be inserted in SQLite after call creation"
        )
        assert record.paciente_id == "pac-persist-1"
        assert record.nombre_completo == "Persistencia Test"
        assert record.procedimiento == "Colecistectomía"
        assert record.dia_postop == 4
        assert record.eps == "Sura EPS"
        assert record.ended_at is None, (
            "Incomplete call must have ended_at=None"
        )
        assert record.total_turns == 0
        assert not record.escalated

        # Clean up
        await global_store.remove(call_id)

    @pytest.mark.asyncio
    async def test_incomplete_call_state_in_sqlite(self):
        """A call that has been created but not yet ended has ended_at=None
        and state reflects the current conversation phase."""
        from backend.persistence.sqlite import get_call_by_id

        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            resp = await client.post(
                "/calls",
                json={
                    "patient_id": "pac-incomplete",
                    "dia_postop": 1,
                    "procedimiento": "Apendicectomía",
                    "nombre_completo": "Incomplete Test",
                    "eps": "EPS",
                },
            )
        call_id = resp.json()["call_id"]
        assert resp.json()["state"] == "GREETING"

        record = get_call_by_id(call_id)
        assert record is not None
        assert record.ended_at is None
        # State at creation is GREETING (after start_call)
        assert record.state in ("GREETING", "IDLE")
        assert record.total_turns == 0

        await global_store.remove(call_id)

    # -- turn persistence ----------------------------------------------------

    @pytest.mark.asyncio
    async def test_turn_persistence(self):
        """After processing a turn, ConversationTurnRecord entries are
        inserted for both patient and agent messages."""
        from backend.persistence.sqlite import get_turns_for_call

        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            # Create call
            resp = await client.post(
                "/calls",
                json={
                    "patient_id": "pac-turn-p",
                    "dia_postop": 2,
                    "procedimiento": "Apendicectomía",
                    "nombre_completo": "Turn Persist",
                    "eps": "EPS",
                },
            )
            call_id = resp.json()["call_id"]

            # First turn
            await client.post(
                f"/calls/{call_id}/turn",
                json={"audio_base64": _MOCK_AUDIO_B64},
            )

        turns = get_turns_for_call(call_id)
        assert len(turns) == 2, (
            f"Expected 2 turns (1 patient + 1 agent), got {len(turns)}"
        )

        # First turn is patient (index 0), second is agent (index 1)
        roles = [t.role for t in turns]
        assert "PATIENT" in roles
        assert "AGENT" in roles

        # Patient turn has the STT transcription
        patient_turn = [t for t in turns if t.role == "PATIENT"][0]
        assert "acepto" in patient_turn.text.lower()

        # Agent turn has the consent request
        agent_turn = [t for t in turns if t.role == "AGENT"][0]
        assert "continuar" in agent_turn.text.lower()

        await global_store.remove(call_id)

    @pytest.mark.asyncio
    async def test_turn_indices_are_sequential(self):
        """Turn indices across multiple HTTP turns are sequential
        (patient=even, agent=odd)."""
        from backend.persistence.sqlite import get_turns_for_call

        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            resp = await client.post(
                "/calls",
                json={
                    "patient_id": "pac-seq",
                    "dia_postop": 3,
                    "procedimiento": "Apendicectomía",
                    "nombre_completo": "Seq Test",
                    "eps": "EPS",
                },
            )
            call_id = resp.json()["call_id"]

            # Process 3 turns (greeting→consent, consent→questions, q0_answer)
            for _ in range(3):
                await client.post(
                    f"/calls/{call_id}/turn",
                    json={"audio_base64": _MOCK_AUDIO_B64},
                )

        turns = get_turns_for_call(call_id)
        assert len(turns) >= 6, (
            f"Expected at least 6 turns for 3 HTTP calls, got {len(turns)}"
        )
        indices = [t.turn_index for t in turns]
        assert indices == sorted(indices), (
            "Turn indices must be in ascending order"
        )

        await global_store.remove(call_id)

    # -- escalation alert persistence ----------------------------------------

    @pytest.mark.asyncio
    async def test_escalation_alert_persisted_for_red(self):
        """When a RED escalation is classified, an EscalationAlertRecord
        is inserted into SQLite."""
        from backend.persistence.sqlite import get_alerts_for_call

        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            # Create call
            resp = await client.post(
                "/calls",
                json={
                    "patient_id": "pac-red-a",
                    "dia_postop": 3,
                    "procedimiento": "Apendicectomía",
                    "nombre_completo": "Red Alert",
                    "eps": "EPS",
                },
            )
            call_id = resp.json()["call_id"]

            # Advance through greeting + consent
            await client.post(
                f"/calls/{call_id}/turn",
                json={"audio_base64": _MOCK_AUDIO_B64},
            )
            await client.post(
                f"/calls/{call_id}/turn",
                json={"audio_base64": _MOCK_AUDIO_B64},
            )

            # Send RED response for pain question (NRS 9)
            async def _pain_red_stt(audio_data):
                return TranscriptionResult(
                    text="Me duele muchísimo, un 9 de 10, no soporto el dolor.",
                    language="es",
                    duration_seconds=2.0,
                    model="whisper-large-v3",
                )

            with patch("backend.api.calls._stt", _pain_red_stt):
                await client.post(
                    f"/calls/{call_id}/turn",
                    json={"audio_base64": _MOCK_AUDIO_B64},
                )

        alerts = get_alerts_for_call(call_id)
        assert len(alerts) >= 1, (
            f"Expected at least 1 escalation alert for RED pain, got {len(alerts)}"
        )
        red_alert = [a for a in alerts if a.severity == "RED"]
        assert len(red_alert) >= 1
        assert red_alert[0].domain == "dolor"

        await global_store.remove(call_id)

    @pytest.mark.asyncio
    async def test_no_alert_persisted_for_green(self):
        """GREEN escalation classifications are not persisted as alerts."""
        from backend.persistence.sqlite import get_alerts_for_call

        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            resp = await client.post(
                "/calls",
                json={
                    "patient_id": "pac-green",
                    "dia_postop": 3,
                    "procedimiento": "Apendicectomía",
                    "nombre_completo": "Green Test",
                    "eps": "EPS",
                },
            )
            call_id = resp.json()["call_id"]

            # Advance through greeting + consent → QUESTIONS
            await client.post(
                f"/calls/{call_id}/turn",
                json={"audio_base64": _MOCK_AUDIO_B64},
            )
            await client.post(
                f"/calls/{call_id}/turn",
                json={"audio_base64": _MOCK_AUDIO_B64},
            )

            # Answer pain with benign response
            async def _benign_stt(audio_data):
                return TranscriptionResult(
                    text="Muy bien, no tengo dolor, un nivel 1.",
                    language="es",
                    duration_seconds=1.0,
                    model="whisper-large-v3",
                )

            with patch("backend.api.calls._stt", _benign_stt):
                await client.post(
                    f"/calls/{call_id}/turn",
                    json={"audio_base64": _MOCK_AUDIO_B64},
                )

        alerts = get_alerts_for_call(call_id)
        # No alerts should be persisted for GREEN
        assert len(alerts) == 0, (
            f"GREEN classifications must not be persisted as alerts, got {len(alerts)}"
        )

        await global_store.remove(call_id)

    # -- completed-call summary persistence ----------------------------------

    @pytest.mark.asyncio
    async def test_summary_persisted_on_call_end(self):
        """When a call completes through to ENDED, a SummaryRecord is
        inserted into SQLite.  RAG citations accumulated across all turns
        are reflected in the ``sources_json`` field."""
        from backend.persistence.sqlite import (
            get_summary_for_call,
            get_call_by_id,
        )
        from backend.conversation.orchestrator import OrchestratorTurn

        # ------------------------------------------------------------------
        # Mock citations: each turn returns a different set so the
        # per-call accumulator collects unique document IDs across turns
        # and deduplicates on document_id.  Duplicate document_id
        # entries simulate the same source being cited across multiple
        # turns — the accumulator must keep only one.
        # ------------------------------------------------------------------
        _CITATIONS_BY_TURN: tuple[list[dict], ...] = (
            [  # turn 0 — greeting → consent
                {"chunk_id": "ch-a1", "document_id": "doc-rag-a",
                 "source_filename": "guia_clinica_postop.pdf", "page_number": 3},
            ],
            [  # turn 1 — consent → questions (same doc-a = dedup)
                {"chunk_id": "ch-a2", "document_id": "doc-rag-a",
                 "source_filename": "guia_clinica_postop.pdf", "page_number": 7},
            ],
            [  # turn 2 — answer pain (new source)
                {"chunk_id": "ch-b1", "document_id": "doc-rag-b",
                 "source_filename": "manejo_dolor.pdf", "page_number": 2},
            ],
            [  # turn 3 — answer fever (another new source)
                {"chunk_id": "ch-c1", "document_id": "doc-rag-c",
                 "source_filename": "protocolo_fiebre.pdf", "page_number": 5},
            ],
            [  # turn 4 — answer wound (dup of doc-b)
                {"chunk_id": "ch-b2", "document_id": "doc-rag-b",
                 "source_filename": "manejo_dolor.pdf", "page_number": 10},
            ],
            [],  # turn 5 — answer appetite
            [  # turn 6 — answer sleep (new source)
                {"chunk_id": "ch-d1", "document_id": "doc-rag-d",
                 "source_filename": "cuidados_generales.pdf", "page_number": 1},
            ],
            [],  # turn 7 — answer mobility
            [],  # turn 8 — closing
        )

        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            # Create call
            resp = await client.post(
                "/calls",
                json={
                    "patient_id": "pac-summary",
                    "dia_postop": 3,
                    "procedimiento": "Apendicectomía",
                    "nombre_completo": "Summary Test",
                    "eps": "Coosalud EPS",
                },
            )
            call_id = resp.json()["call_id"]

            # ---- Patch the orchestrator instance to inject mock citations ----
            orch = await global_store.get(call_id)
            assert orch is not None, "Orchestrator must be in call_store"
            _original_process = orch.process_patient_message
            _call_count = [0]

            def _patched_process(patient_text: str) -> OrchestratorTurn:
                idx = _call_count[0]
                _call_count[0] += 1
                result = _original_process(patient_text)
                if idx < len(_CITATIONS_BY_TURN) and _CITATIONS_BY_TURN[idx]:
                    result.citations = _CITATIONS_BY_TURN[idx]
                return result

            orch.process_patient_message = _patched_process  # type: ignore[method-assign]

            # Walk through all turns to ENDED
            # greeting → consent → 6 questions → closing → ENDED = 9 turns
            for _ in range(9):
                resp = await client.post(
                    f"/calls/{call_id}/turn",
                    json={"audio_base64": _MOCK_AUDIO_B64},
                )
                if resp.status_code != 200:
                    break

        assert resp.status_code == 200
        assert resp.json()["call_ended"] is True

        # Verify call record is updated
        call = get_call_by_id(call_id)
        assert call is not None
        assert call.ended_at is not None, "Ended call must have ended_at set"
        assert call.state == "ENDED"
        assert call.total_turns >= 2

        # Verify summary is persisted
        summary = get_summary_for_call(call_id)
        assert summary is not None, (
            "SummaryRecord must be persisted when call ends"
        )
        assert summary.call_id == call_id
        assert len(summary.patient_summary) > 0
        assert len(summary.procedure_summary) > 0
        assert len(summary.symptoms_summary) > 0
        assert len(summary.decision_summary) > 0
        assert len(summary.next_steps) > 0

        # sources_json must be a valid JSON array containing
        # the unique document IDs accumulated across turns.
        import json as _json
        sources = _json.loads(summary.sources_json)
        assert isinstance(sources, list)

        # Expected unique document IDs after deduplication:
        # doc-rag-a, doc-rag-b, doc-rag-c, doc-rag-d
        expected_doc_ids = {"doc-rag-a", "doc-rag-b", "doc-rag-c", "doc-rag-d"}
        actual_doc_ids = {s[0] for s in sources}
        assert actual_doc_ids == expected_doc_ids, (
            f"Summary sources must include all unique RAG citation "
            f"document IDs accumulated across turns.\n"
            f"Expected: {sorted(expected_doc_ids)}\n"
            f"Got:      {sorted(actual_doc_ids)}"
        )

        # Verify expected filenames are present
        expected_filenames = {
            "guia_clinica_postop.pdf",
            "manejo_dolor.pdf",
            "protocolo_fiebre.pdf",
            "cuidados_generales.pdf",
        }
        actual_filenames = {s[1] for s in sources}
        assert actual_filenames == expected_filenames, (
            f"Summary sources must include source filenames.\n"
            f"Expected: {sorted(expected_filenames)}\n"
            f"Got:      {sorted(actual_filenames)}"
        )

    # -- restart-safe retrieval ----------------------------------------------

    @pytest.mark.asyncio
    async def test_restart_safe_retrieval(self):
        """After a call completes, clearing the in-memory store does not
        affect data retrieval from SQLite.  This simulates an application
        restart: the CallStore is empty, but call data, turns, summary,
        and alerts can still be read from the database."""
        from backend.persistence.sqlite import (
            get_call_by_id,
            get_turns_for_call,
            get_summary_for_call,
            get_alerts_for_call,
        )

        # Step 1: run a complete call with RED escalation
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            resp = await client.post(
                "/calls",
                json={
                    "patient_id": "pac-restart",
                    "dia_postop": 5,
                    "procedimiento": "Colecistectomía",
                    "nombre_completo": "Restart Test",
                    "eps": "Nueva EPS",
                },
            )
            call_id = resp.json()["call_id"]

            # greeting → consent
            await client.post(
                f"/calls/{call_id}/turn",
                json={"audio_base64": _MOCK_AUDIO_B64},
            )
            await client.post(
                f"/calls/{call_id}/turn",
                json={"audio_base64": _MOCK_AUDIO_B64},
            )

            # Send RED pain response
            async def _red_stt(audio_data):
                return TranscriptionResult(
                    text="Me duele muchísimo, un 9 de 10.",
                    language="es",
                    duration_seconds=2.0,
                    model="whisper-large-v3",
                )

            with patch("backend.api.calls._stt", _red_stt):
                await client.post(
                    f"/calls/{call_id}/turn",
                    json={"audio_base64": _MOCK_AUDIO_B64},
                )

            # Walk remaining turns to ENDED using default STT
            for _ in range(6):
                resp = await client.post(
                    f"/calls/{call_id}/turn",
                    json={"audio_base64": _MOCK_AUDIO_B64},
                )
                if resp.status_code != 200:
                    break

        assert resp.status_code == 200
        assert resp.json()["call_ended"] is True

        # Step 2: simulate restart — clear the in-memory CallStore
        # The global_store should already be empty (call ended removes it),
        # but let's be explicit.
        from backend.api.call_store import call_store as cs
        await cs.remove(call_id)

        # Data must still be retrievable from SQLite
        call = get_call_by_id(call_id)
        assert call is not None
        assert call.paciente_id == "pac-restart"
        assert call.ended_at is not None
        assert call.escalated is True, (
            "RED escalation should mark call.escalated=True"
        )

        turns = get_turns_for_call(call_id)
        assert len(turns) >= 2
        assert any(t.role == "PATIENT" for t in turns)
        assert any(t.role == "AGENT" for t in turns)

        summary = get_summary_for_call(call_id)
        assert summary is not None

        alerts = get_alerts_for_call(call_id)
        assert len(alerts) >= 1
        assert any(a.severity == "RED" for a in alerts)

        # Call store should be empty (simulating restart)
        assert not await cs.exists(call_id)
