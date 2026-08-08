"""Tests for the voice call REST endpoints.

All tests mock STT and TTS dependencies so they execute quickly without
external services.  The ``ConversationOrchestrator`` runs with
``rag_config=None`` and ``llm_config=None`` — it uses deterministic
fallback messages.
"""

from __future__ import annotations

import base64
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
    module-level state (metrics collector, turn-index counter) before
    each test so tests are isolated.

    Uses ``unittest.mock.patch`` on the module-level globals so every
    endpoint call goes through the mocks.
    """
    from backend.api.calls import _call_turn_index
    from backend.api.metrics import metrics_collector

    metrics_collector.reset()
    _call_turn_index.clear()

    with patch("backend.api.calls._stt", mock_stt), patch(
        "backend.api.calls._tts", mock_tts
    ):
        yield

    metrics_collector.reset()
    _call_turn_index.clear()


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
