"""Integration tests for the frontend-backend voice call contract.

These tests exercise the exact HTTP contract that the vanilla frontend
(call.js) consumes: POST /calls to create a call and POST /calls/{call_id}/turn
to send patient audio and receive agent audio.  STT and TTS are mocked so
tests run without external services.  RagConfig, LlmConfig, and the patient
loader are also mocked so tests do not trigger model downloads, XLSX reads,
or external API connections.

The tests verify:

- POST /calls returns 201 with valid CreateCallResponse shape.
- POST /calls/{call_id}/turn returns 200 with valid TurnResponse shape.
- Base64 audio round-trip: encode → send → decode response audio.
- Full call flow from GREETING through ENDED.
- Escalation info presence and shape during QUESTIONS phase.
- Citation structure when present.
- Error handling for missing calls, empty audio, invalid base64.

These are *contract* tests: they validate the API surface the frontend depends
on without testing browser JavaScript.
"""

from __future__ import annotations

import base64
from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from backend.api.call_store import call_store as global_store
from backend.conversation.state import State
from backend.voice.models import TranscriptionResult
from backend.voice.tts.protocol import TTSResult
from backend.main import app

# ---------------------------------------------------------------------------
# Test constants that mirror the frontend's data.js PATIENTS catalogue
# ---------------------------------------------------------------------------

FRONTEND_PATIENTS = [
    {
        "id": "P001",
        "name": "Paciente 001",
        "age": 45,
        "procedure": "Apendicectomía laparoscópica",
        "postopDay": 3,
    },
    {
        "id": "P002",
        "name": "Paciente 002",
        "age": 62,
        "procedure": "Colecistectomía",
        "postopDay": 5,
    },
    {
        "id": "P003",
        "name": "Paciente 003",
        "age": 38,
        "procedure": "Hernioplastia inguinal",
        "postopDay": 2,
    },
    {
        "id": "P004",
        "name": "Paciente 004",
        "age": 55,
        "procedure": "Cesárea",
        "postopDay": 4,
    },
    {
        "id": "P005",
        "name": "Paciente 005",
        "age": 71,
        "procedure": "Reemplazo total de cadera",
        "postopDay": 7,
    },
]

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

    Also patches RagConfig/LlmConfig to return None and _get_patients
    to return an empty dict so tests do not trigger model downloads,
    XLSX reads, or external API connections.
    """
    from backend.api.calls import _call_turn_index
    from backend.api.metrics import metrics_collector

    metrics_collector.reset()
    _call_turn_index.clear()

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


# ---------------------------------------------------------------------------
# POST /calls — frontend contract tests
# ---------------------------------------------------------------------------


class TestCreateCallFrontendContract:
    """Verify POST /calls returns the shape the frontend expects."""

    @pytest.mark.asyncio
    async def test_create_call_returns_expected_fields(self):
        """CreateCallResponse must include all fields indexed by call.js."""
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
                    "nombre_completo": "Paciente 001",
                    "eps": "EPS",
                },
            )

        assert response.status_code == 201
        data = response.json()

        # Fields consumed by call.js via sessionStorage
        assert "call_id" in data
        assert isinstance(data["call_id"], str)
        assert len(data["call_id"]) > 0

        assert "audio_base64" in data
        assert isinstance(data["audio_base64"], str)
        assert len(data["audio_base64"]) > 0

        assert "state" in data
        assert "requires_response" in data
        assert "question_index" in data
        assert "total_questions" in data
        assert "call_ended" in data

        assert data["total_questions"] == 6
        assert data["call_ended"] is False

        # Verify the greeting audio decodes to valid bytes
        decoded = base64.b64decode(data["audio_base64"])
        assert len(decoded) > 0

    @pytest.mark.asyncio
    async def test_all_frontend_patients_create_call(self):
        """Every patient in the frontend PATIENTS array can create a call."""
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            for patient in FRONTEND_PATIENTS:
                response = await client.post(
                    "/calls",
                    json={
                        "patient_id": patient["id"],
                        "dia_postop": patient["postopDay"],
                        "procedimiento": patient["procedure"],
                        "nombre_completo": patient["name"],
                        "eps": "EPS",
                    },
                )
                assert response.status_code == 201, (
                    f"Patient {patient['id']} failed: {response.text}"
                )
                data = response.json()
                assert data["call_id"]
                assert data["state"] == State.GREETING.value

                # Clean up orchestrator from store
                await global_store.remove(data["call_id"])


# ---------------------------------------------------------------------------
# POST /calls/{call_id}/turn — frontend contract tests
# ---------------------------------------------------------------------------


class TestTurnFrontendContract:
    """Verify POST /calls/{call_id}/turn returns the shape the frontend expects."""

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
                    "patient_id": "P001",
                    "dia_postop": 3,
                    "procedimiento": "Apendicectomía laparoscópica",
                    "nombre_completo": "Paciente 001",
                    "eps": "EPS",
                },
            )
        assert response.status_code == 201
        cid = response.json()["call_id"]
        yield cid
        await global_store.remove(cid)

    @pytest.mark.asyncio
    async def test_turn_returns_expected_fields(self, call_id):
        """TurnResponse must include all fields consumed by call.js."""
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

        # Fields consumed by call.js
        assert "call_id" in data
        assert data["call_id"] == call_id

        assert "audio_base64" in data
        assert isinstance(data["audio_base64"], str)
        assert len(data["audio_base64"]) > 0

        assert "transcription" in data
        assert isinstance(data["transcription"], str)
        assert len(data["transcription"]) > 0

        # patient_transcription added for frontend display
        assert "patient_transcription" in data

        assert "state" in data
        assert "citations" in data
        assert isinstance(data["citations"], list)

        assert "requires_response" in data
        assert "question_index" in data
        assert "total_questions" in data
        assert "call_ended" in data

        # escalation may be null or an object
        assert "escalation" in data
        assert "esalation" not in data  # guard against misspelling regression

    @pytest.mark.asyncio
    async def test_base64_audio_round_trip(self, call_id):
        """Frontend sends base64 audio, backend returns base64 WAV.

        This verifies the encode → POST → decode → play pipeline that
        call.js implements.
        """
        # Encode audio as the frontend would
        test_audio = b"\x52\x49\x46\x46" + b"\x00" * 40  # Fake WAV header
        encoded = base64.b64encode(test_audio).decode("ascii")

        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            response = await client.post(
                f"/calls/{call_id}/turn",
                json={"audio_base64": encoded},
            )

        assert response.status_code == 200
        data = response.json()

        # Decode response audio (as call.js does with atob + Uint8Array)
        decoded = base64.b64decode(data["audio_base64"])
        assert len(decoded) > 0

    @pytest.mark.asyncio
    async def test_media_recorder_webm_mime_type(self, call_id):
        """The backend accepts audio/webm base64 (what MediaRecorder produces).

        The frontend's MediaRecorder typically produces audio/webm;codecs=opus.
        The Groq Whisper adapter handles multiple formats, so the backend
        must accept the base64 payload regardless of audio codec.
        """
        # Simulate a webm-like payload (the mock STT doesn't actually decode)
        webm_audio = b"\x1a\x45\xdf\xa3" + b"\x00" * 100  # EBML/webm header
        encoded = base64.b64encode(webm_audio).decode("ascii")

        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            response = await client.post(
                f"/calls/{call_id}/turn",
                json={"audio_base64": encoded},
            )

        # Should succeed — the mock STT doesn't inspect the audio data
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# Full call flow (contract test)
# ---------------------------------------------------------------------------


class TestFullCallFlowContract:
    """Verify a complete call from GREETING to ENDED using the frontend contract."""

    @pytest.mark.asyncio
    async def test_full_call_flow(self):
        """Walk through a complete call and verify every response shape."""
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            # 1. Create call (as app.js does)
            resp = await client.post(
                "/calls",
                json={
                    "patient_id": "P003",
                    "dia_postop": 2,
                    "procedimiento": "Hernioplastia inguinal",
                    "nombre_completo": "Paciente 003",
                    "eps": "EPS",
                },
            )
            assert resp.status_code == 201
            create_data = resp.json()
            call_id = create_data["call_id"]
            assert create_data["state"] == State.GREETING.value
            assert create_data["total_questions"] == 6

            # 2. Greeting response → CONSENT
            resp = await client.post(
                f"/calls/{call_id}/turn",
                json={"audio_base64": _MOCK_AUDIO_B64},
            )
            assert resp.status_code == 200
            d = resp.json()
            assert d["state"] == State.CONSENT.value
            assert "consentimiento" in d["transcription"].lower() or "continuar" in d["transcription"].lower()

            # 3. Consent response → QUESTIONS (q_index=0)
            resp = await client.post(
                f"/calls/{call_id}/turn",
                json={"audio_base64": _MOCK_AUDIO_B64},
            )
            assert resp.status_code == 200
            d = resp.json()
            assert d["state"] == State.QUESTIONS.value
            assert d["question_index"] == 0
            assert d["escalation"] is None  # No answer yet to classify

            # 4-9. Answer all 6 questions
            for i in range(6):
                resp = await client.post(
                    f"/calls/{call_id}/turn",
                    json={"audio_base64": _MOCK_AUDIO_B64},
                )
                assert resp.status_code == 200
                d = resp.json()

                if i < 5:
                    # Next question asked
                    assert d["question_index"] == i + 1
                    # Escalation must be present (patient answered question i)
                    assert d["escalation"] is not None, (
                        f"Expected escalation for question {i}, got None"
                    )
                    esc = d["escalation"]
                    assert "severity" in esc
                    assert esc["severity"] in ("GREEN", "YELLOW", "RED")
                    assert "should_escalate" in esc
                    assert "reason" in esc
                    assert "next_action" in esc
                    assert "domain" in esc
                else:
                    # Last answer → CLOSING (or still QUESTIONS)
                    assert d["state"] in (State.CLOSING.value, State.QUESTIONS.value)
                    assert d["escalation"] is not None
                    assert d["escalation"]["domain"] == "movilidad"

            # 10. Closing response → ENDED
            resp = await client.post(
                f"/calls/{call_id}/turn",
                json={"audio_base64": _MOCK_AUDIO_B64},
            )
            assert resp.status_code == 200
            d = resp.json()
            assert d["state"] == State.ENDED.value
            assert d["call_ended"] is True
            assert d["requires_response"] is False

            # Orchestrator cleaned up
            assert not await global_store.exists(call_id)


# ---------------------------------------------------------------------------
# Escalation shape contract
# ---------------------------------------------------------------------------


class TestEscalationContract:
    """Verify escalation info shape matches what the frontend expects."""

    @pytest.fixture
    async def call_id_questions(self):
        """Create a call advanced to the first question."""
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            # Create
            resp = await client.post(
                "/calls",
                json={
                    "patient_id": "P001",
                    "dia_postop": 3,
                    "procedimiento": "Apendicectomía laparoscópica",
                    "nombre_completo": "Paciente 001",
                    "eps": "EPS",
                },
            )
            cid = resp.json()["call_id"]
            # Greeting → Consent
            await client.post(f"/calls/{cid}/turn", json={"audio_base64": _MOCK_AUDIO_B64})
            # Consent → Questions (q_index=0)
            await client.post(f"/calls/{cid}/turn", json={"audio_base64": _MOCK_AUDIO_B64})

        yield cid
        await global_store.remove(cid)

    @pytest.mark.asyncio
    async def test_escalation_fields_match_frontend_contract(self, call_id_questions):
        """The escalation object shape must match what call.js renders."""
        async def _red_stt(audio_data: bytes) -> TranscriptionResult:
            return TranscriptionResult(
                text="Me duele mucho, un 9 de 10, no aguanto.",
                language="es",
                duration_seconds=2.0,
                model="whisper-large-v3",
            )

        with patch("backend.api.calls._stt", _red_stt):
            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                response = await client.post(
                    f"/calls/{call_id_questions}/turn",
                    json={"audio_base64": _MOCK_AUDIO_B64},
                )

        assert response.status_code == 200
        esc = response.json()["escalation"]

        # Fields call.js reads
        assert esc is not None
        assert "severity" in esc
        assert "should_escalate" in esc
        assert "reason" in esc
        assert "next_action" in esc
        assert "domain" in esc

        # Severity values the frontend styles for
        assert esc["severity"] in ("GREEN", "YELLOW", "RED")

    @pytest.mark.asyncio
    async def test_green_escalation_displayed(self, call_id_questions):
        """Even GREEN results are returned (not null) so the frontend can
        show a benign-status banner."""
        async def _benign_stt(audio_data: bytes) -> TranscriptionResult:
            return TranscriptionResult(
                text="Todo bien, casi nada de dolor, un 1.",
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
                    f"/calls/{call_id_questions}/turn",
                    json={"audio_base64": _MOCK_AUDIO_B64},
                )

        assert response.status_code == 200
        esc = response.json()["escalation"]
        assert esc is not None
        assert esc["severity"] == "GREEN"
        assert esc["should_escalate"] is False


# ---------------------------------------------------------------------------
# Citation contract
# ---------------------------------------------------------------------------


class TestCitationContract:
    """Verify citation shape matches what the frontend expects (even if empty)."""

    @pytest.fixture
    async def call_id(self):
        """Create a call for citation testing."""
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            resp = await client.post(
                "/calls",
                json={
                    "patient_id": "P005",
                    "dia_postop": 7,
                    "procedimiento": "Reemplazo total de cadera",
                    "nombre_completo": "Paciente 005",
                    "eps": "EPS",
                },
            )
        cid = resp.json()["call_id"]
        yield cid
        await global_store.remove(cid)

    @pytest.mark.asyncio
    async def test_citations_is_always_list(self, call_id):
        """The citations field must always be a list (even empty) so
        call.js's ``data.citations || []`` works safely."""
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
        assert isinstance(data["citations"], list)

    @pytest.mark.asyncio
    async def test_citation_fields_when_present(self, call_id):
        """When citations are returned, each must have the expected fields."""
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            # Advance through all turns to reach CLOSING (citations may appear)
            for _ in range(9):
                response = await client.post(
                    f"/calls/{call_id}/turn",
                    json={"audio_base64": _MOCK_AUDIO_B64},
                )
                if response.status_code != 200:
                    break

            assert response.status_code == 200
            data = response.json()

            # If citations exist, validate their shape
            for citation in data["citations"]:
                assert "chunk_id" in citation
                assert "document_id" in citation
                assert "source_filename" in citation
                assert "page_number" in citation
                assert isinstance(citation["page_number"], int)
                assert citation["page_number"] >= 1


# ---------------------------------------------------------------------------
# Error handling contract
# ---------------------------------------------------------------------------


class TestErrorHandlingContract:
    """Verify error response shapes the frontend handles."""

    @pytest.mark.asyncio
    async def test_missing_call_returns_404(self):
        """A turn on a non-existent call must return 404 with detail message."""
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            response = await client.post(
                "/calls/nonexistent-call-id/turn",
                json={"audio_base64": _MOCK_AUDIO_B64},
            )

        assert response.status_code == 404
        data = response.json()
        assert "detail" in data
        assert "not found" in data["detail"].lower()

    @pytest.mark.asyncio
    async def test_invalid_base64_returns_400(self):
        """Invalid base64 must return 400 — frontend shows error to user."""
        # First create a call
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            resp = await client.post(
                "/calls",
                json={
                    "patient_id": "P001",
                    "dia_postop": 3,
                    "procedimiento": "Apendicectomía laparoscópica",
                    "nombre_completo": "Paciente 001",
                    "eps": "EPS",
                },
            )
            call_id = resp.json()["call_id"]

            # Send invalid base64
            resp = await client.post(
                f"/calls/{call_id}/turn",
                json={"audio_base64": "!!!not-valid!!!"},
            )

        assert resp.status_code == 400
        assert "detail" in resp.json()

        # Clean up
        await global_store.remove(call_id)

    @pytest.mark.asyncio
    async def test_empty_audio_base64_returns_422(self):
        """Empty audio_base64 is a validation error — 422."""
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            resp = await client.post(
                "/calls",
                json={
                    "patient_id": "P001",
                    "dia_postop": 3,
                    "procedimiento": "Apendicectomía laparoscópica",
                    "nombre_completo": "Paciente 001",
                    "eps": "EPS",
                },
            )
            call_id = resp.json()["call_id"]

            resp = await client.post(
                f"/calls/{call_id}/turn",
                json={"audio_base64": ""},
            )

        assert resp.status_code == 422
        await global_store.remove(call_id)

    @pytest.mark.asyncio
    async def test_create_call_validation_errors(self):
        """Invalid patient data returns 422 — frontend shows error."""
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
                    "patient_id": "P001",
                    "dia_postop": -1,
                    "procedimiento": "Test",
                    "nombre_completo": "Test",
                },
            )
            assert resp.status_code == 422

            # Empty patient_id
            resp = await client.post(
                "/calls",
                json={
                    "patient_id": "",
                    "dia_postop": 0,
                    "procedimiento": "Test",
                    "nombre_completo": "Test",
                },
            )
            assert resp.status_code == 422


# ---------------------------------------------------------------------------
# SessionStorage-like state preservation
# ---------------------------------------------------------------------------


class TestCreateCallResponseIsSelfContained:
    """Verify the CreateCallResponse contains everything the frontend needs
    to pass via sessionStorage and start the call page without additional
    API calls."""

    @pytest.mark.asyncio
    async def test_response_has_all_session_storage_fields(self):
        """All fields stored by app.js in sessionStorage must be present."""
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
                    "nombre_completo": "Paciente 001",
                    "eps": "EPS",
                },
            )

        assert response.status_code == 201
        data = response.json()

        # These are the fields app.js stores in sessionStorage.callData:
        assert "call_id" in data
        assert "audio_base64" in data  # stored as greeting_audio_b64
        assert "total_questions" in data

        # Verify call_id is a non-empty string (used as URL-safe identifier)
        call_id = data["call_id"]
        assert isinstance(call_id, str)
        assert len(call_id) > 0
        assert " " not in call_id  # URL-safe

        # Verify greeting audio is valid base64
        decoded = base64.b64decode(data["audio_base64"])
        assert len(decoded) > 0


# ---------------------------------------------------------------------------
# Patient transcription rendering contract
# ---------------------------------------------------------------------------


class TestPatientTranscriptionContract:
    """Verify the patient_transcription field contract that call.js depends on.

    call.js line 415-418 reads::

        addMessage(
            "patient",
            data.patient_transcription || "🎤 [grabación enviada]"
        );

    The field must always be present and its value must faithfully reflect the
    STT output for the patient's speech.  The frontend uses JavaScript
    false-coalescing (``||``), so ``null``, ``undefined`` (missing key), and
    ``""`` would all trigger the fallback string.
    """

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
                    "patient_id": "P001",
                    "dia_postop": 3,
                    "procedimiento": "Apendicectomía laparoscópica",
                    "nombre_completo": "Paciente 001",
                    "eps": "EPS",
                },
            )
        assert response.status_code == 201
        cid = response.json()["call_id"]
        yield cid
        await global_store.remove(cid)

    @pytest.mark.asyncio
    async def test_patient_transcription_contains_stt_output(self, call_id):
        """The patient_transcription field must be a non-empty string that
        matches the STT provider's transcription output.

        This is the primary path — when STT succeeds the frontend renders
        the patient's exact transcribed words in the conversation history.
        """
        original_stt_output = "Sí, acepto continuar con la llamada."

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

        # Key assertion: the field exists and matches the mock STT output
        assert "patient_transcription" in data
        assert data["patient_transcription"] == original_stt_output
        assert isinstance(data["patient_transcription"], str)
        assert len(data["patient_transcription"]) > 0

    @pytest.mark.asyncio
    async def test_patient_transcription_is_never_missing_key(self, call_id):
        """The key 'patient_transcription' must always be present in every
        TurnResponse.  Even when the value is ``None`` the key itself must
        exist so that ``data.patient_transcription || "..."`` works without
        a ReferenceError in the browser."""
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

        # The key is unconditionally present
        assert "patient_transcription" in data
        # Value is either a non-empty string or None (never absent / undefined)
        assert data["patient_transcription"] is None or (
            isinstance(data["patient_transcription"], str)
            and len(data["patient_transcription"]) > 0
        )

    @pytest.mark.asyncio
    async def test_patient_transcription_value_across_multiple_turns(
        self, call_id
    ):
        """The patient_transcription reflects the STT output for *each*
        turn, not a stale value from a previous turn."""
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            # Turn 1: greeting response → CONSENT
            resp1 = await client.post(
                f"/calls/{call_id}/turn",
                json={"audio_base64": _MOCK_AUDIO_B64},
            )
            assert resp1.status_code == 200
            pt1 = resp1.json()["patient_transcription"]

            # Turn 2: consent → QUESTIONS (q=0)
            resp2 = await client.post(
                f"/calls/{call_id}/turn",
                json={"audio_base64": _MOCK_AUDIO_B64},
            )
            assert resp2.status_code == 200
            pt2 = resp2.json()["patient_transcription"]

        # Both turns used the same mock STT output, so values match
        assert pt1 == pt2 == "Sí, acepto continuar con la llamada."
        assert isinstance(pt1, str) and len(pt1) > 0
        assert isinstance(pt2, str) and len(pt2) > 0


# ---------------------------------------------------------------------------
# Agent transcription preservation
# ---------------------------------------------------------------------------


class TestAgentTranscriptionContract:
    """Verify that the agent ``transcription`` field is always a non-empty
    string in every TurnResponse.

    call.js line 421-423 reads::

        if (data.transcription) {
            addMessage("agent", data.transcription, data.citations || []);
        }

    A missing, null, or empty transcription would silently skip the agent
    message, breaking the conversation history the patient sees.
    """

    @pytest.mark.asyncio
    async def test_agent_transcription_never_empty(self):
        """Walk through a full call and verify *every* TurnResponse has a
        non-empty agent transcription string."""
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            # Create call
            resp = await client.post(
                "/calls",
                json={
                    "patient_id": "P002",
                    "dia_postop": 5,
                    "procedimiento": "Colecistectomía",
                    "nombre_completo": "Paciente 002",
                    "eps": "EPS",
                },
            )
            assert resp.status_code == 201
            call_id = resp.json()["call_id"]

            # Advance through every turn and check agent transcription
            responses = []
            for _ in range(10):
                r = await client.post(
                    f"/calls/{call_id}/turn",
                    json={"audio_base64": _MOCK_AUDIO_B64},
                )
                responses.append(r)
                if r.json().get("call_ended"):
                    break

            # Clean up
            await global_store.remove(call_id)

        turn_responses = [
            r for r in responses if r.status_code == 200
        ]
        assert len(turn_responses) >= 7, (
            f"Expected at least 7 turns, got {len(turn_responses)}"
        )

        for i, r in enumerate(turn_responses):
            data = r.json()
            transcription = data.get("transcription")
            assert transcription is not None, (
                f"Turn {i}: agent transcription is missing"
            )
            assert isinstance(transcription, str), (
                f"Turn {i}: agent transcription is {type(transcription)}"
            )
            assert len(transcription) > 0, (
                f"Turn {i}: agent transcription is empty"
            )

    @pytest.mark.asyncio
    async def test_agent_transcription_across_state_transitions(self):
        """The agent transcription field is populated at every state
        transition: GREETING→CONSENT, CONSENT→QUESTIONS, QUESTIONS→…,
        CLOSING→ENDED."""
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            resp = await client.post(
                "/calls",
                json={
                    "patient_id": "P003",
                    "dia_postop": 2,
                    "procedimiento": "Hernioplastia inguinal",
                    "nombre_completo": "Paciente 003",
                    "eps": "EPS",
                },
            )
            assert resp.status_code == 201
            call_id = resp.json()["call_id"]

            # GREETING → CONSENT
            r1 = await client.post(
                f"/calls/{call_id}/turn",
                json={"audio_base64": _MOCK_AUDIO_B64},
            )
            assert r1.status_code == 200
            assert r1.json()["state"] == State.CONSENT.value
            assert isinstance(r1.json()["transcription"], str)
            assert len(r1.json()["transcription"]) > 0

            # CONSENT → QUESTIONS (q=0)
            r2 = await client.post(
                f"/calls/{call_id}/turn",
                json={"audio_base64": _MOCK_AUDIO_B64},
            )
            assert r2.status_code == 200
            assert r2.json()["state"] == State.QUESTIONS.value
            assert isinstance(r2.json()["transcription"], str)
            assert len(r2.json()["transcription"]) > 0

            # Answer all questions until ENDED
            for _attempt in range(8):
                r = await client.post(
                    f"/calls/{call_id}/turn",
                    json={"audio_base64": _MOCK_AUDIO_B64},
                )
                assert r.status_code == 200
                assert isinstance(r.json()["transcription"], str)
                assert len(r.json()["transcription"]) > 0
                if r.json().get("call_ended"):
                    break

            await global_store.remove(call_id)


# ---------------------------------------------------------------------------
# Citations through call-flow states
# ---------------------------------------------------------------------------


class TestCitationsCallFlowContract:
    """Verify citations shape at each conversation phase.

    call.js passes ``data.citations || []`` to ``addMessage`` and renders
    each citation's ``source_filename``, ``document_id``, and ``page_number``
    in a ``.citations-list`` div.
    """

    @pytest.mark.asyncio
    async def test_citations_list_through_all_phases(self):
        """The ``citations`` field must be a list at every phase and each
        citation, when present, must have the fields call.js renders."""
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            resp = await client.post(
                "/calls",
                json={
                    "patient_id": "P004",
                    "dia_postop": 4,
                    "procedimiento": "Cesárea",
                    "nombre_completo": "Paciente 004",
                    "eps": "EPS",
                },
            )
            assert resp.status_code == 201
            call_id = resp.json()["call_id"]

            phases_checked = 0
            for _turn in range(10):
                r = await client.post(
                    f"/calls/{call_id}/turn",
                    json={"audio_base64": _MOCK_AUDIO_B64},
                )
                if r.status_code != 200:
                    break
                data = r.json()
                # citations is always a list
                assert isinstance(data["citations"], list), (
                    f"State {data['state']}: citations is {type(data['citations'])}, expected list"
                )
                # validate each citation when present
                for cit in data["citations"]:
                    assert "chunk_id" in cit
                    assert "document_id" in cit
                    assert "source_filename" in cit
                    assert "page_number" in cit
                    assert isinstance(cit["page_number"], int)
                    assert cit["page_number"] >= 1
                phases_checked += 1
                if data.get("call_ended"):
                    break

            await global_store.remove(call_id)

        # Must have covered at least CONSENT, QUESTIONS, CLOSING, ENDED
        assert phases_checked >= 4, (
            f"Expected at least 4 phases, got {phases_checked}"
        )


# ---------------------------------------------------------------------------
# Escalation timing contract
# ---------------------------------------------------------------------------


class TestEscalationTimingContract:
    """Verify when the ``escalation`` field is None vs populated.

    call.js line 426 reads::

        showEscalation(data.escalation || null);

    and ``showEscalation`` hides the banner when the value is falsy.
    The escalation verdict is only populated during the QUESTIONS phase
    (one per answered question) and at CLOSING.
    """

    @pytest.mark.asyncio
    async def test_escalation_null_before_questions(self):
        """During GREETING and CONSENT phases, escalation must be None."""
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            resp = await client.post(
                "/calls",
                json={
                    "patient_id": "P005",
                    "dia_postop": 7,
                    "procedimiento": "Reemplazo total de cadera",
                    "nombre_completo": "Paciente 005",
                    "eps": "EPS",
                },
            )
            assert resp.status_code == 201
            call_id = resp.json()["call_id"]

            # Turn 1: GREETING response → CONSENT
            r1 = await client.post(
                f"/calls/{call_id}/turn",
                json={"audio_base64": _MOCK_AUDIO_B64},
            )
            assert r1.status_code == 200
            assert r1.json()["state"] == State.CONSENT.value
            assert r1.json()["escalation"] is None, (
                "escalation must be None during CONSENT response"
            )

            # Turn 2: CONSENT → QUESTIONS (q=0)
            r2 = await client.post(
                f"/calls/{call_id}/turn",
                json={"audio_base64": _MOCK_AUDIO_B64},
            )
            assert r2.status_code == 200
            assert r2.json()["state"] == State.QUESTIONS.value
            assert r2.json()["escalation"] is None, (
                "escalation must be None after consent (no answer to classify yet)"
            )

            await global_store.remove(call_id)

    @pytest.mark.asyncio
    async def test_escalation_present_for_every_question(self):
        """Every answered question during the QUESTIONS phase must produce
        a non-None escalation verdict so the banner is never silently hidden
        for a turn that should show severity."""
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            resp = await client.post(
                "/calls",
                json={
                    "patient_id": "P001",
                    "dia_postop": 3,
                    "procedimiento": "Apendicectomía laparoscópica",
                    "nombre_completo": "Paciente 001",
                    "eps": "EPS",
                },
            )
            assert resp.status_code == 201
            call_id = resp.json()["call_id"]

            # Skip to QUESTIONS
            await client.post(
                f"/calls/{call_id}/turn",
                json={"audio_base64": _MOCK_AUDIO_B64},
            )  # GREETING → CONSENT
            await client.post(
                f"/calls/{call_id}/turn",
                json={"audio_base64": _MOCK_AUDIO_B64},
            )  # CONSENT → QUESTIONS q=0

            # Answer all 6 questions
            for i in range(6):
                r = await client.post(
                    f"/calls/{call_id}/turn",
                    json={"audio_base64": _MOCK_AUDIO_B64},
                )
                assert r.status_code == 200
                data = r.json()

                # Every answered question must have an escalation verdict
                assert data["escalation"] is not None, (
                    f"Question {i}: escalation is None — frontend would fail "
                    f"to show severity banner"
                )
                esc = data["escalation"]
                assert "severity" in esc
                assert esc["severity"] in ("GREEN", "YELLOW", "RED")
                assert "should_escalate" in esc
                assert isinstance(esc["should_escalate"], bool)
                assert "reason" in esc
                assert "next_action" in esc
                assert "domain" in esc
                # domain must be a non-empty string for the banner label
                assert isinstance(esc["domain"], str)
                assert len(esc["domain"]) > 0

            await global_store.remove(call_id)

    @pytest.mark.asyncio
    async def test_escalation_at_closing_and_ended(self):
        """At CLOSING the escalation verdict for the final question is
        present.  Once the call reaches ENDED the field may be None or
        the final verdict — the frontend's showEscalation handles both."""
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            resp = await client.post(
                "/calls",
                json={
                    "patient_id": "P002",
                    "dia_postop": 5,
                    "procedimiento": "Colecistectomía",
                    "nombre_completo": "Paciente 002",
                    "eps": "EPS",
                },
            )
            assert resp.status_code == 201
            call_id = resp.json()["call_id"]

            # Walk through every turn until ENDED
            for _turn in range(10):
                r = await client.post(
                    f"/calls/{call_id}/turn",
                    json={"audio_base64": _MOCK_AUDIO_B64},
                )
                if r.status_code != 200:
                    break
                data = r.json()
                state = data["state"]

                if state == State.CLOSING.value:
                    # At CLOSING: the final question's escalation is present
                    assert data["escalation"] is not None, (
                        "escalation must be present at CLOSING"
                    )
                    assert data["escalation"]["domain"] == "movilidad"
                elif state == State.ENDED.value:
                    # At ENDED: field exists (None or final verdict — both ok)
                    assert "escalation" in data
                    break

            await global_store.remove(call_id)


# ---------------------------------------------------------------------------
# Call-state progression contract
# ---------------------------------------------------------------------------


class TestCallStateProgressionContract:
    """Verify the exact sequence of state transitions the frontend observes.

    call.js maintains a ``currentState`` variable and updates it from every
    TurnResponse like::

        if (data.state) {
            setCallState(data.state);
        }

    ``setCallState`` updates the badge CSS class and text content.  States
    outside the known ``STATES`` array are silently ignored (console.warn),
    so every state value at every turn must be one of the six allowed
    values.
    """

    VALID_STATES = frozenset(
        {"IDLE", "GREETING", "CONSENT", "QUESTIONS", "CLOSING", "ENDED"}
    )

    @pytest.mark.asyncio
    async def test_state_progression_greeting_to_ended(self):
        """The backend must return a strict, reproducible state sequence:
        GREETING → CONSENT → QUESTIONS → CLOSING → ENDED."""
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            # Create call
            resp = await client.post(
                "/calls",
                json={
                    "patient_id": "P003",
                    "dia_postop": 2,
                    "procedimiento": "Hernioplastia inguinal",
                    "nombre_completo": "Paciente 003",
                    "eps": "EPS",
                },
            )
            assert resp.status_code == 201
            create_data = resp.json()
            call_id = create_data["call_id"]

            # CreateCallResponse initial state
            assert create_data["state"] == State.GREETING.value
            assert create_data["state"] in self.VALID_STATES

            # Walk the full flow and record states
            observed = [State.GREETING.value]
            for _turn in range(10):
                r = await client.post(
                    f"/calls/{call_id}/turn",
                    json={"audio_base64": _MOCK_AUDIO_B64},
                )
                if r.status_code != 200:
                    break
                state = r.json()["state"]
                assert state in self.VALID_STATES, (
                    f"Unknown state '{state}' returned — frontend would ignore it"
                )
                observed.append(state)
                if r.json().get("call_ended"):
                    break

            await global_store.remove(call_id)

        # Minimum valid sequence
        assert observed[0] == State.GREETING.value
        assert State.CONSENT.value in observed
        assert State.QUESTIONS.value in observed
        assert State.CLOSING.value in observed
        assert observed[-1] == State.ENDED.value

        # No regression to earlier states (monotonic progression)
        state_order = {
            State.IDLE.value: 0,
            State.GREETING.value: 1,
            State.CONSENT.value: 2,
            State.QUESTIONS.value: 3,
            State.CLOSING.value: 4,
            State.ENDED.value: 5,
        }
        prev_idx = -1
        for s in observed:
            idx = state_order[s]
            assert idx >= prev_idx, (
                f"State regression: {observed[observed.index(s)-1] if observed.index(s) > 0 else 'start'} "
                f"→ {s}"
            )
            prev_idx = idx

    @pytest.mark.asyncio
    async def test_call_ended_flag_only_at_end(self):
        """The ``call_ended`` flag must be False throughout the call and
        only become True at the ENDED state."""
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            resp = await client.post(
                "/calls",
                json={
                    "patient_id": "P004",
                    "dia_postop": 4,
                    "procedimiento": "Cesárea",
                    "nombre_completo": "Paciente 004",
                    "eps": "EPS",
                },
            )
            assert resp.status_code == 201
            call_id = resp.json()["call_id"]

            for _turn in range(10):
                r = await client.post(
                    f"/calls/{call_id}/turn",
                    json={"audio_base64": _MOCK_AUDIO_B64},
                )
                if r.status_code != 200:
                    break
                data = r.json()
                if data.get("call_ended"):
                    assert data["state"] == State.ENDED.value, (
                        "call_ended=True but state is not ENDED"
                    )
                    assert data["requires_response"] is False
                    break
                else:
                    assert data["state"] != State.ENDED.value, (
                        "call_ended=False but state is already ENDED"
                    )

            await global_store.remove(call_id)
