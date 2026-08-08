"""Voice call REST endpoints.

POST   /calls                  — Create a new voice call.
POST   /calls/{call_id}/turn   — Process a voice turn (audio in → audio out).

The endpoints accept base64-encoded WAV audio, transcribe via the injected STT
provider, delegate to the ``ConversationOrchestrator`` for dialogue management,
run escalation classification on patient responses during follow-up questions,
and synthesise agent responses via the injected TTS adapter.
"""

from __future__ import annotations

import base64
import binascii
import datetime
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.api.call_store import call_store
from backend.conversation.context import PatientContext
from backend.conversation.orchestrator import (
    ConversationOrchestrator,
    OrchestratorTurn,
)
from backend.conversation.state import State
from backend.conversation.transitions import InvalidTransitionError
from backend.data.models import Patient as DataPatient
from backend.decision import classify, EscalationResult, Severity
from backend.voice.api import SttDependency, transcribe_audio
from backend.voice.models import SttError
from backend.voice.tts.config import TTSConfig
from backend.voice.tts.protocol import TTSProvider, TTSSynthesisError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Follow-up question index → symptom domain mapping
# ---------------------------------------------------------------------------
# The conversation orchestrator asks six structured questions in order
# (see ``conversation/orchestrator.py`` → ``FOLLOW_UP_QUESTIONS``).  Each
# maps to a symptom domain used by the escalation engine.

_QUESTION_DOMAINS: tuple[str, ...] = (
    "dolor",
    "fiebre",
    "herida",
    "apetito",
    "sueño",
    "movilidad",
)

# ---------------------------------------------------------------------------
# Injectable STT / TTS dependencies
# ---------------------------------------------------------------------------

_stt: SttDependency | None = None
"""Injectable async STT callable (set via ``set_stt()``)."""

_tts: TTSProvider | None = None
"""Injectable TTS adapter (set via ``set_tts()``)."""

_tts_config: TTSConfig = TTSConfig()
"""Default TTS configuration, may be replaced through ``set_tts()``."""


def set_stt(stt: SttDependency) -> None:
    """Configure the STT provider used by the voice endpoints.

    Must be an async callable accepting ``bytes`` and returning a
    ``TranscriptionResult``.
    """
    global _stt
    _stt = stt


def set_tts(tts: TTSProvider, config: TTSConfig | None = None) -> None:
    """Configure the TTS adapter used by the voice endpoints.

    Must be an object with a ``synthesize(text, config)`` method returning a
    ``TTSResult``.
    """
    global _tts, _tts_config
    _tts = tts
    if config is not None:
        _tts_config = config


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

calls_router = APIRouter(prefix="/calls", tags=["calls"])


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class CitationResponse(BaseModel):
    """A traceable source citation returned to the caller."""

    chunk_id: str = Field(..., description="Unique chunk identifier")
    document_id: str = Field(..., description="Stable document identifier")
    source_filename: str = Field(..., description="Original PDF filename")
    page_number: int = Field(..., ge=1, description="1-based page number")


class EscalationInfo(BaseModel):
    """Escalation verdict for a patient turn."""

    severity: str = Field(..., description="GREEN, YELLOW, or RED")
    should_escalate: bool = Field(
        ..., description="True when the call should be escalated"
    )
    reason: str = Field(
        ..., description="Spanish-language clinical rationale"
    )
    next_action: str = Field(
        ..., description="Spanish-language instruction for the agent"
    )
    domain: str | None = Field(
        None, description="Symptom domain assessed"
    )

    @classmethod
    def from_result(cls, result: EscalationResult) -> "EscalationInfo":
        """Build from a domain ``EscalationResult``."""
        return cls(
            severity=result.severity.value,
            should_escalate=result.should_escalate,
            reason=result.reason,
            next_action=result.next_action,
            domain=result.domain,
        )


class CreateCallRequest(BaseModel):
    """Request body to create a new voice call."""

    patient_id: str = Field(
        ..., min_length=1, description="Unique patient identifier"
    )
    dia_postop: int = Field(
        ..., ge=0, description="Post-operative day number (>= 0)"
    )
    procedimiento: str = Field(
        ..., min_length=1, description="Surgical procedure name"
    )
    nombre_completo: str = Field(
        ..., min_length=1, description="Patient full name"
    )
    eps: str = Field(
        default="EPS", min_length=1, description="Health provider name"
    )


class CreateCallResponse(BaseModel):
    """Response returned after a call is created."""

    call_id: str = Field(..., description="Unique call identifier")
    audio_base64: str = Field(
        ..., description="Base64-encoded WAV audio of agent greeting"
    )
    state: str = Field(..., description="Conversation state after greeting")
    requires_response: bool = Field(
        ..., description="True when the patient is expected to respond"
    )
    question_index: int | None = Field(
        None,
        description="Zero-based index of the follow-up question just asked",
    )
    total_questions: int = Field(
        ..., description="Total number of follow-up questions"
    )
    call_ended: bool = Field(..., description="True if the call has ended")


class TurnRequest(BaseModel):
    """Request body for a voice turn."""

    audio_base64: str = Field(
        ...,
        min_length=1,
        description="Base64-encoded WAV audio of patient speech",
    )


class TurnResponse(BaseModel):
    """Response returned after processing a voice turn."""

    call_id: str = Field(..., description="Unique call identifier")
    audio_base64: str = Field(
        ..., description="Base64-encoded WAV audio of agent response"
    )
    transcription: str = Field(
        ..., description="Text transcription of the agent's response"
    )
    state: str = Field(..., description="Conversation state after this turn")
    citations: list[CitationResponse] = Field(
        default_factory=list,
        description="Traceable source citations (may be empty)",
    )
    requires_response: bool = Field(
        ..., description="True when the patient is expected to respond"
    )
    question_index: int | None = Field(
        None,
        description="Zero-based index of the follow-up question just asked",
    )
    total_questions: int = Field(
        ..., description="Total number of follow-up questions"
    )
    call_ended: bool = Field(..., description="True if the call has ended")
    escalation: EscalationInfo | None = Field(
        None, description="Escalation verdict (only during QUESTIONS phase)"
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_data_patient(body: CreateCallRequest) -> DataPatient:
    """Build a minimal ``DataPatient`` from request fields.

    Only ``nombre_completo``, ``eps``, and ``procedimiento`` are consumed
    by the orchestrator; remaining fields are filled with safe defaults.
    """
    return DataPatient(
        paciente_id=body.patient_id,
        bundle_id="",
        synthea_runtime="",
        modulo_synthea="",
        procedimiento=body.procedimiento,
        fecha_cirugia=datetime.date.today(),
        edad=0,
        genero="",
        comorbilidades=[],
        complicacion_encounter=False,
        nombre_completo=body.nombre_completo,
        direccion="",
        ciudad="",
        departamento="",
        documento_cc="",
        eps=body.eps,
        source_country="CO",
        adapted_country="CO",
        adaptation_fields=[],
    )


async def _transcribe(audio_bytes: bytes) -> str:
    """Transcribe audio bytes to text using the injected STT provider.

    Returns:
        Whitespace-stripped transcription.

    Raises:
        HTTPException (502): If the STT provider fails.
        HTTPException (500): If no STT provider is configured.
    """
    if _stt is None:
        raise HTTPException(
            status_code=500,
            detail="STT provider not configured.",
        )
    try:
        result = await transcribe_audio(audio_bytes, _stt)
        return result.text.strip()
    except SttError as exc:
        logger.error("STT transcription failed: %s", exc)
        raise HTTPException(
            status_code=502,
            detail=f"Error de transcripción: {exc}",
        ) from exc


def _synthesize(text: str) -> bytes:
    """Synthesise text to WAV bytes using the injected TTS adapter.

    Returns:
        Valid WAV container bytes (16-bit PCM mono).

    Raises:
        HTTPException (502): If the TTS adapter fails.
        HTTPException (500): If no TTS adapter is configured.
    """
    if _tts is None:
        raise HTTPException(
            status_code=500,
            detail="TTS provider not configured.",
        )
    try:
        result = _tts.synthesize(text, _tts_config)
        return result.audio_bytes
    except TTSSynthesisError as exc:
        logger.error("TTS synthesis failed: %s", exc)
        raise HTTPException(
            status_code=502,
            detail=f"Error de síntesis de voz: {exc}",
        ) from exc


def _classify_response(
    patient_text: str,
    question_index: int | None,
    dia_postop: int,
    procedimiento: str,
) -> EscalationInfo | None:
    """Run escalation classification on a patient response.

    Classification is only performed during the QUESTIONS phase when the
    patient has just answered a follow-up question (i.e. *question_index*
    is not ``None`` and greater than zero, meaning the previous turn's
    question was just answered).
    """
    if question_index is None or question_index == 0:
        return None

    domain_idx = question_index - 1
    if domain_idx < 0 or domain_idx >= len(_QUESTION_DOMAINS):
        return None

    domain = _QUESTION_DOMAINS[domain_idx]

    result = classify(
        patient_text=patient_text,
        domain=domain,
        dia_postop=dia_postop,
        procedimiento=procedimiento,
    )
    return EscalationInfo.from_result(result)


def _decode_base64_audio(encoded: str) -> bytes:
    """Decode a base64 string to raw audio bytes.

    Raises:
        HTTPException (400): If the string is not valid base64 or is empty
            after decoding.
    """
    try:
        audio_bytes = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail="audio_base64 must be valid base64-encoded audio.",
        ) from exc
    if not audio_bytes:
        raise HTTPException(
            status_code=400,
            detail="Audio data is empty after decoding.",
        )
    return audio_bytes


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@calls_router.post("", response_model=CreateCallResponse, status_code=201)
async def create_call(body: CreateCallRequest) -> CreateCallResponse:
    """Create a new voice call and return the agent's greeting as audio.

    A ``ConversationOrchestrator`` is instantiated for the patient, the call
    is started (``start_call()``), and the agent greeting is synthesised to
    WAV audio, base64-encoded, and returned.
    """
    patient = _build_data_patient(body)

    patient_context = PatientContext(
        patient=patient,
        dia_postop=body.dia_postop,
        procedimiento=body.procedimiento,
    )

    orchestrator = ConversationOrchestrator(
        patient_context=patient_context,
        rag_config=None,
        llm_config=None,
    )

    turn: OrchestratorTurn = orchestrator.start_call()

    # Synthesise the agent greeting to WAV audio bytes
    wav_bytes = _synthesize(turn.agent_message)
    audio_b64 = base64.b64encode(wav_bytes).decode("ascii")

    # Store the orchestrator for subsequent turn requests
    await call_store.put(patient_context.call_id, orchestrator)

    logger.info(
        "Call %s created for patient %s (state=%s)",
        patient_context.call_id,
        body.patient_id,
        turn.state.name,
    )

    return CreateCallResponse(
        call_id=patient_context.call_id,
        audio_base64=audio_b64,
        state=turn.state.value,
        requires_response=turn.requires_response,
        question_index=turn.question_index,
        total_questions=turn.total_questions,
        call_ended=turn.call_ended,
    )


@calls_router.post("/{call_id}/turn", response_model=TurnResponse)
async def process_turn(
    call_id: str,
    body: TurnRequest,
) -> TurnResponse:
    """Process a patient voice turn and return the agent's spoken response.

    Flow:
    1. Look up the orchestrator by *call_id* (404 if not found).
    2. Decode base64 audio → bytes.
    3. Transcribe via STT → patient text.
    4. Feed patient text into the orchestrator → ``OrchestratorTurn``.
    5. Classify escalation on the patient response (when applicable).
    6. Synthesise agent response → WAV bytes.
    7. Base64-encode and return.
    """
    # 1. Look up orchestrator
    orchestrator = await call_store.get(call_id)
    if orchestrator is None:
        raise HTTPException(
            status_code=404,
            detail=f"Call '{call_id}' not found. Create a call first with POST /calls.",
        )

    # 2. Decode audio
    audio_bytes = _decode_base64_audio(body.audio_base64)

    # 3. Transcribe
    patient_text = await _transcribe(audio_bytes)
    if not patient_text:
        raise HTTPException(
            status_code=400,
            detail="Transcription returned empty text. "
            "Please provide audible speech.",
        )

    logger.info(
        "Call %s turn: patient=%r (state=%s)",
        call_id,
        patient_text[:120],
        orchestrator.state.name,
    )

    # 4. Process through orchestrator
    try:
        turn: OrchestratorTurn = orchestrator.process_patient_message(
            patient_text
        )
    except InvalidTransitionError as exc:
        logger.warning("Invalid transition in call %s: %s", call_id, exc)
        raise HTTPException(
            status_code=400,
            detail=f"Invalid conversation transition: {exc}",
        ) from exc
    except ValueError as exc:
        logger.warning("ValueError in call %s: %s", call_id, exc)
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    # 5. Classify escalation (only during QUESTIONS phase)
    pc = orchestrator.call_context.patient_context
    escalation = _classify_response(
        patient_text=patient_text,
        question_index=turn.question_index,
        dia_postop=pc.dia_postop,
        procedimiento=pc.procedimiento,
    )

    # 6. Synthesise agent response
    wav_bytes = _synthesize(turn.agent_message)
    audio_b64 = base64.b64encode(wav_bytes).decode("ascii")

    # 7. If call has ended, clean up the orchestrator
    if turn.call_ended:
        await call_store.remove(call_id)
        logger.info("Call %s ended — orchestrator removed from store.", call_id)

    return TurnResponse(
        call_id=call_id,
        audio_base64=audio_b64,
        transcription=turn.agent_message,
        state=turn.state.value,
        citations=[
            CitationResponse(
                chunk_id=c.get("chunk_id", ""),
                document_id=c.get("document_id", ""),
                source_filename=c.get("source_filename", ""),
                page_number=c.get("page_number", 1),
            )
            for c in turn.citations
        ],
        requires_response=turn.requires_response,
        question_index=turn.question_index,
        total_questions=turn.total_questions,
        call_ended=turn.call_ended,
        escalation=escalation,
    )
