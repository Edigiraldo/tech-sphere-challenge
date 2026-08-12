"""Voice call REST endpoints.

POST   /calls                  — Create a new voice call.
POST   /calls/{call_id}/turn   — Process a voice turn (audio in → audio out).

The endpoints accept base64-encoded WAV audio, transcribe via the injected STT
provider, delegate to the ``ConversationOrchestrator`` for dialogue management,
run escalation classification on patient responses during follow-up questions,
and synthesise agent responses via the injected TTS adapter.

The ``create_call`` endpoint loads the patient's real dataset profile when
available (via ``backend.data.load_patients``), falling back to the
request-body fields when the patient is not found in the dataset.  The
orchestrator is wired with live ``RagConfig`` and ``LlmConfig`` (from
environment variables); safe fallbacks built into the orchestrator handle
cases where RAG or LLM are unavailable.
"""

from __future__ import annotations

import base64
import binascii
import datetime
import json
import logging
import time
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.api.call_store import call_store
from backend.api.metrics import metrics_collector
from backend.conversation.context import PatientContext
from backend.conversation.messages import MessageRole
from backend.conversation.orchestrator import (
    ConversationOrchestrator,
    OrchestratorTurn,
)
from backend.conversation.state import State
from backend.conversation.transitions import InvalidTransitionError
from backend.data.loader import load_patients
from backend.data.models import Patient as DataPatient
from backend.decision import classify, EscalationResult, Severity
from backend.llm.config import LlmConfig
from backend.metrics.models import TurnMetrics
from backend.persistence.sqlite import (
    CallRecord,
    ConversationTurnRecord,
    EscalationAlertRecord,
    SummaryRecord,
    insert_call as _db_insert_call,
    insert_call_metrics,
    insert_escalation_alert,
    insert_summary,
    insert_turn_metrics_row,
    insert_turns,
    update_call_ended,
    update_call_metrics_ended,
)
from backend.rag.config import RagConfig
from backend.summaries.generator import generate_summary
from backend.summaries.models import SourceReference
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

_call_turn_index: dict[str, int] = {}
"""Per-call turn index counter for metrics instrumentation."""

_call_escalations: dict[str, list[EscalationInfo]] = {}
"""Per-call accumulated escalation results for summary generation."""

_call_citations: dict[str, list[CitationResponse]] = {}
"""Per-call accumulated citations (deduplicated by document_id) for summary generation."""

_call_consecutive_yellows: dict[str, int] = {}
"""Per-call consecutive YELLOW count at the API boundary.

Used by ``_classify_response`` to enforce the two-consecutive-YELLOW
escalation rule when the orchestrator does not provide an escalation
verdict (fallback path).  Reset on GREEN, RED, consent refusal, and
call completion.  The orchestrator's ``_consecutive_yellows`` counter
controls state transitions; this counter is authoritative for the HTTP
response escalation verdict in the fallback path.
"""

_DEFAULT_MODEL: str = "llama-3.3-70b-versatile"
"""Default model identifier for metrics when LLM is not invoked."""


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
    patient_transcription: str | None = Field(
        None,
        description="Text transcription of the patient's speech (as "
        "returned by STT).  ``None`` when the turn did not involve "
        "patient audio input (e.g. the call-creation greeting).",
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
# Lazy patient catalogue (dataset-backed, fallback to request body)
# ---------------------------------------------------------------------------

_patients_cache: dict[str, DataPatient] | None = None
"""Module-level cache of all 40 synthetic patients loaded once."""


def _get_patients() -> dict[str, DataPatient]:
    """Return the dataset patient catalogue, loading it lazily.

    Loading is deferred until the first call-creation request so that
    application startup stays fast and the health endpoint is available
    immediately.
    """
    global _patients_cache
    if _patients_cache is None:
        _patients_cache = load_patients()
        logger.info(
            "Loaded %d synthetic patients from dataset.", len(_patients_cache)
        )
    return _patients_cache


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _persist_call_record(
    call_id: str,
    body: CreateCallRequest,
    state: State,
) -> None:
    """Insert a ``CallRecord`` into SQLite for a newly created call.

    Called from ``create_call`` after the orchestrator is stored so the
    call row exists even if the first turn never arrives.
    """
    now = datetime.datetime.now(datetime.timezone.utc)
    record = CallRecord(
        call_id=call_id,
        paciente_id=body.patient_id,
        nombre_completo=body.nombre_completo,
        procedimiento=body.procedimiento,
        dia_postop=body.dia_postop,
        eps=body.eps,
        state=state.value,
        started_at=now,
        ended_at=None,
        total_turns=0,
        escalated=False,
    )
    try:
        _db_insert_call(record)
        logger.debug("CallRecord inserted for %s.", call_id)
    except Exception:
        logger.exception("Failed to persist CallRecord for %s.", call_id)


def _persist_call_turns(
    call_id: str,
    turn_index: int,
    patient_text: str,
    agent_text: str,
    severity: str | None,
    domain: str | None,
) -> None:
    """Persist patient and agent ``ConversationTurnRecord`` entries.

    The patient turn is recorded with ``severity`` / ``domain`` (when
    an escalation classification was performed).  The agent turn records
    only the response text.
    """
    now = datetime.datetime.now(datetime.timezone.utc)
    try:
        insert_turns([
            ConversationTurnRecord(
                turn_id=f"{call_id}-pt-{turn_index}",
                call_id=call_id,
                turn_index=turn_index * 2,
                role="PATIENT",
                text=patient_text,
                timestamp=now,
                severity=severity,
                domain=domain,
            ),
            ConversationTurnRecord(
                turn_id=f"{call_id}-at-{turn_index}",
                call_id=call_id,
                turn_index=turn_index * 2 + 1,
                role="AGENT",
                text=agent_text,
                timestamp=now,
            ),
        ])
        logger.debug(
            "Turn %d persisted for call %s.", turn_index, call_id
        )
    except Exception:
        logger.exception(
            "Failed to persist turns for call %s turn %d.", call_id, turn_index
        )


def _persist_escalation_alert(
    call_id: str,
    escalation: EscalationInfo,
) -> None:
    """Persist an ``EscalationAlertRecord`` when severity is YELLOW or RED.

    GREEN classifications are informational only and are not persisted as
    standalone alerts.
    """
    if escalation.severity == "GREEN":
        return
    now = datetime.datetime.now(datetime.timezone.utc)
    try:
        insert_escalation_alert(
            EscalationAlertRecord(
                alert_id=uuid.uuid4().hex,
                call_id=call_id,
                created_at=now,
                severity=escalation.severity,
                reason=escalation.reason,
                domain=escalation.domain,
            )
        )
        logger.info(
            "Escalation alert (%s) persisted for call %s in domain %s.",
            escalation.severity,
            call_id,
            escalation.domain,
        )
    except Exception:
        logger.exception(
            "Failed to persist escalation alert for call %s.", call_id
        )


def _persist_call_summary(
    orchestrator: ConversationOrchestrator,
    escalation_results: list[EscalationInfo],
    citations: list[CitationResponse],
) -> None:
    """Generate and persist a ``SummaryRecord`` when the call ends.

    Uses the existing ``generate_summary()`` to produce a
    ``SummaryResult``, then maps it to a ``SummaryRecord`` for SQLite
    persistence.

    Persistence failures are logged but never raised — the call has
    already ended and the HTTP response must not be affected.
    """
    call_id = orchestrator.call_context.patient_context.call_id
    pc = orchestrator.call_context.patient_context

    # Build turn records from orchestrator history
    turn_records: list[ConversationTurnRecord] = []
    for msg in orchestrator.history:
        role = "AGENT" if msg.role is MessageRole.AGENT else "PATIENT"
        turn_records.append(
            ConversationTurnRecord(
                turn_id=uuid.uuid4().hex,
                call_id=call_id,
                turn_index=msg.turn_index,
                role=role,
                text=msg.text,
                timestamp=msg.timestamp,
            )
        )

    # Convert EscalationInfo → EscalationResult for the summary generator
    esc_results: list[EscalationResult] = []
    for info in escalation_results:
        esc_results.append(
            EscalationResult(
                severity=Severity(info.severity),
                should_escalate=info.should_escalate,
                reason=info.reason,
                next_action=info.next_action,
                domain=info.domain,
                source="rule",
            )
        )

    # Build source references from citations
    sources: list[SourceReference] = []
    seen: set[str] = set()
    for c in citations:
        key = c.document_id
        if key and key not in seen:
            seen.add(key)
            sources.append(
                SourceReference(
                    document_id=c.document_id,
                    source_filename=c.source_filename,
                    page_number=c.page_number,
                )
            )

    try:
        summary = generate_summary(
            call_id=call_id,
            patient_context=pc,
            turns=turn_records,
            escalation_results=esc_results,
            sources=sources,
        )
    except Exception:
        logger.exception(
            "Failed to generate summary for call %s.", call_id
        )
        return

    try:
        insert_summary(
            SummaryRecord(
                summary_id=summary.summary_id,
                call_id=summary.call_id,
                created_at=summary.created_at,
                patient_summary=summary.patient_summary.content,
                procedure_summary=summary.procedure.content,
                symptoms_summary="\n".join(
                    s.content for s in summary.symptoms
                ),
                decision_summary=summary.decision.content,
                sources_json=json.dumps(
                    [
                        [s.document_id, s.source_filename, s.page_number]
                        for s in summary.sources
                    ],
                    ensure_ascii=False,
                ),
                next_steps=summary.next_steps.content,
            )
        )
        logger.info("Summary %s persisted for call %s.", summary.summary_id, call_id)
    except Exception:
        logger.exception("Failed to persist summary for call %s.", call_id)


def _build_data_patient(body: CreateCallRequest) -> DataPatient:
    """Build a ``DataPatient``, preferring the real dataset profile.

    Looks up *body.patient_id* in the dataset patients catalogue.  When
    found the full patient profile is returned as-is (real demographics and
    clinical data).  When not found, the request-body fields are used to
    construct a minimal fallback ``DataPatient`` so the call still works
    with manually entered patient information.

    The *dia_postop* and *procedimiento* from the request body are **not**
    applied here; the caller (``create_call``) passes them separately to
    ``PatientContext``, which is the object that governs the conversation.
    This allows the dataset profile to be used for demographics while the
    request body controls the post-operative context.
    """
    try:
        patients = _get_patients()
        if body.patient_id in patients:
            real = patients[body.patient_id]
            logger.info(
                "Call for patient %s: using real dataset profile.", body.patient_id
            )
            return real
    except Exception:
        logger.exception(
            "Failed to load dataset patients — falling back to request body "
            "for patient %s.",
            body.patient_id,
        )

    # Fallback: construct a minimal DataPatient from request fields
    logger.info(
        "Patient %s not found in dataset — using request-body fields.",
        body.patient_id,
    )
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
    call_id: str,
    patient_text: str,
    question_index: int | None,
    dia_postop: int,
    procedimiento: str,
    state: str | None = None,
) -> EscalationInfo | None:
    """Run escalation classification on a patient response.

    Classification is only performed during the QUESTIONS phase when the
    patient has just answered a follow-up question (i.e. *question_index*
    is not ``None`` and greater than zero).

    *question_index* semantics depend on the orchestrator state:
    - During **QUESTIONS**: *question_index* is the index of the next
      question just asked, so the answered domain is ``question_index - 1``.
    - During **CLOSING** (after escalation): *question_index* is the index
      of the question just answered, so the answered domain is
      ``question_index`` itself.

    When *state* is ``None`` (or not ``"CLOSING"``) the function assumes
    QUESTIONS-phase semantics for backward compatibility.

    **API-boundary consecutive-YELLOW accumulation:** This function tracks
    per-call consecutive YELLOW classifications via the module-level
    ``_call_consecutive_yellows`` dict.  When two YELLOW results occur
    without an intervening GREEN or RED, ``should_escalate`` is set to
    ``True`` for the second YELLOW.  This accumulation is only used in
    the fallback path (when ``turn.escalation`` is ``None``).  The
    counter is reset on GREEN, RED, and call completion.
    """
    if question_index is None or question_index == 0:
        return None

    if state == State.CLOSING.value:
        # question_index is the answered question (escalation path)
        domain_idx = question_index
    else:
        # question_index is the next question just asked
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

    # --- API-boundary consecutive-YELLOW accumulation --------------------
    if result.severity is Severity.GREEN or result.severity is Severity.RED:
        _call_consecutive_yellows[call_id] = 0
    elif result.severity is Severity.YELLOW:
        current = _call_consecutive_yellows.get(call_id, 0) + 1
        _call_consecutive_yellows[call_id] = current
        if current >= 2:
            # Second consecutive YELLOW → override should_escalate.
            # EscalationResult is frozen — construct a new instance.
            result = EscalationResult(
                severity=result.severity,
                should_escalate=True,
                reason=result.reason,
                next_action=result.next_action,
                domain=result.domain,
                source=result.source,
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

    # Wire the real RAG and LLM configurations (reads env vars).  The
    # orchestrator has built-in safe fallbacks for missing API keys,
    # empty ChromaDB stores, and LLM failures — a misconfigured
    # provider never crashes the call.
    try:
        rag_config = RagConfig()
    except Exception:
        logger.exception("Failed to build RagConfig — RAG will be unavailable.")
        rag_config = None

    try:
        llm_config = LlmConfig()
    except Exception:
        logger.exception("Failed to build LlmConfig — LLM will be unavailable.")
        llm_config = None

    orchestrator = ConversationOrchestrator(
        patient_context=patient_context,
        rag_config=rag_config,
        llm_config=llm_config,
    )

    turn: OrchestratorTurn = orchestrator.start_call()

    # Synthesise the agent greeting to WAV audio bytes
    wav_bytes = _synthesize(turn.agent_message)
    audio_b64 = base64.b64encode(wav_bytes).decode("ascii")

    # Store the orchestrator for subsequent turn requests
    await call_store.put(patient_context.call_id, orchestrator)

    # Persist the call record to SQLite (even before the first turn)
    _persist_call_record(
        patient_context.call_id, body, turn.state
    )

    # Register with metrics collector
    metrics_collector.start_call(patient_context.call_id, body.patient_id)
    _call_turn_index[patient_context.call_id] = 0

    # Persist the metrics-registration row so the call survives restart
    try:
        insert_call_metrics(patient_context.call_id, body.patient_id)
    except Exception:
        logger.warning(
            "Failed to persist call-metrics row for %s",
            patient_context.call_id,
            exc_info=True,
        )

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
    turn_start_ms = time.time() * 1000.0
    stt_start_ms = turn_start_ms
    patient_text = await _transcribe(audio_bytes)
    stt_duration_ms = max(0.0, time.time() * 1000.0 - stt_start_ms)
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

    # 5. Classify escalation — prefer orchestrator's classification when
    #    available (it classifies before RAG/LLM during QUESTIONS), falling
    #    back to the endpoint-level classifier with API-boundary
    #    consecutive-YELLOW accumulation for backward compatibility.
    pc = orchestrator.call_context.patient_context
    if turn.escalation is not None:
        escalation = EscalationInfo.from_result(turn.escalation)
    else:
        escalation = _classify_response(
            call_id=call_id,
            patient_text=patient_text,
            question_index=turn.question_index,
            dia_postop=pc.dia_postop,
            procedimiento=pc.procedimiento,
            state=turn.state.value,
        )

    # --- Reset consecutive-YELLOW counter on consent refusal -----------
    # When the patient refuses consent the orchestrator transitions to
    # CLOSING without performing any classification.  The state is
    # CLOSING and *escalation* is ``None`` (the ``_classify_response``
    # fallback returns ``None`` because ``question_index`` is ``None``
    # during consent).  This combination is unique to consent refusal
    # among all CLOSING transitions (all-question-done, RED, and
    # two-YELLOW paths all carry a non-``None`` *escalation*).
    if turn.state is State.CLOSING and escalation is None:
        _call_consecutive_yellows[call_id] = 0
    # ------------------------------------------------------------------

    # 6. Synthesise agent response
    tts_start_ms = time.time() * 1000.0
    wav_bytes = _synthesize(turn.agent_message)
    tts_duration_ms = max(0.0, time.time() * 1000.0 - tts_start_ms)
    audio_b64 = base64.b64encode(wav_bytes).decode("ascii")

    # Read turn_index *before* call-ended cleanup so the final turn
    # retains its sequential index instead of falling back to 0.
    turn_index = _call_turn_index.get(call_id, 0)

    # --- Persist conversation turns to SQLite ---------------------------
    _persist_call_turns(
        call_id=call_id,
        turn_index=turn_index,
        patient_text=patient_text,
        agent_text=turn.agent_message,
        severity=escalation.severity if escalation else None,
        domain=escalation.domain if escalation else None,
    )

    # --- Persist escalation alert (YELLOW/RED only) ---------------------
    if escalation is not None:
        _persist_escalation_alert(call_id, escalation)

    # Track escalation results for summary generation (module-level dict)
    if escalation is not None:
        _call_escalations.setdefault(call_id, []).append(escalation)

    # Build citations for later summary use
    turn_citations = [
        CitationResponse(
            chunk_id=c.get("chunk_id", ""),
            document_id=c.get("document_id", ""),
            source_filename=c.get("source_filename", ""),
            page_number=c.get("page_number", 1),
        )
        for c in turn.citations
    ]

    # Accumulate citations per call (deduplicated by document_id)
    # so the end-of-call summary includes sources from all turns,
    # not only the last one.
    if turn_citations:
        per_call = _call_citations.setdefault(call_id, [])
        existing_ids = {c.document_id for c in per_call}
        for tc in turn_citations:
            if tc.document_id and tc.document_id not in existing_ids:
                per_call.append(tc)
                existing_ids.add(tc.document_id)

    # 7. If call has ended, clean up the orchestrator and persist
    #    summary + final call state.
    if turn.call_ended:
        # Generate and persist structured summary before removing
        # the orchestrator (the orchestrator's history drives the
        # summary content).
        all_escalations_for_call = _call_escalations.get(call_id, [])
        all_citations_for_call = _call_citations.get(call_id, [])
        _persist_call_summary(
            orchestrator,
            escalation_results=all_escalations_for_call,
            citations=all_citations_for_call,
        )

        # Update the call record: mark ended, set final state and
        # total turn count (each HTTP turn = 2 conversation turns).
        _total = (
            (_call_turn_index.get(call_id, 0) + 1) * 2
        )
        _any_escalated = any(
            e.should_escalate for e in all_escalations_for_call
        )
        try:
            update_call_ended(
                call_id=call_id,
                state=turn.state.value,
                ended_at=datetime.datetime.now(datetime.timezone.utc),
                total_turns=_total,
                escalated=_any_escalated,
            )
            logger.info(
                "Call %s ended — SQLite update persisted.", call_id
            )
        except Exception:
            logger.exception(
                "Failed to update call-ended for call %s.", call_id
            )

        await call_store.remove(call_id)
        logger.info("Call %s ended — orchestrator removed from store.", call_id)

    # Record turn metrics with component timings
    turn_end_ms = time.time() * 1000.0
    total_latency_ms = max(0.0, turn_end_ms - turn_start_ms)

    # Determine the model identifier: use the LLM config when available,
    # otherwise fall back to the default placeholder.
    model_id: str = _DEFAULT_MODEL
    if orchestrator._llm_config is not None:
        model_id = orchestrator._llm_config.model_name

    try:
        metrics_collector.record_turn(
            TurnMetrics(
                call_id=call_id,
                turn_index=turn_index,
                total_latency_ms=total_latency_ms,
                model=model_id,
                rag_queries=turn.rag_queries,
                tts_duration_ms=tts_duration_ms,
                stt_duration_ms=stt_duration_ms,
                llm_duration_ms=turn.llm_duration_ms,
                input_tokens=turn.prompt_tokens,
                output_tokens=turn.completion_tokens,
            )
        )
        # Only advance the index when the call has *not* ended.  When
        # the call has ended the cleanup below removes the key, so
        # re-adding it would leak a stale counter.
        if not turn.call_ended:
            _call_turn_index[call_id] = turn_index + 1
    except ValueError as exc:
        # Silently skip only the known "call not started in collector"
        # case; re-raise or log unexpected ValueErrors so that
        # validation / instrumentation defects are visible.
        if "has not been started" in str(exc):
            pass
        else:
            logger.warning(
                "Unexpected ValueError recording metrics for call %s: %s",
                call_id,
                exc,
            )

    # Persist this turn's metrics to SQLite so they survive restart
    try:
        insert_turn_metrics_row(
            call_id=call_id,
            turn_index=turn_index,
            total_latency_ms=total_latency_ms,
            model=model_id,
            rag_queries=turn.rag_queries,
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            tts_duration_ms=tts_duration_ms,
            stt_duration_ms=stt_duration_ms,
            llm_duration_ms=turn.llm_duration_ms,
            input_tokens=turn.prompt_tokens,
            output_tokens=turn.completion_tokens,
        )
    except Exception:
        logger.warning(
            "Failed to persist turn metrics for call %s turn %d",
            call_id,
            turn_index,
            exc_info=True,
        )

    # --- Mark call as ended in metrics *after* the final turn has been
    #     recorded so that queries never observe a completed call that
    #     is missing its last turn.
    if turn.call_ended:
        metrics_collector.end_call(call_id)
        _call_turn_index.pop(call_id, None)
        _call_escalations.pop(call_id, None)
        _call_citations.pop(call_id, None)
        _call_consecutive_yellows.pop(call_id, None)

        # Mark the call as ended in SQLite metrics persistence as well
        try:
            update_call_metrics_ended(call_id)
        except Exception:
            logger.warning(
                "Failed to persist call-ended marker for %s",
                call_id,
                exc_info=True,
            )

    return TurnResponse(
        call_id=call_id,
        audio_base64=audio_b64,
        transcription=turn.agent_message,
        patient_transcription=patient_text,
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
