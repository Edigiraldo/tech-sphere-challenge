"""Conversation orchestrator: deterministic Spanish text-only call flow.

``ConversationOrchestrator`` connects the existing domain primitives —
``PatientContext``, ``CallContext``, ``State`` / ``Event`` state machine,
``History`` / ``Message``, RAG retrieval, LLM answer generation, and
**LLM second-approval** (``backend/llm/approval.py``) — into a single
coordinated dialogue flow for postoperative follow-up.

The orchestrator is text-only and deterministic: it uses a fixed sequence of
structured questions (in Spanish), drives state transitions safely, records
every turn in the history, and classifies patient answers **before** any
RAG/LLM call or doubt-intent gate.  RED answers short-circuit immediately
to ENDED with an urgent safety message, ``call_ended=True``, and no further
processing — RED never passes through LLM approval nor the doubt gate.

Every non-RED answer during QUESTIONS goes through the doubt-intent gate
(clinical questions trigger RAG and repeat the same question) and then
**LLM second-approval**
(``llm_second_approval()``), a conservative safety reviewer that may:
* **confirm** the deterministic classification,
* **escalate** severity (upgrade GREEN→YELLOW, GREEN→RED, or YELLOW→RED;
  downgrades are rejected),
* **request clarification** from the patient (at most one per question;
  stays on the same question), or
* **request RAG** for clinical doubt (runs RAG retrieval in QUESTIONS, then
  continues).

LLM failures, timeouts, invalid output, or severity-downgrade attempts fall
back automatically to the deterministic classification.  GREEN and
first-YELLOW answers confirmed by approval receive deterministic
acknowledgments without RAG+LLM answer generation.  Two consecutive YELLOW
classifications trigger escalation with ``should_escalate=True``.

Clinical questions during CLOSING are answered with RAG+LLM (with citations)
and the call remains in CLOSING; non-questions end the call.

Fallback behaviour:
* Consent refused → polite closing, call ends.
* RED classification → short-circuit to ENDED with urgent safety message
  and ``call_ended=True``.  No RAG/LLM call; no LLM approval; no further
  questions.
* LLM second-approval failure → deterministic classification used as-is
  (safe fallback; the approval is conservative — it only downgrades to the
  original deterministic result, never below it).
* Two consecutive YELLOW → escalation to CLOSING.
* No RAG chunks retrieved → the agent states it lacks information and
  advises consulting the treating physician.
* LLM unreachable → a safe fallback message is returned with
  ``insufficient_knowledge=True`` (handled by the LLM adapter).
"""

from __future__ import annotations

import datetime
import logging
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Optional

from backend.conversation.context import CallContext, PatientContext
from backend.conversation.messages import History, Message, MessageRole
from backend.conversation.state import Event, State
from backend.conversation.transitions import InvalidTransitionError, next_state
from backend.decision import classify as _decision_classify
from backend.decision import EscalationResult, Severity
from backend.llm.adapter import RagAnswer, generate_rag_answer
from backend.llm.approval import (
    llm_confirm_doubt,
    llm_second_approval,
    DoubtApprovalResult,
    LlmApprovalResult,
    _has_explicit_doubt_markers,
    _build_doubt_rag_query,
)
from backend.llm.config import LlmConfig
from backend.rag.config import RagConfig
from backend.rag.retrieval import RetrievalResult, retrieve

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Spanish text normalisation helper (stdlib only, no external deps)
# ---------------------------------------------------------------------------


def _normalize_spanish_text(text: str) -> str:
    """Remove diacritics and normalise Spanish text for question detection.

    STT transcriptions (Groq Whisper Large V3) frequently omit accents
    and punctuation, so ``"cómo"`` becomes ``"como"`` and ``"¿Qué?"``
    becomes ``"que"``.  This helper strips combining characters so
    question-marker matching works on both accented and unaccented input
    without requiring the caller to duplicate every pattern.

    Parameters
    ----------
    text : str
        Raw input text (may contain diacritics, punctuation, mixed case).

    Returns
    -------
    str
        Lower-case, diacritic-free, whitespace-stripped text suitable
        for heuristic substring matching.  Punctuation other than
        question marks is *not* removed — those are handled separately
        in ``_is_clinical_question``.
    """
    # Unicode NFD decomposes accented chars into base + combining diacritic
    # (e.g. 'é' → 'e' + U+0301).  Filtering out Unicode category 'Mn'
    # (Non-Spacing Mark) strips all diacritics while preserving the base
    # letters and whitespace.
    nfd = unicodedata.normalize("NFD", text)
    stripped = "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn")
    return stripped.lower().strip()


# ---------------------------------------------------------------------------
# Structured postoperative questions (Spanish, Colombian regionalisms)
# ---------------------------------------------------------------------------
# Each question targets a specific follow-up domain.  The orchestrator asks
# them in order during the QUESTIONS state, using the patient's procedure
# name for personalisation.

FOLLOW_UP_QUESTIONS: list[str] = [
    "¿Cómo ha sido su nivel de dolor en los últimos días? Por favor, "
    "descríbame la intensidad, de 0 (sin dolor) a 10 (el peor dolor "
    "imaginable), y en qué parte del cuerpo lo siente.",

    "¿Ha tenido fiebre o escalofríos? Si ha medido su temperatura, "
    "¿cuánto le marcó el termómetro?",

    "¿Cómo está su herida quirúrgica? ¿Ha notado enrojecimiento, "
    "hinchazón, secreción, mal olor o que se haya abierto?",

    "¿Cómo ha estado su apetito? ¿Ha podido comer y tomar líquidos "
    "sin problema?",

    "¿Cómo ha dormido estas noches? ¿El dolor u otra molestia le ha "
    "impedido descansar?",

    "¿Ha podido movilizarse o caminar? ¿Siente mareos, debilidad o "
    "dificultad para ponerse de pie?",
]

_NUM_QUESTIONS: int = len(FOLLOW_UP_QUESTIONS)

# Follow-up question index → symptom domain mapping (must stay aligned
# with ``FOLLOW_UP_QUESTIONS`` order).  Mirrors ``backend.api.calls._QUESTION_DOMAINS``.
_QUESTION_DOMAINS: tuple[str, ...] = (
    "dolor",
    "fiebre",
    "herida",
    "apetito",
    "sueño",
    "movilidad",
)


# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------


@dataclass
class OrchestratorTurn:
    """Result of processing one conversation turn.

    Attributes
    ----------
    agent_message : str
        What the agent says next (always non-empty Spanish text).
    state : State
        The conversation state **after** this turn has been processed.
    citations : list[dict]
        Traceable source citations from RAG (may be empty when no RAG was
        performed or no chunks were retrieved).
    call_ended : bool
        ``True`` when the call has reached ``State.ENDED``.
    requires_response : bool
        ``True`` when the patient is expected to provide the next input.
    question_index : int or ``None``
        Zero-based index of the follow-up question just asked (``None``
        during non-QUESTION states).
    total_questions : int
        Total number of follow-up questions (always ``len(FOLLOW_UP_QUESTIONS)``
        once known, 0 otherwise).
    llm_duration_ms : float or None
        Optional duration of the LLM inference call in milliseconds.
    prompt_tokens : int or None
        Optional number of input tokens consumed by the LLM.
    completion_tokens : int or None
        Optional number of output tokens consumed by the LLM.
    rag_queries : int
        Number of RAG retrieval queries executed during this turn (>= 0).
    escalation : EscalationResult or None
        Escalation classification for this turn (set during QUESTIONS when
        the patient's answer was classified before RAG/LLM).  ``None``
        when no classification was performed.
    """

    agent_message: str
    state: State
    citations: list[dict[str, Any]] = field(default_factory=list)
    call_ended: bool = False
    requires_response: bool = True
    question_index: Optional[int] = None
    total_questions: int = _NUM_QUESTIONS
    escalation: Optional[EscalationResult] = None
    llm_duration_ms: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    rag_queries: int = 0

    def __post_init__(self) -> None:
        if not self.agent_message.strip():
            raise ValueError("agent_message must be non-empty")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class ConversationOrchestrator:
    """Deterministic Spanish text-only orchestrator for a postoperative call.

    The orchestrator owns a ``CallContext`` and drives the state machine
    through the standard dialogue phases:

    1. **IDLE** → ``start_call()`` → **GREETING** (agent greets patient)
    2. **GREETING** → ``process_patient_message()`` → **CONSENT**
       (agent requests consent)
    3. **CONSENT** → → **QUESTIONS** (consent given) or **CLOSING** (refused)
    4. **QUESTIONS** → sequence of structured questions, one per turn.
       After the last question → **CLOSING**.  **RED** classification
       → immediate **ENDED**.  Second consecutive **YELLOW** → **CLOSING**.
    5. **CLOSING** → ``process_patient_message()`` → **ENDED**

    The orchestrator delegates to:
    * ``backend.rag.retrieval.retrieve`` for clinical knowledge retrieval
    * ``backend.llm.adapter.generate_rag_answer`` for validated answer
      generation with traceable citations

    Parameters
    ----------
    patient_context : PatientContext
        Wrapped ``backend.data.models.Patient`` with per-call constraints.
    rag_config : RagConfig or None
        RAG configuration for retrieval.  When ``None``, RAG retrieval is
        skipped and the agent answers with ``insufficient_knowledge``.
    llm_config : LlmConfig or None
        LLM configuration for answer generation.  When ``None``, the LLM is
        not called and the agent uses pre-defined fallback messages.
    """

    def __init__(
        self,
        patient_context: PatientContext,
        rag_config: Optional[RagConfig] = None,
        llm_config: Optional[LlmConfig] = None,
    ) -> None:
        if not isinstance(patient_context, PatientContext):
            raise TypeError(
                f"patient_context must be PatientContext, "
                f"got {type(patient_context).__name__}"
            )
        self._call_context = CallContext(
            call_id=patient_context.call_id,
            patient_context=patient_context,
        )
        self._rag_config = rag_config
        self._llm_config = llm_config

        # Internal per-call tracking
        self._question_index: int = 0
        self._consecutive_yellows: int = 0
        self._llm_doubt_clarification_attempts: dict[int, int] = {}
        """Per-question index count of LLM clarification requests (max 1)."""

    # -- public read-only properties -----------------------------------------

    @property
    def call_context(self) -> CallContext:
        """The immutable ``CallContext`` for this call (history is mutable)."""
        return self._call_context

    @property
    def state(self) -> State:
        """Current position in the conversation state machine."""
        return self._call_context.state

    @property
    def history(self) -> History:
        """Append-only ordered message history for this call."""
        return self._call_context.history

    # -- public API ----------------------------------------------------------

    def start_call(self) -> OrchestratorTurn:
        """Begin the call: transition ``IDLE → GREETING`` and greet the patient.

        Returns
        -------
        OrchestratorTurn
            The agent's greeting message with the new state.

        Raises
        ------
        InvalidTransitionError
            If the current state does not allow ``START_CALL``.
        """
        self._transition(Event.START_CALL)

        pc = self._call_context.patient_context
        greeting = (
            f"¡Buenos días, {pc.patient.nombre_completo}! "
            f"Le hablo del equipo de seguimiento postoperatorio de "
            f"{pc.patient.eps}. ¿Me escucha bien?"
        )

        self._record_agent(greeting)
        return self._make_turn(greeting, requires_response=True)

    def process_patient_message(self, text: str) -> OrchestratorTurn:
        """Process a patient message and advance the conversation.

        This is the main turn-processing method.  It:
        1. Validates and records the patient message.
        2. Determines the appropriate action based on current state.
        3. Performs RAG retrieval and LLM answer generation (when applicable).
        4. Transitions the state machine and returns the agent's response.

        Parameters
        ----------
        text : str
            The patient's message (raw text, will be whitespace-stripped).

        Returns
        -------
        OrchestratorTurn
            The agent's response with the new state and traceable citations.

        Raises
        ------
        ValueError
            If *text* is empty after stripping.
        InvalidTransitionError
            If the current state cannot accept patient input.
        """
        stripped = text.strip()
        if not stripped:
            raise ValueError("Patient message must be non-empty")

        self._record_patient(stripped)

        current_state = self._call_context.state

        if current_state is State.IDLE:
            raise InvalidTransitionError(current_state, Event.START_CALL)
        elif current_state is State.GREETING:
            return self._handle_greeting_response(stripped)
        elif current_state is State.CONSENT:
            return self._handle_consent_response(stripped)
        elif current_state is State.QUESTIONS:
            return self._handle_question_response(stripped)
        elif current_state is State.CLOSING:
            return self._handle_closing_response(stripped)
        elif current_state is State.ENDED:
            return self._make_turn(
                "La llamada ya ha finalizado. Si necesita asistencia, "
                "por favor comuníquese con su médico tratante.",
                requires_response=False,
            )
        else:
            raise ValueError(f"Unknown state: {current_state}")

    # -- state-specific handlers ---------------------------------------------

    def _handle_greeting_response(self, patient_text: str) -> OrchestratorTurn:
        """Respond to patient greeting → request consent."""
        self._transition(Event.GREETING_COMPLETE)

        pc = self._call_context.patient_context
        consent_msg = (
            f"Gracias por confirmar, {pc.patient.nombre_completo}. "
            f"Antes de continuar, necesito informarle que esta llamada "
            f"es parte de su seguimiento postoperatorio por "
            f"{pc.procedimiento}. "
            f"La información que compartamos será utilizada únicamente "
            f"para monitorear su recuperación. "
            f"¿Acepta usted continuar con esta llamada de seguimiento?"
        )

        self._record_agent(consent_msg)
        return self._make_turn(consent_msg, requires_response=True)

    def _handle_consent_response(self, patient_text: str) -> OrchestratorTurn:
        """Handle consent response → QUESTIONS or CLOSING.

        Consent refusal has priority over positive substrings.  Phrases
        such as ``"No acepto continuar"``, ``"No autorizo"``, or
        ``"No deseo continuar"`` are treated as refusal regardless of
        the presence of positive keywords like ``"acepto"``.
        """
        lower = patient_text.lower()

        # --- Absolute refusal markers (priority over everything) ---
        _absolute_refusal = (
            "no acepto", "no autorizo", "no deseo", "no quiero",
            "no estoy de acuerdo",
        )
        for phrase in _absolute_refusal:
            if phrase in lower:
                return self._refuse_consent()

        # --- Standard refusal detection ---
        consent_refused = any(
            word in lower
            for word in ("no", "rechazo", "rechazar", "termine", "colgar")
        ) and not any(
            word in lower
            for word in ("sí", "si", "acepto", "aceptar", "adelante", "claro", "bueno", "dale", "listo")
        )

        if consent_refused:
            return self._refuse_consent()
        else:
            self._transition(Event.CONSENT_GIVEN)
            self._question_index = 0
            return self._ask_next_question()

    def _refuse_consent(self) -> OrchestratorTurn:
        """Handle consent refusal → CLOSING with a respectful message."""
        self._transition(Event.CONSENT_REFUSED)
        closing_msg = (
            "Entiendo, respeto su decisión. Si en el futuro desea "
            "recibir seguimiento, no dude en comunicarse con "
            f"{self._call_context.patient_context.patient.eps}. "
            "Que tenga un buen día."
        )
        self._record_agent(closing_msg)
        return self._make_turn(
            closing_msg,
            requires_response=True,
        )

    def _handle_question_response(self, patient_text: str) -> OrchestratorTurn:
        """Process patient answer to a follow-up question.

        Steps (safety-first):
        1. **Classify** the patient's answer against the symptom domain
           deterministically.  RED detection runs first so that numeric
           and domain-specific RED signals embedded in question-shaped
           text (e.g. ``"es normal que me duela un 9?"``) are never
           bypassed by the doubt-intent gate.
        2. **RED** → short-circuit: no RAG/LLM, no doubt gate, urgent
           safety message, transition directly to ENDED,
           ``call_ended=True``.
        3. **Doubt-intent gate** (only for non-RED answers): check whether
           the patient's input is a clinical question/doubt rather than a
           symptom answer.
           a. Deterministic check for explicit markers (question marks,
              interrogative words).
           b. LLM confirmation when available (``llm_confirm_doubt()``).
           c. On confirmed doubt: run RAG inline, answer with citations,
              repeat the same follow-up question (do NOT advance index).
           d. Safe fallback preserves explicit doubt markers when LLM fails.
           e. Doubt-intent answers must NOT trigger YELLOW accumulation or
              escalation — they are unanswered questions.
        4. **LLM second-approval** for every non-RED, non-doubt answer:
           a. Call ``llm_second_approval()`` — it may confirm, upgrade severity,
              request RAG for a doubt, or request one clarification.
           b. Failures/timeouts/invalid/low-confidence output fall back to
              deterministic classification.
        5. Process the approval result:
           - **confirm** → proceed normally with final severity.
           - **escalate** → apply upgraded severity.
           - **request_clarification** → stay on same question, ask one
             clarification question (max 1 per question).
           - **request_rag** → run RAG retrieval in QUESTIONS, get clinical
             context, then continue/finish normally.
        6. Final question (index 5, movilidad) proceeds to CLOSING after
           answer/RAG; clarification stays on question 6.
        """
        answered_idx = self._question_index
        domain = _QUESTION_DOMAINS[answered_idx]
        pc = self._call_context.patient_context

        # --- Safety-first: classify before doubt-gate ---
        # RED detection must always run first, even for question-shaped
        # text (e.g. "es normal que me duela un 9?").  The deterministic
        # classifier catches numeric RED signals and domain-specific RED
        # keywords regardless of interrogative phrasing.
        try:
            classification = _decision_classify(
                patient_text=patient_text,
                domain=domain,
                dia_postop=pc.dia_postop,
                procedimiento=pc.procedimiento,
            )
        except Exception:
            # Classifier failed — treat conservatively as YELLOW
            classification = EscalationResult(
                severity=Severity.YELLOW,
                should_escalate=False,
                reason="No se pudo clasificar la respuesta del paciente.",
                next_action="Continuar monitoreo con precaución.",
                domain=domain,
                source="incomplete",
            )

        severity = classification.severity

        # --- RED: short-circuit immediately → ENDED (no doubt gate, no LLM) ---
        if severity is Severity.RED:
            self._consecutive_yellows = 0
            self._transition(Event.EMERGENCY_TERMINATE)
            urgent_msg = self._build_red_message(classification)
            self._record_agent(urgent_msg)
            self._question_index += 1
            return self._make_turn(
                urgent_msg,
                requires_response=False,
                call_ended=True,
                question_index=answered_idx,
                escalation=classification,
            )

        # --- Doubt-intent gate: only for non-RED answers ---
        # Clinical questions are not symptom reports — they should trigger
        # RAG and repeat the same question, never accumulate YELLOW or
        # trigger escalation.  RED was already handled above; only
        # non-RED answers reach this gate.
        doubt_result = self._check_doubt_intent(
            patient_text, answered_idx, domain, pc,
        )
        if doubt_result is not None:
            return doubt_result

        # --- LLM second-approval for every non-RED, non-doubt answer ---
        final_classification = classification
        approval_result: LlmApprovalResult | None = None
        approval_meta: dict[str, Any] = {}

        if self._llm_config is not None:
            try:
                approval_result = llm_second_approval(
                    patient_text=patient_text,
                    domain=domain,
                    deterministic_classification=classification,
                    dia_postop=pc.dia_postop,
                    procedimiento=pc.procedimiento,
                    config=self._llm_config,
                )
            except Exception:
                logger.exception(
                    "LLM second-approval crashed — falling back to deterministic"
                )
                approval_result = None

            if approval_result is not None and approval_result.llm_used:
                approval_meta["llm_duration_ms"] = approval_result.llm_duration_ms
                approval_meta["prompt_tokens"] = approval_result.prompt_tokens
                approval_meta["completion_tokens"] = approval_result.completion_tokens
        else:
            # No LLM config — skip approval, use deterministic only
            approval_result = None

        # --- Determine effective classification after approval ---
        if approval_result is not None and approval_result.llm_used:
            final_severity = approval_result.severity
            final_classification = EscalationResult(
                severity=final_severity,
                should_escalate=approval_result.should_escalate,
                reason=approval_result.reason,
                next_action=approval_result.next_action,
                domain=domain,
                source=classification.source,
            )
        else:
            # Fallback: use deterministic classification as-is
            final_severity = severity

        # --- Process by action type ---

        # --- RED (upgraded from approval) → ENDED ---
        if final_severity is Severity.RED:
            self._consecutive_yellows = 0
            self._transition(Event.EMERGENCY_TERMINATE)
            urgent_msg = self._build_red_message(final_classification)
            self._record_agent(urgent_msg)
            self._question_index += 1
            return self._make_turn(
                urgent_msg,
                requires_response=False,
                call_ended=True,
                question_index=answered_idx,
                escalation=final_classification,
                llm_meta=approval_meta if approval_meta else None,
            )

        # --- Request clarification → stay on same question ---
        if (
            approval_result is not None
            and approval_result.llm_used
            and approval_result.action == "request_clarification"
        ):
            # Limit to 1 clarification per question
            attempts = self._llm_doubt_clarification_attempts.get(answered_idx, 0)
            if attempts >= 1:
                logger.info(
                    "Clarification limit reached for question %d — "
                    "proceeding with deterministic YELLOW.",
                    answered_idx,
                )
            else:
                self._llm_doubt_clarification_attempts[answered_idx] = attempts + 1
                clarification_q = approval_result.clarification_question
                full_msg = (
                    f"{clarification_q}"
                )
                self._record_agent(full_msg)
                # Do NOT advance _question_index — patient answers same domain again
                return self._make_turn(
                    full_msg,
                    requires_response=True,
                    question_index=answered_idx,
                    escalation=final_classification,
                    llm_meta=approval_meta if approval_meta else None,
                )

        # --- Request RAG for doubt → run RAG, then continue ---
        if (
            approval_result is not None
            and approval_result.llm_used
            and approval_result.action == "request_rag"
        ):
            rag_response, rag_citations, rag_meta = self._run_doubt_rag(
                patient_text=patient_text,
                rag_query=approval_result.rag_query,
                domain=domain,
            )
            # Merge approval LLM metrics with RAG response metrics
            merged_meta = dict(approval_meta)
            if rag_meta.get("rag_queries", 0) > 0:
                merged_meta["rag_queries"] = rag_meta.get("rag_queries", 0)
            if rag_meta.get("llm_duration_ms") is not None:
                merged_meta["llm_duration_ms"] = rag_meta.get("llm_duration_ms")
            if rag_meta.get("prompt_tokens") is not None:
                merged_meta["prompt_tokens"] = rag_meta.get("prompt_tokens")
            if rag_meta.get("completion_tokens") is not None:
                merged_meta["completion_tokens"] = rag_meta.get("completion_tokens")

            # Advance the question index
            self._question_index += 1

            # Reset/accumulate yellows
            self._consecutive_yellows = 0  # RAG resolved the doubt

            # After RAG: if this was the last question → CLOSING
            if self._question_index >= _NUM_QUESTIONS:
                # Build closing message with RAG response as prefix
                closing_prefix = rag_response if rag_response else ""
                return self._close_questions(
                    final_message=closing_prefix if closing_prefix else None,
                    citations=rag_citations,
                    question_index=self._question_index,
                    llm_meta=merged_meta if merged_meta else None,
                    escalation=final_classification,
                )
            else:
                return self._ask_next_question(
                    after_message=rag_response if rag_response else None,
                    citations=rag_citations,
                    question_index=self._question_index,
                    llm_meta=merged_meta if merged_meta else None,
                    escalation=final_classification,
                )

        # --- Normal flow (confirm / escalate without RED) ---
        # Reset clarification attempts for this question (successfully answered)
        self._llm_doubt_clarification_attempts.pop(answered_idx, None)

        if final_severity is Severity.YELLOW:
            self._consecutive_yellows += 1

            if self._consecutive_yellows >= 2:
                # Two consecutive YELLOW → escalate with should_escalate=True
                escalated_classification = EscalationResult(
                    severity=final_classification.severity,
                    should_escalate=True,
                    reason=final_classification.reason,
                    next_action=final_classification.next_action,
                    domain=final_classification.domain,
                    source=final_classification.source,
                )
                self._transition(Event.ESCALATION_TRIGGER)
                escalate_msg = self._build_consecutive_yellow_message(final_classification)
                self._record_agent(escalate_msg)
                self._question_index += 1
                return self._make_turn(
                    escalate_msg,
                    requires_response=True,
                    question_index=answered_idx,
                    escalation=escalated_classification,
                    llm_meta=approval_meta if approval_meta else None,
                )
            else:
                # First YELLOW: deterministic ack
                ack = self._build_yellow_ack(domain, final_classification)
                self._question_index += 1
                if self._question_index >= _NUM_QUESTIONS:
                    return self._close_questions(
                        ack,
                        question_index=self._question_index,
                        escalation=final_classification,
                        llm_meta=approval_meta if approval_meta else None,
                    )
                else:
                    return self._ask_next_question(
                        after_message=ack,
                        question_index=self._question_index,
                        escalation=final_classification,
                        llm_meta=approval_meta if approval_meta else None,
                    )

        # --- GREEN: deterministic ack ---
        self._consecutive_yellows = 0
        ack = self._build_green_ack(domain)
        self._question_index += 1
        if self._question_index >= _NUM_QUESTIONS:
            return self._close_questions(
                ack,
                question_index=self._question_index,
                escalation=final_classification,
                llm_meta=approval_meta if approval_meta else None,
            )
        else:
            return self._ask_next_question(
                after_message=ack,
                question_index=self._question_index,
                escalation=final_classification,
                llm_meta=approval_meta if approval_meta else None,
            )

    def _handle_closing_response(self, patient_text: str) -> OrchestratorTurn:
        """Process patient message during CLOSING.

        If the patient asks a clinical question, use RAG+LLM to answer,
        include citations, and remain in CLOSING.
        Otherwise, end the call.
        """
        if self._is_clinical_question(patient_text):
            return self._handle_closing_question(patient_text)
        else:
            return self._end_call()

    def _is_closing_negation(self, normalized: str) -> bool:
        """Detect phrases that express having no questions or doubts.

        In the CLOSING state the agent asks ``"¿Tiene alguna pregunta
        antes de finalizar la llamada?"``.  Patients may respond with
        negation phrases such as ``"no tengo preguntas"``,
        ``"sin dudas"``, or ``"ninguna pregunta"``.  These must end the
        call, not be treated as clinical questions.

        Parameters
        ----------
        normalized : str
            Diacritic-stripped, lower-cased Spanish text (already
            produced by ``_normalize_spanish_text``).

        Returns
        -------
        bool
            ``True`` when the text expresses "no questions/doubts".
        """
        # -- Direct negation templates ------------------------------------
        # Order matters: longer patterns first so "ninguna pregunta"
        # matches before the bare "pregunta" keyword (handled by the
        # caller, not here).
        negation_templates = (
            # "no tengo preguntas", "no tengo dudas"
            "no tengo preguntas", "no tengo dudas",
            # "no, no tengo preguntas"
            "no tengo ninguna pregunta", "no tengo ninguna duda",
            # "sin preguntas", "sin dudas"
            "sin preguntas", "sin dudas",
            # "sin ninguna pregunta", "sin ninguna duda"
            "sin ninguna pregunta", "sin ninguna duda",
            # "ninguna pregunta", "ninguna duda"
            "ninguna pregunta", "ninguna duda",
            # "no tengo inquietudes"
            "no tengo inquietudes", "sin inquietudes",
            "ninguna inquietud",
            # "no tengo consultas"
            "no tengo consultas", "sin consultas",
            # "no gracias" alone (common polite ending)
            "no, nada mas", "no nada mas",
            "nada mas", "nada, gracias", "nada gracias",
            "no, gracias", "no gracias",
            "no senor", "no senora", "no doctor", "no doctora",
            "todo claro", "todo bien",
            "estoy bien", "estamos bien",
            "asi esta bien", "asi estamos bien",
            "no necesito nada",
            "por ahora no", "por el momento no",
        )
        for tpl in negation_templates:
            if tpl in normalized:
                return True

        # -- "no ..." + keyword within the same short span ---------------
        # Handles variations like "no, creo que no tengo preguntas"
        negated_keyword_patterns = (
            ("no", "pregunta"),
            ("no", "preguntas"),
            ("no", "duda"),
            ("no", "dudas"),
            ("no", "inquietud"),
            ("no", "inquietudes"),
            ("no", "consulta"),
            ("no", "consultas"),
        )
        for neg_word, kw in negated_keyword_patterns:
            neg_idx = normalized.find(neg_word)
            if neg_idx < 0:
                continue
            kw_idx = normalized.find(kw, neg_idx)
            if kw_idx < 0:
                continue
            # Require the keyword to appear within 30 chars of the
            # negation word so we don't catch completely unrelated
            # uses of "no" in long inputs.
            if kw_idx - neg_idx <= 30:
                # Also check there's no "tengo una" or "tengo un"
                # immediately before the keyword — that would be a
                # positive question, not a closing negation.
                prefix = normalized[max(0, kw_idx - 20):kw_idx]
                if "tengo una " in prefix or "tengo un " in prefix:
                    continue
                return True

        return False

    def _is_clinical_question(self, text: str) -> bool:
        """Detect whether the patient is asking a clinical question.

        Handles STT output that lacks accents and punctuation by
        normalising diacritics before matching.  Uses compound patterns
        (e.g. ``"que cuidados"`` rather than bare ``"que "``) to avoid
        false positives from relative-pronoun uses like
        ``"la herida que tengo"``.

        Negation phrases such as ``"no tengo preguntas"``, ``"sin dudas"``
        are detected **before** compound question patterns (``que``,
        ``como``, ``cual``, etc.) so they correctly end the call instead
        of being treated as clinical questions.  Only explicit question
        marks and ``"por que"`` receive higher priority than negation
        detection.
        """
        lowered = text.lower().strip()

        # -- Direct question marks (written/typed input) ------------------
        if "?" in lowered or "¿" in lowered:
            return True

        # -- Normalise away diacritics for STT-style matching ------------
        normalized = _normalize_spanish_text(text)

        # -- Explicit question words (accented forms still work in
        #    normalised input because the normaliser strips diacritics) --
        explicit_q_words = ("por que",)
        for w in explicit_q_words:
            if w in normalized:
                return True

        # -- Closing negation: detect phrases that express "no questions"
        #    BEFORE any compound question pattern matching.  Compound
        #    patterns such as ``"o que"`` (which appears as a substring
        #    in ``"creo que"``) and ``"y que"`` must not trump an
        #    explicit negation like ``"no tengo preguntas"``.
        #    Examples: "no tengo preguntas", "sin dudas", "ninguna duda",
        #    "no tengo ninguna pregunta", "no, no tengo preguntas".
        #    "creo que no tengo preguntas".
        if self._is_closing_negation(normalized):
            return False

        # -- Compound "que" patterns (interrogative, avoids relative
        #    pronoun false positives like "la herida que tengo") ---------
        que_patterns = (
            # clinical-care questions (catches "que cuidados debo seguir")
            "que cuidados", "que cuidos", "que cuidado",
            "que debo", "que puedo", "que hago", "que hacer",
            "que significa", "que quiere decir", "que pasa",
            "que paso", "que pasaria", "que sucede", "que ocurre",
            # medication / symptom questions
            "que medicamento", "que medicina", "que remedio",
            "que pastilla", "que analgesico",
            "que sintoma", "que malestar", "que molestia",
            "que enfermedad", "que infeccion",
            # treatment / recovery questions
            "que tratamiento", "que ejercicio", "que actividad",
            "que dieta", "que alimentacion", "que comida", "que comer",
            "que bebida", "que tomar", "que liquido",
            "que recomienda", "que aconseja", "que sugiere",
            "que tipo", "que clase", "que examenes", "que examen",
            "que es ", "que son ", "que esta ",
            # urgency indicators
            "que urgencia", "que emergencia", "que gravedad",
            # common interrogative anchors
            "que tengo que", "que debo de", "que puedo hacer",
            "y que", "o que",
        )
        for p in que_patterns:
            if p in normalized:
                return True

        # -- Compound "como" patterns (interrogative; "como" alone can
        #    mean "I eat", so we only match it with following context) --
        como_patterns = (
            "como debo", "como deberia", "como puedo", "como limpio",
            "como cuidarme", "como esta", "como estan",
            "como me", "como le", "como se", "como hago",
            "como va", "como van", "como es", "como son",
            "como saber", "como saber si", "como reconozco",
            "como identificar", "como trato", "como tratar",
            "como cuidar", "como curo", "como curar",
            "como aliviar", "como manejar", "como controlar",
            "como evitar", "como prevenir",
            "como proceder", "como seguir", "como continuar",
            "como queda", "como quedo",
            "como funciona",
        )
        for p in como_patterns:
            if p in normalized:
                return True

        # -- Other interrogatives (non-accented forms for STT) -----------
        other_q_words = (
            "cual ", "cuales ", "cuanta ", "cuantas ", "cuanto ",
            "cuantos ", "quien ", "quienes ", "adonde ", "adónde ",
        )
        for w in other_q_words:
            if w in normalized:
                return True

        # -- Common clinical-question sub-phrases ------------------------
        clinical_q_phrases = (
            "cuanto tiempo", "cuanto dura", "cuanto tarda",
            "cada cuanto", "hasta cuando", "desde cuando",
            "cuantos dias", "cuantas veces", "cuantas horas",
            "cuando puedo", "cuando debo", "a donde", "a donde debo",
            "me toca ir",
        )
        for p in clinical_q_phrases:
            if p in normalized:
                return True

        # -- Explicit inquiry / "I have a question" patterns -------------
        inquiry_patterns = (
            "tengo una duda", "tengo una pregunta",
            "quisiera saber", "quisiera preguntar",
            "necesito saber", "quiero saber", "quiero preguntar",
            "me puede decir", "me podría decir",
            "me puede explicar", "me podría explicar",
            "me puede ayudar", "me podría ayudar",
            "me puede orientar", "me podría orientar",
            "me puede aclarar", "me podría aclarar",
            "me puede contar", "me podría contar",
            "puede decirme", "podria decirme",
            "puede explicarme", "podria explicarme",
            "me dice", "me explica", "me cuenta",
            "me ayudas", "me ayudas con",
            # imperative inquiry forms
            "explíqueme", "expliqueme", "cuénteme", "cuenteme",
            "digame", "diga me", "aconseje", "aconsejeme",
            "recomiende", "recomiendeme", "explique",
            "indique", "indiqueme",
            "orienteme", "informeme", "aclarame", "acláreme",
            "expliqueme",
        )
        for p in inquiry_patterns:
            if p in normalized:
                return True

        # -- General question/inquiry keywords (broad but low-risk in
        #    CLOSING context where patient is explicitly invited to ask).
        #    Negation was already checked above — anything reaching here
        #    with these keywords is treated as a real question.  ---------
        general_q_keywords = (
            "pregunta", "duda", "inquietud", "consultar",
            "consulta",
        )
        for kw in general_q_keywords:
            if kw in normalized:
                return True

        # -- Colombian-regionalism inquiry openers -----------------------
        colombian_inquiries = (
            "oiga doctor", "oiga doc", "oiga", "vea pues",
            "vea doctor",
        )
        for p in colombian_inquiries:
            if p in normalized:
                return True

        # -- "Is it normal?" patterns -----------------------------------
        normal_question_patterns = (
            "es normal", "sera normal", "es grave",
            "sera grave", "es peligroso", "sera peligroso",
            "sera malo", "es malo",
            "puedo comer", "puedo tomar", "puedo hacer",
            "se puede", "se pueden",
        )
        for p in normal_question_patterns:
            if p in normalized:
                return True

        return False

    def _handle_closing_question(self, patient_text: str) -> OrchestratorTurn:
        """Answer a clinical question during CLOSING with RAG+LLM.

        Remains in CLOSING so the patient can ask follow-up questions.
        The patient message has already been recorded by
        ``process_patient_message``.
        """
        response_msg, citations, meta = self._generate_closing_rag_response(
            patient_text=patient_text,
        )

        self._record_agent(response_msg)
        return self._make_turn(
            response_msg,
            citations=citations,
            requires_response=True,
            question_index=None,
            llm_meta=meta,
        )

    def _generate_closing_rag_response(
        self,
        patient_text: str,
    ) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
        """Retrieve and generate a RAG+LLM answer for a CLOSING clinical question.

        Returns a ``(response_text, citations, metadata)`` tuple.
        """
        meta: dict[str, Any] = {
            "rag_queries": 0,
            "llm_duration_ms": None,
            "prompt_tokens": None,
            "completion_tokens": None,
        }

        pc = self._call_context.patient_context
        query = (
            f"Procedimiento: {pc.procedimiento}. "
            f"Día postoperatorio: {pc.dia_postop}. "
            f"Pregunta del paciente: {patient_text}"
        )

        # --- Retrieve ---
        retrieval_result = self._retrieve(patient_text, query)
        if retrieval_result is not None:
            meta["rag_queries"] = 1

        if not retrieval_result or not retrieval_result.has_results or not retrieval_result.sufficient:
            return (
                "Gracias por su pregunta. No cuento con información "
                "suficiente para responderla en este momento. Le "
                "recomiendo consultar con su médico tratante.",
                [],
                meta,
            )

        # --- Generate ---
        if self._llm_config is None:
            return (
                "Gracias por su pregunta. Le recuerdo que cualquier "
                "inquietud debe ser consultada con su médico tratante.",
                [],
                meta,
            )

        context_chunks: list[dict[str, Any]] = [
            {
                "chunk_id": c.chunk_id,
                "document_id": c.document_id,
                "source_filename": c.source_filename,
                "page_number": c.page_number,
                "text": c.text,
                "similarity": c.similarity,
            }
            for c in retrieval_result.chunks
        ]

        try:
            answer: RagAnswer = generate_rag_answer(
                query=query,
                context_chunks=context_chunks,
                config=self._llm_config,
                debug=False,
            )
        except Exception:
            logger.exception("LLM call failed during closing question")
            return (
                "Gracias por su pregunta. No puedo procesar su consulta "
                "en este momento. Por favor, consulte a su médico "
                "tratante si tiene dudas.",
                [],
                meta,
            )

        meta["llm_duration_ms"] = answer.llm_duration_ms
        meta["prompt_tokens"] = answer.prompt_tokens
        meta["completion_tokens"] = answer.completion_tokens

        if answer.insufficient_knowledge:
            return (
                "Gracias por su pregunta. Basado en la información "
                "disponible, no tengo detalles adicionales. Le sugiero "
                "consultar a su médico si tiene inquietudes.",
                [],
                meta,
            )

        response = f"{answer.answer}"

        if answer.citations:
            cited = ", ".join(
                f"{c.source_filename} (p. {c.page_number})"
                for c in answer.citations
            )
            response += f"\n\n(Fuentes consultadas: {cited})"

        citations_list: list[dict[str, Any]] = [
            {
                "chunk_id": c.chunk_id,
                "document_id": c.document_id,
                "source_filename": c.source_filename,
                "page_number": c.page_number,
            }
            for c in answer.citations
        ]

        return response, citations_list, meta

    # -- helper: ask next question -------------------------------------------

    def _ask_next_question(
        self,
        after_message: Optional[str] = None,
        citations: Optional[list[dict[str, Any]]] = None,
        llm_meta: Optional[dict[str, Any]] = None,
        question_index: Optional[int] = None,
        escalation: Optional[EscalationResult] = None,
    ) -> OrchestratorTurn:
        """Ask the current follow-up question.

        If *after_message* is provided, it is recorded as the agent's
        response to the previous answer, then the next question is appended.
        If *citations* are provided, they are included in the turn.
        If *llm_meta* is provided, its LLM metrics (``llm_duration_ms``,
        ``prompt_tokens``, ``completion_tokens``, ``rag_queries``) are
        propagated into the turn.
        If *escalation* is provided, it is propagated into the turn.
        """
        qidx = question_index if question_index is not None else self._question_index
        if qidx >= _NUM_QUESTIONS:
            return self._close_questions(
                after_message,
                citations=citations,
                question_index=qidx,
                llm_meta=llm_meta,
                escalation=escalation,
            )

        question = FOLLOW_UP_QUESTIONS[qidx]

        if after_message:
            # Prepend the clinical response, then ask the next question
            full = f"{after_message}\n\n{question}"
        else:
            full = question

        self._record_agent(full)
        return self._make_turn(
            full,
            citations=citations or [],
            question_index=qidx,
            requires_response=True,
            llm_meta=llm_meta,
            escalation=escalation,
        )

    # -- helper: close questions phase ---------------------------------------

    def _close_questions(
        self,
        final_message: Optional[str] = None,
        citations: Optional[list[dict[str, Any]]] = None,
        question_index: Optional[int] = None,
        llm_meta: Optional[dict[str, Any]] = None,
        escalation: Optional[EscalationResult] = None,
    ) -> OrchestratorTurn:
        """Transition QUESTIONS → CLOSING and produce closing message.

        Parameters
        ----------
        question_index : int or None
            When provided (after the last follow-up question was answered),
            this equals ``_NUM_QUESTIONS`` so the escalation layer can
            infer the ``movilidad`` domain for the final answer.
        llm_meta : dict or None
            Optional LLM metrics from the last question's clinical
            response.  Propagates ``rag_queries``, ``llm_duration_ms``,
            ``prompt_tokens``, and ``completion_tokens`` into the turn.
        escalation : EscalationResult or None
            Optional escalation classification for the turn.
        """
        self._transition(Event.QUESTIONS_COMPLETE)

        pc = self._call_context.patient_context
        closing = (
            f"Hemos terminado las preguntas de seguimiento, "
            f"{pc.patient.nombre_completo}. Gracias por su tiempo y "
            f"por compartir esta información. Recuerde que cualquier "
            f"síntoma nuevo o que empeore debe ser consultado con su "
            f"médico tratante. ¿Tiene alguna pregunta antes de "
            f"finalizar la llamada?"
        )

        if final_message:
            full = f"{final_message}\n\n{closing}"
        else:
            full = closing

        self._record_agent(full)
        return self._make_turn(
            full,
            citations=citations or [],
            question_index=question_index,
            requires_response=True,
            llm_meta=llm_meta,
            escalation=escalation,
        )

    # -- doubt-intent detection and handling ----------------------------------

    def _check_doubt_intent(
        self,
        patient_text: str,
        answered_idx: int,
        domain: str,
        pc: PatientContext,
    ) -> OrchestratorTurn | None:
        """Check whether the patient input is a clinical doubt rather than
        an answer to the follow-up question.

        Returns an ``OrchestratorTurn`` when doubt is confirmed (with RAG
        answer and same question repeated), or ``None`` when the input
        should proceed to normal escalation classification.

        Two-stage detection:
        1. Deterministic check for explicit markers (question marks,
           interrogative words, ``"es normal"``, ``"puedo"``, etc.).
        2. LLM confirmation when available — the LLM can overrule the
           deterministic check in either direction.

        On LLM failure the deterministic result is preserved — explicit
        doubts stay as doubts (safe fallback).
        """
        has_explicit = _has_explicit_doubt_markers(patient_text)

        if self._llm_config is not None:
            try:
                doubt_check = llm_confirm_doubt(
                    patient_text=patient_text,
                    domain=domain,
                    follow_up_question=FOLLOW_UP_QUESTIONS[answered_idx],
                    dia_postop=pc.dia_postop,
                    procedimiento=pc.procedimiento,
                    config=self._llm_config,
                )
            except Exception:
                logger.exception(
                    "Doubt-check LLM crashed — falling back to deterministic"
                )
                if has_explicit:
                    return self._handle_confirmed_doubt(
                        patient_text, answered_idx, domain,
                        rag_query=_build_doubt_rag_query(patient_text),
                    )
                return None

            if doubt_check.is_doubt:
                return self._handle_confirmed_doubt(
                    patient_text, answered_idx, domain,
                    rag_query=doubt_check.rag_query,
                    clarification_text=doubt_check.clarification_text,
                    doubt_meta={
                        "llm_duration_ms": doubt_check.llm_duration_ms,
                        "prompt_tokens": doubt_check.prompt_tokens,
                        "completion_tokens": doubt_check.completion_tokens,
                    },
                )
            return None

        # No LLM config — fall back to deterministic markers only
        if has_explicit:
            return self._handle_confirmed_doubt(
                patient_text, answered_idx, domain,
                rag_query=_build_doubt_rag_query(patient_text),
            )
        return None

    def _handle_confirmed_doubt(
        self,
        patient_text: str,
        answered_idx: int,
        domain: str,
        rag_query: str,
        clarification_text: str = "",
        doubt_meta: dict[str, Any] | None = None,
    ) -> OrchestratorTurn:
        """Handle a confirmed clinical doubt during QUESTIONS.

        Runs RAG retrieval with the provided query, returns the RAG answer
        with traceable citations, then repeats the same follow-up question.
        Does **not** advance the question index — the patient still needs
        to answer the original follow-up question.

        Doubt turns carry an ``EscalationResult`` with ``source="doubt"``
        and ``should_escalate=False`` so they never persist as alerts.
        Consecutive-YELLOW accumulation is reset because a doubt is not
        a symptom report.
        """
        # Run RAG for the doubt
        rag_response, rag_citations, rag_meta = self._run_doubt_rag(
            patient_text=patient_text,
            rag_query=rag_query,
            domain=domain,
        )

        # Merge LLM doubt metrics with RAG response metrics
        merged_meta: dict[str, Any] = dict(doubt_meta or {})
        if rag_meta.get("rag_queries", 0) > 0:
            merged_meta["rag_queries"] = rag_meta.get("rag_queries", 0)
        if rag_meta.get("llm_duration_ms") is not None:
            merged_meta["llm_duration_ms"] = rag_meta.get("llm_duration_ms")
        if rag_meta.get("prompt_tokens") is not None:
            merged_meta["prompt_tokens"] = rag_meta.get("prompt_tokens")
        if rag_meta.get("completion_tokens") is not None:
            merged_meta["completion_tokens"] = rag_meta.get("completion_tokens")

        # Build response: clarification + RAG answer + repeat question
        parts: list[str] = []
        if clarification_text:
            parts.append(clarification_text)
        if rag_response:
            parts.append(rag_response)
        question = FOLLOW_UP_QUESTIONS[answered_idx]
        parts.append(question)

        full_msg = "\n\n".join(parts)
        self._record_agent(full_msg)

        # Doubt turns get a non-conclusive escalation marker.
        # source="doubt" ensures the API layer does not reclassify.
        doubt_escalation = EscalationResult(
            severity=Severity.YELLOW,
            should_escalate=False,
            reason=(
                "El paciente tiene una duda clínica — "
                "no es un reporte de síntoma."
            ),
            next_action="Responder la duda con RAG y repetir la pregunta.",
            domain=domain,
            source="doubt",
        )

        # Reset consecutive yellows — doubt is not a symptom report
        self._consecutive_yellows = 0

        return self._make_turn(
            full_msg,
            citations=rag_citations,
            question_index=answered_idx,
            requires_response=True,
            escalation=doubt_escalation,
            llm_meta=merged_meta if merged_meta else None,
        )

    # -- safety-first message builders ---------------------------------------

    def _build_green_ack(self, domain: str) -> str:
        """Build a deterministic Spanish acknowledgment for GREEN answers."""
        ack_templates: dict[str, str] = {
            "dolor": (
                "Gracias por compartir esa información. Me alegra saber "
                "que su nivel de dolor está controlado."
            ),
            "fiebre": (
                "Gracias. Es una buena señal que no presente fiebre ni "
                "escalofríos."
            ),
            "herida": (
                "Gracias. Qué bueno que su herida quirúrgica esté "
                "cicatrizando sin complicaciones."
            ),
            "apetito": (
                "Gracias por informarme. Me alegra saber que su apetito "
                "y tolerancia a los alimentos están bien."
            ),
            "sueño": (
                "Gracias. Es positivo que esté descansando y su sueño "
                "sea reparador."
            ),
            "movilidad": (
                "Gracias. Es excelente que pueda movilizarse sin "
                "dificultad y se sienta con fuerzas."
            ),
        }
        return ack_templates.get(
            domain,
            "Gracias por compartir esta información. Continúe así.",
        )

    def _build_yellow_ack(
        self, domain: str, classification: EscalationResult
    ) -> str:
        """Build a deterministic acknowledgment for a first-YELLOW answer."""
        base = (
            "Gracias por compartir esta información. Tendré en cuenta lo "
            "que me comenta y le sugiero que esté atento a cualquier "
            "cambio."
        )
        return base

    def _build_red_message(self, classification: EscalationResult) -> str:
        """Build a clear Spanish urgent safety message for RED escalations.

        The call ends immediately after this message, so there is no
        expectation of a patient response.
        """
        pc = self._call_context.patient_context
        return (
            f"{pc.patient.nombre_completo}, he identificado información "
            f"que requiere atención médica urgente. "
            f"{classification.reason} "
            f"Por favor, comuníquese de inmediato con su médico tratante "
            f"en {pc.patient.eps} o acuda al servicio de urgencias más "
            f"cercano. No espere a que los síntomas empeoren. "
            f"Esta llamada finaliza aquí. Cuídese."
        )

    def _build_consecutive_yellow_message(
        self, classification: EscalationResult
    ) -> str:
        """Build an escalation message for two consecutive YELLOW results."""
        pc = self._call_context.patient_context
        return (
            f"{pc.patient.nombre_completo}, he notado que varios de sus "
            f"síntomas requieren atención. "
            f"{classification.reason} "
            f"Le recomiendo comunicarse con su médico tratante en "
            f"{pc.patient.eps} lo antes posible para una evaluación "
            f"más detallada. No espere a que los síntomas empeoren. "
            f"¿Tiene alguna pregunta antes de finalizar?"
        )

    # -- helper: end call ----------------------------------------------------

    def _end_call(self) -> OrchestratorTurn:
        """Complete the CLOSING → ENDED transition."""
        self._transition(Event.CLOSING_COMPLETE)

        pc = self._call_context.patient_context
        farewell = (
            f"La llamada ha finalizado. {pc.patient.nombre_completo}, "
            f"le recuerdo que su procedimiento de {pc.procedimiento} "
            f"requiere seguimiento continuo. "
            f"Cualquier duda, comuníquese con {pc.patient.eps}. "
            f"¡Que se recupere pronto!"
        )

        self._record_agent(farewell)
        return self._make_turn(
            farewell,
            call_ended=True,
            requires_response=False,
        )

    # -- RAG + LLM -----------------------------------------------------------

    def _generate_clinical_response(
        self,
        patient_text: str,
        question_index: int,
    ) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
        """Retrieve clinical knowledge and generate a validated response.

        Returns a ``(response_text, citations, metadata)`` tuple where
        *citations* is a list of dicts with ``chunk_id``, ``document_id``,
        ``source_filename``, and ``page_number`` and *metadata* is a dict
        with optional keys ``rag_queries``, ``llm_duration_ms``,
        ``prompt_tokens``, and ``completion_tokens``.

        Falls back to safe messages when RAG is unavailable or returns no
        results.
        """
        question = FOLLOW_UP_QUESTIONS[question_index]

        # Default metadata
        meta: dict[str, Any] = {
            "rag_queries": 0,
            "llm_duration_ms": None,
            "prompt_tokens": None,
            "completion_tokens": None,
        }

        # --- Retrieve ---
        retrieval_result = self._retrieve(patient_text, question)
        if retrieval_result is not None:
            meta["rag_queries"] = 1

        if not retrieval_result or not retrieval_result.has_results or not retrieval_result.sufficient:
            return (
                "Gracias por compartir esta información. No cuento con "
                "material de consulta suficiente para darle una "
                "orientación más detallada sobre este punto en este "
                "momento. Si tiene dudas, consulte a su médico tratante.",
                [],
                meta,
            )

        # --- Generate ---
        if self._llm_config is None:
            return (
                "Gracias por compartir esta información. Le recuerdo "
                "que cualquier síntoma que le preocupe debe ser "
                "consultado con su médico tratante.",
                [],
                meta,
            )

        query = (
            f"Paciente postoperatorio de {self._call_context.patient_context.procedimiento}, "
            f"día {self._call_context.patient_context.dia_postop} postoperatorio. "
            f"Pregunta del seguimiento: {question}. "
            f"El paciente respondió: {patient_text}"
        )

        context_chunks: list[dict[str, Any]] = [
            {
                "chunk_id": c.chunk_id,
                "document_id": c.document_id,
                "source_filename": c.source_filename,
                "page_number": c.page_number,
                "text": c.text,
                "similarity": c.similarity,
            }
            for c in retrieval_result.chunks
        ]

        try:
            answer: RagAnswer = generate_rag_answer(
                query=query,
                context_chunks=context_chunks,
                config=self._llm_config,
                debug=False,
            )
        except Exception:
            logger.exception("LLM call failed in orchestrator")
            return (
                "Gracias por compartir esta información. No puedo "
                "procesar su respuesta en este momento. Por favor, "
                "consulte a su médico tratante si tiene dudas.",
                [],
                meta,
            )

        # Propagate LLM metrics
        meta["llm_duration_ms"] = answer.llm_duration_ms
        meta["prompt_tokens"] = answer.prompt_tokens
        meta["completion_tokens"] = answer.completion_tokens

        if answer.insufficient_knowledge:
            return (
                "Gracias. Basado en la información disponible, no "
                "tengo detalles adicionales sobre este aspecto de su "
                "recuperación. Le sugiero consultar a su médico si "
                "tiene inquietudes.",
                [],
                meta,
            )

        # Build response with citations
        response = f"{answer.answer}"

        if answer.citations:
            cited = ", ".join(
                f"{c.source_filename} (p. {c.page_number})"
                for c in answer.citations
            )
            response += f"\n\n(Fuentes consultadas: {cited})"

        citations_list: list[dict[str, Any]] = [
            {
                "chunk_id": c.chunk_id,
                "document_id": c.document_id,
                "source_filename": c.source_filename,
                "page_number": c.page_number,
            }
            for c in answer.citations
        ]

        return response, citations_list, meta

    def _run_doubt_rag(
        self,
        patient_text: str,
        rag_query: str,
        domain: str,
    ) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
        """Run RAG retrieval in response to an LLM approval 'request_rag' action.

        The LLM is uncertain about the patient's answer and wants clinical
        context before continuing.  We run RAG with the LLM-provided query,
        then call the LLM again to get guidance on the patient's condition.

        Returns a ``(response_text, citations, metadata)`` tuple.
        """
        meta: dict[str, Any] = {
            "rag_queries": 0,
            "llm_duration_ms": None,
            "prompt_tokens": None,
            "completion_tokens": None,
        }

        pc = self._call_context.patient_context
        query = (
            f"Procedimiento: {pc.procedimiento}. "
            f"Día postoperatorio: {pc.dia_postop}. "
            f"Dominio: {domain}. "
            f"Respuesta del paciente: {patient_text}. "
            f"Consulta clínica: {rag_query}"
        )

        # --- Retrieve ---
        retrieval_result = self._retrieve(patient_text, rag_query)
        if retrieval_result is not None:
            meta["rag_queries"] = 1

        if not retrieval_result or not retrieval_result.has_results or not retrieval_result.sufficient:
            # No RAG results — return a warning but continue
            return (
                "He consultado la información disponible pero no encuentro "
                "detalles adicionales sobre este aspecto. Le sugiero "
                "consultar con su médico tratante si tiene dudas.",
                [],
                meta,
            )

        # --- Generate ---
        if self._llm_config is None:
            return (
                "Gracias por compartir esta información. Le recuerdo que "
                "cualquier síntoma que le preocupe debe ser consultado "
                "con su médico tratante.",
                [],
                meta,
            )

        context_chunks: list[dict[str, Any]] = [
            {
                "chunk_id": c.chunk_id,
                "document_id": c.document_id,
                "source_filename": c.source_filename,
                "page_number": c.page_number,
                "text": c.text,
                "similarity": c.similarity,
            }
            for c in retrieval_result.chunks
        ]

        try:
            answer: RagAnswer = generate_rag_answer(
                query=query,
                context_chunks=context_chunks,
                config=self._llm_config,
                debug=False,
            )
        except Exception:
            logger.exception("LLM call failed during doubt RAG")
            return (
                "He consultado la información disponible. Le recomiendo "
                "consultar con su médico tratante si tiene dudas sobre "
                "este aspecto de su recuperación.",
                [],
                meta,
            )

        meta["llm_duration_ms"] = answer.llm_duration_ms
        meta["prompt_tokens"] = answer.prompt_tokens
        meta["completion_tokens"] = answer.completion_tokens

        if answer.insufficient_knowledge:
            return (
                "He consultado la información clínica disponible. Basado en "
                "ella, no encuentro detalles adicionales. Le sugiero "
                "consultar a su médico tratante.",
                [],
                meta,
            )

        response = f"{answer.answer}"

        if answer.citations:
            cited = ", ".join(
                f"{c.source_filename} (p. {c.page_number})"
                for c in answer.citations
            )
            response += f"\n\n(Fuentes consultadas: {cited})"

        citations_list: list[dict[str, Any]] = [
            {
                "chunk_id": c.chunk_id,
                "document_id": c.document_id,
                "source_filename": c.source_filename,
                "page_number": c.page_number,
            }
            for c in answer.citations
        ]

        return response, citations_list, meta

    def _retrieve(
        self,
        patient_text: str,
        question: str,
    ) -> Optional[RetrievalResult]:
        """Perform RAG retrieval if configured, else return ``None``."""
        if self._rag_config is None:
            return None

        pc = self._call_context.patient_context
        query = (
            f"Procedimiento: {pc.procedimiento}. "
            f"Día postoperatorio: {pc.dia_postop}. "
            f"Pregunta de seguimiento: {question}. "
            f"Respuesta del paciente: {patient_text}"
        )

        try:
            try:
                from backend.persistence.sqlite import get_active_document_ids
                valid_ids = get_active_document_ids()
            except RuntimeError:
                # SQLite not initialised — fall back to no filtering
                valid_ids = None
            return retrieve(
                query=query, config=self._rag_config,
                valid_document_ids=valid_ids,
            )
        except Exception:
            logger.exception("RAG retrieval failed in orchestrator")
            return None

    # -- state transition helper ---------------------------------------------

    def _transition(self, event: Event) -> None:
        """Advance the state machine, mutating the ``CallContext``.

        Because ``CallContext`` is frozen, we replace it with a new
        instance that shares the same history and patient context.
        """
        old_state = self._call_context.state
        new_state = next_state(old_state, event)
        self._call_context = CallContext(
            call_id=self._call_context.call_id,
            patient_context=self._call_context.patient_context,
            state=new_state,
            history=self._call_context.history,
            created_at=self._call_context.created_at,
        )
        logger.info(
            "Transition: %s --(%s)--> %s",
            old_state.name,
            event.name,
            new_state.name,
        )

    # -- message recording helpers -------------------------------------------

    def _record_agent(self, text: str) -> None:
        """Record an agent message in the call history."""
        msg = Message(
            turn_index=len(self._call_context.history),
            role=MessageRole.AGENT,
            text=text,
            timestamp=datetime.datetime.now(datetime.timezone.utc),
        )
        self._call_context.history.append(msg)

    def _record_patient(self, text: str) -> None:
        """Record a patient message in the call history."""
        msg = Message(
            turn_index=len(self._call_context.history),
            role=MessageRole.PATIENT,
            text=text,
            timestamp=datetime.datetime.now(datetime.timezone.utc),
        )
        self._call_context.history.append(msg)

    # -- turn construction helper --------------------------------------------

    def _make_turn(
        self,
        agent_message: str,
        *,
        requires_response: bool = True,
        citations: Optional[list[dict[str, Any]]] = None,
        question_index: Optional[int] = None,
        call_ended: bool = False,
        llm_meta: Optional[dict[str, Any]] = None,
        escalation: Optional[EscalationResult] = None,
    ) -> OrchestratorTurn:
        """Build an ``OrchestratorTurn`` from the current call state.

        Parameters
        ----------
        llm_meta : dict or None
            Optional LLM metrics dict with keys ``rag_queries``,
            ``llm_duration_ms``, ``prompt_tokens``, and
            ``completion_tokens``.  When ``None`` (non-question turns),
            the metrics fields default to zero / ``None``.
        escalation : EscalationResult or None
            Optional escalation classification for this turn.
        """
        extra: dict[str, Any] = {}
        if llm_meta is not None:
            extra["rag_queries"] = llm_meta.get("rag_queries", 0)
            extra["llm_duration_ms"] = llm_meta.get("llm_duration_ms")
            extra["prompt_tokens"] = llm_meta.get("prompt_tokens")
            extra["completion_tokens"] = llm_meta.get("completion_tokens")

        return OrchestratorTurn(
            agent_message=agent_message,
            state=self._call_context.state,
            citations=citations or [],
            call_ended=call_ended or self._call_context.state is State.ENDED,
            requires_response=requires_response,
            question_index=question_index,
            total_questions=_NUM_QUESTIONS,
            escalation=escalation,
            **extra,
        )
