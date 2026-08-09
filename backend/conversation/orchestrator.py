"""Conversation orchestrator: deterministic Spanish text-only call flow.

``ConversationOrchestrator`` connects the existing domain primitives —
``PatientContext``, ``CallContext``, ``State`` / ``Event`` state machine,
``History`` / ``Message``, RAG retrieval, and LLM answer generation — into
a single coordinated dialogue flow for postoperative follow-up.

The orchestrator is text-only and deterministic: it uses a fixed sequence of
structured questions (in Spanish), drives state transitions safely, records
every turn in the history, and classifies patient answers **before** any
RAG/LLM call.  RED answers short-circuit immediately to ENDED with an urgent
safety message, ``call_ended=True``, and no further processing.  GREEN and
first-YELLOW answers receive deterministic acknowledgments without RAG/LLM.
Two consecutive YELLOW classifications trigger escalation with
``should_escalate=True``.

Clinical questions during CLOSING are answered with RAG+LLM (with citations)
and the call remains in CLOSING; non-questions end the call.

Fallback behaviour:
* Consent refused → polite closing, call ends.
* RED classification → short-circuit to ENDED with urgent safety message
  and ``call_ended=True``.  No RAG/LLM call; no further questions.
* Two consecutive YELLOW → escalation to CLOSING.
* No RAG chunks retrieved → the agent states it lacks information and
  advises consulting the treating physician.
* LLM unreachable → a safe fallback message is returned with
  ``insufficient_knowledge=True`` (handled by the LLM adapter).
"""

from __future__ import annotations

import datetime
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from backend.conversation.context import CallContext, PatientContext
from backend.conversation.messages import History, Message, MessageRole
from backend.conversation.state import Event, State
from backend.conversation.transitions import InvalidTransitionError, next_state
from backend.decision import classify as _decision_classify
from backend.decision import EscalationResult, Severity
from backend.llm.adapter import RagAnswer, generate_rag_answer
from backend.llm.config import LlmConfig
from backend.rag.config import RagConfig
from backend.rag.retrieval import RetrievalResult, retrieve

logger = logging.getLogger(__name__)

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
        1. Classify the patient's answer against the symptom domain.
        2. **RED** → short-circuit: no RAG/LLM, urgent safety message,
           transition directly to ENDED, ``call_ended=True``.
        3. **YELLOW (first)** → deterministic acknowledgment, no RAG/LLM,
           ask next question.
        4. **YELLOW (second consecutive)** → escalate to CLOSING.
        5. **GREEN** → deterministic acknowledgment, no RAG/LLM,
           ask next question.
        """
        answered_idx = self._question_index
        domain = _QUESTION_DOMAINS[answered_idx]
        pc = self._call_context.patient_context

        # --- Safety-first: classify before any RAG/LLM ---
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

        # --- RED: short-circuit immediately → ENDED ---
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

        # --- YELLOW: accumulate or escalate ---
        if severity is Severity.YELLOW:
            self._consecutive_yellows += 1

            if self._consecutive_yellows >= 2:
                # Two consecutive YELLOW → escalate with should_escalate=True
                escalated_classification = EscalationResult(
                    severity=classification.severity,
                    should_escalate=True,
                    reason=classification.reason,
                    next_action=classification.next_action,
                    domain=classification.domain,
                    source=classification.source,
                )
                self._transition(Event.ESCALATION_TRIGGER)
                escalate_msg = self._build_consecutive_yellow_message(classification)
                self._record_agent(escalate_msg)
                self._question_index += 1
                return self._make_turn(
                    escalate_msg,
                    requires_response=True,
                    question_index=answered_idx,
                    escalation=escalated_classification,
                )
            else:
                # First YELLOW: deterministic ack
                ack = self._build_yellow_ack(domain, classification)
                self._question_index += 1
                if self._question_index >= _NUM_QUESTIONS:
                    return self._close_questions(
                        ack,
                        question_index=self._question_index,
                        escalation=classification,
                    )
                else:
                    return self._ask_next_question(
                        after_message=ack,
                        question_index=self._question_index,
                        escalation=classification,
                    )

        # --- GREEN: deterministic ack ---
        self._consecutive_yellows = 0
        ack = self._build_green_ack(domain)
        self._question_index += 1
        if self._question_index >= _NUM_QUESTIONS:
            return self._close_questions(
                ack,
                question_index=self._question_index,
                escalation=classification,
            )
        else:
            return self._ask_next_question(
                after_message=ack,
                question_index=self._question_index,
                escalation=classification,
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

    def _is_clinical_question(self, text: str) -> bool:
        """Detect whether the patient is asking a clinical question.

        Uses simple heuristics: question marks, Spanish question words,
        and inquiry patterns that suggest the patient wants information.
        """
        lowered = text.lower().strip()

        # Direct question markers
        if "?" in lowered or "¿" in lowered:
            return True

        # Spanish question words / inquiry patterns (avoid ambiguous
        # matches like "como" meaning "I eat", "que" as relative pronoun).
        question_markers = (
            "cómo ", "qué ", "cuándo ", "cuando ",
            "dónde ", "donde ", "por qué", "por que",
            "pregunta", "duda", "quisiera saber",
            "necesito saber", "me puede decir",
            "me podría decir", "me podría explicar",
            "me puede explicar", "explíqueme",
            "expliqueme", "cuénteme", "cuenteme",
        )
        for marker in question_markers:
            if marker in lowered:
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
