"""Conversation orchestrator: deterministic Spanish text-only call flow.

``ConversationOrchestrator`` connects the existing domain primitives —
``PatientContext``, ``CallContext``, ``State`` / ``Event`` state machine,
``History`` / ``Message``, RAG retrieval, and LLM answer generation — into
a single coordinated dialogue flow for postoperative follow-up.

The orchestrator is text-only and deterministic: it uses a fixed sequence of
structured questions (in Spanish), drives state transitions safely, records
every turn in the history, retrieves clinical knowledge through RAG, and
generates validated answers via the LLM adapter with traceable citations.

Fallback behaviour:
* Consent refused → polite closing, call ends.
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
    """

    agent_message: str
    state: State
    citations: list[dict[str, Any]] = field(default_factory=list)
    call_ended: bool = False
    requires_response: bool = True
    question_index: Optional[int] = None
    total_questions: int = _NUM_QUESTIONS

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
       After the last question → **CLOSING**
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
            f"{pc.patient.eps}. ¿Cómo se siente el día de hoy?"
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
        """Handle consent response → QUESTIONS or CLOSING."""
        lower = patient_text.lower()
        consent_refused = any(
            word in lower
            for word in ("no", "rechazo", "rechazar", "termine", "colgar")
        ) and not any(
            word in lower
            for word in ("sí", "si", "acepto", "aceptar", "adelante", "claro", "bueno", "dale", "listo")
        )

        if consent_refused:
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
        else:
            self._transition(Event.CONSENT_GIVEN)
            self._question_index = 0
            return self._ask_next_question()

    def _handle_question_response(self, patient_text: str) -> OrchestratorTurn:
        """Process patient answer to a follow-up question.

        Steps:
        1. Retrieve RAG context and generate an LLM answer.
        2. Ask the next question (or close if done).
        """
        # --- RAG + LLM response for this answer ---
        response_msg, citations = self._generate_clinical_response(
            patient_text=patient_text,
            question_index=self._question_index,
        )

        # Advance question index and ask the next one
        self._question_index += 1
        if self._question_index >= _NUM_QUESTIONS:
            return self._close_questions(response_msg, citations=citations)
        else:
            return self._ask_next_question(after_message=response_msg, citations=citations)

    def _handle_closing_response(self, patient_text: str) -> OrchestratorTurn:
        """Last patient acknowledgment → end the call."""
        return self._end_call()

    # -- helper: ask next question -------------------------------------------

    def _ask_next_question(
        self,
        after_message: Optional[str] = None,
        citations: Optional[list[dict[str, Any]]] = None,
    ) -> OrchestratorTurn:
        """Ask the current follow-up question.

        If *after_message* is provided, it is recorded as the agent's
        response to the previous answer, then the next question is appended.
        If *citations* are provided, they are included in the turn.
        """
        if self._question_index >= _NUM_QUESTIONS:
            return self._close_questions(after_message, citations=citations)

        question = FOLLOW_UP_QUESTIONS[self._question_index]

        if after_message:
            # Prepend the clinical response, then ask the next question
            full = f"{after_message}\n\n{question}"
        else:
            full = question

        self._record_agent(full)
        return self._make_turn(
            full,
            citations=citations or [],
            question_index=self._question_index,
            requires_response=True,
        )

    # -- helper: close questions phase ---------------------------------------

    def _close_questions(
        self,
        final_message: Optional[str] = None,
        citations: Optional[list[dict[str, Any]]] = None,
    ) -> OrchestratorTurn:
        """Transition QUESTIONS → CLOSING and produce closing message."""
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
        return self._make_turn(full, citations=citations or [], requires_response=True)

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
    ) -> tuple[str, list[dict[str, Any]]]:
        """Retrieve clinical knowledge and generate a validated response.

        Returns a ``(response_text, citations)`` tuple where *citations* is
        a list of dicts with ``chunk_id``, ``document_id``,
        ``source_filename``, and ``page_number``.

        Falls back to safe messages when RAG is unavailable or returns no
        results.
        """
        question = FOLLOW_UP_QUESTIONS[question_index]

        # --- Retrieve ---
        retrieval_result = self._retrieve(patient_text, question)

        if not retrieval_result or not retrieval_result.has_results:
            return (
                "Gracias por compartir esta información. No cuento con "
                "material de consulta suficiente para darle una "
                "orientación más detallada sobre este punto en este "
                "momento. Si tiene dudas, consulte a su médico tratante.",
                [],
            )

        # --- Generate ---
        if self._llm_config is None:
            return (
                "Gracias por compartir esta información. Le recuerdo "
                "que cualquier síntoma que le preocupe debe ser "
                "consultado con su médico tratante.",
                [],
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
            )

        if answer.insufficient_knowledge:
            return (
                "Gracias. Basado en la información disponible, no "
                "tengo detalles adicionales sobre este aspecto de su "
                "recuperación. Le sugiero consultar a su médico si "
                "tiene inquietudes.",
                [],
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

        return response, citations_list

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
            return retrieve(query=query, config=self._rag_config)
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
    ) -> OrchestratorTurn:
        """Build an ``OrchestratorTurn`` from the current call state."""
        return OrchestratorTurn(
            agent_message=agent_message,
            state=self._call_context.state,
            citations=citations or [],
            call_ended=call_ended or self._call_context.state is State.ENDED,
            requires_response=requires_response,
            question_index=question_index,
            total_questions=_NUM_QUESTIONS,
        )
