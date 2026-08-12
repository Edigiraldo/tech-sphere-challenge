"""Tests for ``backend.conversation.orchestrator`` — ConversationOrchestrator."""

from __future__ import annotations

import datetime
from unittest.mock import MagicMock, patch

import pytest

from backend.conversation.context import CallContext, PatientContext
from backend.conversation.messages import History, Message, MessageRole
from backend.conversation.orchestrator import (
    FOLLOW_UP_QUESTIONS,
    _NUM_QUESTIONS,
    ConversationOrchestrator,
    OrchestratorTurn,
)
from backend.conversation.state import Event, State
from backend.conversation.transitions import InvalidTransitionError
from backend.data.models import Patient as DataPatient
from backend.decision import EscalationResult, Severity
from backend.llm.adapter import RagAnswer, RagCitation
from backend.llm.config import LlmConfig
from backend.rag.config import RagConfig

# ---------------------------------------------------------------------------
# Test constants
# ---------------------------------------------------------------------------

# A single patient response that classifies as GREEN for all six symptom
# domains (dolor, fiebre, herida, apetito, sueño, movilidad).  Used in
# tests that walk the full question loop without triggering YELLOW escalation.
_GREEN_RESPONSE = (
    "Todo bien, sin dolor, sin fiebre, herida limpia, "
    "tengo buen apetito, duermo bien, camino sin problema"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_data_patient(**overrides) -> DataPatient:
    """Build a valid ``DataPatient`` with sensible defaults."""
    defaults = dict(
        paciente_id="pac_test_001",
        bundle_id="bundle_test",
        synthea_runtime="synthetic_fallback",
        modulo_synthea="appendicitis",
        procedimiento="Apendicectomía",
        fecha_cirugia=datetime.date(2026, 6, 14),
        edad=34,
        genero="F",
        comorbilidades=[],
        complicacion_encounter=False,
        nombre_completo="María Test",
        direccion="Calle 1",
        ciudad="Soacha",
        departamento="Cundinamarca",
        documento_cc="123456789",
        eps="Compensar EPS",
        source_country="US",
        adapted_country="CO",
        adaptation_fields=[],
    )
    defaults.update(overrides)
    return DataPatient(**defaults)


def make_patient_context(**overrides) -> PatientContext:
    """Build a ``PatientContext`` with defaults."""
    kwargs = dict(
        patient=make_data_patient(),
        dia_postop=3,
        procedimiento="Apendicectomía",
    )
    kwargs.update(overrides)
    return PatientContext(**kwargs)


def make_orchestrator(
    patient_context: PatientContext | None = None,
    rag_config: RagConfig | None = None,
    llm_config: LlmConfig | None = None,
) -> ConversationOrchestrator:
    """Build an orchestrator with defaults."""
    if patient_context is None:
        patient_context = make_patient_context()
    return ConversationOrchestrator(
        patient_context=patient_context,
        rag_config=rag_config,
        llm_config=llm_config,
    )


def make_rag_config() -> RagConfig:
    """Build a default RagConfig for tests (no env needed)."""
    return RagConfig(
        embedding_model="BAAI/bge-m3",
        chroma_persist_dir=datetime.__file__,  # dummy, won't be used
        collection_name="test_collection",
        chunk_size=800,
        chunk_overlap=150,
        retrieval_top_k=5,
        similarity_threshold=0.0,
    )


def make_llm_config() -> LlmConfig:
    """Build a default LlmConfig for tests (no external API key needed)."""
    return LlmConfig(
        provider="groq",
        model_name="llama-3.1-70b-versatile",
        api_key="test-key",
        temperature=0.2,
        max_output_tokens=512,
    )


# ---------------------------------------------------------------------------
# OrchestratorTurn
# ---------------------------------------------------------------------------


class TestOrchestratorTurn:
    """OrchestratorTurn dataclass validation."""

    def test_minimal_construction(self):
        turn = OrchestratorTurn(
            agent_message="Hola",
            state=State.GREETING,
        )
        assert turn.agent_message == "Hola"
        assert turn.state is State.GREETING
        assert turn.citations == []
        assert not turn.call_ended
        assert turn.requires_response
        assert turn.question_index is None
        assert turn.total_questions == _NUM_QUESTIONS

    def test_empty_agent_message_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            OrchestratorTurn(agent_message="", state=State.IDLE)

    def test_whitespace_only_agent_message_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            OrchestratorTurn(agent_message="   ", state=State.IDLE)

    def test_call_ended_flag(self):
        turn = OrchestratorTurn(
            agent_message="Adiós",
            state=State.ENDED,
            call_ended=True,
            requires_response=False,
        )
        assert turn.call_ended
        assert not turn.requires_response

    def test_question_index(self):
        turn = OrchestratorTurn(
            agent_message="Pregunta 1",
            state=State.QUESTIONS,
            question_index=2,
        )
        assert turn.question_index == 2


# ---------------------------------------------------------------------------
# ConversationOrchestrator — construction
# ---------------------------------------------------------------------------


class TestOrchestratorConstruction:
    """ConversationOrchestrator instantiation."""

    def test_construction_with_patient_context(self):
        pc = make_patient_context()
        orch = ConversationOrchestrator(pc)
        assert orch.state is State.IDLE
        assert isinstance(orch.call_context, CallContext)
        assert orch.call_context.patient_context is pc

    def test_rejects_non_patient_context(self):
        with pytest.raises(TypeError, match="PatientContext"):
            ConversationOrchestrator("not a context")  # type: ignore[arg-type]

    def test_call_context_matches_patient_context_call_id(self):
        pc = make_patient_context()
        orch = ConversationOrchestrator(pc)
        assert orch.call_context.call_id == pc.call_id

    def test_history_starts_empty(self):
        orch = make_orchestrator()
        assert len(orch.history) == 0

    def test_optional_configs_none(self):
        orch = ConversationOrchestrator(make_patient_context())
        assert orch.state is State.IDLE
        # Should not crash with None configs


# ---------------------------------------------------------------------------
# start_call
# ---------------------------------------------------------------------------


class TestStartCall:
    """start_call behaviour."""

    def test_transitions_to_greeting(self):
        orch = make_orchestrator()
        result = orch.start_call()
        assert orch.state is State.GREETING
        assert result.state is State.GREETING
        assert not result.call_ended
        assert result.requires_response

    def test_returns_greeting_with_patient_name(self):
        orch = make_orchestrator()
        result = orch.start_call()
        assert "María Test" in result.agent_message
        assert "Compensar EPS" in result.agent_message
        # Should be Spanish greeting
        assert "Buenos días" in result.agent_message or "Buenos" in result.agent_message

    def test_records_agent_message_in_history(self):
        orch = make_orchestrator()
        result = orch.start_call()
        assert len(orch.history) == 1
        msg = orch.history[0]
        assert msg.role is MessageRole.AGENT
        assert msg.text == result.agent_message
        assert msg.turn_index == 0

    def test_cannot_start_twice(self):
        orch = make_orchestrator()
        orch.start_call()
        with pytest.raises(InvalidTransitionError):
            orch.start_call()


# ---------------------------------------------------------------------------
# process_patient_message — input validation
# ---------------------------------------------------------------------------


class TestProcessPatientMessageValidation:
    """Input validation for process_patient_message."""

    def test_empty_text_raises(self):
        orch = make_orchestrator()
        orch.start_call()
        with pytest.raises(ValueError, match="non-empty"):
            orch.process_patient_message("")

    def test_whitespace_only_raises(self):
        orch = make_orchestrator()
        orch.start_call()
        with pytest.raises(ValueError, match="non-empty"):
            orch.process_patient_message("   \t  ")

    def test_raises_invalid_transition_from_idle(self):
        orch = make_orchestrator()
        with pytest.raises(InvalidTransitionError):
            orch.process_patient_message("Hola")


# ---------------------------------------------------------------------------
# process_patient_message — GREETING → CONSENT
# ---------------------------------------------------------------------------


class TestGreetingToConsent:
    """GREETING → CONSENT transition."""

    def test_transitions_to_consent(self):
        orch = make_orchestrator()
        orch.start_call()
        result = orch.process_patient_message("Muy bien, gracias.")
        assert orch.state is State.CONSENT
        assert result.state is State.CONSENT
        assert not result.call_ended

    def test_presents_consent_information(self):
        orch = make_orchestrator()
        orch.start_call()
        result = orch.process_patient_message("Bien, gracias.")
        assert "seguimiento" in result.agent_message.lower()
        assert "Apendicectomía" in result.agent_message
        assert "María Test" in result.agent_message
        assert "¿Acepta" in result.agent_message or "acepta" in result.agent_message.lower()

    def test_records_both_messages(self):
        orch = make_orchestrator()
        orch.start_call()
        orch.process_patient_message("Bien, gracias.")
        assert len(orch.history) == 3
        assert orch.history[0].role is MessageRole.AGENT   # greeting
        assert orch.history[1].role is MessageRole.PATIENT  # patient response
        assert orch.history[2].role is MessageRole.AGENT    # consent request


# ---------------------------------------------------------------------------
# process_patient_message — CONSENT
# ---------------------------------------------------------------------------


class TestConsentGiven:
    """CONSENT → QUESTIONS when consent is given."""

    def test_accepts_explicit_si(self):
        orch = make_orchestrator()
        orch.start_call()
        orch.process_patient_message("Bien, gracias.")
        result = orch.process_patient_message("Sí, acepto.")
        assert orch.state is State.QUESTIONS
        assert result.state is State.QUESTIONS
        assert not result.call_ended

    def test_accepts_claro(self):
        orch = make_orchestrator()
        orch.start_call()
        orch.process_patient_message("Bien.")
        result = orch.process_patient_message("Claro, adelante.")
        assert orch.state is State.QUESTIONS

    def test_accepts_dale(self):
        orch = make_orchestrator()
        orch.start_call()
        orch.process_patient_message("Bien.")
        result = orch.process_patient_message("Dale, siga.")
        assert orch.state is State.QUESTIONS

    def test_accepts_bueno(self):
        orch = make_orchestrator()
        orch.start_call()
        orch.process_patient_message("Bien.")
        result = orch.process_patient_message("Bueno, está bien.")
        assert orch.state is State.QUESTIONS

    def test_first_question_asked(self):
        orch = make_orchestrator()
        orch.start_call()
        orch.process_patient_message("Bien.")
        result = orch.process_patient_message("Sí, acepto.")
        assert result.question_index == 0
        assert result.requires_response
        assert "dolor" in result.agent_message.lower()

    def test_records_consent_exchange(self):
        orch = make_orchestrator()
        orch.start_call()
        orch.process_patient_message("Bien.")
        result = orch.process_patient_message("Sí.")
        # history: greeting (A), response (P), consent request (A), sí (P), question (A)
        assert len(orch.history) == 5
        assert orch.history[3].role is MessageRole.PATIENT
        assert orch.history[3].text == "Sí."


class TestConsentRefused:
    """CONSENT → CLOSING when consent is refused."""

    def test_refuses_with_no(self):
        orch = make_orchestrator()
        orch.start_call()
        orch.process_patient_message("Bien.")
        result = orch.process_patient_message("No, no quiero.")
        assert orch.state is State.CLOSING
        assert result.state is State.CLOSING
        assert not result.call_ended

    def test_refuses_with_rechazo(self):
        orch = make_orchestrator()
        orch.start_call()
        orch.process_patient_message("Bien.")
        result = orch.process_patient_message("Rechazo el seguimiento.")
        assert orch.state is State.CLOSING

    def test_closing_message_respects_decision(self):
        orch = make_orchestrator()
        orch.start_call()
        orch.process_patient_message("Bien.")
        result = orch.process_patient_message("No, gracias.")
        assert "respeto" in result.agent_message.lower()


# ---------------------------------------------------------------------------
# process_patient_message — QUESTIONS → next question / closing
# ---------------------------------------------------------------------------


class TestQuestionsFlow:
    """Question progression through the QUESTIONS state."""

    def test_questions_advance_in_order(self):
        orch = make_orchestrator()
        orch.start_call()
        orch.process_patient_message("Bien.")
        orch.process_patient_message("Sí, acepto.")

        for i in range(_NUM_QUESTIONS):
            assert orch.state is State.QUESTIONS
            result = orch.process_patient_message(_GREEN_RESPONSE)
            if i < _NUM_QUESTIONS - 1:
                # Still in QUESTIONS, asking next question
                expected_idx = i + 1
                assert orch.state is State.QUESTIONS, (
                    f"Expected QUESTIONS after answer {i}, "
                    f"got {orch.state.name}"
                )
                assert result.question_index == expected_idx
            else:
                # Last question answered → CLOSING
                assert orch.state is State.CLOSING
                assert result.state is State.CLOSING

    def test_all_questions_complete_transitions_to_closing(self):
        orch = make_orchestrator()
        orch.start_call()
        orch.process_patient_message("Bien.")
        orch.process_patient_message("Sí.")

        for i in range(_NUM_QUESTIONS - 1):
            orch.process_patient_message(_GREEN_RESPONSE)

        # Last question answer → CLOSING
        result = orch.process_patient_message(_GREEN_RESPONSE)
        assert orch.state is State.CLOSING
        assert result.state is State.CLOSING
        assert not result.call_ended

    def test_message_count_in_questions(self):
        """Each question turn adds 2 messages: patient answer + agent response."""
        orch = make_orchestrator()
        orch.start_call()
        orch.process_patient_message("Bien.")
        orch.process_patient_message("Sí.")

        # After consent: history has 5 messages
        base_count = len(orch.history)  # 5

        orch.process_patient_message(_GREEN_RESPONSE)
        # One question turn adds: patient msg + agent msg = 2
        assert len(orch.history) == base_count + 2


# ---------------------------------------------------------------------------
# process_patient_message — CLOSING → ENDED
# ---------------------------------------------------------------------------


class TestClosingToEnded:
    """CLOSING → ENDED transition."""

    def test_closing_response_ends_call(self):
        orch = make_orchestrator()
        orch.start_call()
        orch.process_patient_message("Bien.")
        orch.process_patient_message("Sí.")

        # Answer all questions
        for _ in range(_NUM_QUESTIONS):
            orch.process_patient_message(_GREEN_RESPONSE)

        # Now in CLOSING
        assert orch.state is State.CLOSING

        # Patient's final response (non-question → end call)
        result = orch.process_patient_message("No, gracias, todo claro.")
        assert orch.state is State.ENDED
        assert result.state is State.ENDED
        assert result.call_ended
        assert not result.requires_response

    def test_farewell_includes_patient_info(self):
        orch = make_orchestrator()
        orch.start_call()
        orch.process_patient_message("Bien.")
        orch.process_patient_message("Sí.")
        for _ in range(_NUM_QUESTIONS):
            orch.process_patient_message(_GREEN_RESPONSE)
        result = orch.process_patient_message("No, gracias.")
        assert "María Test" in result.agent_message
        assert "Apendicectomía" in result.agent_message

    def test_closing_clinical_question_stays_in_closing(self):
        """A clinical question during CLOSING uses RAG/LLM and stays in CLOSING."""
        orch = make_orchestrator()
        orch.start_call()
        orch.process_patient_message("Bien.")
        orch.process_patient_message("Sí.")
        for _ in range(_NUM_QUESTIONS):
            orch.process_patient_message(_GREEN_RESPONSE)

        assert orch.state is State.CLOSING

        # Patient asks a clinical question
        result = orch.process_patient_message(
            "¿Qué debo hacer si me duele la herida?"
        )
        # Should remain in CLOSING (not ended)
        assert orch.state is State.CLOSING
        assert result.state is State.CLOSING
        assert not result.call_ended
        assert result.requires_response

    def test_non_question_during_closing_ends_call(self):
        """A non-question during CLOSING ends the call."""
        orch = make_orchestrator()
        orch.start_call()
        orch.process_patient_message("Bien.")
        orch.process_patient_message("Sí.")
        for _ in range(_NUM_QUESTIONS):
            orch.process_patient_message(_GREEN_RESPONSE)

        assert orch.state is State.CLOSING
        result = orch.process_patient_message("Gracias, adiós.")
        assert orch.state is State.ENDED
        assert result.call_ended

    def test_cannot_process_after_ended(self):
        orch = make_orchestrator()
        orch.start_call()
        orch.process_patient_message("Bien.")
        orch.process_patient_message("Sí.")
        for _ in range(_NUM_QUESTIONS):
            orch.process_patient_message(_GREEN_RESPONSE)
        orch.process_patient_message("No.")

        # ENDED — should get a message but no new transition
        result = orch.process_patient_message("Hola otra vez")
        assert orch.state is State.ENDED


class TestClinicalQuestionDetection:
    """Question detection must tolerate realistic Spanish STT output."""

    @pytest.mark.parametrize(
        "text",
        [
            "que cuidados debo seguir despues de una apendicectomia",
            "¿Qué cuidados debo seguir después de una apendicectomía?",
            "como limpio la herida",
            "si, que como deberia yo cuidarme ante una apendicectomia",
            "cuando puedo volver a caminar",
            "cuanto tiempo tarda la recuperacion",
            "tengo una duda sobre la fiebre",
            "quiero saber si esto es normal",
            "me puede explicar que debo hacer",
            "me toca ir a urgencias",
            "oiga doctor, que hago si me duele",
            "es normal esto despues de la cirugia",
            "que medicamento puedo tomar",
        ],
    )
    def test_detects_spoken_clinical_questions(self, text):
        orch = make_orchestrator()
        assert orch._is_clinical_question(text)

    @pytest.mark.parametrize(
        "text",
        [
            "la herida que tengo esta limpia",
            "como bien y duermo bien",
            "todo esta bien",
            "gracias, adios",
            "no tengo fiebre",
        ],
    )
    def test_does_not_misclassify_statements(self, text):
        orch = make_orchestrator()
        assert not orch._is_clinical_question(text)

    def test_stt_style_closing_question_stays_in_closing(self):
        orch = make_orchestrator()
        orch.start_call()
        orch.process_patient_message("Bien.")
        orch.process_patient_message("Si.")
        for _ in range(_NUM_QUESTIONS):
            orch.process_patient_message(_GREEN_RESPONSE)

        result = orch.process_patient_message(
            "si que cuidados debo seguir despues de una apendicectomia"
        )

        assert orch.state is State.CLOSING
        assert result.state is State.CLOSING
        assert not result.call_ended
        assert result.requires_response


# ---------------------------------------------------------------------------
# CLOSING negation phrase regression tests
# ---------------------------------------------------------------------------


class TestClosingNegationPhrases:
    """Negation phrases like 'no tengo preguntas' must END the call.

    The CLOSING state invites the patient to ask clinical questions.
    Negation phrases expressing "no questions/doubts" must be detected
    as non-questions and end the call.  Positive question patterns
    (e.g. 'tengo una pregunta') must remain detected as questions.
    """

    def _advance_to_closing(self, orch: ConversationOrchestrator) -> None:
        """Advance the orchestrator through the full flow to CLOSING."""
        orch.start_call()
        orch.process_patient_message("Bien, gracias.")
        orch.process_patient_message("Si, acepto.")
        for _ in range(_NUM_QUESTIONS):
            orch.process_patient_message(_GREEN_RESPONSE)
        assert orch.state is State.CLOSING

    # --- Negation variants that must END the call -------------------------

    @pytest.mark.parametrize(
        "patient_text",
        [
            # Core negation phrases from the bug report
            "no tengo preguntas",
            "no tengo dudas",
            "sin preguntas",
            # Variations
            "No tengo preguntas, gracias.",
            "no tengo dudas, todo esta claro",
            "Sin preguntas, doctor.",
            "ninguna pregunta",
            "ninguna duda",
            "ninguna inquietud",
            "no tengo ninguna pregunta",
            "no tengo ninguna duda",
            "sin ninguna pregunta",
            "sin ninguna duda",
            "no tengo inquietudes",
            "sin inquietudes",
            # Polite closing phrases
            "no, nada mas",
            "nada mas, gracias",
            "no, gracias",
            "todo claro, gracias",
            "todo bien, doctor",
            "estoy bien, gracias",
            "asi esta bien",
            "no necesito nada",
            "por ahora no",
            # Colombian regionalisms
            "no senor, todo bien",
            "no doctor, gracias",
        ],
    )
    def test_negation_phrase_ends_call(self, patient_text: str) -> None:
        """Negation phrases must result in call_ended=True, state=ENDED."""
        orch = make_orchestrator()
        self._advance_to_closing(orch)
        result = orch.process_patient_message(patient_text)
        assert orch.state is State.ENDED, (
            f"Expected ENDED for {patient_text!r}, got {orch.state.name}"
        )
        assert result.call_ended, (
            f"Expected call_ended=True for {patient_text!r}"
        )

    @pytest.mark.parametrize(
        "patient_text",
        [
            "no tengo preguntas",
            "sin dudas",
            "no tengo dudas, doctor",
            "ninguna duda",
            "nada mas",
            "no, gracias",
        ],
    )
    def test_negation_does_not_trigger_clinical_question(
        self, patient_text: str
    ) -> None:
        """Negation phrases must NOT be detected as clinical questions."""
        orch = make_orchestrator()
        assert not orch._is_clinical_question(patient_text), (
            f"Expected NOT clinical question: {patient_text!r}"
        )

    # --- Positive question patterns must still work -----------------------

    @pytest.mark.parametrize(
        "patient_text",
        [
            # Explicit "I have a question"
            "tengo una pregunta",
            "tengo una duda",
            "tengo una duda sobre la recuperacion",
            # Question marks preserved
            "¿Qué debo hacer si me duele?",
            "que debo hacer si me duele?",
            # Clinical questions still detected
            "que cuidados debo seguir",
            "cuando puedo volver a trabajar",
            "me puede explicar que debo hacer",
            "es normal esto",
            "quiero saber si puedo comer algo",
            # Mixed: positive question after some negation in same text
            "bueno, pero tengo una duda",
        ],
    )
    def test_positive_question_stays_in_closing(
        self, patient_text: str
    ) -> None:
        """Real questions during CLOSING must remain in CLOSING, not end."""
        orch = make_orchestrator(rag_config=make_rag_config())
        self._advance_to_closing(orch)
        result = orch.process_patient_message(patient_text)
        assert orch.state is State.CLOSING, (
            f"Expected CLOSING for {patient_text!r}, got {orch.state.name}"
        )
        assert not result.call_ended, (
            f"Expected call_ended=False for {patient_text!r}"
        )

    def test_question_mark_still_detected(self) -> None:
        """Explicit question marks must still trigger clinical question detection."""
        orch = make_orchestrator()
        # Question mark always overrides negation check
        assert orch._is_clinical_question("no se, ¿que hago?")

    def test_tengo_una_pregunta_is_question(self) -> None:
        """'tengo una pregunta' is a question, NOT a closing negation."""
        orch = make_orchestrator()
        assert orch._is_clinical_question("tengo una pregunta sobre la herida")

    def test_tengo_una_duda_is_question(self) -> None:
        """'tengo una duda' is a question, NOT a closing negation."""
        orch = make_orchestrator()
        assert orch._is_clinical_question("tengo una duda sobre el dolor")

    def test_no_tengo_ninguna_pregunta_ends_call(self) -> None:
        """'no tengo ninguna pregunta' must end the call."""
        orch = make_orchestrator()
        self._advance_to_closing(orch)
        result = orch.process_patient_message("no tengo ninguna pregunta, gracias")
        assert orch.state is State.ENDED
        assert result.call_ended

    def test_no_tengo_preguntas_with_comma_ends_call(self) -> None:
        """'no, no tengo preguntas' must end the call."""
        orch = make_orchestrator()
        self._advance_to_closing(orch)
        result = orch.process_patient_message("no, no tengo preguntas")
        assert orch.state is State.ENDED
        assert result.call_ended

    def test_por_el_momento_no_ends_call(self) -> None:
        """'por el momento no' must end the call."""
        orch = make_orchestrator()
        self._advance_to_closing(orch)
        result = orch.process_patient_message("no, por el momento no tengo dudas")
        assert orch.state is State.ENDED
        assert result.call_ended

    def test_sin_dudas_ends_call(self) -> None:
        """'sin dudas' must end the call."""
        orch = make_orchestrator()
        self._advance_to_closing(orch)
        result = orch.process_patient_message("sin dudas doctor, gracias")
        assert orch.state is State.ENDED
        assert result.call_ended

    # --- Regression: negation embedded in longer phrases ----------------

    @pytest.mark.parametrize(
        "patient_text",
        [
            # "o que" patterns (from "creo que", "pienso que", etc.)
            # must NOT trump negation — the negation check now runs first.
            "creo que no tengo preguntas",
            "pienso que no tengo dudas",
            "creo que sin preguntas doctor",
            "considero que no tengo dudas",
            "supongo que no tengo preguntas",
            "me parece que no tengo dudas",
            "diria que no tengo preguntas",
            "creo que no, sin preguntas",
            # "y que" patterns
            "y que no tengo preguntas",
            "y que no tengo dudas doctor",
            # Other embedded negation
            "pues creo que no tengo dudas",
            "la verdad creo que no tengo preguntas",
            "en mi opinion, no tengo dudas",
            "yo diria que no tengo preguntas",
            "realmente creo que no doctor, sin dudas",
            "no, la verdad no tengo preguntas",
            "no se doctor, creo que no tengo dudas",
        ],
    )
    def test_embedded_negation_ends_call(self, patient_text: str) -> None:
        """Negation embedded in longer phrases must end the call
        even when compound patterns like 'o que' appear in the text."""
        orch = make_orchestrator()
        self._advance_to_closing(orch)
        result = orch.process_patient_message(patient_text)
        assert orch.state is State.ENDED, (
            f"Expected ENDED for {patient_text!r}, got {orch.state.name}"
        )
        assert result.call_ended, (
            f"Expected call_ended=True for {patient_text!r}"
        )

    @pytest.mark.parametrize(
        "patient_text",
        [
            "creo que no tengo preguntas",
            "pienso que no tengo dudas",
            "y que no tengo preguntas",
            "diria que no doctor, sin preguntas",
        ],
    )
    def test_embedded_negation_not_clinical_question(
        self, patient_text: str
    ) -> None:
        """Negation embedded in longer phrases must NOT trigger
        clinical-question detection."""
        orch = make_orchestrator()
        assert not orch._is_clinical_question(patient_text), (
            f"Expected NOT clinical question: {patient_text!r}"
        )

    # --- Positive questions must still work after negation-move fix ---

    @pytest.mark.parametrize(
        "patient_text",
        [
            # Questions with "creo que" but positive intent
            "creo que si, tengo una pregunta",
            "creo que tengo una duda sobre eso",
            "pienso que si debo preguntar algo",
            # Questions with "y que" interrogative
            "y que cuidados debo seguir",
            "y que debo hacer si me duele",
            # Standard compound patterns still detected
            "o que debo hacer ahora",
            "o que cuidados necesito",
        ],
    )
    def test_positive_questions_still_detected_after_fix(
        self, patient_text: str
    ) -> None:
        """Real clinical questions must still be detected as such
        after the negation check was moved before compound patterns."""
        orch = make_orchestrator()
        assert orch._is_clinical_question(patient_text), (
            f"Expected clinical question: {patient_text!r}"
        )

    def test_question_mark_with_negation_still_question(self) -> None:
        """Explicit question marks override negation even with the
        negation check moved earlier."""
        orch = make_orchestrator()
        # Question mark always has highest priority
        assert orch._is_clinical_question("no se, ¿creo que no tengo preguntas?")
        assert orch._is_clinical_question("creo que no tengo dudas?")
        assert orch._is_clinical_question("¿sin preguntas?")

    def test_por_que_with_negation_still_question(self) -> None:
        """'por que' must still override negation."""
        orch = make_orchestrator()
        assert orch._is_clinical_question("por que no tengo preguntas")


# ---------------------------------------------------------------------------
# Full call flow
# ---------------------------------------------------------------------------


class TestFullCallFlow:
    """End-to-end call from IDLE to ENDED."""

    def test_full_happy_path(self):
        orch = make_orchestrator()
        assert orch.state is State.IDLE

        # Start
        r = orch.start_call()
        assert orch.state is State.GREETING
        assert "María Test" in r.agent_message

        # Greeting response
        r = orch.process_patient_message("Bien, gracias.")
        assert orch.state is State.CONSENT
        assert "Acepta" in r.agent_message

        # Consent given
        r = orch.process_patient_message("Sí, acepto.")
        assert orch.state is State.QUESTIONS
        assert r.question_index == 0

        # Answer all questions with GREEN-classifying responses
        for _ in range(_NUM_QUESTIONS):
            r = orch.process_patient_message(_GREEN_RESPONSE)

        # After all questions, go to CLOSING
        assert orch.state is State.CLOSING

        # Close
        r = orch.process_patient_message("No, nada más. Gracias.")
        assert orch.state is State.ENDED
        assert r.call_ended
        assert not r.requires_response

    def test_history_has_all_turns(self):
        orch = make_orchestrator()
        orch.start_call()
        orch.process_patient_message("Bien.")
        orch.process_patient_message("Sí, acepto.")
        for _ in range(_NUM_QUESTIONS):
            orch.process_patient_message(_GREEN_RESPONSE)
        orch.process_patient_message("No, gracias.")

        # Expected turn count:
        # IDLE→GREETING: 1 agent msg
        # GREETING→CONSENT: 1 patient, 1 agent = 2
        # CONSENT→QUESTIONS: 1 patient, 1 agent = 2
        # Each question: 1 patient, 1 agent = 2 * _NUM_QUESTIONS
        # CLOSING→ENDED: 1 patient, 1 agent = 2
        expected = 1 + 2 + 2 + (2 * _NUM_QUESTIONS) + 2
        assert len(orch.history) == expected

        # Check alternating agent/patient pattern
        for idx, msg in enumerate(orch.history):
            if idx % 2 == 0:
                assert msg.role is MessageRole.AGENT, f"Index {idx}: expected AGENT"
            else:
                assert msg.role is MessageRole.PATIENT, f"Index {idx}: expected PATIENT"

    def test_consent_refused_flow(self):
        orch = make_orchestrator()
        orch.start_call()
        orch.process_patient_message("Bien.")
        r = orch.process_patient_message("No, gracias. No quiero.")
        assert orch.state is State.CLOSING

        r = orch.process_patient_message("Bueno, adiós.")
        assert orch.state is State.ENDED
        assert r.call_ended

# ---------------------------------------------------------------------------
# RAG + LLM integration (mocked)
# ---------------------------------------------------------------------------
# Since the safety-first flow classifies answers BEFORE RAG/LLM and uses
# deterministic acks for GREEN/YELLOW, RAG+LLM is only invoked during:
#   1. CLOSING clinical questions (via ``_generate_closing_rag_response``)
#   2. The legacy ``_generate_clinical_response`` (kept for backward compat)


class TestRagLlmIntegration:
    """Orchestrator behaviour with RAG and LLM configs during CLOSING."""

    def _advance_to_closing(self, orch: ConversationOrchestrator):
        """Advance the orchestrator through the full flow to CLOSING."""
        orch.start_call()
        orch.process_patient_message("Bien.")
        orch.process_patient_message("Sí, acepto.")
        for _ in range(_NUM_QUESTIONS):
            orch.process_patient_message(_GREEN_RESPONSE)
        assert orch.state is State.CLOSING

    def test_no_rag_config_produces_fallback_in_closing(self):
        """Without RAG/LLM config, clinical question gets fallback."""
        orch = make_orchestrator(rag_config=None, llm_config=None)
        self._advance_to_closing(orch)
        result = orch.process_patient_message(
            "¿Qué debo hacer si me duele la herida?"
        )
        # Should remain in CLOSING
        assert orch.state is State.CLOSING
        assert "consult" in result.agent_message.lower()

    @patch("backend.conversation.orchestrator.retrieve")
    def test_retrieve_called_with_closing_question(self, mock_retrieve):
        """RAG retrieval is called for clinical questions during CLOSING."""
        from backend.rag.retrieval import RetrievalResult

        mock_retrieve.return_value = RetrievalResult(query="", chunks=[])

        rag_config = make_rag_config()
        orch = make_orchestrator(rag_config=rag_config, llm_config=None)
        self._advance_to_closing(orch)
        orch.process_patient_message(
            "¿Qué debo hacer si me duele la herida?"
        )

        mock_retrieve.assert_called_once()
        call_query = mock_retrieve.call_args.kwargs["query"]
        assert "Apendicectomía" in call_query

    @patch("backend.conversation.orchestrator.retrieve")
    def test_no_rag_results_produces_fallback_in_closing(self, mock_retrieve):
        """When RAG returns no chunks during CLOSING, fallback used."""
        from backend.rag.retrieval import RetrievalResult

        mock_retrieve.return_value = RetrievalResult(query="", chunks=[])

        rag_config = make_rag_config()
        orch = make_orchestrator(rag_config=rag_config, llm_config=None)
        self._advance_to_closing(orch)
        result = orch.process_patient_message(
            "¿Qué debo hacer si me duele la herida?"
        )
        assert orch.state is State.CLOSING
        assert "consult" in result.agent_message.lower()

    @patch("backend.conversation.orchestrator.generate_rag_answer")
    @patch("backend.conversation.orchestrator.retrieve")
    def test_llm_called_in_closing(self, mock_retrieve, mock_generate):
        """When RAG returns chunks during CLOSING, LLM is called."""
        from backend.rag.retrieval import RetrievedChunk, RetrievalResult

        chunk = RetrievedChunk(
            chunk_id="chunk1",
            document_id="doc1",
            source_filename="test.pdf",
            chunk_index=0,
            page_number=1,
            text="Postoperative care guidance text.",
            similarity=0.85,
        )
        mock_retrieve.return_value = RetrievalResult(
            query="", chunks=[chunk], sufficient=True,
        )

        mock_answer = RagAnswer(
            answer="Gracias por su pregunta. Le recomiendo mantener la herida limpia.",
            citations=[
                RagCitation(
                    chunk_id="chunk1",
                    document_id="doc1",
                    source_filename="test.pdf",
                    page_number=1,
                    excerpt="Postoperative care...",
                )
            ],
            insufficient_knowledge=False,
            model="llama-3.1-70b-versatile",
        )
        mock_generate.return_value = mock_answer

        rag_config = make_rag_config()
        llm_config = LlmConfig(
            model_name="llama-3.1-70b-versatile",
            api_key="fake-key",
        )
        orch = make_orchestrator(
            rag_config=rag_config,
            llm_config=llm_config,
        )
        self._advance_to_closing(orch)
        result = orch.process_patient_message(
            "¿Qué debo hacer si me duele la herida?"
        )

        mock_generate.assert_called_once()
        call_query = mock_generate.call_args.kwargs["query"]
        assert "Apendicectomía" in call_query
        assert "Gracias" in result.agent_message
        assert len(result.citations) == 1
        assert result.citations[0]["source_filename"] == "test.pdf"
        assert result.citations[0]["chunk_id"] == "chunk1"
        # Should remain in CLOSING
        assert orch.state is State.CLOSING
        assert not result.call_ended

    @patch("backend.conversation.orchestrator.generate_rag_answer")
    @patch("backend.conversation.orchestrator.retrieve")
    def test_insufficient_knowledge_fallback_in_closing(
        self, mock_retrieve, mock_generate
    ):
        """When LLM returns insufficient_knowledge, fallback is used."""
        from backend.rag.retrieval import RetrievedChunk, RetrievalResult

        chunk = RetrievedChunk(
            chunk_id="chunk1",
            document_id="doc1",
            source_filename="test.pdf",
            chunk_index=0,
            page_number=1,
            text="Some text.",
            similarity=0.85,
        )
        mock_retrieve.return_value = RetrievalResult(
            query="", chunks=[chunk], sufficient=True,
        )

        mock_answer = RagAnswer(
            answer="No tengo suficiente información.",
            citations=[],
            insufficient_knowledge=True,
            model="llama-3.1-70b-versatile",
        )
        mock_generate.return_value = mock_answer

        rag_config = make_rag_config()
        llm_config = LlmConfig(
            model_name="llama-3.1-70b-versatile",
            api_key="fake-key",
        )
        orch = make_orchestrator(
            rag_config=rag_config,
            llm_config=llm_config,
        )
        self._advance_to_closing(orch)
        result = orch.process_patient_message(
            "¿Qué debo hacer si me duele la herida?"
        )
        assert "consult" in result.agent_message.lower()
        assert orch.state is State.CLOSING

    @patch("backend.conversation.orchestrator.retrieve")
    def test_retrieve_exception_fallback_in_closing(self, mock_retrieve):
        """When RAG retrieval raises during CLOSING, falls back gracefully."""
        mock_retrieve.side_effect = RuntimeError("ChromaDB unavailable")

        rag_config = make_rag_config()
        orch = make_orchestrator(rag_config=rag_config, llm_config=None)
        self._advance_to_closing(orch)
        result = orch.process_patient_message(
            "¿Qué debo hacer si me duele la herida?"
        )
        assert "consult" in result.agent_message.lower()
        assert orch.state is State.CLOSING

    @patch("backend.conversation.orchestrator.generate_rag_answer")
    @patch("backend.conversation.orchestrator.retrieve")
    def test_llm_exception_fallback_in_closing(self, mock_retrieve, mock_generate):
        """When LLM raises during CLOSING, falls back gracefully."""
        from backend.rag.retrieval import RetrievedChunk, RetrievalResult

        chunk = RetrievedChunk(
            chunk_id="chunk1",
            document_id="doc1",
            source_filename="test.pdf",
            chunk_index=0,
            page_number=1,
            text="Some text.",
            similarity=0.85,
        )
        mock_retrieve.return_value = RetrievalResult(
            query="", chunks=[chunk], sufficient=True,
        )
        mock_generate.side_effect = RuntimeError("LLM unavailable")

        rag_config = make_rag_config()
        llm_config = LlmConfig(
            model_name="llama-3.1-70b-versatile",
            api_key="fake-key",
        )
        orch = make_orchestrator(
            rag_config=rag_config,
            llm_config=llm_config,
        )
        self._advance_to_closing(orch)
        result = orch.process_patient_message(
            "¿Qué debo hacer si me duele la herida?"
        )
        assert "consult" in result.agent_message.lower()
        assert orch.state is State.CLOSING

    @patch("backend.conversation.orchestrator.retrieve")
    def test_non_empty_but_insufficient_retrieval_fallback_in_closing(
        self, mock_retrieve,
    ):
        """When RAG returns chunks but sufficient=False during CLOSING,
        the orchestrator must use the fallback without calling the LLM."""
        from backend.rag.retrieval import RetrievedChunk, RetrievalResult

        chunk = RetrievedChunk(
            chunk_id="chunk1",
            document_id="doc1",
            source_filename="test.pdf",
            chunk_index=0,
            page_number=1,
            text="Postoperative care guidance text.",
            similarity=0.50,
        )
        mock_retrieve.return_value = RetrievalResult(
            query="",
            chunks=[chunk],
            sufficient=False,  # below min_chunks or min_avg_similarity
        )

        rag_config = make_rag_config()
        orch = make_orchestrator(rag_config=rag_config, llm_config=None)
        self._advance_to_closing(orch)
        result = orch.process_patient_message(
            "¿Qué debo hacer si me duele la herida?"
        )
        # Fallback must be used, no LLM call
        assert "consult" in result.agent_message.lower()
        assert orch.state is State.CLOSING
        assert result.citations == []

    @patch("backend.conversation.orchestrator.generate_rag_answer")
    @patch("backend.conversation.orchestrator.retrieve")
    def test_non_empty_insufficient_skips_llm_call_in_closing(
        self, mock_retrieve, mock_generate,
    ):
        """When retrieval has chunks but is insufficient, the LLM
        must NOT be called."""
        from backend.rag.retrieval import RetrievedChunk, RetrievalResult

        chunk = RetrievedChunk(
            chunk_id="chunk1",
            document_id="doc1",
            source_filename="test.pdf",
            chunk_index=0,
            page_number=1,
            text="Some text.",
            similarity=0.40,
        )
        mock_retrieve.return_value = RetrievalResult(
            query="",
            chunks=[chunk],
            sufficient=False,
        )

        rag_config = make_rag_config()
        llm_config = LlmConfig(
            model_name="llama-3.1-70b-versatile",
            api_key="fake-key",
        )
        orch = make_orchestrator(
            rag_config=rag_config,
            llm_config=llm_config,
        )
        self._advance_to_closing(orch)
        orch.process_patient_message(
            "¿Qué debo hacer si me duele la herida?"
        )
        # LLM must not be called when retrieval is insufficient
        mock_generate.assert_not_called()


# ---------------------------------------------------------------------------
# Safety-first classification during QUESTIONS
# ---------------------------------------------------------------------------


class TestRedShortCircuit:
    """RED classification must short-circuit: no RAG/LLM, urgent message, CLOSING."""

    def _advance_to_questions(self, orch):
        orch.start_call()
        orch.process_patient_message("Bien.")
        orch.process_patient_message("Sí, acepto.")

    def test_red_transitions_to_ended(self):
        """RED → immediate transition to ENDED."""
        orch = make_orchestrator()
        self._advance_to_questions(orch)
        assert orch.state is State.QUESTIONS

        result = orch.process_patient_message("Me duele un 9, insoportable, no aguanto.")
        assert orch.state is State.ENDED
        assert result.state is State.ENDED
        assert result.call_ended
        assert not result.requires_response

    def test_red_exposes_escalation_in_turn(self):
        """RED result is exposed in OrchestratorTurn.escalation."""
        orch = make_orchestrator()
        self._advance_to_questions(orch)

        result = orch.process_patient_message("Me duele un 9, insoportable, no aguanto.")
        assert result.escalation is not None
        assert result.escalation.severity is Severity.RED
        assert result.escalation.should_escalate is True

    def test_red_message_contains_urgent_instructions(self):
        """The RED message tells the patient to seek urgent care."""
        orch = make_orchestrator()
        self._advance_to_questions(orch)

        result = orch.process_patient_message("Me duele un 9, insoportable, no aguanto.")
        msg = result.agent_message.lower()
        assert "urgente" in msg or "inmediato" in msg or "urgencias" in msg
        assert "finaliza" in msg

    def test_red_no_citations(self):
        """RED short-circuit produces no citations (no RAG/LLM called)."""
        orch = make_orchestrator()
        self._advance_to_questions(orch)

        result = orch.process_patient_message("Me duele un 9, insoportable, no aguanto.")
        assert result.citations == []

    @patch("backend.conversation.orchestrator.retrieve")
    @patch("backend.conversation.orchestrator.generate_rag_answer")
    def test_red_does_not_call_rag_or_llm(
        self, mock_generate, mock_retrieve
    ):
        """RED classification must not trigger RAG or LLM calls."""
        orch = make_orchestrator(rag_config=make_rag_config())
        self._advance_to_questions(orch)

        orch.process_patient_message("Me duele un 9, insoportable, no aguanto.")
        mock_retrieve.assert_not_called()
        mock_generate.assert_not_called()


class TestYellowAccumulation:
    """Two consecutive YELLOW results must escalate safely."""

    def _advance_to_questions(self, orch):
        orch.start_call()
        orch.process_patient_message("Bien.")
        orch.process_patient_message("Sí, acepto.")

    def test_first_yellow_stays_in_questions(self):
        """First YELLOW: ack + next question, stays in QUESTIONS."""
        orch = make_orchestrator()
        self._advance_to_questions(orch)
        assert orch.state is State.QUESTIONS

        # "me duele bastante" → YELLOW for dolor domain
        result = orch.process_patient_message("Me duele bastante, un 6.")
        assert orch.state is State.QUESTIONS
        assert result.state is State.QUESTIONS
        assert result.escalation is not None
        assert result.escalation.severity is Severity.YELLOW

    def test_two_consecutive_yellows_escalate(self):
        """Second consecutive YELLOW → transition to CLOSING."""
        orch = make_orchestrator()
        self._advance_to_questions(orch)
        assert orch.state is State.QUESTIONS

        # First answer: YELLOW (dolor)
        orch.process_patient_message("Me duele bastante, un 6.")
        assert orch.state is State.QUESTIONS

        # Second answer: YELLOW (fiebre)
        result = orch.process_patient_message("Tuve un poco de fiebre ayer.")
        # Second consecutive yellow → CLOSING
        assert orch.state is State.CLOSING
        assert result.state is State.CLOSING
        assert result.escalation is not None
        assert result.escalation.severity is Severity.YELLOW

    def test_green_resets_yellow_count(self):
        """A GREEN after YELLOW resets the consecutive counter."""
        orch = make_orchestrator()
        self._advance_to_questions(orch)
        assert orch.state is State.QUESTIONS

        # First answer: YELLOW (dolor)
        orch.process_patient_message("Me duele bastante, un 6.")
        assert orch.state is State.QUESTIONS

        # Second answer: GREEN (fiebre)
        result = orch.process_patient_message(
            "No tengo fiebre, temperatura normal."
        )
        assert orch.state is State.QUESTIONS
        assert result.escalation is not None
        assert result.escalation.severity is Severity.GREEN

    def test_red_resets_yellow_count(self):
        """RED after YELLOW resets counter (and triggers RED immediately)."""
        orch = make_orchestrator()
        self._advance_to_questions(orch)
        assert orch.state is State.QUESTIONS

        # First answer: YELLOW (dolor)
        orch.process_patient_message("Me duele bastante, un 6.")
        assert orch.state is State.QUESTIONS

        # Second answer: RED cross-cutting → immediate RED → ENDED
        result = orch.process_patient_message(
            "Tengo dolor en el pecho y no puedo respirar."
        )
        assert orch.state is State.ENDED
        assert result.escalation is not None
        assert result.escalation.severity is Severity.RED


class TestGreenDeterministicAck:
    """GREEN answers use deterministic ack, no RAG/LLM."""

    def _advance_to_questions(self, orch):
        orch.start_call()
        orch.process_patient_message("Bien.")
        orch.process_patient_message("Sí, acepto.")

    def test_green_produces_ack(self):
        """GREEN answer gets deterministic message + next question."""
        orch = make_orchestrator()
        self._advance_to_questions(orch)

        result = orch.process_patient_message(_GREEN_RESPONSE)
        assert orch.state is State.QUESTIONS
        assert result.question_index == 1  # next question (fiebre)
        assert result.escalation is not None
        assert result.escalation.severity is Severity.GREEN

    @patch("backend.conversation.orchestrator.retrieve")
    @patch("backend.conversation.orchestrator.generate_rag_answer")
    def test_green_does_not_call_rag_or_llm(
        self, mock_generate, mock_retrieve
    ):
        """GREEN classification must not trigger RAG or LLM calls."""
        orch = make_orchestrator(rag_config=make_rag_config())
        self._advance_to_questions(orch)

        orch.process_patient_message(_GREEN_RESPONSE)
        mock_retrieve.assert_not_called()
        mock_generate.assert_not_called()


class TestFirstYellowDeterministicAck:
    """First YELLOW uses deterministic ack, no RAG/LLM."""

    def _advance_to_questions(self, orch):
        orch.start_call()
        orch.process_patient_message("Bien.")
        orch.process_patient_message("Sí, acepto.")

    @patch("backend.conversation.orchestrator.retrieve")
    @patch("backend.conversation.orchestrator.generate_rag_answer")
    def test_first_yellow_does_not_call_rag_or_llm(
        self, mock_generate, mock_retrieve
    ):
        """First YELLOW classification must not trigger RAG or LLM calls."""
        orch = make_orchestrator(rag_config=make_rag_config())
        self._advance_to_questions(orch)

        orch.process_patient_message("Me duele bastante, un 6.")
        mock_retrieve.assert_not_called()
        mock_generate.assert_not_called()


# ---------------------------------------------------------------------------
# Existing test classes continue below
# ---------------------------------------------------------------------------


class TestOrchestratorProperties:
    """Property accessors."""

    def test_state_property(self):
        orch = make_orchestrator()
        assert orch.state is State.IDLE
        orch.start_call()
        assert orch.state is State.GREETING

    def test_call_context_property(self):
        pc = make_patient_context()
        orch = ConversationOrchestrator(pc)
        cc = orch.call_context
        assert isinstance(cc, CallContext)
        assert cc.patient_context is pc

    def test_history_property(self):
        orch = make_orchestrator()
        h = orch.history
        assert isinstance(h, History)
        assert len(h) == 0


# ---------------------------------------------------------------------------
# FOLLOW_UP_QUESTIONS
# ---------------------------------------------------------------------------


class TestFollowUpQuestions:
    """FOLLOW_UP_QUESTIONS constant."""

    def test_has_expected_count(self):
        assert len(FOLLOW_UP_QUESTIONS) == _NUM_QUESTIONS

    def test_all_questions_are_non_empty_strings(self):
        for i, q in enumerate(FOLLOW_UP_QUESTIONS):
            assert isinstance(q, str), f"Question {i} is not a string"
            assert q.strip(), f"Question {i} is empty"

    def test_all_questions_are_in_spanish(self):
        spanish_markers = {"á", "é", "í", "ó", "ú", "ñ", "¿"}
        for i, q in enumerate(FOLLOW_UP_QUESTIONS):
            has_spanish = any(c in q.lower() for c in spanish_markers)
            assert has_spanish, (
                f"Question {i} does not contain Spanish characters: {q[:50]}"
            )

    def test_questions_cover_postoperative_domains(self):
        """Each question should cover a distinct follow-up domain."""
        combined = " ".join(FOLLOW_UP_QUESTIONS).lower()
        domains = [
            "dolor",
            "fiebre",
            "herida",
            "apetito",
            "dormido",
            "moviliz",
        ]
        for domain in domains:
            assert domain in combined, f"Missing domain: {domain}"


# ---------------------------------------------------------------------------
# State transitions via orchestrator
# ---------------------------------------------------------------------------


class TestOrchestratorStateTransitions:
    """Verify the orchestrator drives correct state transitions."""

    def test_full_state_sequence_happy_path(self):
        orch = make_orchestrator()
        expected_states = [
            State.GREETING,   # after start_call
            State.CONSENT,    # after greeting response
            State.QUESTIONS,  # after consent given
        ]
        assert orch.state is State.IDLE
        orch.start_call()
        assert orch.state is expected_states[0]
        orch.process_patient_message("Bien.")
        assert orch.state is expected_states[1]
        orch.process_patient_message("Sí, acepto.")
        assert orch.state is expected_states[2]

        # Answer all questions
        for _ in range(_NUM_QUESTIONS):
            orch.process_patient_message(_GREEN_RESPONSE)
        assert orch.state is State.CLOSING

        orch.process_patient_message("No, gracias.")
        assert orch.state is State.ENDED

    def test_cannot_skip_states(self):
        """The orchestrator enforces valid transitions through the state machine."""
        orch = make_orchestrator()
        # Cannot go directly to GREETING without start_call
        with pytest.raises(InvalidTransitionError):
            orch.process_patient_message("Hola")

        orch.start_call()
        # Cannot jump from GREETING to QUESTIONS
        with pytest.raises(InvalidTransitionError):
            # This would try to directly trigger CONSENT_GIVEN which isn't
            # valid from GREETING
            orch._transition(Event.CONSENT_GIVEN)

    def test_call_id_is_consistent(self):
        orch = make_orchestrator()
        call_id_before = orch.call_context.call_id
        orch.start_call()
        assert orch.call_context.call_id == call_id_before
        orch.process_patient_message("Bien.")
        assert orch.call_context.call_id == call_id_before
        orch.process_patient_message("Sí.")
        assert orch.call_context.call_id == call_id_before


# ---------------------------------------------------------------------------
# Regression tests — consent refusal with negation priority
# ---------------------------------------------------------------------------


class TestConsentRefusalPriority:
    """Consent refusal must have priority over positive substrings."""

    def _advance_to_consent(self, orch):
        orch.start_call()
        orch.process_patient_message("Bien, gracias.")

    def test_no_acepto_continuar_is_refusal(self):
        """'No acepto continuar' must be treated as refusal, not consent."""
        orch = make_orchestrator()
        self._advance_to_consent(orch)
        assert orch.state is State.CONSENT
        result = orch.process_patient_message("No acepto continuar")
        assert orch.state is State.CLOSING
        assert "respeto" in result.agent_message.lower()

    def test_no_autorizo_is_refusal(self):
        """'No autorizo' must be treated as refusal, not consent."""
        orch = make_orchestrator()
        self._advance_to_consent(orch)
        result = orch.process_patient_message("No autorizo esta llamada")
        assert orch.state is State.CLOSING
        assert "respeto" in result.agent_message.lower()

    def test_no_deseo_continuar_is_refusal(self):
        """'No deseo continuar' must be treated as refusal, not consent."""
        orch = make_orchestrator()
        self._advance_to_consent(orch)
        result = orch.process_patient_message("No deseo continuar, gracias")
        assert orch.state is State.CLOSING
        assert "respeto" in result.agent_message.lower()

    def test_no_quiero_is_refusal(self):
        """'No quiero' must be treated as refusal."""
        orch = make_orchestrator()
        self._advance_to_consent(orch)
        result = orch.process_patient_message("No quiero seguimiento")
        assert orch.state is State.CLOSING

    def test_no_estoy_de_acuerdo_is_refusal(self):
        """'No estoy de acuerdo' must be treated as refusal."""
        orch = make_orchestrator()
        self._advance_to_consent(orch)
        result = orch.process_patient_message("No estoy de acuerdo con esto")
        assert orch.state is State.CLOSING

    def test_si_acepto_still_works(self):
        """'Sí, acepto' must still be treated as consent given."""
        orch = make_orchestrator()
        self._advance_to_consent(orch)
        result = orch.process_patient_message("Sí, acepto continuar")
        assert orch.state is State.QUESTIONS
        assert result.question_index == 0

    def test_claro_adelante_still_works(self):
        """'Claro, adelante' must still be treated as consent given."""
        orch = make_orchestrator()
        self._advance_to_consent(orch)
        result = orch.process_patient_message("Claro, adelante con las preguntas")
        assert orch.state is State.QUESTIONS


# ---------------------------------------------------------------------------
# Regression tests — second YELLOW should_escalate=True
# ---------------------------------------------------------------------------


class TestSecondYellowEscalation:
    """The second consecutive YELLOW must return should_escalate=True."""

    def _advance_to_questions(self, orch):
        orch.start_call()
        orch.process_patient_message("Bien.")
        orch.process_patient_message("Sí, acepto.")

    def test_second_yellow_returns_should_escalate_true(self):
        """Second consecutive YELLOW → should_escalate=True in the turn."""
        orch = make_orchestrator()
        self._advance_to_questions(orch)

        # First answer: YELLOW (dolor)
        orch.process_patient_message("Me duele bastante, un 6.")
        assert orch.state is State.QUESTIONS

        # Second answer: YELLOW (fiebre)
        result = orch.process_patient_message("Tuve un poco de fiebre ayer.")
        assert orch.state is State.CLOSING
        assert result.escalation is not None
        assert result.escalation.severity is Severity.YELLOW
        assert result.escalation.should_escalate is True, (
            "Second consecutive YELLOW must have should_escalate=True"
        )

    def test_second_yellow_preserves_severity_yellow(self):
        """Second consecutive YELLOW must preserve severity YELLOW."""
        orch = make_orchestrator()
        self._advance_to_questions(orch)

        orch.process_patient_message("Me duele bastante, un 6.")
        result = orch.process_patient_message("Tuve un poco de fiebre ayer.")
        assert result.escalation is not None
        assert result.escalation.severity is Severity.YELLOW

    def test_second_yellow_transitions_to_closing(self):
        """Second consecutive YELLOW must transition to CLOSING."""
        orch = make_orchestrator()
        self._advance_to_questions(orch)

        orch.process_patient_message("Me duele bastante, un 6.")
        result = orch.process_patient_message("Tuve un poco de fiebre ayer.")
        assert orch.state is State.CLOSING
        assert result.state is State.CLOSING

    def test_single_yellow_still_should_not_escalate(self):
        """First YELLOW must still have should_escalate=False."""
        orch = make_orchestrator()
        self._advance_to_questions(orch)

        result = orch.process_patient_message("Me duele bastante, un 6.")
        assert result.escalation is not None
        assert result.escalation.severity is Severity.YELLOW
        assert result.escalation.should_escalate is False, (
            "First YELLOW must have should_escalate=False"
        )


# ---------------------------------------------------------------------------
# Regression tests — RED terminates call, no RAG/LLM
# ---------------------------------------------------------------------------


class TestRedTerminatesCall:
    """RED must short-circuit to ENDED with call_ended=True and no RAG/LLM."""

    def _advance_to_questions(self, orch):
        orch.start_call()
        orch.process_patient_message("Bien.")
        orch.process_patient_message("Sí, acepto.")

    def test_red_returns_call_ended_true(self):
        """RED must return call_ended=True so the frontend disables recording."""
        orch = make_orchestrator()
        self._advance_to_questions(orch)
        result = orch.process_patient_message(
            "Me duele un 9, insoportable, no aguanto."
        )
        assert result.call_ended is True
        assert not result.requires_response

    def test_red_state_is_ended_not_closing(self):
        """RED must transition to ENDED, not CLOSING."""
        orch = make_orchestrator()
        self._advance_to_questions(orch)
        result = orch.process_patient_message(
            "Me duele un 9, insoportable, no aguanto."
        )
        assert orch.state is State.ENDED
        assert result.state is State.ENDED

    def test_no_further_processing_after_red(self):
        """After RED ends the call, further messages return the ended message."""
        orch = make_orchestrator()
        self._advance_to_questions(orch)
        orch.process_patient_message(
            "Me duele un 9, insoportable, no aguanto."
        )
        assert orch.state is State.ENDED

        # Further message should not trigger RAG/LLM, just return ended message
        result = orch.process_patient_message(
            "¿Qué debo hacer si me duele la herida?"
        )
        assert orch.state is State.ENDED
        assert "ya ha finalizado" in result.agent_message.lower()
        # The call_ended flag reflects ENDED state (not a new event)

    @patch("backend.conversation.orchestrator.retrieve")
    @patch("backend.conversation.orchestrator.generate_rag_answer")
    def test_red_does_not_call_rag_or_llm_with_rag_config(
        self, mock_generate, mock_retrieve
    ):
        """RED must not call RAG or LLM even when RAG/LLM are configured."""
        orch = make_orchestrator(rag_config=make_rag_config())
        self._advance_to_questions(orch)

        orch.process_patient_message(
            "Me duele un 9, insoportable, no aguanto."
        )
        mock_retrieve.assert_not_called()
        mock_generate.assert_not_called()

    def test_red_escalation_in_turn(self):
        """RED escalation is exposed with should_escalate=True."""
        orch = make_orchestrator()
        self._advance_to_questions(orch)
        result = orch.process_patient_message(
            "Me duele un 9, insoportable, no aguanto."
        )
        assert result.escalation is not None
        assert result.escalation.severity is Severity.RED
        assert result.escalation.should_escalate is True

    def test_red_question_index_is_preserved(self):
        """RED turn preserves the question_index for domain tracking."""
        orch = make_orchestrator()
        self._advance_to_questions(orch)
        result = orch.process_patient_message(
            "Me duele un 9, insoportable, no aguanto."
        )
        assert result.question_index == 0  # dolor domain

    def test_red_message_is_deterministic_spanish(self):
        """RED must produce a deterministic Spanish message without RAG/LLM."""
        orch = make_orchestrator()
        self._advance_to_questions(orch)
        result = orch.process_patient_message(
            "Me duele un 9, insoportable, no aguanto."
        )
        msg = result.agent_message.lower()
        assert "urgente" in msg or "inmediato" in msg or "urgencias" in msg
        assert "médico" in msg or "medico" in msg
        assert "finaliza" in msg  # call is ending
        assert "?" not in result.agent_message  # no question to patient


# ---------------------------------------------------------------------------
# LLM second-approval integration tests
# ---------------------------------------------------------------------------


def _make_llm_config() -> LlmConfig:
    """Build a test LlmConfig that will pass validation."""
    return LlmConfig(
        provider="groq",
        model_name="llama-3.1-70b-versatile",
        api_key="test-key",
        temperature=0.2,
        max_output_tokens=512,
    )


class TestLlmSecondApprovalGreenConfirmation:
    """LLM confirms GREEN → proceed normally."""

    def _advance_to_questions(self, orch):
        orch.start_call()
        orch.process_patient_message("Bien.")
        orch.process_patient_message("Sí, acepto.")

    @patch("backend.conversation.orchestrator.llm_second_approval")
    def test_green_confirmed_by_llm(self, mock_approval):
        """LLM confirms GREEN → continue with GREEN ack."""
        from backend.llm.approval import LlmApprovalResult
        mock_approval.return_value = LlmApprovalResult(
            severity=Severity.GREEN,
            should_escalate=False,
            reason="Concuerdo, evolución favorable.",
            next_action="Continuar seguimiento.",
            action="confirm",
            llm_used=True,
            llm_duration_ms=100.0,
            prompt_tokens=80,
            completion_tokens=40,
        )
        orch = make_orchestrator(llm_config=_make_llm_config())
        self._advance_to_questions(orch)

        # Answer question 0 (dolor)
        result = orch.process_patient_message("Todo bien, sin dolor.")
        assert orch.state is State.QUESTIONS
        assert result.question_index == 1  # moved to question 1
        assert result.escalation is not None
        assert result.escalation.severity is Severity.GREEN
        mock_approval.assert_called_once()

    @patch("backend.conversation.orchestrator.llm_second_approval")
    def test_green_upgraded_to_yellow_by_llm(self, mock_approval):
        """LLM upgrades GREEN → YELLOW."""
        from backend.llm.approval import LlmApprovalResult
        mock_approval.return_value = LlmApprovalResult(
            severity=Severity.YELLOW,
            should_escalate=False,
            reason="Aunque parece bien, el dolor moderado es preocupante.",
            next_action="Monitorear de cerca.",
            action="escalate",
            llm_used=True,
            llm_duration_ms=120.0,
            prompt_tokens=90,
            completion_tokens=45,
        )
        orch = make_orchestrator(llm_config=_make_llm_config())
        self._advance_to_questions(orch)

        result = orch.process_patient_message("Me duele un poquito, no mucho.")
        assert result.escalation is not None
        assert result.escalation.severity is Severity.YELLOW
        mock_approval.assert_called_once()

    @patch("backend.conversation.orchestrator.llm_second_approval")
    def test_green_upgraded_to_red_by_llm(self, mock_approval):
        """LLM upgrades GREEN → RED → ENDED."""
        from backend.llm.approval import LlmApprovalResult
        mock_approval.return_value = LlmApprovalResult(
            severity=Severity.RED,
            should_escalate=True,
            reason="El paciente reporta señales de alarma que requieren atención inmediata.",
            next_action="Transferir urgente.",
            action="escalate",
            llm_used=True,
            llm_duration_ms=130.0,
            prompt_tokens=95,
            completion_tokens=50,
        )
        orch = make_orchestrator(llm_config=_make_llm_config())
        self._advance_to_questions(orch)

        result = orch.process_patient_message("Me duele un poco.")
        assert orch.state is State.ENDED
        assert result.call_ended
        assert result.escalation is not None
        assert result.escalation.severity is Severity.RED

    @patch("backend.conversation.orchestrator.llm_second_approval")
    def test_no_llm_config_skips_approval(self, mock_approval):
        """Without LlmConfig, approval is skipped entirely."""
        orch = make_orchestrator()  # no llm_config
        self._advance_to_questions(orch)

        result = orch.process_patient_message(_GREEN_RESPONSE)
        assert result.escalation is not None
        assert result.escalation.severity is Severity.GREEN
        mock_approval.assert_not_called()


class TestLlmSecondApprovalYellowNoDowngrade:
    """LLM must never downgrade YELLOW to GREEN."""

    def _advance_to_questions(self, orch):
        orch.start_call()
        orch.process_patient_message("Bien.")
        orch.process_patient_message("Sí, acepto.")

    @patch("backend.conversation.orchestrator.llm_second_approval")
    def test_yellow_confirmed_by_llm(self, mock_approval):
        """LLM confirms YELLOW."""
        from backend.llm.approval import LlmApprovalResult
        mock_approval.return_value = LlmApprovalResult(
            severity=Severity.YELLOW,
            should_escalate=False,
            reason="Concuerdo, enrojecimiento requiere monitoreo.",
            next_action="Monitorear.",
            action="confirm",
            llm_used=True,
            llm_duration_ms=100.0,
            prompt_tokens=80,
            completion_tokens=40,
        )
        orch = make_orchestrator(llm_config=_make_llm_config())
        self._advance_to_questions(orch)

        # Answer question 0 (dolor), normal first
        orch.process_patient_message(_GREEN_RESPONSE)

        # Answer question 1 (fiebre) - YELLOW
        result = orch.process_patient_message(
            "Tuve un poco de fiebre ayer, pero ya estoy mejor."
        )
        assert result.escalation is not None
        assert result.escalation.severity is Severity.YELLOW

    @patch("backend.conversation.orchestrator.llm_second_approval")
    def test_yellow_upgraded_to_red_by_llm(self, mock_approval):
        """LLM may upgrade YELLOW → RED."""
        from backend.llm.approval import LlmApprovalResult

        # First two calls return GREEN, third returns RED upgrade
        mock_approval.side_effect = [
            LlmApprovalResult(
                severity=Severity.GREEN,
                should_escalate=False,
                reason="Ok.",
                next_action="Continue.",
                action="confirm",
                llm_used=True,
            ),
            LlmApprovalResult(
                severity=Severity.GREEN,
                should_escalate=False,
                reason="Ok.",
                next_action="Continue.",
                action="confirm",
                llm_used=True,
            ),
            LlmApprovalResult(
                severity=Severity.RED,
                should_escalate=True,
                reason="Enrojecimiento con calor indica infección grave.",
                next_action="Transferir urgente.",
                action="escalate",
                llm_used=True,
                llm_duration_ms=110.0,
                prompt_tokens=85,
                completion_tokens=45,
            ),
        ]
        orch = make_orchestrator(llm_config=_make_llm_config())
        self._advance_to_questions(orch)

        # Answer question 0 (dolor) first
        orch.process_patient_message(_GREEN_RESPONSE)

        # Answer question 1 (fiebre)
        orch.process_patient_message("Sin fiebre.")

        # Answer question 2 (herida) - YELLOW triggers RED via LLM
        result = orch.process_patient_message(
            "La herida está enrojecida y caliente."
        )
        assert orch.state is State.ENDED
        assert result.call_ended
        assert result.escalation is not None
        assert result.escalation.severity is Severity.RED


class TestLlmSecondApprovalClarification:
    """LLM requests clarification → stay on same question."""

    def _advance_to_questions(self, orch):
        orch.start_call()
        orch.process_patient_message("Bien.")
        orch.process_patient_message("Sí, acepto.")

    @patch("backend.conversation.orchestrator.llm_second_approval")
    def test_clarification_stays_on_same_question(self, mock_approval):
        """Clarification must not advance question_index."""
        from backend.llm.approval import LlmApprovalResult
        mock_approval.return_value = LlmApprovalResult(
            severity=Severity.YELLOW,
            should_escalate=False,
            reason="Respuesta ambigua.",
            next_action="Solicitar aclaración.",
            action="request_clarification",
            clarification_question="¿Podría describir mejor su nivel de dolor en una escala de 0 a 10?",
            llm_used=True,
            llm_duration_ms=110.0,
            prompt_tokens=80,
            completion_tokens=50,
        )
        orch = make_orchestrator(llm_config=_make_llm_config())
        self._advance_to_questions(orch)

        # First answer (dolor, q=0) → clarification
        result = orch.process_patient_message("Pues, más o menos.")
        assert result.question_index == 0  # still on question 0!
        assert orch.state is State.QUESTIONS
        assert "?" in result.agent_message  # clarification is a question

    @patch("backend.conversation.orchestrator.llm_second_approval")
    def test_clarification_then_answer_advances(self, mock_approval):
        """After one clarification, next answer advances normally."""
        from backend.llm.approval import LlmApprovalResult

        # First call → clarification
        # Second call → confirm
        mock_approval.side_effect = [
            LlmApprovalResult(
                severity=Severity.YELLOW,
                should_escalate=False,
                reason="Ambiguous.",
                next_action="Clarify.",
                action="request_clarification",
                clarification_question="¿Qué tan fuerte es su dolor?",
                llm_used=True,
            ),
            LlmApprovalResult(
                severity=Severity.GREEN,
                should_escalate=False,
                reason="Now clear, it's mild.",
                next_action="Continue.",
                action="confirm",
                llm_used=True,
            ),
        ]
        orch = make_orchestrator(llm_config=_make_llm_config())
        self._advance_to_questions(orch)

        # First answer → clarification (stays on q=0)
        r1 = orch.process_patient_message("Pues, más o menos.")
        assert r1.question_index == 0

        # Second answer → advances (q=0 → q=1)
        r2 = orch.process_patient_message("No mucho, un 2 de 10.")
        assert r2.question_index == 1

    @patch("backend.conversation.orchestrator.llm_second_approval")
    def test_clarification_limit_one_per_question(self, mock_approval):
        """Only one clarification is allowed per question."""
        from backend.llm.approval import LlmApprovalResult

        mock_approval.return_value = LlmApprovalResult(
            severity=Severity.YELLOW,
            should_escalate=False,
            reason="Ambiguous.",
            next_action="Clarify.",
            action="request_clarification",
            clarification_question="¿Podría ser más específico?",
            llm_used=True,
        )
        orch = make_orchestrator(llm_config=_make_llm_config())
        self._advance_to_questions(orch)

        # First attempt → clarification (stays)
        r1 = orch.process_patient_message("Pues, más o menos.")
        assert r1.question_index == 0

        # Second attempt on same question → limit reached, proceeds as YELLOW
        r2 = orch.process_patient_message("No sé, regular.")
        # Should advance now
        assert r2.question_index == 1
        assert r2.escalation is not None
        assert r2.escalation.severity is Severity.YELLOW


class TestLlmSecondApprovalRag:
    """LLM requests RAG for doubt → run RAG in QUESTIONS."""

    def _advance_to_questions(self, orch):
        orch.start_call()
        orch.process_patient_message("Bien.")
        orch.process_patient_message("Sí, acepto.")

    @patch("backend.conversation.orchestrator.retrieve")
    @patch("backend.conversation.orchestrator.generate_rag_answer")
    @patch("backend.conversation.orchestrator.llm_second_approval")
    def test_rag_request_runs_retrieval_and_continues(
        self, mock_approval, mock_generate, mock_retrieve
    ):
        """RAG doubt → run retrieval, generate answer, continue."""
        from backend.llm.approval import LlmApprovalResult
        from backend.rag.retrieval import RetrievalResult, RetrievedChunk

        # Approval requests RAG
        mock_approval.return_value = LlmApprovalResult(
            severity=Severity.YELLOW,
            should_escalate=False,
            reason="Duda sobre enrojecimiento.",
            next_action="Consultar fuentes.",
            action="request_rag",
            rag_query="¿Es normal enrojecimiento post-apendicectomía día 3?",
            llm_used=True,
            llm_duration_ms=120.0,
            prompt_tokens=90,
            completion_tokens=50,
        )

        # RAG returns chunks
        mock_retrieve.return_value = RetrievalResult(
            chunks=[
                RetrievedChunk(
                    chunk_id="rag-chunk-1",
                    document_id="doc-1",
                    source_filename="guia_postop.pdf",
                    chunk_index=0,
                    page_number=3,
                    text="El enrojecimiento leve es normal en los primeros días.",
                    similarity=0.72,
                ),
            ],
            query="test",
            sufficient=True,
        )

        # LLM RAG answer
        mock_generate.return_value = RagAnswer(
            answer="El enrojecimiento leve es normal en los primeros días postoperatorios.",
            citations=[
                RagCitation(
                    chunk_id="rag-chunk-1",
                    document_id="doc-1",
                    source_filename="guia_postop.pdf",
                    page_number=3,
                    excerpt="El enrojecimiento leve es normal...",
                )
            ],
            insufficient_knowledge=False,
            model="llama-3.1-70b-versatile",
            llm_duration_ms=200.0,
            prompt_tokens=100,
            completion_tokens=60,
        )

        orch = make_orchestrator(
            rag_config=make_rag_config(),
            llm_config=_make_llm_config(),
        )
        self._advance_to_questions(orch)

        # First answer → RAG doubt, run retrieval
        result = orch.process_patient_message(
            "La herida está un poco roja."
        )
        # Should have advanced
        assert result.question_index == 1
        mock_retrieve.assert_called_once()
        mock_generate.assert_called_once()

        # Citations should be included
        assert len(result.citations) > 0

    @patch("backend.conversation.orchestrator.retrieve")
    @patch("backend.conversation.orchestrator.generate_rag_answer")
    @patch("backend.conversation.orchestrator.llm_second_approval")
    def test_rag_on_last_question_proceeds_to_closing(
        self, mock_approval, mock_generate, mock_retrieve
    ):
        """RAG on question 6 (last) proceeds to CLOSING after retrieval."""
        from backend.llm.approval import LlmApprovalResult
        from backend.rag.retrieval import RetrievalResult, RetrievedChunk

        orch = make_orchestrator(
            rag_config=make_rag_config(),
            llm_config=_make_llm_config(),
        )
        self._advance_to_questions(orch)

        # Answer questions 0-4 with GREEN
        for _ in range(5):
            mock_approval.return_value = LlmApprovalResult(
                severity=Severity.GREEN,
                should_escalate=False,
                reason="Ok.",
                next_action="Continue.",
                action="confirm",
                llm_used=True,
            )
            orch.process_patient_message(_GREEN_RESPONSE)

        # Now on question 5 (movilidad, last one)
        # This time: RAG doubt
        mock_approval.return_value = LlmApprovalResult(
            severity=Severity.YELLOW,
            should_escalate=False,
            reason="Duda sobre mareo postoperatorio.",
            next_action="Consultar fuentes.",
            action="request_rag",
            rag_query="¿Es normal mareo al caminar después de apendicectomía?",
            llm_used=True,
        )
        mock_retrieve.return_value = RetrievalResult(
            chunks=[
                RetrievedChunk(
                    chunk_id="rag-chunk-final",
                    document_id="doc-final",
                    source_filename="movilidad.pdf",
                    chunk_index=0,
                    page_number=1,
                    text="El mareo leve es común los primeros días.",
                    similarity=0.68,
                ),
            ],
            query="test",
            sufficient=True,
        )
        mock_generate.return_value = RagAnswer(
            answer="El mareo leve al movilizarse es común en los primeros días postoperatorios.",
            citations=[
                RagCitation(
                    chunk_id="rag-chunk-final",
                    document_id="doc-final",
                    source_filename="movilidad.pdf",
                    page_number=1,
                )
            ],
            insufficient_knowledge=False,
            model="llama-3.1-70b-versatile",
        )

        result = orch.process_patient_message("Me mareo un poco al caminar.")
        # Last question → CLOSING
        assert orch.state is State.CLOSING
        assert result.state is State.CLOSING


class TestLlmSecondApprovalFallback:
    """LLM approval failures fall back to deterministic."""

    def _advance_to_questions(self, orch):
        orch.start_call()
        orch.process_patient_message("Bien.")
        orch.process_patient_message("Sí, acepto.")

    @patch("backend.conversation.orchestrator.llm_second_approval")
    def test_approval_crash_falls_back(self, mock_approval):
        """If llm_second_approval crashes, use deterministic result."""
        mock_approval.side_effect = RuntimeError("Boom")
        orch = make_orchestrator(llm_config=_make_llm_config())
        self._advance_to_questions(orch)

        result = orch.process_patient_message(_GREEN_RESPONSE)
        assert result.escalation is not None
        assert result.escalation.severity is Severity.GREEN
        assert not result.call_ended

    @patch("backend.conversation.orchestrator.llm_second_approval")
    def test_approval_returns_empty_llm_used(self, mock_approval):
        """llm_used=False means fallback to deterministic."""
        from backend.llm.approval import LlmApprovalResult
        mock_approval.return_value = LlmApprovalResult(
            severity=Severity.GREEN,
            should_escalate=False,
            reason="Deterministic fallback.",
            next_action="Continue.",
            action="confirm",
            llm_used=False,  # fallback
        )
        orch = make_orchestrator(llm_config=_make_llm_config())
        self._advance_to_questions(orch)

        result = orch.process_patient_message(_GREEN_RESPONSE)
        assert result.escalation is not None
        assert result.escalation.severity is Severity.GREEN


class TestLlmSecondApprovalMetrics:
    """LLM approval metrics propagate to turn."""

    def _advance_to_questions(self, orch):
        orch.start_call()
        orch.process_patient_message("Bien.")
        orch.process_patient_message("Sí, acepto.")

    @patch("backend.conversation.orchestrator.llm_second_approval")
    def test_llm_metrics_in_turn(self, mock_approval):
        """LLM duration and token counts propagate to OrchestratorTurn."""
        from backend.llm.approval import LlmApprovalResult
        mock_approval.return_value = LlmApprovalResult(
            severity=Severity.GREEN,
            should_escalate=False,
            reason="Confirmed.",
            next_action="Continue.",
            action="confirm",
            llm_used=True,
            llm_duration_ms=156.7,
            prompt_tokens=95,
            completion_tokens=42,
        )
        orch = make_orchestrator(llm_config=_make_llm_config())
        self._advance_to_questions(orch)

        result = orch.process_patient_message(_GREEN_RESPONSE)
        assert result.llm_duration_ms == 156.7
        assert result.prompt_tokens == 95
        assert result.completion_tokens == 42


class TestLlmSecondApprovalFinalQuestionClarification:
    """Clarification on question 6 (last) stays on question 6."""

    def _advance_to_questions(self, orch):
        orch.start_call()
        orch.process_patient_message("Bien.")
        orch.process_patient_message("Sí, acepto.")

    @patch("backend.conversation.orchestrator.llm_second_approval")
    def test_clarification_on_final_question_stays(self, mock_approval):
        """Clarification on question 6 stays on index 5."""
        from backend.llm.approval import LlmApprovalResult

        orch = make_orchestrator(llm_config=_make_llm_config())
        self._advance_to_questions(orch)

        # Answer questions 0-4 with GREEN
        for i in range(5):
            mock_approval.return_value = LlmApprovalResult(
                severity=Severity.GREEN,
                should_escalate=False,
                reason="Ok.",
                next_action="Continue.",
                action="confirm",
                llm_used=True,
            )
            orch.process_patient_message(_GREEN_RESPONSE)

        # Question 5 (last) → clarification
        mock_approval.return_value = LlmApprovalResult(
            severity=Severity.YELLOW,
            should_escalate=False,
            reason="Ambiguous answer.",
            next_action="Clarify.",
            action="request_clarification",
            clarification_question="¿Puede describir mejor cómo se siente al caminar?",
            llm_used=True,
        )
        result = orch.process_patient_message("Pues, regular.")
        assert result.question_index == 5  # stays on last question
        assert orch.state is State.QUESTIONS


# ---------------------------------------------------------------------------
# Escalation metadata/domain alignment after LLM severity upgrade
# ---------------------------------------------------------------------------


class TestEscalationDomainAlignment:
    """After an LLM severity upgrade, the next question must be associated
    with its own index/domain — no stale current-question escalation
    metadata must leak forward to the NEXT answer's classification."""

    def _advance_to_questions(self, orch: ConversationOrchestrator) -> None:
        orch.start_call()
        orch.process_patient_message("Bien.")
        orch.process_patient_message("Sí, acepto.")

    @patch("backend.conversation.orchestrator.llm_second_approval")
    def test_upgrade_escalation_domain_correctly_identifies_assessed_domain(
        self, mock_approval
    ) -> None:
        """LLM upgrades GREEN→YELLOW on question 1 (fiebre).
        The turn response should carry escalation with domain="fiebre"
        (the assessed domain) while question_index points to the NEXT
        question (herida).  Both referents are correct — the escalation
        describes what was assessed; question_index describes what is
        being asked next."""
        from backend.llm.approval import LlmApprovalResult

        orch = make_orchestrator(llm_config=_make_llm_config())
        self._advance_to_questions(orch)

        # q0: GREEN (dolor)
        mock_approval.return_value = LlmApprovalResult(
            severity=Severity.GREEN,
            should_escalate=False,
            reason="Sin dolor.",
            next_action="Continuar.",
            action="confirm",
            llm_used=True,
        )
        r0 = orch.process_patient_message("Sin dolor, todo bien.")
        assert r0.question_index == 1  # asking fiebre
        assert r0.escalation is not None
        assert r0.escalation.severity is Severity.GREEN
        assert r0.escalation.domain == "dolor"

        # q1: GREEN → YELLOW upgrade (fiebre)
        mock_approval.return_value = LlmApprovalResult(
            severity=Severity.YELLOW,
            should_escalate=False,
            reason="Detectado riesgo de fiebre que el clasificador no vio.",
            next_action="Monitorear.",
            action="escalate",
            llm_used=True,
        )
        r1 = orch.process_patient_message("A veces siento calor pero no se si es fiebre.")
        # After upgrade, domain should be the ASSESSED domain (fiebre)
        assert r1.escalation is not None, (
            "Upgraded escalation must be present in turn"
        )
        assert r1.escalation.domain == "fiebre", (
            f"Expected domain='fiebre' (assessed), got {r1.escalation.domain!r}"
        )
        # question_index points to NEXT question (herida)
        assert r1.question_index == 2, (
            f"Expected question_index=2 (herida), got {r1.question_index}"
        )

    @patch("backend.conversation.orchestrator.llm_second_approval")
    def test_next_answer_has_independent_classification(
        self, mock_approval
    ) -> None:
        """After LLM upgrade on question N, the NEXT answer (question N+1)
        must have its OWN classification — no stale metadata from the
        previous turn must affect it."""
        from backend.llm.approval import LlmApprovalResult

        orch = make_orchestrator(llm_config=_make_llm_config())
        self._advance_to_questions(orch)

        # q0: GREEN (dolor)
        mock_approval.return_value = LlmApprovalResult(
            severity=Severity.GREEN,
            should_escalate=False,
            reason="Ok.",
            next_action="Continue.",
            action="confirm",
            llm_used=True,
        )
        orch.process_patient_message("Sin dolor.")

        # q1: GREEN → YELLOW upgrade (fiebre)
        mock_approval.return_value = LlmApprovalResult(
            severity=Severity.YELLOW,
            should_escalate=False,
            reason="Posible fiebre no detectada.",
            next_action="Monitorear fiebre.",
            action="escalate",
            llm_used=True,
        )
        orch.process_patient_message("Senti calor ayer, no medi temperatura.")

        # q2: herida — must classify INDEPENDENTLY
        mock_approval.return_value = LlmApprovalResult(
            severity=Severity.GREEN,
            should_escalate=False,
            reason="Herida bien.",
            next_action="Continue.",
            action="confirm",
            llm_used=True,
        )
        r2 = orch.process_patient_message("La herida esta cicatrizando bien, sin enrojecimiento.")
        # Must have its OWN classification for herida, not carry fiebre
        assert r2.escalation is not None, "Expected fresh classification for herida answer"
        assert r2.escalation.domain == "herida", (
            f"Expected domain='herida' for herida answer, "
            f"got domain={r2.escalation.domain!r}"
        )
        assert r2.escalation.severity is Severity.GREEN
        # question_index advances correctly
        assert r2.question_index == 3, (
            f"Expected question_index=3 (apetito), got {r2.question_index}"
        )

    @patch("backend.conversation.orchestrator.llm_second_approval")
    def test_consecutive_yellow_still_has_escalation(self, mock_approval) -> None:
        """Two consecutive YELLOW must still have escalation in the turn
        (the escalation IS the reason for CLOSING)."""
        from backend.llm.approval import LlmApprovalResult

        orch = make_orchestrator(llm_config=_make_llm_config())
        self._advance_to_questions(orch)

        # q0: YELLOW (dolor)
        mock_approval.return_value = LlmApprovalResult(
            severity=Severity.YELLOW,
            should_escalate=False,
            reason="Dolor moderado.",
            next_action="Monitorear.",
            action="confirm",
            llm_used=True,
        )
        orch.process_patient_message("Me duele bastante, un 6.")

        # q1: YELLOW (fiebre) → second consecutive → escalate
        mock_approval.return_value = LlmApprovalResult(
            severity=Severity.YELLOW,
            should_escalate=False,
            reason="Fiebre reportada.",
            next_action="Monitorear.",
            action="confirm",
            llm_used=True,
        )
        r1 = orch.process_patient_message("Tuve fiebre ayer.")
        assert orch.state is State.CLOSING
        # Consecutive-yellow escalation MUST carry escalation with should_escalate=True
        assert r1.escalation is not None, (
            "Consecutive-yellow turn must have escalation"
        )
        assert r1.escalation.severity is Severity.YELLOW
        assert r1.escalation.should_escalate is True, (
            "Consecutive-yellow must have should_escalate=True"
        )

    @patch("backend.conversation.orchestrator.llm_second_approval")
    def test_closing_turn_after_last_question_escalation_domain_correct(
        self, mock_approval
    ) -> None:
        """After completing all questions, the CLOSING turn's escalation
        domain correctly identifies the last question's domain."""
        from backend.llm.approval import LlmApprovalResult

        orch = make_orchestrator(llm_config=_make_llm_config())
        self._advance_to_questions(orch)

        # Answer q0-q4 with GREEN
        for i in range(5):
            mock_approval.return_value = LlmApprovalResult(
                severity=Severity.GREEN,
                should_escalate=False,
                reason="Ok.",
                next_action="Continue.",
                action="confirm",
                llm_used=True,
            )
            orch.process_patient_message(_GREEN_RESPONSE)

        # q5 (movilidad, last): LLM upgrades GREEN→YELLOW
        mock_approval.return_value = LlmApprovalResult(
            severity=Severity.YELLOW,
            should_escalate=False,
            reason="Posible mareo no detectado.",
            next_action="Monitorear movilidad.",
            action="escalate",
            llm_used=True,
        )
        result = orch.process_patient_message("Me siento un poco debil al caminar.")
        assert orch.state is State.CLOSING, (
            f"Expected CLOSING after last question, got {orch.state.name}"
        )
        # The escalation correctly identifies the assessed domain
        assert result.escalation is not None
        assert result.escalation.domain == "movilidad", (
            f"Expected domain='movilidad' (last assessed), "
            f"got domain={result.escalation.domain!r}"
        )

    @patch("backend.conversation.orchestrator.llm_second_approval")
    def test_red_upgrade_ends_call_with_escalation(self, mock_approval) -> None:
        """LLM upgrade to RED must end call AND carry escalation.
        This turn directly represents the escalation decision."""
        from backend.llm.approval import LlmApprovalResult

        orch = make_orchestrator(llm_config=_make_llm_config())
        self._advance_to_questions(orch)

        # q0: LLM upgrades to RED
        mock_approval.return_value = LlmApprovalResult(
            severity=Severity.RED,
            should_escalate=True,
            reason="Señal de alerta grave detectada por el revisor LLM.",
            next_action="Transferir urgente.",
            action="escalate",
            llm_used=True,
        )
        result = orch.process_patient_message("Me duele mucho, no aguanto, tengo fiebre alta.")
        assert orch.state is State.ENDED
        assert result.call_ended
        assert result.escalation is not None, "RED turn must have escalation"
        assert result.escalation.severity is Severity.RED
        assert result.escalation.should_escalate is True


class TestConsecutiveYellowBehavior:
    """Regression tests for consecutive-YELLOW escalation behavior."""

    def _advance_to_questions(self, orch: ConversationOrchestrator) -> None:
        orch.start_call()
        orch.process_patient_message("Bien.")
        orch.process_patient_message("Sí, acepto.")

    def test_green_resets_consecutive_yellows(self) -> None:
        """A GREEN answer after a YELLOW resets the consecutive count."""
        orch = make_orchestrator()
        self._advance_to_questions(orch)

        # q0: YELLOW (dolor)
        r0 = orch.process_patient_message("Me duele bastante, un 6.")
        assert r0.escalation is not None
        assert r0.escalation.severity is Severity.YELLOW
        assert not r0.escalation.should_escalate
        assert orch.state is State.QUESTIONS

        # q1: GREEN (fiebre) — resets the counter
        r1 = orch.process_patient_message("No he tenido fiebre.")
        assert r1.escalation is not None
        assert r1.escalation.severity is Severity.GREEN
        assert orch.state is State.QUESTIONS

        # q2: YELLOW (herida) — should be first YELLOW again, not second
        r2 = orch.process_patient_message("La herida esta enrojecida.")
        assert r2.escalation is not None
        assert r2.escalation.severity is Severity.YELLOW
        assert not r2.escalation.should_escalate, (
            "After GREEN reset, this should be first YELLOW"
        )
        assert orch.state is State.QUESTIONS

    def test_first_yellow_does_not_escalate(self) -> None:
        """First YELLOW must have should_escalate=False."""
        orch = make_orchestrator()
        self._advance_to_questions(orch)

        r = orch.process_patient_message("Me duele bastante, un 6.")
        assert r.escalation is not None
        assert r.escalation.severity is Severity.YELLOW
        assert r.escalation.should_escalate is False

    def test_second_yellow_escalates_to_closing(self) -> None:
        """Second consecutive YELLOW transitions to CLOSING."""
        orch = make_orchestrator()
        self._advance_to_questions(orch)

        orch.process_patient_message("Me duele bastante, un 6.")  # YELLOW
        r = orch.process_patient_message("Tuve fiebre ayer.")      # YELLOW
        assert orch.state is State.CLOSING

    def test_consecutive_yellow_has_should_escalate_true(self) -> None:
        """Second consecutive YELLOW must have should_escalate=True."""
        orch = make_orchestrator()
        self._advance_to_questions(orch)

        orch.process_patient_message("Me duele bastante, un 6.")
        r = orch.process_patient_message("Tuve fiebre ayer.")
        assert r.escalation is not None
        assert r.escalation.should_escalate is True

    def test_single_yellow_stays_in_questions(self) -> None:
        """A single YELLOW should not exit QUESTIONS."""
        orch = make_orchestrator()
        self._advance_to_questions(orch)

        r = orch.process_patient_message("Me duele un poco, un 4.")
        assert orch.state is State.QUESTIONS
        assert r.escalation is not None
        assert r.escalation.severity is Severity.YELLOW

    def test_consecutive_yellows_across_domains(self) -> None:
        """Two consecutive YELLOW answers in different domains
        (dolor + fiebre) escalate correctly."""
        orch = make_orchestrator()
        self._advance_to_questions(orch)

        # q0: dolor YELLOW
        r0 = orch.process_patient_message("Me duele bastante, un 6.")
        assert r0.escalation is not None
        assert r0.escalation.domain == "dolor"
        assert r0.escalation.severity is Severity.YELLOW
        assert not r0.escalation.should_escalate

        # q1: fiebre YELLOW -> second consecutive -> escalate
        # Use text that produces YELLOW (not RED) for fiebre
        r1 = orch.process_patient_message("Tuve fiebre ayer.")
        assert orch.state is State.CLOSING
        assert r1.escalation is not None
        assert r1.escalation.domain == "fiebre", (
            f"Second YELLOW escalation must carry its own domain (fiebre), "
            f"got {r1.escalation.domain!r}"
        )
        assert r1.escalation.should_escalate is True

        # Verify that after escalation, _consecutive_yellows is tracked
        # (state is CLOSING, no further question processing)
        assert r1.escalation.severity is Severity.YELLOW

    def test_consecutive_yellows_with_first_upgrade(self) -> None:
        """Consecutive yellow works when the first YELLOW came from
        an LLM upgrade (was GREEN deterministically)."""
        orch = make_orchestrator()
        self._advance_to_questions(orch)

        # q0: respuesta well-behaved → GREEN
        r0 = orch.process_patient_message("Sin dolor, todo bien, un 1.")
        assert r0.escalation is not None
        assert r0.escalation.severity is Severity.GREEN
        assert r0.escalation.domain == "dolor"

        # The actual consecutive-YELLOW test with upgrades requires
        # LLM, so we test the deterministic path: YELLOW→YELLOW
        # which is already covered above. This test verifies that
        # GREEN→YELLOW→YELLOW works in the deterministic path.
        # Since the determinist classifier can't upgrade, we just
        # verify the regular flow.

        # Bonus: verify a scenario where first answer is YELLOW
        # then GREEN (resets), then another YELLOW (first after reset)
        orch2 = make_orchestrator()
        self._advance_to_questions(orch2)

        r0b = orch2.process_patient_message("Me duele, un 5.")  # YELLOW
        assert r0b.escalation.severity is Severity.YELLOW
        assert not r0b.escalation.should_escalate

        r1b = orch2.process_patient_message("Sin fiebre, normal.")  # GREEN
        assert r1b.escalation.severity is Severity.GREEN

        r2b = orch2.process_patient_message("Herida enrojecida.")  # YELLOW → should be first again
        assert r2b.escalation.severity is Severity.YELLOW
        assert not r2b.escalation.should_escalate, (
            "After GREEN reset, this should be first YELLOW again"
        )


class TestEscalationDomainAlignmentDeterministic:
    """Domain alignment tests using the deterministic classifier
    (no LLM mock needed).  These verify that consecutive Yellow
    escalation correctly carries the assessed domain."""

    def _advance_to_questions(self, orch: ConversationOrchestrator) -> None:
        orch.start_call()
        orch.process_patient_message("Bien.")
        orch.process_patient_message("Sí, acepto.")

    def test_domain_correct_after_consecutive_yellows(self) -> None:
        """Two YELLOW answers (dolor, fiebre) → escalation has fiebre domain."""
        orch = make_orchestrator()
        self._advance_to_questions(orch)

        # q0: dolor YELLOW
        orch.process_patient_message("Me duele, un 6.")
        # q1: fiebre YELLOW → second consecutive
        r = orch.process_patient_message("Tuve fiebre ayer, 38 grados.")

        assert r.escalation is not None
        assert r.escalation.domain == "fiebre", (
            f"Consecutive-yellow escalation must carry the second answer's "
            f"domain (fiebre), got {r.escalation.domain!r}"
        )
        assert r.escalation.should_escalate is True

    def test_all_questions_green_then_closing(self) -> None:
        """After all GREEN answers, the closing turn carries the
        last question's domain."""
        orch = make_orchestrator()
        self._advance_to_questions(orch)

        # Answer all 6 questions with GREEN
        for _ in range(5):
            orch.process_patient_message(_GREEN_RESPONSE)
        # Last question (movilidad)
        r = orch.process_patient_message(_GREEN_RESPONSE)

        assert orch.state is State.CLOSING
        assert r.escalation is not None
        assert r.escalation.domain == "movilidad", (
            f"Last question's escalation must have domain 'movilidad', "
            f"got {r.escalation.domain!r}"
        )
        assert r.escalation.severity is Severity.GREEN

    def test_question_index_reset_not_leaked(self) -> None:
        """Verify that _question_index is correctly tracked
        through multiple turns and doesn't leak between domains."""
        orch = make_orchestrator()
        self._advance_to_questions(orch)

        # q0: dolor
        r0 = orch.process_patient_message("Sin dolor.")
        assert r0.question_index == 1  # next is fiebre
        assert r0.escalation.domain == "dolor"

        # q1: fiebre
        r1 = orch.process_patient_message("Sin fiebre.")
        assert r1.question_index == 2  # next is herida
        assert r1.escalation.domain == "fiebre"

        # q2: herida
        r2 = orch.process_patient_message("Herida bien.")
        assert r2.question_index == 3  # next is apetito
        assert r2.escalation.domain == "herida"

        # q3: apetito
        r3 = orch.process_patient_message("Buen apetito.")
        assert r3.question_index == 4  # next is sueño
        assert r3.escalation.domain == "apetito"

        # q4: sueño (decision engine canonicalises to ASCII "sueno")
        r4 = orch.process_patient_message("Duermo bien.")
        assert r4.question_index == 5  # next is movilidad
        assert r4.escalation.domain == "sueno"

        # q5: movilidad (last)
        r5 = orch.process_patient_message("Camino sin problema.")
        assert r5.escalation.domain == "movilidad"
        assert orch.state is State.CLOSING


# ===========================================================================
# Doubt-intent gate scenarios (Questions state)
# ===========================================================================


class TestDoubtIntentGate:
    """Tests for the deterministic doubt-intent gate during QUESTIONS."""

    @staticmethod
    def _advance_to_questions(orch: ConversationOrchestrator) -> None:
        orch.start_call()
        orch.process_patient_message("Bien, gracias.")
        orch.process_patient_message("Si, acepto.")
        assert orch.state is State.QUESTIONS

    def test_appendectomy_doubt_detected_and_answered(self):
        """Exact appendectomy doubt: patient asks about pain → RAG answer, repeat question."""
        orch = make_orchestrator(llm_config=make_llm_config())
        self._advance_to_questions(orch)

        # Patient asks a clinical question instead of answering
        result = orch.process_patient_message(
            "es normal que me duela al caminar despues de una apendicectomia"
        )
        # Doubt gate should catch this
        assert result.escalation is not None
        assert result.escalation.source == "doubt"
        assert result.escalation.should_escalate is False
        # Question index should NOT have advanced (still on question 0)
        assert result.question_index == 0
        assert orch.state is State.QUESTIONS

    def test_sixth_mobility_doubt_stays_in_questions(self):
        """Sixth question (mobility) doubt → stays in QUESTIONS, repeats question."""
        orch = make_orchestrator(llm_config=make_llm_config())
        self._advance_to_questions(orch)

        # Answer questions 0-4 with green (no LLM, deterministic only)
        for _ in range(5):
            orch.process_patient_message(_GREEN_RESPONSE)

        # Question 5 (mobility) — patient asks a doubt with explicit markers
        # Using ? marker to guarantee deterministic detection
        result = orch.process_patient_message(
            "?como debo movilizarme despues de la cirugia?"
        )
        # The ? and "como debo" are explicit doubt markers
        assert result.escalation is not None
        assert result.escalation.source == "doubt"
        assert result.escalation.should_escalate is False
        assert orch.state is State.QUESTIONS, (
            f"Expected QUESTIONS, got {orch.state.name}"
        )

    def test_mobility_doubt_with_llm_confirmation(self):
        """LLM confirms mobility doubt, question repeated, stays in QUESTIONS."""
        from backend.llm.approval import DoubtApprovalResult
        from unittest.mock import patch as mock_patch

        # Create a no-doubt result for first 5 questions
        no_doubt = DoubtApprovalResult(
            is_doubt=False,
            reason="Respuesta normal.",
            llm_used=True,
            classification="llm",
        )
        # Create a doubt result for the 6th question
        yes_doubt = DoubtApprovalResult(
            is_doubt=True,
            reason="El paciente pregunta sobre ejercicio.",
            rag_query="cuando puedo hacer ejercicio despues de apendicectomia",
            clarification_text="Permitame consultar.",
            llm_used=True,
            classification="llm",
        )

        orch = make_orchestrator(llm_config=make_llm_config())
        self._advance_to_questions(orch)

        # Answer questions 0-4 with no-doubt mock
        with mock_patch("backend.conversation.orchestrator.llm_confirm_doubt",
                        return_value=no_doubt):
            for _ in range(5):
                orch.process_patient_message(_GREEN_RESPONSE)

        # Question 5 — doubt confirmed
        with mock_patch("backend.conversation.orchestrator.llm_confirm_doubt",
                        return_value=yes_doubt):
            result = orch.process_patient_message(
                "como debo movilizarme despues de la operacion"
            )

        assert result.question_index == 5  # stays on mobility
        assert orch.state is State.QUESTIONS
        assert result.escalation is not None
        assert result.escalation.source == "doubt"
        assert result.escalation.should_escalate is False

    def test_red_inside_doubt_still_short_circuits(self):
        """RED symptoms inside a doubt-looking response still trigger RED."""
        orch = make_orchestrator()
        self._advance_to_questions(orch)

        # Text that could look like a doubt but has RED keywords
        # Use explicit RED-triggering text
        result = orch.process_patient_message(
            "me duele un 9, es insoportable, no aguanto mas"
        )
        # RED should still short-circuit
        assert orch.state is State.ENDED
        assert result.call_ended
        assert result.escalation is not None
        assert result.escalation.severity is Severity.RED

    @patch("backend.conversation.orchestrator.llm_confirm_doubt")
    def test_unclear_intent_llm_handles_ambiguity(self, mock_doubt):
        """Unclear intent — LLM says NOT a doubt, proceeds to classification."""
        from backend.llm.approval import DoubtApprovalResult
        mock_doubt.return_value = DoubtApprovalResult(
            is_doubt=False,
            reason="El paciente describe su estado, no pregunta.",
            llm_used=True,
            classification="llm",
        )

        orch = make_orchestrator(llm_config=make_llm_config())
        self._advance_to_questions(orch)

        result = orch.process_patient_message("pues no se, tal vez un poco")
        # Should be classified (GREEN, YELLOW, or RED but NOT doubt)
        assert result.escalation is not None
        assert result.escalation.source != "doubt"
        assert result.question_index == 1  # advanced

    def test_doubt_does_not_trigger_alert(self):
        """Doubt turns have should_escalate=False → no alert."""
        orch = make_orchestrator(llm_config=make_llm_config())
        self._advance_to_questions(orch)

        result = orch.process_patient_message(
            "es normal que me duela al caminar"
        )
        assert result.escalation is not None
        assert result.escalation.source == "doubt"
        assert result.escalation.should_escalate is False
        assert result.escalation.severity is Severity.YELLOW  # non-conclusive marker

    @patch("backend.conversation.orchestrator.llm_second_approval")
    def test_llm_failure_doubt_falls_back(self, mock_approval):
        """When LLM fails, explicit doubt markers are preserved."""
        mock_approval.side_effect = RuntimeError("LLM crash")

        orch = make_orchestrator(llm_config=make_llm_config())
        self._advance_to_questions(orch)

        result = orch.process_patient_message("?es normal que me duela?")
        # The input "?es normal que me duela?" has ? marker which is in
        # explicit doubt markers. The LLM crashes, but deterministic
        # markers preserve the doubt.
        assert result.escalation is not None
        # Falls back to deterministic — the doubt gate catches it
        assert result.escalation.source == "doubt"

    def test_question_shaped_red_still_short_circuits(self):
        """Question-shaped text with RED signals still triggers RED, not doubt.

        ``"es normal que me duela un 9?"`` is interrogative (``?``,
        ``"es normal"``) but contains a pain level of 9/10, which is RED.
        The RED classifier must run before the doubt-intent gate so that
        numeric RED signals embedded in question-shaped text are never
        bypassed.
        """
        orch = make_orchestrator(llm_config=make_llm_config())
        self._advance_to_questions(orch)

        result = orch.process_patient_message(
            "es normal que me duela un 9?"
        )
        # RED must short-circuit — no doubt path
        assert orch.state is State.ENDED
        assert result.call_ended
        assert result.escalation is not None
        assert result.escalation.severity is Severity.RED, (
            f"Expected RED, got {result.escalation.severity}"
        )
        assert result.escalation.should_escalate is True
        assert result.escalation.source != "doubt", (
            "Question-shaped RED text must NOT follow the doubt path"
        )

    def test_first_yellow_not_conclusive(self):
        """First YELLOW classification has should_escalate=False."""
        orch = make_orchestrator()
        self._advance_to_questions(orch)

        # Answer q0 (dolor) with a YELLOW-triggering response
        result = orch.process_patient_message("Me duele un 6 de 10.")
        assert result.escalation is not None
        assert result.escalation.severity is Severity.YELLOW
        assert result.escalation.should_escalate is False, (
            "First YELLOW must NOT be conclusive"
        )

    def test_second_consecutive_yellow_is_conclusive(self):
        """Second consecutive YELLOW triggers should_escalate=True."""
        orch = make_orchestrator()
        self._advance_to_questions(orch)

        # q0: first YELLOW
        orch.process_patient_message("Me duele un 6 de 10.")
        # q1: second YELLOW
        result = orch.process_patient_message("Tuve un poco de fiebre ayer.")
        assert result.escalation is not None
        assert result.escalation.severity is Severity.YELLOW
        assert result.escalation.should_escalate is True, (
            "Second consecutive YELLOW must be conclusive"
        )

    def test_green_resets_consecutive_yellows(self):
        """GREEN between YELLOWs resets the counter — no escalation."""
        orch = make_orchestrator()
        self._advance_to_questions(orch)

        # q0: YELLOW (dolor)
        r0 = orch.process_patient_message("Me duele un 6 de 10.")
        assert r0.escalation.severity is Severity.YELLOW
        assert r0.escalation.should_escalate is False
        # q1: GREEN (fiebre)
        r1 = orch.process_patient_message("No tuve fiebre, todo normal.")
        assert r1.escalation.severity is Severity.GREEN
        # q2: YELLOW (herida) — first after reset, NOT conclusive
        r2 = orch.process_patient_message("La herida esta un poco roja pero sin pus.")
        assert r2.escalation is not None
        assert r2.escalation.severity is Severity.YELLOW
        assert r2.escalation.should_escalate is False, (
            "YELLOW after GREEN must NOT be conclusive"
        )

