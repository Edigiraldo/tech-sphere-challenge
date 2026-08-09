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
    "como bien, duermo bien, camino sin problema"
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
        assert "ya ha finalizado" in result.agent_message.lower()




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
            query="", chunks=[chunk]
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
            query="", chunks=[chunk]
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
            query="", chunks=[chunk]
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
