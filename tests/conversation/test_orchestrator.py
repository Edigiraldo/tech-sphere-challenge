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
from backend.llm.adapter import RagAnswer, RagCitation
from backend.llm.config import LlmConfig
from backend.rag.config import RagConfig


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
            result = orch.process_patient_message(f"Respuesta de prueba {i}")
            if i < _NUM_QUESTIONS - 1:
                # Still in QUESTIONS, asking next question
                expected_idx = i + 1
                assert orch.state is State.QUESTIONS
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
            orch.process_patient_message(f"Respuesta {i}")

        # Last question answer → CLOSING
        result = orch.process_patient_message("Última respuesta")
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

        orch.process_patient_message("Respuesta 0")
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
        for i in range(_NUM_QUESTIONS):
            orch.process_patient_message(f"Respuesta {i}")

        # Now in CLOSING
        assert orch.state is State.CLOSING

        # Patient's final response
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
        for i in range(_NUM_QUESTIONS):
            orch.process_patient_message(f"OK {i}")
        result = orch.process_patient_message("No, gracias.")
        assert "María Test" in result.agent_message
        assert "Apendicectomía" in result.agent_message

    def test_cannot_process_after_ended(self):
        orch = make_orchestrator()
        orch.start_call()
        orch.process_patient_message("Bien.")
        orch.process_patient_message("Sí.")
        for i in range(_NUM_QUESTIONS):
            orch.process_patient_message(f"OK {i}")
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

        # Answer all questions
        for i in range(_NUM_QUESTIONS):
            r = orch.process_patient_message(f"Respuesta para pregunta {i}.")

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
        for i in range(_NUM_QUESTIONS):
            orch.process_patient_message(f"Respuesta {i}")
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


class TestRagLlmIntegration:
    """Orchestrator behaviour with RAG and LLM configs."""

    def _advance_to_questions(self, orch: ConversationOrchestrator):
        orch.start_call()
        orch.process_patient_message("Bien.")
        orch.process_patient_message("Sí, acepto.")

    def test_no_rag_config_produces_fallback(self):
        """Without RAG config, the agent gives a fallback message."""
        orch = make_orchestrator(rag_config=None, llm_config=None)
        self._advance_to_questions(orch)
        result = orch.process_patient_message("Me duele un poco.")
        assert "consulte" in result.agent_message.lower()

    @patch("backend.conversation.orchestrator.retrieve")
    def test_retrieve_called_with_context(self, mock_retrieve):
        """RAG retrieval is called with patient context in query."""
        from backend.rag.retrieval import RetrievalResult

        mock_retrieve.return_value = RetrievalResult(query="", chunks=[])

        rag_config = make_rag_config()
        orch = make_orchestrator(rag_config=rag_config, llm_config=None)
        self._advance_to_questions(orch)
        orch.process_patient_message("Me duele un poco.")

        mock_retrieve.assert_called_once()
        call_query = mock_retrieve.call_args.kwargs["query"]
        assert "Apendicectomía" in call_query
        assert "3" in call_query  # dia_postop

    @patch("backend.conversation.orchestrator.retrieve")
    def test_no_rag_results_produces_fallback(self, mock_retrieve):
        """When RAG returns no chunks, fallback message is used."""
        from backend.rag.retrieval import RetrievalResult

        mock_retrieve.return_value = RetrievalResult(query="", chunks=[])

        rag_config = make_rag_config()
        orch = make_orchestrator(rag_config=rag_config, llm_config=None)
        self._advance_to_questions(orch)
        result = orch.process_patient_message("Me duele un poco.")
        assert "consulte" in result.agent_message.lower()

    @patch("backend.conversation.orchestrator.generate_rag_answer")
    @patch("backend.conversation.orchestrator.retrieve")
    def test_llm_called_with_context(self, mock_retrieve, mock_generate):
        """When RAG returns chunks, LLM is called."""
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
            answer="Gracias por compartir. Su recuperación parece normal.",
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
            model="gemini-1.5-flash",
        )
        mock_generate.return_value = mock_answer

        rag_config = make_rag_config()
        llm_config = LlmConfig(
            model_name="gemini-1.5-flash",
            api_key="fake-key",
        )
        orch = make_orchestrator(
            rag_config=rag_config,
            llm_config=llm_config,
        )
        self._advance_to_questions(orch)
        result = orch.process_patient_message("Me duele un poco, nivel 3.")

        mock_generate.assert_called_once()
        call_query = mock_generate.call_args.kwargs["query"]
        assert "Apendicectomía" in call_query
        # The response should include the LLM answer
        assert "Gracias" in result.agent_message
        # Structured citations should be propagated into the turn
        assert len(result.citations) == 1
        assert result.citations[0]["source_filename"] == "test.pdf"
        assert result.citations[0]["chunk_id"] == "chunk1"

    @patch("backend.conversation.orchestrator.generate_rag_answer")
    @patch("backend.conversation.orchestrator.retrieve")
    def test_insufficient_knowledge_fallback(self, mock_retrieve, mock_generate):
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
            model="gemini-1.5-flash",
        )
        mock_generate.return_value = mock_answer

        rag_config = make_rag_config()
        llm_config = LlmConfig(
            model_name="gemini-1.5-flash",
            api_key="fake-key",
        )
        orch = make_orchestrator(
            rag_config=rag_config,
            llm_config=llm_config,
        )
        self._advance_to_questions(orch)
        result = orch.process_patient_message("Me duele un poco.")
        # The fallback uses "consultar" (infinitive) rather than "consulte"
        # (subjunctive).  Both are valid Spanish for medical advice.
        assert "consult" in result.agent_message.lower()

    @patch("backend.conversation.orchestrator.retrieve")
    def test_retrieve_exception_fallback(self, mock_retrieve):
        """When RAG retrieval raises, orchestrator continues with fallback."""
        mock_retrieve.side_effect = RuntimeError("ChromaDB unavailable")

        rag_config = make_rag_config()
        orch = make_orchestrator(rag_config=rag_config, llm_config=None)
        self._advance_to_questions(orch)
        result = orch.process_patient_message("Me duele un poco.")
        # Should not crash; should give fallback message
        assert "consulte" in result.agent_message.lower()

    @patch("backend.conversation.orchestrator.generate_rag_answer")
    @patch("backend.conversation.orchestrator.retrieve")
    def test_llm_exception_fallback(self, mock_retrieve, mock_generate):
        """When LLM raises, orchestrator continues with fallback."""
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
            model_name="gemini-1.5-flash",
            api_key="fake-key",
        )
        orch = make_orchestrator(
            rag_config=rag_config,
            llm_config=llm_config,
        )
        self._advance_to_questions(orch)
        result = orch.process_patient_message("Me duele un poco.")
        assert "consulte" in result.agent_message.lower()


# ---------------------------------------------------------------------------
# Properties
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
        for i in range(_NUM_QUESTIONS):
            orch.process_patient_message(f"Respuesta {i}")
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
