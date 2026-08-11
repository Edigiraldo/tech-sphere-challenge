"""Live scenario integration tests for LLM second-approval.

Runs 6 sequential full-call scenarios exercising all LLM approval paths:
  1. GREEN confirmation
  2. YELLOW escalation (two consecutive)
  3. RED short-circuit
  4. Disagreement upgrade (LLM upgrades severity)
  5. Clarification/retry
  6. Doubt/RAG (including final-question behavior)

Uses mocked Groq calls to simulate LLM approval responses while exercising
the complete orchestrator flow from IDLE to ENDED/CLOSING.
"""

from __future__ import annotations

import datetime
import time
from unittest.mock import patch

from backend.conversation.context import PatientContext
from backend.conversation.orchestrator import (
    _NUM_QUESTIONS,
    ConversationOrchestrator,
)
from backend.conversation.state import State
from backend.data.models import Patient as DataPatient
from backend.decision import Severity
from backend.llm.approval import LlmApprovalResult
from backend.llm.config import LlmConfig
from backend.rag.config import RagConfig


def make_data_patient(**overrides) -> DataPatient:
    defaults = dict(
        paciente_id="pac_live_001",
        bundle_id="bundle_test",
        synthea_runtime="synthetic",
        modulo_synthea="appendicitis",
        procedimiento="Apendicectomia",
        fecha_cirugia=datetime.date(2026, 6, 14),
        edad=34,
        genero="F",
        comorbilidades=[],
        complicacion_encounter=False,
        nombre_completo="Maria Test",
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


def make_llm_config() -> LlmConfig:
    return LlmConfig(
        provider="groq",
        model_name="llama-3.1-70b-versatile",
        api_key="test-key",
        temperature=0.2,
        max_output_tokens=512,
    )


def make_rag_config() -> RagConfig:
    return RagConfig(
        embedding_model="BAAI/bge-m3",
        chroma_persist_dir=datetime.__file__,
        collection_name="test_collection",
        chunk_size=800,
        chunk_overlap=150,
        retrieval_top_k=5,
        similarity_threshold=0.0,
    )


def _advance_to_questions(orch: ConversationOrchestrator):
    """Advance the orchestrator past greeting and consent to QUESTIONS."""
    orch.start_call()
    orch.process_patient_message("Bien, gracias.")
    orch.process_patient_message("Si, acepto.")
    assert orch.state is State.QUESTIONS, f"Expected QUESTIONS, got {orch.state.name}"


def green_reply() -> str:
    return "Todo bien, sin dolor, sin fiebre, herida limpia, como bien, duermo bien, camino sin problema"


def _green_approval() -> LlmApprovalResult:
    return LlmApprovalResult(
        severity=Severity.GREEN,
        should_escalate=False,
        reason="Concuerdo, evoluciona favorablemente.",
        next_action="Continuar seguimiento normal.",
        action="confirm",
        llm_used=True,
        llm_duration_ms=100.0,
        prompt_tokens=80,
        completion_tokens=40,
    )


def _yellow_approval() -> LlmApprovalResult:
    return LlmApprovalResult(
        severity=Severity.YELLOW,
        should_escalate=False,
        reason="Sintoma amarillo confirmado.",
        next_action="Monitorear de cerca.",
        action="confirm",
        llm_used=True,
        llm_duration_ms=100.0,
        prompt_tokens=80,
        completion_tokens=40,
    )


def _red_upgrade_approval() -> LlmApprovalResult:
    return LlmApprovalResult(
        severity=Severity.RED,
        should_escalate=True,
        reason="Senal de alerta detectada por el revisor LLM.",
        next_action="Transferir urgente.",
        action="escalate",
        llm_used=True,
        llm_duration_ms=120.0,
        prompt_tokens=90,
        completion_tokens=45,
    )


def _clarification_approval(question: str) -> LlmApprovalResult:
    return LlmApprovalResult(
        severity=Severity.YELLOW,
        should_escalate=False,
        reason="Respuesta ambigua, necesito aclaracion.",
        next_action="Solicitar aclaracion.",
        action="request_clarification",
        clarification_question=question,
        llm_used=True,
        llm_duration_ms=110.0,
        prompt_tokens=85,
        completion_tokens=50,
    )


def _rag_approval(query: str) -> LlmApprovalResult:
    return LlmApprovalResult(
        severity=Severity.YELLOW,
        should_escalate=False,
        reason="Duda clinica, necesito informacion adicional.",
        next_action="Consultar fuentes clinicas.",
        action="request_rag",
        rag_query=query,
        llm_used=True,
        llm_duration_ms=150.0,
        prompt_tokens=95,
        completion_tokens=55,
    )


def _print_turn(idx: int, state: str, question_idx, severity_str: str, agent_msg: str):
    qi = question_idx if question_idx is not None else "-"
    sev = severity_str if severity_str else "-"
    short = agent_msg[:100].replace("\n", " ")
    print(f"  Turn {idx:2d} | state={state:12s} | q={qi} | sev={sev:8s} | {short}...")


# =============================================================================
# Scenario 1: GREEN confirmation (full happy path)
# =============================================================================

@patch("backend.conversation.orchestrator.llm_second_approval")
def test_scenario_1_green_confirmation(mock_approval):
    """All 6 questions answer GREEN; LLM confirms every one."""
    print("\n" + "=" * 70)
    print("  SCENARIO 1: GREEN confirmation (LLM confirms all GREEN)")
    print("=" * 70)

    mock_approval.return_value = _green_approval()

    orch = ConversationOrchestrator(
        PatientContext(
            patient=make_data_patient(),
            dia_postop=3,
            procedimiento="Apendicectomia",
        ),
        rag_config=None,
        llm_config=make_llm_config(),
    )

    _advance_to_questions(orch)

    for i in range(_NUM_QUESTIONS):
        result = orch.process_patient_message("Todo bien con eso. Sin problemas.")
        sev = result.escalation.severity.value if result.escalation else "N/A"
        _print_turn(i + 1, result.state.value, result.question_index, sev, result.agent_message)

        if i < _NUM_QUESTIONS - 1:
            assert result.escalation is not None
            assert result.escalation.severity is Severity.GREEN, (
                f"Question {i}: expected GREEN, got {result.escalation.severity}"
            )

    # After all questions -> CLOSING
    assert orch.state is State.CLOSING
    # End the call
    final = orch.process_patient_message("No, gracias, todo claro.")
    assert orch.state is State.ENDED
    assert final.call_ended
    _print_turn(_NUM_QUESTIONS + 1, final.state.value, None, "-", final.agent_message)

    print(f"\n  PASS Scenario 1 -- {_NUM_QUESTIONS} questions, all GREEN, call ended")


# =============================================================================
# Scenario 2: YELLOW escalation (two consecutive)
# =============================================================================

@patch("backend.conversation.orchestrator.llm_second_approval")
def test_scenario_2_yellow_escalation(mock_approval):
    """Two consecutive YELLOW -> escalation to CLOSING with should_escalate=True."""
    print("\n" + "=" * 70)
    print("  SCENARIO 2: YELLOW escalation (two consecutive -> CLOSING)")
    print("=" * 70)

    # First two questions -> GREEN, then two YELLOW -> escalation
    mock_approval.side_effect = [
        _green_approval(),   # q0 dolor -> GREEN
        _green_approval(),   # q1 fiebre -> GREEN
        _yellow_approval(),  # q2 herida -> YELLOW (first)
        _yellow_approval(),  # q3 apetito -> YELLOW (second -> escalate!)
    ]

    orch = ConversationOrchestrator(
        PatientContext(patient=make_data_patient(), dia_postop=3, procedimiento="Apendicectomia"),
        rag_config=None,
        llm_config=make_llm_config(),
    )
    _advance_to_questions(orch)

    # q0: GREEN
    r = orch.process_patient_message("Sin dolor, todo bien.")
    assert r.escalation.severity is Severity.GREEN
    _print_turn(1, r.state.value, r.question_index, "GREEN", r.agent_message)

    # q1: GREEN
    r = orch.process_patient_message("Sin fiebre.")
    assert r.escalation.severity is Severity.GREEN
    _print_turn(2, r.state.value, r.question_index, "GREEN", r.agent_message)

    # q2: YELLOW (first)
    r = orch.process_patient_message("La herida esta enrojecida.")
    assert r.escalation.severity is Severity.YELLOW
    assert not r.escalation.should_escalate
    _print_turn(3, r.state.value, r.question_index, "YELLOW", r.agent_message)

    # q3: YELLOW (second) -> escalation
    r = orch.process_patient_message("No tengo hambre, me cuesta comer.")
    assert orch.state is State.CLOSING
    assert r.escalation is not None
    assert r.escalation.severity is Severity.YELLOW
    assert r.escalation.should_escalate is True, "Second YELLOW must trigger escalation!"
    _print_turn(4, r.state.value, r.question_index, "YELLOW*", r.agent_message)

    print("\n  PASS Scenario 2 -- YELLOW accumulated, escalation triggered")


# =============================================================================
# Scenario 3: RED short-circuit (deterministic RED -> no LLM call)
# =============================================================================

@patch("backend.conversation.orchestrator.llm_second_approval")
def test_scenario_3_red_short_circuit(mock_approval):
    """RED from deterministic classifier bypasses LLM entirely."""
    print("\n" + "=" * 70)
    print("  SCENARIO 3: RED short-circuit (deterministic RED -> ENDED, no LLM)")
    print("=" * 70)

    orch = ConversationOrchestrator(
        PatientContext(patient=make_data_patient(), dia_postop=3, procedimiento="Apendicectomia"),
        rag_config=None,
        llm_config=make_llm_config(),
    )
    _advance_to_questions(orch)

    # RED answer to q0 (dolor)
    r = orch.process_patient_message("Me duele un 9, es insoportable, no aguanto mas.")
    assert orch.state is State.ENDED
    assert r.call_ended
    assert r.escalation.severity is Severity.RED
    _print_turn(1, r.state.value, r.question_index, "RED", r.agent_message)

    # LLM approval should NOT have been called for RED
    mock_approval.assert_not_called()

    print("\n  PASS Scenario 3 -- RED bypasses LLM, call ended immediately")


# =============================================================================
# Scenario 4: Disagreement upgrade (LLM detects danger classifier missed)
# =============================================================================

@patch("backend.conversation.orchestrator.llm_second_approval")
def test_scenario_4_disagreement_upgrade(mock_approval):
    """LLM detects red flags the deterministic classifier missed -> upgrades to RED."""
    print("\n" + "=" * 70)
    print("  SCENARIO 4: Disagreement upgrade (LLM detects danger -> RED)")
    print("=" * 70)

    # q0: GREEN, q1: LLM upgrades to RED
    mock_approval.side_effect = [
        _green_approval(),       # q0 dolor -> GREEN
        _red_upgrade_approval(), # q1 fiebre -> LLM upgrades to RED
    ]

    orch = ConversationOrchestrator(
        PatientContext(patient=make_data_patient(), dia_postop=3, procedimiento="Apendicectomia"),
        rag_config=None,
        llm_config=make_llm_config(),
    )
    _advance_to_questions(orch)

    # q0: GREEN
    r = orch.process_patient_message("Sin dolor, todo bien.")
    _print_turn(1, r.state.value, r.question_index, "GREEN", r.agent_message)

    # q1: deterministic might be GREEN but LLM upgrades to RED
    r = orch.process_patient_message("Tuve fiebre de 40 grados con escalofrios y confusion.")
    assert orch.state is State.ENDED
    assert r.call_ended
    assert r.escalation is not None
    assert r.escalation.severity is Severity.RED, (
        f"Expected RED from LLM upgrade, got {r.escalation.severity}"
    )
    _print_turn(2, r.state.value, r.question_index, "RED-up", r.agent_message)

    print("\n  PASS Scenario 4 -- LLM upgraded severity, call ended")


# =============================================================================
# Scenario 5: Clarification/retry (LLM requests one clarification)
# =============================================================================

@patch("backend.conversation.orchestrator.llm_second_approval")
def test_scenario_5_clarification_retry(mock_approval):
    """LLM requests clarification -> stays on same question, patient answers again."""
    print("\n" + "=" * 70)
    print("  SCENARIO 5: Clarification/retry (LLM requests clarification -> same domain)")
    print("=" * 70)

    orch = ConversationOrchestrator(
        PatientContext(patient=make_data_patient(), dia_postop=3, procedimiento="Apendicectomia"),
        rag_config=None,
        llm_config=make_llm_config(),
    )
    _advance_to_questions(orch)

    # q0: GREEN (use direct mock)
    mock_approval.return_value = _green_approval()
    r = orch.process_patient_message("Sin dolor.")
    _print_turn(1, r.state.value, r.question_index, "GREEN", r.agent_message)

    # q1 (fiebre): LLM requests clarification
    mock_approval.return_value = _clarification_approval(
        "Podria decirme en una escala del 0 al 10 que tan fuerte es su dolor?"
    )
    r = orch.process_patient_message("Pues, mas o menos, a veces.")
    assert r.question_index == 1  # still on question 1!
    assert orch.state is State.QUESTIONS
    assert "?" in r.agent_message
    _print_turn(2, r.state.value, r.question_index, "CLARIFY", r.agent_message)

    # Patient clarifies -> advances normally
    mock_approval.return_value = _green_approval()
    r = orch.process_patient_message("No he tenido fiebre, todo normal.")
    assert r.question_index == 2  # advanced to question 2
    _print_turn(3, r.state.value, r.question_index, "GREEN", r.agent_message)

    # Process remaining questions (q2 through q5 = 4 questions)
    for _ in range(4):
        r = orch.process_patient_message(green_reply())
        qi = r.question_index
        st = r.state.value
        _print_turn(qi + 1 if qi else 99, st, qi, "GREEN", r.agent_message)

    assert orch.state is State.CLOSING, f"Expected CLOSING, got {orch.state.name}"

    # End the call
    r = orch.process_patient_message("Gracias, nada mas.")
    assert orch.state is State.ENDED
    _print_turn(99, r.state.value, None, "-", r.agent_message)

    print("\n  PASS Scenario 5 -- clarification requested, patient answered, continued")


# =============================================================================
# Scenario 6: Doubt/RAG (LLM requests RAG, including final-question behavior)
# =============================================================================

@patch("backend.conversation.orchestrator.retrieve")
@patch("backend.conversation.orchestrator.generate_rag_answer")
@patch("backend.conversation.orchestrator.llm_second_approval")
def test_scenario_6_doubt_rag(mock_approval, mock_generate, mock_retrieve):
    """LLM requests RAG for doubt -> RAG runs, answer included, call continues.
    Final question (q5) with RAG proceeds to CLOSING."""
    print("\n" + "=" * 70)
    print("  SCENARIO 6: Doubt/RAG (LLM requests RAG -> includes answer, final -> CLOSING)")
    print("=" * 70)

    from backend.llm.adapter import RagAnswer, RagCitation
    from backend.rag.retrieval import RetrievedChunk, RetrievalResult

    orch = ConversationOrchestrator(
        PatientContext(patient=make_data_patient(), dia_postop=3, procedimiento="Apendicectomia"),
        rag_config=make_rag_config(),
        llm_config=make_llm_config(),
    )
    _advance_to_questions(orch)

    # RAG returns chunks
    mock_retrieve.return_value = RetrievalResult(
        chunks=[
            RetrievedChunk(
                chunk_id="rag-chunk-1",
                document_id="doc-1",
                source_filename="guia_postop.pdf",
                chunk_index=0,
                page_number=3,
                text="Es normal tener dificultad para dormir los primeros dias postoperatorios.",
                similarity=0.72,
            ),
        ],
        query="test",
        sufficient=True,
    )

    mock_generate.return_value = RagAnswer(
        answer="Segun la guia clinica, es normal tener alteraciones del sueno en los primeros dias despues de la cirugia.",
        citations=[
            RagCitation(
                chunk_id="rag-chunk-1",
                document_id="doc-1",
                source_filename="guia_postop.pdf",
                page_number=3,
            )
        ],
        insufficient_knowledge=False,
        model="llama-3.1-70b-versatile",
        llm_duration_ms=200.0,
        prompt_tokens=100,
        completion_tokens=60,
    )

    # q0-q3: GREEN
    mock_approval.return_value = _green_approval()
    for i in range(4):
        r = orch.process_patient_message(green_reply())
        _print_turn(i + 1, r.state.value, r.question_index, "GREEN", r.agent_message)

    # q4: RAG doubt (sueno)
    mock_approval.return_value = _rag_approval(
        "Es normal dificultad para dormir despues de apendicectomia?"
    )
    r = orch.process_patient_message("No he podido dormir bien, me despierto mucho.")
    assert r.question_index == 5  # advanced to q5 after RAG
    assert r.citations  # citations from RAG
    _print_turn(5, r.state.value, r.question_index, "RAG", r.agent_message)

    # q5: RAG doubt on last question (movilidad) -> should go to CLOSING
    mock_approval.return_value = _rag_approval(
        "Es normal mareo al caminar despues de apendicectomia?"
    )
    mock_retrieve.return_value = RetrievalResult(
        chunks=[
            RetrievedChunk(
                chunk_id="rag-chunk-2",
                document_id="doc-2",
                source_filename="movilidad.pdf",
                chunk_index=0,
                page_number=1,
                text="El mareo leve al movilizarse es comun.",
                similarity=0.68,
            ),
        ],
        query="test",
        sufficient=True,
    )
    mock_generate.return_value = RagAnswer(
        answer="El mareo leve al movilizarse es comun en los primeros dias postoperatorios.",
        citations=[
            RagCitation(
                chunk_id="rag-chunk-2",
                document_id="doc-2",
                source_filename="movilidad.pdf",
                page_number=1,
            )
        ],
        insufficient_knowledge=False,
        model="llama-3.1-70b-versatile",
    )

    r = orch.process_patient_message("Me mareo un poco al caminar.")
    assert orch.state is State.CLOSING, (
        f"Expected CLOSING after last question, got {orch.state.name}"
    )
    assert r.state is State.CLOSING
    _print_turn(6, r.state.value, r.question_index, "RAG->CL", r.agent_message)

    # End the call from CLOSING
    r = orch.process_patient_message("Gracias, todo claro.")
    assert orch.state is State.ENDED
    _print_turn(7, r.state.value, None, "-", r.agent_message)

    print("\n  PASS Scenario 6 -- RAG executed in QUESTIONS, final -> CLOSING")


# =============================================================================
# Main runner
# =============================================================================

def run_all_scenarios():
    """Run all 6 sequential full-call live scenarios."""
    print("\n")
    print("=" * 70)
    print("     LLM SECOND-APPROVAL -- 6 LIVE SCENARIOS")
    print("=" * 70)
    start = time.time()

    results: dict[str, bool] = {}

    scenarios = [
        ("GREEN confirmation", test_scenario_1_green_confirmation),
        ("YELLOW escalation", test_scenario_2_yellow_escalation),
        ("RED short-circuit", test_scenario_3_red_short_circuit),
        ("Disagreement upgrade", test_scenario_4_disagreement_upgrade),
        ("Clarification/retry", test_scenario_5_clarification_retry),
        ("Doubt/RAG (incl. final)", test_scenario_6_doubt_rag),
    ]

    for name, func in scenarios:
        try:
            func()
            results[name] = True
        except Exception as exc:
            print(f"\n  FAIL {name}: {exc}")
            import traceback
            traceback.print_exc()
            results[name] = False

    elapsed = time.time() - start
    print(f"\n{'=' * 70}")
    print(f"  RESULTS ({elapsed:.2f}s):")
    for name, passed in results.items():
        icon = "PASS" if passed else "FAIL"
        print(f"    {icon}  {name}")
    all_passed = all(results.values())
    print(f"\n  {'ALL SCENARIOS PASSED' if all_passed else 'SOME SCENARIOS FAILED'}")
    print(f"{'=' * 70}\n")
    return all_passed


if __name__ == "__main__":
    success = run_all_scenarios()
    exit(0 if success else 1)
