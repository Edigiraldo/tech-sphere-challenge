"""Tests for the summary generator (``backend.summaries.generator``).

Covers:
- ``generate_summary`` with full and empty inputs.
- ``SummaryResult`` model validation.
- ``SummarySection`` and ``SourceReference`` construction.
- Domain heading mapping.
- Escalation decision summaries (GREEN, YELLOW, RED, accumulation).
- Next-steps generation.
- Truncation of very long patient responses.
"""

from __future__ import annotations

import datetime

import pytest

from backend.conversation.context import PatientContext
from backend.data.models import Patient as DataPatient
from backend.decision.models import EscalationResult, Severity
from backend.persistence.sqlite import ConversationTurnRecord
from backend.summaries.generator import generate_summary, _severity_gt
from backend.summaries.models import (
    SourceReference,
    SummaryResult,
    SummarySection,
)


# ======================================================================
# Helpers
# ======================================================================


def make_data_patient(**overrides) -> DataPatient:
    defaults = dict(
        paciente_id="pac_test_001",
        bundle_id="bundle_test",
        synthea_runtime="synthetic_fallback",
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


def make_patient_context(**overrides) -> PatientContext:
    dp = make_data_patient(**overrides)
    return PatientContext(patient=dp, dia_postop=3, procedimiento="Apendicectomia")


def make_turn(
    turn_id: str,
    call_id: str,
    turn_index: int,
    role: str,
    text: str,
    severity: str | None = None,
    domain: str | None = None,
) -> ConversationTurnRecord:
    return ConversationTurnRecord(
        turn_id=turn_id,
        call_id=call_id,
        turn_index=turn_index,
        role=role,
        text=text,
        timestamp=datetime.datetime.now(datetime.timezone.utc),
        severity=severity,
        domain=domain,
    )


def make_escalation_result(
    severity: Severity,
    domain: str | None = None,
    reason: str = "Motivo de prueba",
) -> EscalationResult:
    return EscalationResult(
        severity=severity,
        should_escalate=(severity is Severity.RED),
        reason=reason,
        next_action="Accion recomendada",
        domain=domain,
        source="rule",
    )


def make_source(
    document_id: str = "doc-1",
    source_filename: str = "guia.pdf",
    page_number: int = 3,
) -> SourceReference:
    return SourceReference(
        document_id=document_id,
        source_filename=source_filename,
        page_number=page_number,
    )


# ======================================================================
# Model validation
# ======================================================================


class TestSummarySection:
    """``SummarySection`` construction."""

    def test_minimal_construction(self):
        s = SummarySection(heading="Dolor", content="El paciente reporto dolor leve.")
        assert s.heading == "Dolor"
        assert s.content == "El paciente reporto dolor leve."

    def test_empty_heading_raises(self):
        with pytest.raises(ValueError, match="heading"):
            SummarySection(heading="", content="content")

    def test_empty_content_raises(self):
        with pytest.raises(ValueError, match="content"):
            SummarySection(heading="H", content="   ")

    def test_immutable(self):
        s = SummarySection(heading="H", content="C")
        with pytest.raises(Exception):
            s.heading = "X"  # type: ignore[misc]


class TestSourceReference:
    """``SourceReference`` construction."""

    def test_minimal_construction(self):
        ref = SourceReference(document_id="doc-1", source_filename="guia.pdf", page_number=3)
        assert ref.document_id == "doc-1"
        assert ref.page_number == 3

    def test_empty_document_id_raises(self):
        with pytest.raises(ValueError, match="document_id"):
            SourceReference(document_id="", source_filename="f.pdf")

    def test_negative_page_number_raises(self):
        with pytest.raises(ValueError, match="page_number"):
            SourceReference(document_id="d", source_filename="f.pdf", page_number=-1)


class TestSummaryResult:
    """``SummaryResult`` construction and properties."""

    def test_minimal_construction(self):
        r = SummaryResult(
            summary_id="s1",
            call_id="c1",
            patient_summary=SummarySection(heading="P", content="P"),
            procedure=SummarySection(heading="Pr", content="Pr"),
        )
        assert r.summary_id == "s1"
        assert r.symptoms == []
        assert r.sources == []
        assert not r.has_escalation

    def test_empty_summary_id_raises(self):
        with pytest.raises(ValueError, match="summary_id"):
            SummaryResult(
                summary_id="", call_id="c1",
                patient_summary=SummarySection(heading="P", content="P"),
                procedure=SummarySection(heading="Pr", content="Pr"),
            )

    def test_all_section_headings(self):
        r = SummaryResult(
            summary_id="s1", call_id="c1",
            patient_summary=SummarySection(heading="Paciente", content="P"),
            procedure=SummarySection(heading="Procedimiento", content="Pr"),
            symptoms=[
                SummarySection(heading="Dolor", content="D"),
                SummarySection(heading="Fiebre", content="F"),
            ],
        )
        headings = r.all_section_headings
        assert headings[:4] == ("Paciente", "Procedimiento", "Dolor", "Fiebre")
        assert "escalamiento" in headings[4].lower()
        assert "pasos" in headings[5].lower()

    def test_total_sources(self):
        r = SummaryResult(
            summary_id="s1", call_id="c1",
            patient_summary=SummarySection(heading="P", content="P"),
            procedure=SummarySection(heading="Pr", content="Pr"),
            sources=[
                SourceReference("d1", "f1.pdf", 1),
                SourceReference("d2", "f2.pdf", 2),
            ],
        )
        assert r.total_sources == 2

    def test_has_escalation_true(self):
        r = SummaryResult(
            summary_id="s1", call_id="c1",
            patient_summary=SummarySection(heading="P", content="P"),
            procedure=SummarySection(heading="Pr", content="Pr"),
            decision=SummarySection(
                heading="Dec",
                content="ESCALAMIENTO INMEDIATO (ROJO). Razon: dolor severo.",
            ),
        )
        assert r.has_escalation

    def test_has_escalation_false_default(self):
        r = SummaryResult(
            summary_id="s1", call_id="c1",
            patient_summary=SummarySection(heading="P", content="P"),
            procedure=SummarySection(heading="Pr", content="Pr"),
        )
        assert not r.has_escalation

    def test_invalid_symptom_type_raises(self):
        with pytest.raises(TypeError, match="SummarySection"):
            SummaryResult(
                summary_id="s1", call_id="c1",
                patient_summary=SummarySection(heading="P", content="P"),
                procedure=SummarySection(heading="Pr", content="Pr"),
                symptoms=["not a SummarySection"],  # type: ignore[list-item]
            )


# ======================================================================
# generate_summary — basic and error paths
# ======================================================================


class TestGenerateSummaryBasic:
    """Basic generation and error paths."""

    def test_empty_call_id_raises(self):
        pc = make_patient_context()
        with pytest.raises(ValueError, match="call_id"):
            generate_summary("", pc, [], [], [])

    def test_minimal_input_produces_valid_summary(self):
        pc = make_patient_context()
        result = generate_summary("call-1", pc, [], [], [])
        assert result.call_id == "call-1"
        assert result.patient_summary.heading == "Paciente"
        assert "Maria Test" in result.patient_summary.content
        assert result.procedure.heading == "Procedimiento"
        assert "Apendicectomia" in result.procedure.content
        assert "postoperatorio: 3" in result.procedure.content.lower()
        assert len(result.symptoms) == 6  # all domains present
        assert result.decision.heading == "Decision de escalamiento"
        assert not result.has_escalation

    def test_different_patient(self):
        pc = make_patient_context(
            nombre_completo="Juan Perez",
            edad=45,
            ciudad="Bogota",
            eps="Salud Total EPS",
        )
        result = generate_summary("c1", pc, [], [], [])
        assert "Juan Perez" in result.patient_summary.content
        assert "45 anos" in result.patient_summary.content
        assert "Bogota" in result.patient_summary.content
        assert "Salud Total EPS" in result.patient_summary.content

    def test_different_procedure(self):
        pc = PatientContext(
            patient=make_data_patient(),
            dia_postop=7,
            procedimiento="Colecistectomia",
        )
        result = generate_summary("c1", pc, [], [], [])
        assert "Colecistectomia" in result.procedure.content
        assert "postoperatorio: 7" in result.procedure.content.lower()


# ======================================================================
# generate_summary — symptom sections from turns
# ======================================================================


class TestGenerateSummarySymptoms:
    """Symptom section generation from turns."""

    def test_patient_turns_mapped_to_domains(self):
        pc = make_patient_context()
        turns = [
            make_turn("t0", "c1", 0, "AGENT", "Pregunta 1"),
            make_turn("t1", "c1", 1, "PATIENT", "Me duele un poco, nivel 3"),
            make_turn("t2", "c1", 2, "AGENT", "Pregunta 2"),
            make_turn("t3", "c1", 3, "PATIENT", "No he tenido fiebre"),
            make_turn("t4", "c1", 4, "AGENT", "Pregunta 3"),
            make_turn("t5", "c1", 5, "PATIENT", "La herida esta bien"),
        ]
        result = generate_summary("c1", pc, turns, [], [])
        # First symptom domain (dolor)
        assert "nivel 3" in result.symptoms[0].content.lower()
        # Second (fiebre)
        assert "fiebre" in result.symptoms[1].content.lower()
        # Third (herida)
        assert "herida" in result.symptoms[2].content.lower()

    def test_empty_turns_produces_placeholder(self):
        pc = make_patient_context()
        result = generate_summary("c1", pc, [], [], [])
        # All six domains should have placeholder text
        for s in result.symptoms:
            assert "no se registro" in s.content.lower()

    def test_extra_patient_turns_beyond_six_not_mapped(self):
        pc = make_patient_context()
        turns = [make_turn(f"t{i}", "c1", i, "PATIENT", f"Respuesta {i}")
                 for i in range(10)]
        result = generate_summary("c1", pc, turns, [], [])
        # Only first 6 patient turns are mapped to domains
        for i in range(6):
            assert f"Respuesta {i}" in result.symptoms[i].content
        # The extra turns are simply not mapped

    def test_long_patient_response_truncated(self):
        pc = make_patient_context()
        long_text = "x" * 500
        turns = [make_turn("t0", "c1", 0, "PATIENT", long_text)]
        result = generate_summary("c1", pc, turns, [], [])
        content = result.symptoms[0].content
        assert len(content) < 400  # truncated + annotation
        assert "..." in content

    def test_severity_annotation_added(self):
        pc = make_patient_context()
        turns = [
            make_turn("t0", "c1", 0, "PATIENT", "Dolor intenso nivel 9"),
        ]
        esc_results = [
            make_escalation_result(Severity.RED, domain="dolor", reason="Dolor severo"),
        ]
        result = generate_summary("c1", pc, turns, esc_results, [])
        assert "ALERTA ROJA" in result.symptoms[0].content

    def test_yellow_severity_annotation(self):
        pc = make_patient_context()
        turns = [
            make_turn("t0", "c1", 0, "PATIENT", "Me duele moderado"),
        ]
        esc_results = [
            make_escalation_result(Severity.YELLOW, domain="dolor", reason="Dolor moderado"),
        ]
        result = generate_summary("c1", pc, turns, esc_results, [])
        assert "ALERTA AMARILLA" in result.symptoms[0].content


# ======================================================================
# generate_summary — decision section
# ======================================================================


class TestGenerateSummaryDecision:
    """Decision section generation."""

    def test_no_escalation_results(self):
        pc = make_patient_context()
        result = generate_summary("c1", pc, [], [], [])
        assert "no se requirio escalamiento" in result.decision.content.lower()

    def test_green_only(self):
        pc = make_patient_context()
        esc = [make_escalation_result(Severity.GREEN, domain="dolor", reason="Normal")]
        result = generate_summary("c1", pc, [], esc, [])
        assert "VERDE" in result.decision.content
        assert not result.has_escalation

    def test_red_escalation(self):
        pc = make_patient_context()
        esc = [make_escalation_result(Severity.RED, domain="dolor", reason="Dolor NRS 9")]
        result = generate_summary("c1", pc, [], esc, [])
        assert "ROJO" in result.decision.content
        assert "Dolor NRS 9" in result.decision.content
        assert result.has_escalation

    def test_yellow_accumulation(self):
        pc = make_patient_context()
        esc = [
            make_escalation_result(Severity.YELLOW, domain="dolor", reason="Dolor moderado"),
            make_escalation_result(Severity.YELLOW, domain="fiebre", reason="Temp 37.9"),
        ]
        result = generate_summary("c1", pc, [], esc, [])
        assert "ACUMULACION" in result.decision.content
        assert "AMARILLO x2" in result.decision.content
        assert result.has_escalation

    def test_single_yellow(self):
        pc = make_patient_context()
        esc = [make_escalation_result(Severity.YELLOW, domain="herida", reason="Enrojecimiento")]
        result = generate_summary("c1", pc, [], esc, [])
        assert "PRECAUCION" in result.decision.content
        assert "Enrojecimiento" in result.decision.content
        assert result.has_escalation

    def test_red_takes_precedence_over_yellow(self):
        pc = make_patient_context()
        esc = [
            make_escalation_result(Severity.YELLOW, domain="dolor", reason="Moderado"),
            make_escalation_result(Severity.RED, domain="fiebre", reason="Temp 39.0"),
        ]
        result = generate_summary("c1", pc, [], esc, [])
        assert "ROJO" in result.decision.content
        assert "Temp 39.0" in result.decision.content


# ======================================================================
# generate_summary — next steps
# ======================================================================


class TestGenerateSummaryNextSteps:
    """Next-steps section generation."""

    def test_no_escalation(self):
        pc = make_patient_context()
        result = generate_summary("c1", pc, [], [], [])
        assert "seguimiento" in result.next_steps.content.lower()
        assert "programado" in result.next_steps.content.lower()

    def test_red_next_steps(self):
        pc = make_patient_context()
        esc = [make_escalation_result(Severity.RED, domain="dolor")]
        result = generate_summary("c1", pc, [], esc, [])
        assert "Transferir" in result.next_steps.content
        assert "EPS" in result.next_steps.content
        assert "24 horas" in result.next_steps.content

    def test_yellow_next_steps(self):
        pc = make_patient_context()
        esc = [make_escalation_result(Severity.YELLOW, domain="dolor")]
        result = generate_summary("c1", pc, [], esc, [])
        assert "48 horas" in result.next_steps.content
        assert "urgencias" in result.next_steps.content.lower()

    def test_green_next_steps(self):
        pc = make_patient_context()
        esc = [make_escalation_result(Severity.GREEN, domain="dolor")]
        result = generate_summary("c1", pc, [], esc, [])
        assert "programado" in result.next_steps.content.lower()


# ======================================================================
# generate_summary — sources
# ======================================================================


class TestGenerateSummarySources:
    """Source references preserved in the output."""

    def test_sources_preserved(self):
        pc = make_patient_context()
        sources = [
            make_source("doc-1", "guia_apendicectomia.pdf", 3),
            make_source("doc-2", "cuidados_postop.pdf", 7),
        ]
        result = generate_summary("c1", pc, [], [], sources)
        assert result.total_sources == 2
        assert result.sources[0].document_id == "doc-1"
        assert result.sources[1].source_filename == "cuidados_postop.pdf"


# ======================================================================
# Helper functions
# ======================================================================


class TestSeverityGt:
    """``_severity_gt`` ordering."""

    def test_red_gt_yellow(self):
        assert _severity_gt("RED", "YELLOW")

    def test_red_gt_green(self):
        assert _severity_gt("RED", "GREEN")

    def test_yellow_gt_green(self):
        assert _severity_gt("YELLOW", "GREEN")

    def test_yellow_not_gt_red(self):
        assert not _severity_gt("YELLOW", "RED")

    def test_green_not_gt_anything(self):
        assert not _severity_gt("GREEN", "YELLOW")
        assert not _severity_gt("GREEN", "RED")

    def test_same_not_gt(self):
        assert not _severity_gt("RED", "RED")
        assert not _severity_gt("YELLOW", "YELLOW")

    def test_unknown_low(self):
        assert not _severity_gt("BANANA", "GREEN")


# ======================================================================
# generate_summary — UUID stability / determinism
# ======================================================================


class TestGenerateSummaryDeterminism:
    """Summary generation is deterministic given the same inputs."""

    def test_same_inputs_same_output(self):
        pc = make_patient_context()
        turns = [
            make_turn("t0", "c1", 0, "PATIENT", "Dolor nivel 3"),
            make_turn("t1", "c1", 1, "PATIENT", "Sin fiebre"),
        ]
        esc = [make_escalation_result(Severity.GREEN, domain="dolor")]
        sources = [make_source()]

        result1 = generate_summary("c1", pc, turns, esc, sources)
        result2 = generate_summary("c1", pc, turns, esc, sources)

        # Content is the same (summary_id differs — UUIDs)
        assert result1.patient_summary == result2.patient_summary
        assert result1.decision.content == result2.decision.content
        assert result1.next_steps.content == result2.next_steps.content
        # Each symptom section matches
        for s1, s2 in zip(result1.symptoms, result2.symptoms):
            assert s1.content == s2.content

    def test_different_call_id_different_summary_id(self):
        pc = make_patient_context()
        r1 = generate_summary("c1", pc, [], [], [])
        r2 = generate_summary("c2", pc, [], [], [])
        assert r1.call_id != r2.call_id
        assert r1.summary_id != r2.summary_id
