"""Tests for backend.data.models — dataclass shapes and immutability."""

import datetime
from pathlib import Path

import pytest

from backend.data.models import (
    Conversation,
    ConversationTurn,
    PDFReference,
    Patient,
    Trajectory,
)


class TestPatientModel:
    def test_minimal_construction(self):
        p = Patient(
            paciente_id="pac_42_00000",
            bundle_id="bundle_001",
            synthea_runtime="synthetic_fallback",
            modulo_synthea="appendicitis",
            procedimiento="Apendicectomía",
            fecha_cirugia=datetime.date(2026, 6, 14),
            edad=34,
            genero="F",
            comorbilidades=[],
            complicacion_encounter=False,
            nombre_completo="Mauricio",
            direccion="Calle 1",
            ciudad="Soacha",
            departamento="Cundinamarca",
            documento_cc="944082010",
            eps="Compensar EPS",
            source_country="US",
            adapted_country="CO",
            adaptation_fields=[],
        )
        assert p.paciente_id == "pac_42_00000"
        assert p.edad == 34
        assert p.modulo_synthea == "appendicitis"

    def test_comorbilidades_preserved(self):
        p = Patient(
            paciente_id="x",
            bundle_id="b",
            synthea_runtime="s",
            modulo_synthea="m",
            procedimiento="p",
            fecha_cirugia=datetime.date(2026, 1, 1),
            edad=50,
            genero="M",
            comorbilidades=["hipertension", "diabetes"],
            complicacion_encounter=True,
            nombre_completo="n",
            direccion="d",
            ciudad="c",
            departamento="dp",
            documento_cc="1",
            eps="e",
            source_country="US",
            adapted_country="CO",
            adaptation_fields=["nombre_completo"],
        )
        assert p.comorbilidades == ["hipertension", "diabetes"]

    def test_immutable(self):
        p = Patient(
            paciente_id="x",
            bundle_id="b",
            synthea_runtime="s",
            modulo_synthea="m",
            procedimiento="p",
            fecha_cirugia=datetime.date(2026, 1, 1),
            edad=1,
            genero="M",
            comorbilidades=[],
            complicacion_encounter=False,
            nombre_completo="n",
            direccion="d",
            ciudad="c",
            departamento="dp",
            documento_cc="1",
            eps="e",
            source_country="US",
            adapted_country="CO",
            adaptation_fields=[],
        )
        with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
            p.edad = 99  # type: ignore[misc]


class TestTrajectoryModel:
    def test_construction(self):
        t = Trajectory(
            trayectoria_id="tray_001",
            paciente_id="pac_001",
            dia_postop=1,
            arquetipo_trayectoria="normal",
            dolor_nrs=2,
            fiebre_c=37.0,
            movilidad="normal",
            herida="normal",
            apetito="normal",
            sueno="normal",
            seed=42,
        )
        assert t.dolor_nrs == 2
        assert t.fiebre_c == 37.0


class TestConversationTurnModel:
    def test_runtime_safe_default(self):
        """A turn constructed without label_ground_truth has None."""
        turn = ConversationTurn(
            dialogo_id="dlg_001",
            caso_id="caso_001",
            paciente_id="pac_001",
            dia_postop=1,
            turno_idx=0,
            hablante="agente",
            texto="Hello",
        )
        assert turn.label_ground_truth is None

    def test_evaluation_path(self):
        turn = ConversationTurn(
            dialogo_id="dlg_001",
            caso_id="caso_001",
            paciente_id="pac_001",
            dia_postop=1,
            turno_idx=0,
            hablante="agente",
            texto="Hello",
            label_ground_truth="verde",
        )
        assert turn.label_ground_truth == "verde"


class TestConversationModel:
    def test_with_turns(self):
        turns = [
            ConversationTurn(
                dialogo_id="d_0", caso_id="c1", paciente_id="p1",
                dia_postop=1, turno_idx=0, hablante="agente", texto="Hola",
            ),
            ConversationTurn(
                dialogo_id="d_1", caso_id="c1", paciente_id="p1",
                dia_postop=1, turno_idx=1, hablante="paciente", texto="Bien",
            ),
        ]
        conv = Conversation(caso_id="c1", paciente_id="p1", dia_postop=1, turns=turns)
        assert len(conv.turns) == 2
        assert conv.turns[0].hablante == "agente"


class TestPDFReferenceModel:
    def test_construction(self):
        ref = PDFReference(
            procedure="appendicitis",
            filename="Apendicitis.pdf",
            path=Path("/fake/dataset/textos/Appendicitis/Apendicitis.pdf"),
        )
        assert ref.procedure == "appendicitis"
        assert ref.filename == "Apendicitis.pdf"
        assert ref.path.suffix == ".pdf"
