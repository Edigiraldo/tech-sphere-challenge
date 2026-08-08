"""Tests for ``backend.conversation.context`` — PatientContext and CallContext."""

import datetime

import pytest

from backend.conversation.context import CallContext, PatientContext
from backend.conversation.messages import History
from backend.conversation.state import State
from backend.data.models import Patient as DataPatient


# ---------------------------------------------------------------------------
# Helper — minimal valid DataPatient
# ---------------------------------------------------------------------------

def make_data_patient(**overrides) -> DataPatient:
    """Build a valid ``backend.data.models.Patient`` with sensible defaults."""
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


# ---------------------------------------------------------------------------
# PatientContext
# ---------------------------------------------------------------------------


class TestPatientContextConstruction:
    """PatientContext validation."""

    def test_minimal_construction(self):
        dp = make_data_patient()
        pc = PatientContext(patient=dp, dia_postop=3, procedimiento="Apendicectomía")
        assert pc.patient is dp
        assert pc.dia_postop == 3
        assert pc.procedimiento == "Apendicectomía"
        assert pc.call_id  # auto-generated, non-empty

    def test_dia_postop_zero_is_valid(self):
        dp = make_data_patient()
        pc = PatientContext(patient=dp, dia_postop=0, procedimiento="Apendicectomía")
        assert pc.dia_postop == 0

    def test_negative_dia_postop_raises(self):
        dp = make_data_patient()
        with pytest.raises(ValueError, match="dia_postop"):
            PatientContext(patient=dp, dia_postop=-1, procedimiento="Apendicectomía")

    def test_empty_procedimiento_raises(self):
        dp = make_data_patient()
        with pytest.raises(ValueError, match="procedimiento"):
            PatientContext(patient=dp, dia_postop=1, procedimiento="")

    def test_whitespace_only_procedimiento_raises(self):
        dp = make_data_patient()
        with pytest.raises(ValueError, match="procedimiento"):
            PatientContext(patient=dp, dia_postop=1, procedimiento="   ")

    def test_procedimiento_with_whitespace_is_stripped_and_valid(self):
        dp = make_data_patient()
        pc = PatientContext(patient=dp, dia_postop=1, procedimiento="  Colecistectomía  ")
        assert pc.procedimiento == "  Colecistectomía  "
        # The stripped value is non-empty, so it passes validation.
        # The raw string is preserved (frozen dataclass stores as given by caller).
        assert pc.procedimiento.strip() == "Colecistectomía"

    def test_empty_call_id_raises(self):
        dp = make_data_patient()
        with pytest.raises(ValueError, match="call_id"):
            PatientContext(patient=dp, dia_postop=1, procedimiento="A", call_id="")

    def test_immutable(self):
        dp = make_data_patient()
        pc = PatientContext(patient=dp, dia_postop=1, procedimiento="Apendicectomía")
        with pytest.raises(Exception):
            pc.dia_postop = 99  # type: ignore[misc]

    def test_auto_generated_call_id_is_unique(self):
        dp = make_data_patient()
        pc1 = PatientContext(patient=dp, dia_postop=1, procedimiento="A")
        pc2 = PatientContext(patient=dp, dia_postop=1, procedimiento="A")
        assert pc1.call_id != pc2.call_id


# ---------------------------------------------------------------------------
# CallContext
# ---------------------------------------------------------------------------


class TestCallContextConstruction:
    """CallContext validation and defaults."""

    def test_minimal_construction(self):
        dp = make_data_patient()
        pc = PatientContext(patient=dp, dia_postop=1, procedimiento="Apendicectomía")
        cc = CallContext(call_id=pc.call_id, patient_context=pc)
        assert cc.call_id == pc.call_id
        assert cc.patient_context is pc
        assert cc.state is State.IDLE
        assert isinstance(cc.history, History)
        assert len(cc.history) == 0
        assert cc.created_at.tzinfo is not None

    def test_call_id_is_non_empty(self):
        dp = make_data_patient()
        pc = PatientContext(patient=dp, dia_postop=1, procedimiento="A")
        with pytest.raises(ValueError, match="call_id"):
            CallContext(call_id="", patient_context=pc)

    def test_call_id_whitespace_only_raises(self):
        dp = make_data_patient()
        pc = PatientContext(patient=dp, dia_postop=1, procedimiento="A")
        with pytest.raises(ValueError, match="call_id"):
            CallContext(call_id="   ", patient_context=pc)

    def test_explicit_state(self):
        dp = make_data_patient()
        pc = PatientContext(patient=dp, dia_postop=1, procedimiento="A")
        cc = CallContext(call_id="call-1", patient_context=pc, state=State.GREETING)
        assert cc.state is State.GREETING

    def test_shared_history_is_mutable(self):
        """CallContext holds a History reference; mutating it is allowed."""
        dp = make_data_patient()
        pc = PatientContext(patient=dp, dia_postop=1, procedimiento="A")
        cc = CallContext(call_id="call-1", patient_context=pc)
        from backend.conversation.messages import Message, MessageRole
        msg = Message(
            turn_index=0, role=MessageRole.AGENT,
            text="Hola",
            timestamp=datetime.datetime.now(datetime.timezone.utc),
        )
        cc.history.append(msg)
        assert len(cc.history) == 1

    def test_immutable(self):
        dp = make_data_patient()
        pc = PatientContext(patient=dp, dia_postop=1, procedimiento="A")
        cc = CallContext(call_id="call-1", patient_context=pc)
        with pytest.raises(Exception):
            cc.state = State.ENDED  # type: ignore[misc]

    def test_naive_created_at_raises(self):
        dp = make_data_patient()
        pc = PatientContext(patient=dp, dia_postop=1, procedimiento="A")
        with pytest.raises(ValueError, match="timezone-aware"):
            CallContext(
                call_id="call-1",
                patient_context=pc,
                created_at=datetime.datetime(2026, 6, 15, 10, 0, 0),
            )

    def test_created_at_default_is_utc(self):
        dp = make_data_patient()
        pc = PatientContext(patient=dp, dia_postop=1, procedimiento="A")
        cc = CallContext(call_id="call-1", patient_context=pc)
        assert cc.created_at.tzinfo is datetime.timezone.utc
