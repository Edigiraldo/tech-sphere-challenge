"""Tests for extended SQLite persistence: calls, conversation turns,
summaries, and escalation alerts CRUD operations.

Covers:
- Table creation via ``init_sqlite``.
- ``CallRecord`` insertion, retrieval, update.
- ``ConversationTurnRecord`` insertion (single + batch), retrieval by call.
- ``SummaryRecord`` insertion and retrieval.
- ``EscalationAlertRecord`` insertion and retrieval.
- ``_reset_sqlite`` and re-initialisation lifecycle.
"""

from __future__ import annotations

import datetime
import json
import os
import tempfile
from pathlib import Path

import pytest

from backend.persistence.sqlite import (
    CallRecord,
    ConversationTurnRecord,
    EscalationAlertRecord,
    SummaryRecord,
    _reset_sqlite,
    get_all_calls,
    get_alerts_for_call,
    get_call_by_id,
    get_summary_for_call,
    get_turns_for_call,
    init_sqlite,
    insert_call,
    insert_escalation_alert,
    insert_summary,
    insert_turn,
    insert_turns,
    update_call_ended,
)


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture(autouse=True)
def _reset_persistence():
    """Reset the SQLite module before each test so every test starts
    with a clean database."""
    _reset_sqlite()
    yield
    _reset_sqlite()


@pytest.fixture
def temp_db_path():
    """A temporary file path for the SQLite database."""
    fd, path = tempfile.mkstemp(suffix=".db", prefix="test_persist_")
    os.close(fd)
    yield Path(path)
    try:
        os.unlink(path)
    except OSError:
        pass


@pytest.fixture
def db(temp_db_path):
    """Initialised SQLite database backed by a temp file."""
    init_sqlite(temp_db_path)
    return temp_db_path


@pytest.fixture
def sample_call():
    """A valid ``CallRecord`` for testing."""
    return CallRecord(
        call_id="call-001",
        paciente_id="pac-001",
        nombre_completo="Maria Test",
        procedimiento="Apendicectomia",
        dia_postop=3,
        eps="Compensar EPS",
        state="IDLE",
        started_at=datetime.datetime(2026, 8, 8, 10, 0, 0, tzinfo=datetime.timezone.utc),
    )


@pytest.fixture
def sample_turn(sample_call):
    """A valid ``ConversationTurnRecord`` for testing."""
    return ConversationTurnRecord(
        turn_id="turn-001",
        call_id=sample_call.call_id,
        turn_index=0,
        role="AGENT",
        text="Buenos dias, como se siente?",
        timestamp=datetime.datetime(2026, 8, 8, 10, 0, 5, tzinfo=datetime.timezone.utc),
    )


@pytest.fixture
def sample_summary(sample_call):
    """A valid ``SummaryRecord`` for testing."""
    return SummaryRecord(
        summary_id="sum-001",
        call_id=sample_call.call_id,
        created_at=datetime.datetime(2026, 8, 8, 10, 5, 0, tzinfo=datetime.timezone.utc),
        patient_summary="Paciente: Maria Test.",
        procedure_summary="Procedimiento: Apendicectomia, dia 3.",
        symptoms_summary="Dolor: leve. Fiebre: no reportada.",
        decision_summary="No se requirio escalamiento.",
        sources_json='[["doc-1","guia_apendicectomia.pdf",3]]',
        next_steps="Continuar seguimiento programado.",
    )


@pytest.fixture
def sample_alert(sample_call):
    """A valid ``EscalationAlertRecord`` for testing."""
    return EscalationAlertRecord(
        alert_id="alert-001",
        call_id=sample_call.call_id,
        created_at=datetime.datetime(2026, 8, 8, 10, 2, 0, tzinfo=datetime.timezone.utc),
        severity="RED",
        reason="Dolor intenso reportado (NRS 9)",
        domain="dolor",
    )


# ======================================================================
# Model validation tests
# ======================================================================


class TestCallRecordValidation:
    """Validate ``CallRecord`` construction guards."""

    def test_minimal_construction(self):
        r = CallRecord(
            call_id="c1",
            paciente_id="p1",
            nombre_completo="Test",
            procedimiento="Apendicectomia",
            eps="EPS",
        )
        assert r.call_id == "c1"
        assert r.dia_postop == 0
        assert r.total_turns == 0
        assert not r.escalated

    def test_empty_call_id_raises(self):
        with pytest.raises(ValueError, match="call_id"):
            CallRecord(
                call_id="", paciente_id="p1", nombre_completo="T",
                procedimiento="A", eps="EPS",
            )

    def test_empty_paciente_id_raises(self):
        with pytest.raises(ValueError, match="paciente_id"):
            CallRecord(
                call_id="c1", paciente_id="", nombre_completo="T",
                procedimiento="A", eps="EPS",
            )

    def test_negative_dia_postop_raises(self):
        with pytest.raises(ValueError, match="dia_postop"):
            CallRecord(
                call_id="c1", paciente_id="p1", nombre_completo="T",
                procedimiento="A", eps="EPS", dia_postop=-1,
            )

    def test_negative_total_turns_raises(self):
        with pytest.raises(ValueError, match="total_turns"):
            CallRecord(
                call_id="c1", paciente_id="p1", nombre_completo="T",
                procedimiento="A", eps="EPS", total_turns=-1,
            )

    def test_naive_started_at_raises(self):
        with pytest.raises(ValueError, match="timezone-aware"):
            CallRecord(
                call_id="c1", paciente_id="p1", nombre_completo="T",
                procedimiento="A", eps="EPS",
                started_at=datetime.datetime(2026, 8, 8, 10, 0, 0),
            )

    def test_naive_ended_at_raises(self):
        with pytest.raises(ValueError, match="timezone-aware"):
            CallRecord(
                call_id="c1", paciente_id="p1", nombre_completo="T",
                procedimiento="A", eps="EPS",
                ended_at=datetime.datetime(2026, 8, 8, 10, 0, 0),
            )

    def test_immutable(self):
        r = CallRecord(
            call_id="c1", paciente_id="p1", nombre_completo="T",
            procedimiento="A", eps="EPS",
        )
        with pytest.raises(Exception):
            r.state = "ENDED"  # type: ignore[misc]


class TestConversationTurnRecordValidation:
    """Validate ``ConversationTurnRecord`` construction guards."""

    def test_minimal_construction(self):
        r = ConversationTurnRecord(
            turn_id="t1", call_id="c1", turn_index=0,
            role="AGENT", text="Hola",
        )
        assert r.turn_id == "t1"
        assert r.severity is None
        assert r.domain is None

    def test_empty_turn_id_raises(self):
        with pytest.raises(ValueError, match="turn_id"):
            ConversationTurnRecord(
                turn_id="", call_id="c1", turn_index=0,
                role="AGENT", text="Hola",
            )

    def test_invalid_role_raises(self):
        with pytest.raises(ValueError, match="role"):
            ConversationTurnRecord(
                turn_id="t1", call_id="c1", turn_index=0,
                role="DOCTOR", text="Hola",
            )

    def test_empty_text_raises(self):
        with pytest.raises(ValueError, match="text"):
            ConversationTurnRecord(
                turn_id="t1", call_id="c1", turn_index=0,
                role="AGENT", text="   ",
            )

    def test_negative_turn_index_raises(self):
        with pytest.raises(ValueError, match="turn_index"):
            ConversationTurnRecord(
                turn_id="t1", call_id="c1", turn_index=-1,
                role="AGENT", text="Hola",
            )


class TestSummaryRecordValidation:
    """Validate ``SummaryRecord`` construction guards."""

    def test_minimal_construction(self):
        r = SummaryRecord(
            summary_id="s1",
            call_id="c1",
            patient_summary="Paciente: Test.",
            procedure_summary="Procedimiento: Apendicectomia.",
            symptoms_summary="Sin sintomas relevantes.",
            decision_summary="No se requirio escalamiento.",
            next_steps="Continuar seguimiento.",
        )
        assert r.summary_id == "s1"
        assert r.sources_json == "[]"

    def test_empty_sections_raise(self):
        with pytest.raises(ValueError, match="patient_summary"):
            SummaryRecord(
                summary_id="s1", call_id="c1",
                patient_summary="", procedure_summary="P",
                symptoms_summary="S", decision_summary="D",
                next_steps="N",
            )

    def test_invalid_sources_json_raises(self):
        with pytest.raises(ValueError, match="sources_json"):
            SummaryRecord(
                summary_id="s1", call_id="c1",
                patient_summary="P", procedure_summary="P",
                symptoms_summary="S", decision_summary="D",
                next_steps="N",
                sources_json="not-json",
            )

    def test_sources_json_not_array_raises(self):
        with pytest.raises(ValueError, match="JSON array"):
            SummaryRecord(
                summary_id="s1", call_id="c1",
                patient_summary="P", procedure_summary="P",
                symptoms_summary="S", decision_summary="D",
                next_steps="N",
                sources_json='"string"',
            )


class TestEscalationAlertRecordValidation:
    """Validate ``EscalationAlertRecord`` construction guards."""

    def test_minimal_construction(self):
        r = EscalationAlertRecord(
            alert_id="a1", call_id="c1", reason="Dolor severo",
        )
        assert r.alert_id == "a1"
        assert r.severity == "RED"

    def test_invalid_severity_raises(self):
        with pytest.raises(ValueError, match="severity"):
            EscalationAlertRecord(
                alert_id="a1", call_id="c1", reason="Dolor",
                severity="ORANGE",
            )

    def test_empty_reason_raises(self):
        with pytest.raises(ValueError, match="reason"):
            EscalationAlertRecord(
                alert_id="a1", call_id="c1", reason="   ",
            )


# ======================================================================
# CRUD tests — Calls
# ======================================================================


class TestCallCRUD:
    """CRUD operations for the ``calls`` table."""

    def test_insert_and_retrieve(self, db, sample_call):
        insert_call(sample_call)
        result = get_call_by_id(sample_call.call_id)
        assert result is not None
        assert result.call_id == sample_call.call_id
        assert result.nombre_completo == sample_call.nombre_completo
        assert result.escalated is False

    def test_get_nonexistent_call(self, db):
        result = get_call_by_id("nonexistent")
        assert result is None

    def test_get_all_calls(self, db, sample_call):
        insert_call(sample_call)
        call2 = CallRecord(
            call_id="call-002", paciente_id="pac-002",
            nombre_completo="Juan Test", procedimiento="Colecistectomia",
            eps="EPS2",
        )
        insert_call(call2)
        all_calls = get_all_calls()
        assert len(all_calls) == 2
        # Newest first
        assert all_calls[0].call_id == "call-002"

    def test_duplicate_call_id_raises(self, db, sample_call):
        insert_call(sample_call)
        with pytest.raises(Exception):
            insert_call(sample_call)

    def test_update_call_ended(self, db, sample_call):
        insert_call(sample_call)
        ended_at = datetime.datetime(2026, 8, 8, 10, 10, 0, tzinfo=datetime.timezone.utc)
        update_call_ended(
            sample_call.call_id,
            state="ENDED",
            ended_at=ended_at,
            total_turns=12,
            escalated=True,
        )
        result = get_call_by_id(sample_call.call_id)
        assert result is not None
        assert result.state == "ENDED"
        assert result.ended_at == ended_at
        assert result.total_turns == 12
        assert result.escalated is True


# ======================================================================
# CRUD tests — Conversation Turns
# ======================================================================


class TestTurnCRUD:
    """CRUD operations for the ``conversation_turns`` table."""

    def test_insert_and_retrieve(self, db, sample_call, sample_turn):
        insert_call(sample_call)
        insert_turn(sample_turn)
        turns = get_turns_for_call(sample_call.call_id)
        assert len(turns) == 1
        assert turns[0].turn_id == sample_turn.turn_id
        assert turns[0].text == sample_turn.text

    def test_empty_call_returns_empty(self, db, sample_call):
        insert_call(sample_call)
        turns = get_turns_for_call(sample_call.call_id)
        assert turns == []

    def test_batch_insert(self, db, sample_call):
        insert_call(sample_call)
        t1 = ConversationTurnRecord(
            turn_id="t1", call_id=sample_call.call_id, turn_index=0,
            role="AGENT", text="Hola",
        )
        t2 = ConversationTurnRecord(
            turn_id="t2", call_id=sample_call.call_id, turn_index=1,
            role="PATIENT", text="Buenos dias",
        )
        insert_turns([t1, t2])
        turns = get_turns_for_call(sample_call.call_id)
        assert len(turns) == 2
        assert turns[0].turn_index == 0
        assert turns[1].turn_index == 1

    def test_batch_insert_empty_list_noop(self, db, sample_call):
        insert_call(sample_call)
        insert_turns([])
        turns = get_turns_for_call(sample_call.call_id)
        assert turns == []

    def test_turn_with_severity_and_domain(self, db, sample_call):
        insert_call(sample_call)
        t = ConversationTurnRecord(
            turn_id="t-red", call_id=sample_call.call_id, turn_index=2,
            role="PATIENT", text="Me duele mucho",
            severity="RED", domain="dolor",
        )
        insert_turn(t)
        turns = get_turns_for_call(sample_call.call_id)
        assert turns[0].severity == "RED"
        assert turns[0].domain == "dolor"

    def test_turns_ordered_by_index(self, db, sample_call):
        insert_call(sample_call)
        t0 = ConversationTurnRecord(
            turn_id="t0", call_id=sample_call.call_id, turn_index=2,
            role="AGENT", text="Tercero",
        )
        t1 = ConversationTurnRecord(
            turn_id="t1", call_id=sample_call.call_id, turn_index=0,
            role="AGENT", text="Primero",
        )
        t2 = ConversationTurnRecord(
            turn_id="t2", call_id=sample_call.call_id, turn_index=1,
            role="PATIENT", text="Segundo",
        )
        insert_turns([t0, t1, t2])
        turns = get_turns_for_call(sample_call.call_id)
        assert [t.turn_index for t in turns] == [0, 1, 2]


# ======================================================================
# CRUD tests — Summaries
# ======================================================================


class TestSummaryCRUD:
    """CRUD operations for the ``summaries`` table."""

    def test_insert_and_retrieve(self, db, sample_call, sample_summary):
        insert_call(sample_call)
        insert_summary(sample_summary)
        result = get_summary_for_call(sample_call.call_id)
        assert result is not None
        assert result.summary_id == sample_summary.summary_id
        assert result.decision_summary == sample_summary.decision_summary
        assert result.next_steps == sample_summary.next_steps

    def test_get_summary_nonexistent_call(self, db, sample_call):
        insert_call(sample_call)
        result = get_summary_for_call(sample_call.call_id)
        assert result is None

    def test_duplicate_summary_per_call_raises(self, db, sample_call, sample_summary):
        insert_call(sample_call)
        insert_summary(sample_summary)
        dup = SummaryRecord(
            summary_id="sum-002", call_id=sample_call.call_id,
            created_at=datetime.datetime(2026, 8, 8, 10, 6, 0, tzinfo=datetime.timezone.utc),
            patient_summary="P2", procedure_summary="P2",
            symptoms_summary="S2", decision_summary="D2",
            next_steps="N2",
        )
        with pytest.raises(Exception):
            insert_summary(dup)

    def test_sources_json_roundtrip(self, db, sample_call):
        insert_call(sample_call)
        s = SummaryRecord(
            summary_id="sum-json", call_id=sample_call.call_id,
            patient_summary="P", procedure_summary="P",
            symptoms_summary="S", decision_summary="D",
            next_steps="N",
            sources_json='[["doc1","guia.pdf",3],["doc2","otro.pdf",7]]',
        )
        insert_summary(s)
        result = get_summary_for_call(sample_call.call_id)
        assert result is not None
        parsed = json.loads(result.sources_json)
        assert len(parsed) == 2
        assert parsed[0] == ["doc1", "guia.pdf", 3]
        assert parsed[1] == ["doc2", "otro.pdf", 7]


# ======================================================================
# CRUD tests — Escalation Alerts
# ======================================================================


class TestAlertCRUD:
    """CRUD operations for the ``escalation_alerts`` table."""

    def test_insert_and_retrieve(self, db, sample_call, sample_alert):
        insert_call(sample_call)
        insert_escalation_alert(sample_alert)
        alerts = get_alerts_for_call(sample_call.call_id)
        assert len(alerts) == 1
        assert alerts[0].alert_id == sample_alert.alert_id
        assert alerts[0].severity == "RED"
        assert alerts[0].reason == sample_alert.reason

    def test_multiple_alerts_per_call(self, db, sample_call):
        insert_call(sample_call)
        a1 = EscalationAlertRecord(
            alert_id="a1", call_id=sample_call.call_id,
            created_at=datetime.datetime(2026, 8, 8, 10, 1, 0, tzinfo=datetime.timezone.utc),
            reason="Dolor", severity="YELLOW", domain="dolor",
        )
        a2 = EscalationAlertRecord(
            alert_id="a2", call_id=sample_call.call_id,
            created_at=datetime.datetime(2026, 8, 8, 10, 2, 0, tzinfo=datetime.timezone.utc),
            reason="Fiebre alta", severity="RED", domain="fiebre",
        )
        insert_escalation_alert(a1)
        insert_escalation_alert(a2)
        alerts = get_alerts_for_call(sample_call.call_id)
        assert len(alerts) == 2
        # Newest first (a2 has later timestamp)
        assert alerts[0].alert_id == "a2"

    def test_empty_call_returns_empty(self, db, sample_call):
        insert_call(sample_call)
        alerts = get_alerts_for_call(sample_call.call_id)
        assert alerts == []

    def test_alert_without_domain(self, db, sample_call):
        insert_call(sample_call)
        a = EscalationAlertRecord(
            alert_id="a-nodom", call_id=sample_call.call_id,
            reason="Sintomas criticos generales", severity="RED",
        )
        insert_escalation_alert(a)
        alerts = get_alerts_for_call(sample_call.call_id)
        assert alerts[0].domain is None


# ======================================================================
# Lifecycle tests
# ======================================================================


class TestSqliteLifecycle:
    """Module-level reset and re-initialisation."""

    def test_reset_and_reinit(self, temp_db_path):
        init_sqlite(temp_db_path)
        insert_call(CallRecord(
            call_id="c1", paciente_id="p1", nombre_completo="T",
            procedimiento="A", eps="EPS",
        ))
        assert get_call_by_id("c1") is not None

        _reset_sqlite()
        init_sqlite(temp_db_path)
        # Table still exists (CREATE IF NOT EXISTS), but old data persists
        # because we point to the same file.
        # To verify a full reset, use a new path.
        fd, path2 = tempfile.mkstemp(suffix=".db", prefix="test_reset_")
        os.close(fd)
        try:
            _reset_sqlite()
            init_sqlite(path2)
            assert get_call_by_id("c1") is None
        finally:
            try:
                os.unlink(path2)
            except OSError:
                pass

    def test_uninitialised_raises(self):
        _reset_sqlite()
        with pytest.raises(RuntimeError, match="not initialised"):
            get_all_calls()
