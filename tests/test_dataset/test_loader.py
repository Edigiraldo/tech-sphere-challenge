"""Tests for backend.data.loader — XLSX loading and type correctness."""

from pathlib import Path

from backend.data.loader import (
    get_dataset_path,
    load_conversations,
    load_conversations_for_evaluation,
    load_patients,
    load_trajectories,
)
from backend.data.models import (
    Conversation,
    ConversationTurn,
    Patient,
    Trajectory,
)


# ---------------------------------------------------------------------------
# get_dataset_path
# ---------------------------------------------------------------------------

class TestGetDatasetPath:
    def test_returns_path(self):
        p = get_dataset_path()
        assert isinstance(p, Path)
        assert p.name == "dataset"

    def test_path_exists(self):
        assert get_dataset_path().is_dir()


# ---------------------------------------------------------------------------
# Patients
# ---------------------------------------------------------------------------

class TestLoadPatients:
    def test_returns_dict(self):
        patients = load_patients()
        assert isinstance(patients, dict)

    def test_correct_count(self):
        patients = load_patients()
        assert len(patients) == 40

    def test_all_values_are_patients(self):
        patients = load_patients()
        for pid, p in patients.items():
            assert isinstance(p, Patient), f"{pid} is not a Patient"
            assert p.paciente_id == pid

    def test_keys_match_paciente_id(self):
        patients = load_patients()
        assert all(k == v.paciente_id for k, v in patients.items())

    def test_patient_has_merged_fields(self):
        """Every patient should have both clinical and demographic data."""
        patients = load_patients()
        for p in patients.values():
            assert p.nombre_completo != ""
            assert p.procedimiento != ""
            assert p.edad > 0
            assert p.genero in ("M", "F")
            assert p.eps != ""
            assert p.modulo_synthea in (
                "appendicitis",
                "breast_cancer",
                "cholecystitis",
                "colorectal_cancer",
                "total_joint_replacement",
            )

    def test_comorbilidades_parsed(self):
        """comorbilidades field must be a list of strings, even when empty."""
        patients = load_patients()
        for p in patients.values():
            assert isinstance(p.comorbilidades, list)
            for c in p.comorbilidades:
                assert isinstance(c, str)

    def test_fecha_cirugia_is_date(self):
        import datetime
        patients = load_patients()
        for p in patients.values():
            assert isinstance(p.fecha_cirugia, datetime.date)

    def test_adaptation_fields_parsed(self):
        patients = load_patients()
        for p in patients.values():
            assert isinstance(p.adaptation_fields, list)


# ---------------------------------------------------------------------------
# Trajectories
# ---------------------------------------------------------------------------

class TestLoadTrajectories:
    def test_returns_dict(self):
        trajs = load_trajectories()
        assert isinstance(trajs, dict)

    def test_correct_patient_count(self):
        """Trajectories should cover all 40 patients."""
        trajs = load_trajectories()
        assert len(trajs) == 40

    def test_total_trajectories(self):
        trajs = load_trajectories()
        total = sum(len(v) for v in trajs.values())
        assert total == 160  # 40 patients x ~4 days each

    def test_all_trajectory_types(self):
        trajs = load_trajectories()
        for pid, tlist in trajs.items():
            for t in tlist:
                assert isinstance(t, Trajectory)
                assert t.paciente_id == pid
                assert isinstance(t.dolor_nrs, int)
                assert isinstance(t.fiebre_c, (int, float))
                assert isinstance(t.dia_postop, int)

    def test_dia_postop_values(self):
        """Each patient has varying post-op days (typically 1, 3, 5, 7)."""
        trajs = load_trajectories()
        for pid, tlist in trajs.items():
            days = {t.dia_postop for t in tlist}
            assert 1 in days, f"{pid}: missing day 1"


# ---------------------------------------------------------------------------
# Conversations (runtime-safe — no labels)
# ---------------------------------------------------------------------------

class TestLoadConversations:
    def test_returns_list(self):
        convs = load_conversations()
        assert isinstance(convs, list)
        assert len(convs) > 0

    def test_all_conversation_types(self):
        convs = load_conversations()
        for c in convs:
            assert isinstance(c, Conversation)
            assert c.caso_id != ""
            assert c.paciente_id != ""

    def test_turns_sorted(self):
        convs = load_conversations()
        for c in convs:
            indices = [t.turno_idx for t in c.turns]
            assert indices == sorted(indices), f"{c.caso_id}: turns not sorted"

    def test_no_label_ground_truth(self):
        """Runtime-safe path must never populate label_ground_truth."""
        convs = load_conversations()
        for c in convs:
            for t in c.turns:
                assert t.label_ground_truth is None, (
                    f"label_ground_truth leaked in {t.dialogo_id}"
                )

    def test_turns_have_speaker(self):
        convs = load_conversations()
        for c in convs:
            for t in c.turns:
                assert t.hablante in ("agente", "paciente", "tercero")

    def test_text_is_non_empty(self):
        convs = load_conversations()
        for c in convs:
            for t in c.turns:
                assert t.texto != ""

    def test_total_turns(self):
        convs = load_conversations()
        total_turns = sum(len(c.turns) for c in convs)
        assert total_turns == 3991  # known row count


# ---------------------------------------------------------------------------
# Conversations (evaluation — with labels)
# ---------------------------------------------------------------------------

class TestLoadConversationsForEvaluation:
    def test_returns_list(self):
        convs = load_conversations_for_evaluation()
        assert isinstance(convs, list)
        assert len(convs) > 0

    def test_labels_are_populated(self):
        convs = load_conversations_for_evaluation()
        populated = 0
        for c in convs:
            for t in c.turns:
                if t.label_ground_truth is not None:
                    populated += 1
                    assert t.label_ground_truth in (
                        "verde", "amarillo", "rojo",
                    ), f"unexpected label: {t.label_ground_truth}"
        assert populated > 0, "No turns had label_ground_truth populated"

    def test_same_turn_count_as_runtime(self):
        runtime = load_conversations()
        eval_ = load_conversations_for_evaluation()
        assert len(runtime) == len(eval_)
        for rc, ec in zip(runtime, eval_):
            assert len(rc.turns) == len(ec.turns)
            assert rc.caso_id == ec.caso_id
