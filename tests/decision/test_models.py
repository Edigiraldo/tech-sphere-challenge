"""Tests for ``backend.decision.models`` — EscalationResult and Severity."""

import pytest

from backend.decision.models import EscalationResult, Severity


class TestSeverity:
    """Severity enum values are distinct and ordered by safety."""

    def test_values_exist(self):
        assert Severity.GREEN.value == "GREEN"
        assert Severity.YELLOW.value == "YELLOW"
        assert Severity.RED.value == "RED"

    def test_all_severities_different(self):
        values = {Severity.GREEN, Severity.YELLOW, Severity.RED}
        assert len(values) == 3


class TestEscalationResultConstruction:
    """EscalationResult validates invariants on construction."""

    def test_valid_green(self):
        r = EscalationResult(
            severity=Severity.GREEN,
            should_escalate=False,
            reason="Evolución favorable.",
            next_action="Continuar seguimiento.",
            domain="dolor",
            source="rule",
        )
        assert r.severity is Severity.GREEN
        assert r.should_escalate is False
        assert r.reason == "Evolución favorable."

    def test_valid_yellow(self):
        r = EscalationResult(
            severity=Severity.YELLOW,
            should_escalate=False,
            reason="Dolor moderado.",
            next_action="Monitorear.",
            domain="dolor",
            source="numeric",
        )
        assert r.severity is Severity.YELLOW
        assert r.should_escalate is False

    def test_valid_red(self):
        r = EscalationResult(
            severity=Severity.RED,
            should_escalate=True,
            reason="Dolor intenso.",
            next_action="Transferir al médico.",
            domain="dolor",
            source="rule",
        )
        assert r.severity is Severity.RED
        assert r.should_escalate is True

    def test_domain_none_allowed(self):
        r = EscalationResult(
            severity=Severity.YELLOW,
            should_escalate=False,
            reason="No clasificable.",
            next_action="Aclarar.",
            domain=None,
            source="invalid",
        )
        assert r.domain is None

    # --- invariant violations ---

    def test_red_without_escalation_raises(self):
        with pytest.raises(ValueError, match="RED severity must have"):
            EscalationResult(
                severity=Severity.RED,
                should_escalate=False,
                reason="Dolor.",
                next_action="Algo.",
            )

    def test_green_with_escalation_raises(self):
        with pytest.raises(ValueError, match="GREEN severity must have"):
            EscalationResult(
                severity=Severity.GREEN,
                should_escalate=True,
                reason="Bien.",
                next_action="Algo.",
            )

    def test_empty_reason_raises(self):
        with pytest.raises(ValueError, match="reason must be non-empty"):
            EscalationResult(
                severity=Severity.GREEN,
                should_escalate=False,
                reason="   ",
                next_action="Algo.",
            )

    def test_empty_next_action_raises(self):
        with pytest.raises(ValueError, match="next_action must be non-empty"):
            EscalationResult(
                severity=Severity.GREEN,
                should_escalate=False,
                reason="Bien.",
                next_action="",
            )

    def test_invalid_source_raises(self):
        with pytest.raises(ValueError, match="source must be"):
            EscalationResult(
                severity=Severity.GREEN,
                should_escalate=False,
                reason="Bien.",
                next_action="Algo.",
                source="unknown_source",
            )

    def test_all_valid_sources(self):
        for src in ("rule", "numeric", "ambig", "invalid", "incomplete"):
            r = EscalationResult(
                severity=Severity.YELLOW,
                should_escalate=False,
                reason="Ok.",
                next_action="Algo.",
                source=src,
            )
            assert r.source == src

    def test_frozen(self):
        r = EscalationResult(
            severity=Severity.GREEN,
            should_escalate=False,
            reason="Bien.",
            next_action="Algo.",
        )
        with pytest.raises(Exception):
            r.severity = Severity.RED  # type: ignore[misc]
