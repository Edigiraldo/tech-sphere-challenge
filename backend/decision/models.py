"""Escalation decision result types.

The ``EscalationResult`` dataclass is the sole public return type of the
decision module.  It is frozen, typed, and contains the full escalation
verdict with a Spanish-language reason and next action.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass


class Severity(enum.Enum):
    """Escalation severity levels.

    GREEN  — no red-flag symptoms; continue follow-up normally.
    YELLOW — potentially concerning; accumulate or clarify.
    RED    — immediate escalation; end call and trigger alert.
    """

    GREEN = "GREEN"
    YELLOW = "YELLOW"
    RED = "RED"


@dataclass(frozen=True, slots=True)
class EscalationResult:
    """Deterministic escalation verdict for a single patient turn.

    Attributes
    ----------
    severity : Severity
        The classification level (GREEN, YELLOW, or RED).
    should_escalate : bool
        ``True`` when the call should be escalated immediately (RED or
        accumulated YELLOW).  ``False`` for GREEN and first YELLOW, which
        require the caller to implement the accumulation policy.
    reason : str
        Spanish-language clinical rationale for the decision.  Always
        non-empty.
    next_action : str
        Spanish-language instruction for the agent.  Examples: "Continuar
        seguimiento", "Solicitar aclaración", "Transferir al médico".
    domain : str | None
        The symptom domain being assessed (``"dolor"``, ``"fiebre"``, …),
        or ``None`` when the classification is domain-agnostic
        (invalid / incomplete input).
    source : str
        How the verdict was reached: ``"rule"`` (lexicon match),
        ``"numeric"`` (threshold cross-check), ``"ambig"`` (ambiguous),
        ``"invalid"`` (empty/unparsable), ``"incomplete"`` (missing info).
    """

    severity: Severity
    should_escalate: bool
    reason: str
    next_action: str
    domain: str | None = None
    source: str = "rule"

    def __post_init__(self) -> None:
        # ---------- reason non-empty ----------
        if not self.reason.strip():
            raise ValueError("reason must be non-empty after stripping")

        # ---------- next_action non-empty ----------
        if not self.next_action.strip():
            raise ValueError("next_action must be non-empty after stripping")

        # ---------- source must be a recognised value ----------
        if self.source not in {"rule", "numeric", "ambig", "invalid", "incomplete"}:
            raise ValueError(
                f"source must be 'rule', 'numeric', 'ambig', 'invalid', or "
                f"'incomplete', got {self.source!r}"
            )

        # ---------- consistency: RED always escalates ----------
        if self.severity is Severity.RED and not self.should_escalate:
            raise ValueError("RED severity must have should_escalate=True")

        # ---------- consistency: GREEN never escalates ----------
        if self.severity is Severity.GREEN and self.should_escalate:
            raise ValueError("GREEN severity must have should_escalate=False")
