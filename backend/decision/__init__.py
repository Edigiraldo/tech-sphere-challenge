"""Escalation decision engine.

The ``backend.decision`` module implements the **safety-first, conservative**
escalation classifier required by the architecture.  It is:

* **Text-only**: processes patient speech, no audio.
* **Stdlib-only**: depends only on ``re``, ``enum``, and ``dataclasses``.
* **Deterministic**: no randomness, no ML, no external calls.
* **Conservative**: false negatives (failing to escalate) are catastrophic,
  so the engine biases toward YELLOW and RED.

Public API
----------
``classify(patient_text, domain, dia_postop=0, procedimiento="")``
    Classify a single patient response into ``GREEN``, ``YELLOW``, or
    ``RED``, returning a typed ``EscalationResult``.
``Severity``
    Enum: ``GREEN``, ``YELLOW``, ``RED``.
``EscalationResult``
    Frozen dataclass with ``severity``, ``should_escalate``, ``reason``,
    ``next_action``, ``domain``, and ``source``.

Integration
-----------
The conversation orchestrator calls ``classify()`` after each patient turn
during the ``QUESTIONS`` phase.  Two consecutive ``YELLOW`` verdicts trigger
escalation (accumulation policy is managed by the orchestrator, not this
module).
"""

from __future__ import annotations

from backend.decision.models import EscalationResult, Severity
from backend.decision.rules import classify

__all__ = [
    "classify",
    "EscalationResult",
    "Severity",
]
