"""Conversation state machine: states, events, and registered transitions.

This module defines the finite set of conversation phases (``State``) and the
discrete triggers that drive transitions between them (``Event``).  The
transition table is the single source of truth for valid state-event pairs.
"""

from __future__ import annotations

import enum
from typing import ClassVar, FrozenSet, Mapping, Tuple


class State(enum.Enum):
    """Phases of a postoperative voice call."""

    IDLE = "IDLE"
    GREETING = "GREETING"
    CONSENT = "CONSENT"
    QUESTIONS = "QUESTIONS"
    CLOSING = "CLOSING"
    ENDED = "ENDED"


class Event(enum.Enum):
    """Discrete triggers that drive state transitions."""

    START_CALL = "START_CALL"
    GREETING_COMPLETE = "GREETING_COMPLETE"
    CONSENT_GIVEN = "CONSENT_GIVEN"
    CONSENT_REFUSED = "CONSENT_REFUSED"
    QUESTIONS_COMPLETE = "QUESTIONS_COMPLETE"
    ESCALATION_TRIGGER = "ESCALATION_TRIGGER"
    CLOSING_COMPLETE = "CLOSING_COMPLETE"


# ---------------------------------------------------------------------------
# Transition table — single source of truth for allowed (State, Event) pairs.
# ---------------------------------------------------------------------------

_TRANSITIONS: Mapping[Tuple[State, Event], State] = {
    (State.IDLE, Event.START_CALL): State.GREETING,
    (State.GREETING, Event.GREETING_COMPLETE): State.CONSENT,
    (State.CONSENT, Event.CONSENT_GIVEN): State.QUESTIONS,
    (State.CONSENT, Event.CONSENT_REFUSED): State.CLOSING,
    (State.QUESTIONS, Event.QUESTIONS_COMPLETE): State.CLOSING,
    (State.QUESTIONS, Event.ESCALATION_TRIGGER): State.CLOSING,
    (State.CLOSING, Event.CLOSING_COMPLETE): State.ENDED,
}

# Pre-computed immutable sets for fast validation.
VALID_EVENTS_BY_STATE: Mapping[State, FrozenSet[Event]] = {
    state: frozenset(event for (s, event) in _TRANSITIONS if s is state)
    for state in State
}
