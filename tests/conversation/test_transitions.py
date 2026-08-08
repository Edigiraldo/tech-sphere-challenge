"""Tests for ``backend.conversation.transitions`` — transition logic."""

import pytest

from backend.conversation.state import Event, State
from backend.conversation.transitions import InvalidTransitionError, next_state


class TestValidTransitions:
    """Every documented valid state–event pair returns the expected State."""

    @pytest.mark.parametrize(
        "current, event, expected",
        [
            (State.IDLE, Event.START_CALL, State.GREETING),
            (State.GREETING, Event.GREETING_COMPLETE, State.CONSENT),
            (State.CONSENT, Event.CONSENT_GIVEN, State.QUESTIONS),
            (State.CONSENT, Event.CONSENT_REFUSED, State.CLOSING),
            (State.QUESTIONS, Event.QUESTIONS_COMPLETE, State.CLOSING),
            (State.QUESTIONS, Event.ESCALATION_TRIGGER, State.CLOSING),
            (State.CLOSING, Event.CLOSING_COMPLETE, State.ENDED),
        ],
    )
    def test_valid_transition(self, current, event, expected):
        assert next_state(current, event) is expected


# ---------------------------------------------------------------------------
# Build invalid pairs exhaustively (module-level to avoid class-body scoping
# issues with list comprehensions).
# ---------------------------------------------------------------------------

_VALID_TRANSITIONS: set[tuple[State, Event]] = {
    (State.IDLE, Event.START_CALL),
    (State.GREETING, Event.GREETING_COMPLETE),
    (State.CONSENT, Event.CONSENT_GIVEN),
    (State.CONSENT, Event.CONSENT_REFUSED),
    (State.QUESTIONS, Event.QUESTIONS_COMPLETE),
    (State.QUESTIONS, Event.ESCALATION_TRIGGER),
    (State.CLOSING, Event.CLOSING_COMPLETE),
}

_INVALID_PAIRS = [
    (s, e)
    for s in State
    for e in Event
    if (s, e) not in _VALID_TRANSITIONS
]


class TestInvalidTransitions:
    """Unsupported state–event pairs raise InvalidTransitionError."""

    @pytest.mark.parametrize("current, event", _INVALID_PAIRS)
    def test_invalid_transition_raises(self, current, event):
        with pytest.raises(InvalidTransitionError) as exc_info:
            next_state(current, event)
        assert exc_info.value.current is current
        assert exc_info.value.event is event

    def test_ended_has_no_valid_events(self):
        """No event should be valid from the ENDED state."""
        for event in Event:
            with pytest.raises(InvalidTransitionError):
                next_state(State.ENDED, event)

    def test_invalid_transition_is_value_error(self):
        """InvalidTransitionError must be a ValueError subclass."""
        with pytest.raises(ValueError):
            next_state(State.ENDED, Event.START_CALL)

    def test_error_message_contains_state_event_names(self):
        """The error message should mention the state and event names."""
        with pytest.raises(InvalidTransitionError) as exc_info:
            next_state(State.IDLE, Event.CLOSING_COMPLETE)
        msg = str(exc_info.value)
        assert "IDLE" in msg
        assert "CLOSING_COMPLETE" in msg
