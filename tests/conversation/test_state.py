"""Tests for ``backend.conversation.state`` — State and Event enums."""

import pytest

from backend.conversation.state import (
    Event,
    State,
    VALID_EVENTS_BY_STATE,
)


class TestStateEnum:
    """State enum members and values."""

    def test_all_states_present(self):
        expected = {"IDLE", "GREETING", "CONSENT", "QUESTIONS", "CLOSING", "ENDED"}
        assert {s.name for s in State} == expected

    def test_state_values_are_strings(self):
        for s in State:
            assert isinstance(s.value, str)
            assert s.value == s.name

    def test_state_identity(self):
        assert State.IDLE is State("IDLE")
        assert State.ENDED is State("ENDED")


class TestEventEnum:
    """Event enum members and values."""

    def test_all_events_present(self):
        expected = {
            "START_CALL",
            "GREETING_COMPLETE",
            "CONSENT_GIVEN",
            "CONSENT_REFUSED",
            "QUESTIONS_COMPLETE",
            "ESCALATION_TRIGGER",
            "EMERGENCY_TERMINATE",
            "CLOSING_COMPLETE",
        }
        assert {e.name for e in Event} == expected

    def test_event_values_are_strings(self):
        for e in Event:
            assert isinstance(e.value, str)
            assert e.value == e.name


class TestValidEventsByState:
    """Pre-computed VALID_EVENTS_BY_STATE mapping correctness."""

    def test_idle_only_start_call(self):
        assert VALID_EVENTS_BY_STATE[State.IDLE] == frozenset({Event.START_CALL})

    def test_greeting_only_greeting_complete(self):
        assert VALID_EVENTS_BY_STATE[State.GREETING] == frozenset(
            {Event.GREETING_COMPLETE}
        )

    def test_consent_two_events(self):
        assert VALID_EVENTS_BY_STATE[State.CONSENT] == frozenset(
            {Event.CONSENT_GIVEN, Event.CONSENT_REFUSED}
        )

    def test_questions_three_events(self):
        assert VALID_EVENTS_BY_STATE[State.QUESTIONS] == frozenset(
            {Event.QUESTIONS_COMPLETE, Event.ESCALATION_TRIGGER, Event.EMERGENCY_TERMINATE}
        )

    def test_closing_only_closing_complete(self):
        assert VALID_EVENTS_BY_STATE[State.CLOSING] == frozenset(
            {Event.CLOSING_COMPLETE}
        )

    def test_ended_no_events(self):
        assert VALID_EVENTS_BY_STATE[State.ENDED] == frozenset()

    def test_every_state_has_entry(self):
        """Every State must appear as a key in VALID_EVENTS_BY_STATE."""
        for state in State:
            assert state in VALID_EVENTS_BY_STATE
            assert isinstance(VALID_EVENTS_BY_STATE[state], frozenset)
