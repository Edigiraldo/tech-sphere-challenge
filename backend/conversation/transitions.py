"""State-transition logic for the conversation state machine.

``InvalidTransitionError`` is raised when an event is not valid for the
current state, providing a safe failure mode that the orchestration layer can
catch and handle (e.g. fall through to escalation or human hand-off).
"""

from __future__ import annotations

from .state import Event, State, VALID_EVENTS_BY_STATE, _TRANSITIONS


class InvalidTransitionError(ValueError):
    """Raised when an ``Event`` cannot be applied to the current ``State``.

    Inherits from ``ValueError`` so that callers can catch a single
    ``ValueError`` class for invalid inputs without needing to know about
    this specific exception.
    """

    def __init__(self, current: State, event: Event) -> None:
        self.current = current
        self.event = event
        allowed = sorted(VALID_EVENTS_BY_STATE[current], key=lambda e: e.name)
        super().__init__(
            f"Invalid transition: cannot apply {event.name} in state "
            f"{current.name}. Allowed events: "
            f"{[e.name for e in allowed]}"
        )


def next_state(current: State, event: Event) -> State:
    """Return the next ``State`` after applying ``event``.

    Raises
    ------
    InvalidTransitionError
        If the ``(current, event)`` pair is not in the transition table.
    """
    key = (current, event)
    try:
        return _TRANSITIONS[key]
    except KeyError:
        raise InvalidTransitionError(current, event) from None
