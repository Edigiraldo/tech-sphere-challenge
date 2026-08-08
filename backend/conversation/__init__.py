"""Dialogue orchestration and state machine (``backend.conversation``).

This package provides the foundational domain model for a single
postoperative voice call:

* ``State`` / ``Event`` — finite state machine enum types.
* ``next_state()`` / ``InvalidTransitionError`` — transition logic.
* ``Message`` / ``MessageRole`` / ``History`` — turn-level messaging.
* ``PatientContext`` / ``CallContext`` — per-call context aggregation.

All domain objects are stdlib-only (no voice, LLM, RAG, persistence, or
frontend dependencies).
"""

from .context import CallContext, PatientContext
from .messages import History, Message, MessageRole
from .state import Event, State
from .transitions import InvalidTransitionError, next_state

__all__ = [
    # State machine
    "State",
    "Event",
    "next_state",
    "InvalidTransitionError",
    # Messages
    "MessageRole",
    "Message",
    "History",
    # Context
    "PatientContext",
    "CallContext",
]
