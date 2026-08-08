"""Dialogue orchestration and state machine (``backend.conversation``).

This package provides:

* ``State`` / ``Event`` — finite state machine enum types.
* ``next_state()`` / ``InvalidTransitionError`` — transition logic.
* ``Message`` / ``MessageRole`` / ``History`` — turn-level messaging.
* ``PatientContext`` / ``CallContext`` — per-call context aggregation.
* ``ConversationOrchestrator`` / ``OrchestratorTurn`` — deterministic
  Spanish text-only call flow connecting state machine, history, RAG
  retrieval, and LLM answer generation.
"""

from .context import CallContext, PatientContext
from .messages import History, Message, MessageRole
from .orchestrator import (
    FOLLOW_UP_QUESTIONS,
    ConversationOrchestrator,
    OrchestratorTurn,
)
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
    # Orchestrator
    "ConversationOrchestrator",
    "OrchestratorTurn",
    "FOLLOW_UP_QUESTIONS",
]
