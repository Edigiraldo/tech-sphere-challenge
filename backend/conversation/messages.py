"""Conversation message model and append-only history.

``Message`` is a frozen dataclass representing a single turn in a
conversation.  ``History`` provides strictly-ordered, append-only storage
with read access via iteration, indexing, and an immutable tuple snapshot.
"""

from __future__ import annotations

import datetime
import enum
from dataclasses import dataclass, field
from typing import Iterator, Tuple


class MessageRole(enum.Enum):
    """Who spoke the message."""

    AGENT = "AGENT"
    PATIENT = "PATIENT"


# ---------------------------------------------------------------------------
# Message
# ---------------------------------------------------------------------------


def _validate_turn_index(value: int) -> int:
    """Guard: turn_index must be non-negative."""
    if value < 0:
        raise ValueError(f"turn_index must be >= 0, got {value}")
    return value


def _validate_text(value: str) -> str:
    """Guard: text must be non-empty after stripping whitespace."""
    stripped = value.strip()
    if not stripped:
        raise ValueError("text must be non-empty after stripping whitespace")
    return stripped


def _validate_timestamp(value: datetime.datetime) -> datetime.datetime:
    """Guard: timestamp must be timezone-aware (contain tzinfo)."""
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise ValueError(
            f"timestamp must be timezone-aware, got {value!r} "
            f"(tzinfo={value.tzinfo!r})"
        )
    return value


@dataclass(frozen=True, slots=True)
class Message:
    """A single turn within a conversation flow.

    Attributes
    ----------
    turn_index : int
        Zero-based sequence number within the conversation.
    role : MessageRole
        Whether the message was spoken by the ``AGENT`` or ``PATIENT``.
    text : str
        Non-empty, whitespace-stripped content.
    timestamp : datetime.datetime
        UTC (or other timezone-aware) instant the message was recorded.
    """

    turn_index: int = field(
        default=0,
        metadata={"validate": _validate_turn_index},
    )
    role: MessageRole = field(default=MessageRole.AGENT)
    text: str = field(default="", metadata={"validate": _validate_text})
    timestamp: datetime.datetime = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc),
        metadata={"validate": _validate_timestamp},
    )

    def __post_init__(self) -> None:
        # Run all field-level validators on construction.
        # dataclass __init__ bypasses them when frozen+slots, so we call
        # the validators explicitly.
        _validate_turn_index(self.turn_index)
        _validate_text(self.text)
        _validate_timestamp(self.timestamp)


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------


class History:
    """Append-only ordered collection of ``Message`` objects.

    Supports ``len()``, ``for m in history``, integer indexing via
    ``history[i]``, and an immutable ``tuple`` snapshot via the ``snapshot``
    property.  Once a message is appended it cannot be removed or modified.

    Example
    -------
    >>> h = History()
    >>> msg = Message(turn_index=0, role=MessageRole.AGENT, text="Hello",
    ...               timestamp=datetime.datetime.now(datetime.timezone.utc))
    >>> h.append(msg)
    >>> len(h)
    1
    >>> h[0] is msg
    True
    >>> tuple(m.turn_index for m in h)
    (0,)
    >>> h.snapshot
    (Message(...),)
    """

    __slots__ = ("_messages",)

    def __init__(self) -> None:
        self._messages: list[Message] = []

    # -- public mutator ------------------------------------------------------

    def append(self, message: Message) -> None:
        """Add *message* to the end of the history."""
        if not isinstance(message, Message):
            raise TypeError(
                f"History.append() expects Message, got {type(message).__name__}"
            )
        self._messages.append(message)

    # -- read interface ------------------------------------------------------

    def __len__(self) -> int:
        return len(self._messages)

    def __iter__(self) -> Iterator[Message]:
        return iter(self._messages)

    def __getitem__(self, index: int) -> Message:
        return self._messages[index]

    @property
    def snapshot(self) -> Tuple[Message, ...]:
        """Return an immutable tuple copy of all messages.

        The tuple is a shallow copy: the ``Message`` objects themselves are
        already frozen, so the snapshot is fully immutable.
        """
        return tuple(self._messages)

    # -- repr ----------------------------------------------------------------

    def __repr__(self) -> str:
        return f"History(messages={len(self._messages)})"
