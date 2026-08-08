"""Tests for ``backend.conversation.messages`` — Message and History."""

import datetime

import pytest

from backend.conversation.messages import History, Message, MessageRole


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def make_message(
    turn_index: int = 0,
    role: MessageRole = MessageRole.AGENT,
    text: str = "Hello",
    *,
    timestamp: datetime.datetime | None = None,
) -> Message:
    """Convenience factory with timezone-aware default timestamp."""
    if timestamp is None:
        timestamp = datetime.datetime(2026, 6, 15, 10, 0, 0, tzinfo=datetime.timezone.utc)
    return Message(turn_index=turn_index, role=role, text=text, timestamp=timestamp)


# ---------------------------------------------------------------------------
# Message
# ---------------------------------------------------------------------------


class TestMessageConstruction:
    """Message dataclass validation and defaults."""

    def test_minimal_construction(self):
        msg = Message(
            turn_index=0,
            role=MessageRole.AGENT,
            text="Hello, how are you?",
            timestamp=datetime.datetime(2026, 6, 15, 10, 0, 0, tzinfo=datetime.timezone.utc),
        )
        assert msg.turn_index == 0
        assert msg.role is MessageRole.AGENT
        assert msg.text == "Hello, how are you?"
        assert msg.timestamp == datetime.datetime(2026, 6, 15, 10, 0, 0, tzinfo=datetime.timezone.utc)

    def test_default_timestamp_is_timezone_aware(self):
        msg = Message(
            turn_index=0,
            role=MessageRole.PATIENT,
            text="Sí, muy bien.",
            timestamp=datetime.datetime.now(datetime.timezone.utc),
        )
        assert msg.timestamp.tzinfo is not None

    def test_patient_role(self):
        msg = make_message(role=MessageRole.PATIENT)
        assert msg.role is MessageRole.PATIENT

    def test_whitespace_surrounding_text_is_valid(self):
        """Leading and trailing whitespace is tolerated (text is non-empty after strip)."""
        msg = make_message(text="  ¿Cómo está?  ")
        assert msg.text == "  ¿Cómo está?  "
        assert msg.text.strip() == "¿Cómo está?"

    def test_text_empty_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            make_message(text="")

    def test_text_whitespace_only_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            make_message(text="   \t\n   ")

    def test_negative_turn_index_raises(self):
        with pytest.raises(ValueError, match="turn_index"):
            make_message(turn_index=-1)

    def test_naive_timestamp_raises(self):
        with pytest.raises(ValueError, match="timezone-aware"):
            Message(
                turn_index=0,
                role=MessageRole.AGENT,
                text="Hi",
                timestamp=datetime.datetime(2026, 6, 15, 10, 0, 0),  # naive
            )

    def test_immutable(self):
        msg = make_message()
        with pytest.raises(Exception):
            msg.turn_index = 99  # type: ignore[misc]


class TestMessageRoleEnum:
    """MessageRole enum coverage."""

    def test_two_roles(self):
        roles = {MessageRole.AGENT, MessageRole.PATIENT}
        assert len(roles) == 2

    def test_role_values_are_strings(self):
        assert MessageRole.AGENT.value == "AGENT"
        assert MessageRole.PATIENT.value == "PATIENT"


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------


class TestHistoryBasics:
    """History construction and basic operations."""

    def test_empty_history(self):
        h = History()
        assert len(h) == 0
        assert list(h) == []
        assert h.snapshot == ()

    def test_append_one(self):
        h = History()
        msg = make_message()
        h.append(msg)
        assert len(h) == 1
        assert h[0] is msg

    def test_append_multiple_preserves_order(self):
        h = History()
        msgs = [make_message(turn_index=i) for i in range(5)]
        for m in msgs:
            h.append(m)
        assert len(h) == 5
        for i, m in enumerate(h):
            assert m.turn_index == i

    def test_append_rejects_non_message(self):
        h = History()
        with pytest.raises(TypeError, match="Message"):
            h.append("not a message")  # type: ignore[arg-type]


class TestHistoryReadInterface:
    """History iteration, indexing, and snapshot."""

    def test_iteration_yields_messages(self):
        h = History()
        msgs = [make_message(turn_index=i) for i in range(3)]
        for m in msgs:
            h.append(m)
        for idx, m in enumerate(h):
            assert isinstance(m, Message)
            assert m.turn_index == idx

    def test_len(self):
        h = History()
        assert len(h) == 0
        h.append(make_message())
        assert len(h) == 1
        h.append(make_message(turn_index=1))
        assert len(h) == 2

    def test_indexing(self):
        h = History()
        m0 = make_message(turn_index=0)
        m1 = make_message(turn_index=1)
        h.append(m0)
        h.append(m1)
        assert h[0] is m0
        assert h[1] is m1

    def test_negative_index(self):
        h = History()
        msgs = [make_message(turn_index=i) for i in range(3)]
        for m in msgs:
            h.append(m)
        assert h[-1] is msgs[-1]
        assert h[-2] is msgs[-2]

    def test_index_error(self):
        h = History()
        with pytest.raises(IndexError):
            _ = h[0]

    def test_snapshot_returns_tuple(self):
        h = History()
        msgs = [make_message(turn_index=i) for i in range(3)]
        for m in msgs:
            h.append(m)
        snap = h.snapshot
        assert isinstance(snap, tuple)
        assert len(snap) == 3
        assert snap[0] is msgs[0]
        assert snap[1] is msgs[1]
        assert snap[2] is msgs[2]

    def test_snapshot_is_independent(self):
        """Appending after taking a snapshot does not change the snapshot."""
        h = History()
        h.append(make_message(turn_index=0))
        snap = h.snapshot
        h.append(make_message(turn_index=1))
        assert len(snap) == 1


class TestHistoryRepr:
    """History __repr__ formatting."""

    def test_repr_empty(self):
        assert repr(History()) == "History(messages=0)"

    def test_repr_with_messages(self):
        h = History()
        h.append(make_message())
        h.append(make_message(turn_index=1))
        assert repr(h) == "History(messages=2)"
