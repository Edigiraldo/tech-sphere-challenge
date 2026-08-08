"""Conversation-scoped context objects.

``PatientContext`` wraps an existing ``backend.data.models.Patient`` with
safety constraints for a single call.  ``CallContext`` aggregates the
patient, state machine position, message history, and call metadata into a
single immutable object.
"""

from __future__ import annotations

import datetime
import uuid
from dataclasses import dataclass, field

from backend.data.models import Patient as DataPatient

from .messages import History
from .state import State


# ---------------------------------------------------------------------------
# PatientContext
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class PatientContext:
    """Thin wrapper around ``backend.data.models.Patient`` for use within a
    conversation call.

    Enforces two safety constraints that the raw data model does not:
    * ``dia_postop`` must be >= 0.
    * ``procedimiento`` must be non-empty after stripping.
    """

    patient: DataPatient
    dia_postop: int = 0
    procedimiento: str = ""
    call_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def __post_init__(self) -> None:
        # dia_postop >= 0
        if self.dia_postop < 0:
            raise ValueError(
                f"dia_postop must be >= 0, got {self.dia_postop}"
            )
        # procedimiento non-empty
        if not self.procedimiento.strip():
            raise ValueError(
                f"procedimiento must be non-empty after stripping, "
                f"got {self.procedimiento!r}"
            )
        # call_id non-empty
        if not self.call_id.strip():
            raise ValueError(
                f"call_id must be non-empty, got {self.call_id!r}"
            )


# ---------------------------------------------------------------------------
# CallContext
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class CallContext:
    """Immutable snapshot of a single conversation call's context.

    Wraps the patient context, current state machine position, accumulated
    message history, and creation timestamp into one object.

    Attributes
    ----------
    call_id : str
        Unique identifier for this call (mirrors ``PatientContext.call_id``).
    patient_context : PatientContext
        The patient and procedure being discussed.
    state : State
        Current position in the conversation state machine.
    history : History
        Append-only ordered message log (mutable internally, but the
        reference is frozen on the ``CallContext``).
    created_at : datetime.datetime
        UTC timestamp of when the call context was first created.
    """

    call_id: str
    patient_context: PatientContext
    state: State = State.IDLE
    history: History = field(default_factory=History)
    created_at: datetime.datetime = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc),
    )

    def __post_init__(self) -> None:
        if not self.call_id.strip():
            raise ValueError(
                f"call_id must be non-empty, got {self.call_id!r}"
            )
        if self.created_at.tzinfo is None:
            raise ValueError("created_at must be timezone-aware")
