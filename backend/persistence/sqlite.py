"""SQLite access layer for document metadata and application data.

Provides a thin wrapper around SQLite for the ``documents``, ``calls``,
``conversation_turns``, ``summaries``, and ``escalation_alerts`` tables.

Frozen typed dataclasses are defined in this module for each table row.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from backend.documents.models import Document, DocumentStatus

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_db_path: Optional[Path] = None
"""Path to the SQLite database file. Set by ``init_sqlite``."""


# ===================================================================
# Frozen typed persistence models
# ===================================================================


@dataclass(frozen=True, slots=True)
class CallRecord:
    """A single completed (or in-progress) voice call.

    Attributes
    ----------
    call_id : str
        Unique identifier for this call.
    paciente_id : str
        Dataset ``paciente_id``.
    nombre_completo : str
        Patient full name.
    procedimiento : str
        Surgical procedure name.
    dia_postop : int
        Post-operative day number (>= 0).
    eps : str
        Patient EPS / health provider.
    state : str
        Final conversation ``State`` value (e.g. ``"ENDED"``).
    started_at : datetime
        UTC timestamp when the call was created.
    ended_at : datetime | None
        UTC timestamp when the call ended, or ``None`` if still in progress.
    total_turns : int
        Number of conversation turns recorded.
    escalated : bool
        ``True`` when escalation was triggered during the call.
    """

    call_id: str
    paciente_id: str
    nombre_completo: str
    procedimiento: str
    dia_postop: int = 0
    eps: str = ""
    state: str = "IDLE"
    started_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    ended_at: datetime | None = None
    total_turns: int = 0
    escalated: bool = False

    def __post_init__(self) -> None:
        if not self.call_id.strip():
            raise ValueError("call_id must be non-empty")
        if not self.paciente_id.strip():
            raise ValueError("paciente_id must be non-empty")
        if self.dia_postop < 0:
            raise ValueError(f"dia_postop must be >= 0, got {self.dia_postop}")
        if self.total_turns < 0:
            raise ValueError(
                f"total_turns must be >= 0, got {self.total_turns}"
            )
        if self.started_at.tzinfo is None:
            raise ValueError("started_at must be timezone-aware")
        if self.ended_at is not None and self.ended_at.tzinfo is None:
            raise ValueError("ended_at must be timezone-aware")


@dataclass(frozen=True, slots=True)
class ConversationTurnRecord:
    """A single turn within a call, as persisted to SQLite.

    Attributes
    ----------
    turn_id : str
        Unique identifier for this turn.
    call_id : str
        FK to ``calls.call_id``.
    turn_index : int
        Zero-based sequence number within the call.
    role : str
        ``"AGENT"`` or ``"PATIENT"``.
    text : str
        Non-empty turn content.
    timestamp : datetime
        UTC timestamp when the turn was recorded.
    severity : str | None
        Escalation severity (``"GREEN"``, ``"YELLOW"``, ``"RED"``) for patient
        turns that were classified, or ``None``.
    domain : str | None
        Symptom domain (``"dolor"``, ``"fiebre"``, …) or ``None``.
    """

    turn_id: str
    call_id: str
    turn_index: int
    role: str
    text: str
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    severity: str | None = None
    domain: str | None = None

    def __post_init__(self) -> None:
        if not self.turn_id.strip():
            raise ValueError("turn_id must be non-empty")
        if not self.call_id.strip():
            raise ValueError("call_id must be non-empty")
        if self.turn_index < 0:
            raise ValueError(
                f"turn_index must be >= 0, got {self.turn_index}"
            )
        if self.role not in ("AGENT", "PATIENT"):
            raise ValueError(
                f"role must be 'AGENT' or 'PATIENT', got {self.role!r}"
            )
        if not self.text.strip():
            raise ValueError("text must be non-empty after stripping")
        if self.timestamp.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")


@dataclass(frozen=True, slots=True)
class SummaryRecord:
    """A structured call summary persisted to SQLite.

    Attributes
    ----------
    summary_id : str
        Unique identifier for the summary.
    call_id : str
        FK to ``calls.call_id`` (unique: one summary per call).
    created_at : datetime
        UTC timestamp when the summary was generated.
    patient_summary : str
        Spanish text describing the patient (name, age, EPS, city).
    procedure_summary : str
        Spanish text describing the procedure and post-op day.
    symptoms_summary : str
        Aggregated Spanish text of patient responses across domains.
    decision_summary : str
        Spanish text describing the escalation decision and reason.
    sources_json : str
        JSON array of ``[document_id, source_filename, page_number]`` entries.
    next_steps : str
        Spanish text with recommended next steps.
    """

    summary_id: str
    call_id: str
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    patient_summary: str = ""
    procedure_summary: str = ""
    symptoms_summary: str = ""
    decision_summary: str = ""
    sources_json: str = "[]"
    next_steps: str = ""

    def __post_init__(self) -> None:
        if not self.summary_id.strip():
            raise ValueError("summary_id must be non-empty")
        if not self.call_id.strip():
            raise ValueError("call_id must be non-empty")
        if self.created_at.tzinfo is None:
            raise ValueError("created_at must be timezone-aware")
        if not self.patient_summary.strip():
            raise ValueError("patient_summary must be non-empty")
        if not self.procedure_summary.strip():
            raise ValueError("procedure_summary must be non-empty")
        if not self.symptoms_summary.strip():
            raise ValueError("symptoms_summary must be non-empty")
        if not self.decision_summary.strip():
            raise ValueError("decision_summary must be non-empty")
        if not self.next_steps.strip():
            raise ValueError("next_steps must be non-empty")
        # Validate sources_json is parseable as a JSON array
        try:
            parsed = json.loads(self.sources_json)
            if not isinstance(parsed, list):
                raise ValueError(
                    "sources_json must encode a JSON array"
                )
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError(
                f"sources_json is not valid JSON: {exc}"
            ) from None


@dataclass(frozen=True, slots=True)
class EscalationAlertRecord:
    """An escalation alert persisted to SQLite.

    One or more alerts may be associated with a single call.

    Attributes
    ----------
    alert_id : str
        Unique identifier for the alert.
    call_id : str
        FK to ``calls.call_id``.
    created_at : datetime
        UTC timestamp when the alert was created.
    severity : str
        ``"RED"`` for immediate escalation alerts (YELLOW alerts may also
        be persisted for audit trails).
    reason : str
        Spanish-language clinical rationale.
    domain : str | None
        The symptom domain that triggered the alert, or ``None``.
    """

    alert_id: str
    call_id: str
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    severity: str = "RED"
    reason: str = ""
    domain: str | None = None

    def __post_init__(self) -> None:
        if not self.alert_id.strip():
            raise ValueError("alert_id must be non-empty")
        if not self.call_id.strip():
            raise ValueError("call_id must be non-empty")
        if self.created_at.tzinfo is None:
            raise ValueError("created_at must be timezone-aware")
        if self.severity not in ("GREEN", "YELLOW", "RED"):
            raise ValueError(
                f"severity must be GREEN/YELLOW/RED, got {self.severity!r}"
            )
        if not self.reason.strip():
            raise ValueError("reason must be non-empty")


# ===================================================================
# Initialisation
# ===================================================================


def init_sqlite(db_path: str | Path) -> None:
    """Initialise (or re-initialise) the SQLite database at *db_path*.

    Creates all application tables if they do not exist. Safe to call
    multiple times — subsequent calls re-point the module to a new database
    (useful for tests that use a temp database).
    """
    global _db_path

    _db_path = Path(db_path)
    _db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = _get_conn()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                document_id  TEXT PRIMARY KEY,
                filename     TEXT    NOT NULL,
                status       TEXT    NOT NULL DEFAULT 'pending',
                uploaded_at  TEXT    NOT NULL,
                size_bytes   INTEGER NOT NULL DEFAULT 0,
                content_hash TEXT,
                error_message TEXT
            )
            """
        )
        # Migrate databases created before content-hash identity was added.
        document_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(documents)")
        }
        if "content_hash" not in document_columns:
            conn.execute("ALTER TABLE documents ADD COLUMN content_hash TEXT")
            logger.info("Migrated documents table with content_hash column")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS calls (
                call_id         TEXT PRIMARY KEY,
                paciente_id     TEXT    NOT NULL,
                nombre_completo TEXT    NOT NULL,
                procedimiento   TEXT    NOT NULL,
                dia_postop      INTEGER NOT NULL DEFAULT 0,
                eps             TEXT    NOT NULL,
                state           TEXT    NOT NULL DEFAULT 'IDLE',
                started_at      TEXT    NOT NULL,
                ended_at        TEXT,
                total_turns     INTEGER NOT NULL DEFAULT 0,
                escalated       INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conversation_turns (
                turn_id    TEXT PRIMARY KEY,
                call_id    TEXT NOT NULL,
                turn_index INTEGER NOT NULL,
                role       TEXT NOT NULL,
                text       TEXT NOT NULL,
                timestamp  TEXT NOT NULL,
                severity   TEXT,
                domain     TEXT,
                FOREIGN KEY (call_id) REFERENCES calls(call_id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS summaries (
                summary_id         TEXT PRIMARY KEY,
                call_id            TEXT NOT NULL UNIQUE,
                created_at         TEXT NOT NULL,
                patient_summary    TEXT NOT NULL,
                procedure_summary  TEXT NOT NULL,
                symptoms_summary   TEXT NOT NULL,
                decision_summary   TEXT NOT NULL,
                sources_json       TEXT NOT NULL DEFAULT '[]',
                next_steps         TEXT NOT NULL,
                FOREIGN KEY (call_id) REFERENCES calls(call_id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS escalation_alerts (
                alert_id   TEXT PRIMARY KEY,
                call_id    TEXT NOT NULL,
                created_at TEXT NOT NULL,
                severity   TEXT NOT NULL,
                reason     TEXT NOT NULL,
                domain     TEXT,
                FOREIGN KEY (call_id) REFERENCES calls(call_id)
            )
            """
        )
        conn.commit()
        logger.info("SQLite database initialised at %s", _db_path)
    finally:
        conn.close()


def _reset_sqlite() -> None:
    """Reset the module-level database path (for test teardown).

    After calling this, ``init_sqlite`` must be called again before any
    other function in this module can be used.
    """
    global _db_path
    _db_path = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_conn() -> sqlite3.Connection:
    """Return a new connection to the configured SQLite database.

    Raises ``RuntimeError`` if ``init_sqlite`` has not been called.
    """
    if _db_path is None:
        raise RuntimeError(
            "SQLite database not initialised. Call init_sqlite() first."
        )
    conn = sqlite3.connect(str(_db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _row_to_document(row: sqlite3.Row) -> Document:
    """Convert a ``sqlite3.Row`` to a ``Document`` domain object."""
    return Document(
        document_id=row["document_id"],
        filename=row["filename"],
        status=DocumentStatus(row["status"]),
        uploaded_at=datetime.fromisoformat(row["uploaded_at"]),
        size_bytes=row["size_bytes"],
        content_hash=row["content_hash"] if "content_hash" in row.keys() else None,
        error_message=row["error_message"],
    )


# ---------------------------------------------------------------------------
# Public CRUD operations
# ---------------------------------------------------------------------------


def insert_document(doc: Document) -> None:
    """Insert a new document metadata row.

    Args:
        doc: The document to insert. ``document_id`` must be unique.
    """
    conn = _get_conn()
    try:
        conn.execute(
            """INSERT INTO documents
               (document_id, filename, status, uploaded_at, size_bytes,
                content_hash, error_message)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                doc.document_id,
                doc.filename,
                doc.status.value,
                doc.uploaded_at.isoformat(),
                doc.size_bytes,
                doc.content_hash,
                doc.error_message,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def update_document_status(
    document_id: str,
    status: DocumentStatus,
    error_message: str | None = None,
) -> None:
    """Update the status (and optionally error_message) for a document.

    Args:
        document_id: The document to update.
        status: New lifecycle status.
        error_message: Human-readable error detail (meaningful only for
            ``FAILED`` status). Pass ``None`` to leave unchanged.
    """
    conn = _get_conn()
    try:
        if error_message is not None:
            conn.execute(
                "UPDATE documents SET status = ?, error_message = ? "
                "WHERE document_id = ?",
                (status.value, error_message, document_id),
            )
        else:
            conn.execute(
                "UPDATE documents SET status = ? WHERE document_id = ?",
                (status.value, document_id),
            )
        conn.commit()
    finally:
        conn.close()


def get_all_documents() -> list[Document]:
    """Return all document metadata rows, newest first."""
    conn = _get_conn()
    try:
        rows = conn.execute(
            "SELECT * FROM documents ORDER BY uploaded_at DESC"
        ).fetchall()
        return [_row_to_document(r) for r in rows]
    finally:
        conn.close()


def get_document_by_id(document_id: str) -> Document | None:
    """Return the document with *document_id*, or ``None`` if not found."""
    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT * FROM documents WHERE document_id = ?", (document_id,)
        ).fetchone()
        return _row_to_document(row) if row is not None else None
    finally:
        conn.close()


def get_document_by_content_hash(content_hash: str) -> Document | None:
    """Return the first non-deleted document matching *content_hash*.

    Only considers documents whose status is not ``DELETED``.  Returns
    ``None`` when no active match exists, allowing a new upload to proceed.
    """
    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT * FROM documents WHERE content_hash = ? AND status != ?",
            (content_hash, DocumentStatus.DELETED.value),
        ).fetchone()
        return _row_to_document(row) if row is not None else None
    finally:
        conn.close()


def get_active_document_ids() -> set[str]:
    """Return the set of ``document_id`` values for non-deleted documents.

    This is used by the retrieval pipeline to exclude chunks belonging to
    deleted or unregistered documents from search results.
    """
    conn = _get_conn()
    try:
        rows = conn.execute(
            "SELECT document_id FROM documents WHERE status != ?",
            (DocumentStatus.DELETED.value,),
        ).fetchall()
        return {r["document_id"] for r in rows}
    finally:
        conn.close()


# ===================================================================
# Row converters
# ===================================================================


def _row_to_call(row: sqlite3.Row) -> CallRecord:
    ended_at_raw = row["ended_at"]
    return CallRecord(
        call_id=row["call_id"],
        paciente_id=row["paciente_id"],
        nombre_completo=row["nombre_completo"],
        procedimiento=row["procedimiento"],
        dia_postop=row["dia_postop"],
        eps=row["eps"],
        state=row["state"],
        started_at=datetime.fromisoformat(row["started_at"]),
        ended_at=(
            datetime.fromisoformat(ended_at_raw)
            if ended_at_raw is not None
            else None
        ),
        total_turns=row["total_turns"],
        escalated=bool(row["escalated"]),
    )


def _row_to_turn(row: sqlite3.Row) -> ConversationTurnRecord:
    return ConversationTurnRecord(
        turn_id=row["turn_id"],
        call_id=row["call_id"],
        turn_index=row["turn_index"],
        role=row["role"],
        text=row["text"],
        timestamp=datetime.fromisoformat(row["timestamp"]),
        severity=row["severity"],
        domain=row["domain"],
    )


def _row_to_summary(row: sqlite3.Row) -> SummaryRecord:
    return SummaryRecord(
        summary_id=row["summary_id"],
        call_id=row["call_id"],
        created_at=datetime.fromisoformat(row["created_at"]),
        patient_summary=row["patient_summary"],
        procedure_summary=row["procedure_summary"],
        symptoms_summary=row["symptoms_summary"],
        decision_summary=row["decision_summary"],
        sources_json=row["sources_json"],
        next_steps=row["next_steps"],
    )


def _row_to_alert(row: sqlite3.Row) -> EscalationAlertRecord:
    return EscalationAlertRecord(
        alert_id=row["alert_id"],
        call_id=row["call_id"],
        created_at=datetime.fromisoformat(row["created_at"]),
        severity=row["severity"],
        reason=row["reason"],
        domain=row["domain"],
    )


# ===================================================================
# Public CRUD — Calls
# ===================================================================


def insert_call(record: CallRecord) -> None:
    """Insert a new call row.

    Args:
        record: The call to insert. ``call_id`` must be unique.
    """
    conn = _get_conn()
    try:
        conn.execute(
            """INSERT INTO calls
               (call_id, paciente_id, nombre_completo, procedimiento,
                dia_postop, eps, state, started_at, ended_at,
                total_turns, escalated)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                record.call_id,
                record.paciente_id,
                record.nombre_completo,
                record.procedimiento,
                record.dia_postop,
                record.eps,
                record.state,
                record.started_at.isoformat(),
                record.ended_at.isoformat() if record.ended_at else None,
                record.total_turns,
                int(record.escalated),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def get_call_by_id(call_id: str) -> CallRecord | None:
    """Return the call with *call_id*, or ``None`` if not found."""
    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT * FROM calls WHERE call_id = ?", (call_id,)
        ).fetchone()
        return _row_to_call(row) if row is not None else None
    finally:
        conn.close()


def get_all_calls() -> list[CallRecord]:
    """Return all call rows, newest first."""
    conn = _get_conn()
    try:
        rows = conn.execute(
            "SELECT * FROM calls ORDER BY started_at DESC"
        ).fetchall()
        return [_row_to_call(r) for r in rows]
    finally:
        conn.close()


def update_call_ended(
    call_id: str,
    *,
    state: str,
    ended_at: datetime,
    total_turns: int,
    escalated: bool,
) -> None:
    """Mark a call as ended with its final state and turn count.

    Args:
        call_id: The call to update.
        state: Final conversation state (e.g. ``"ENDED"``).
        ended_at: UTC timestamp when the call ended.
        total_turns: Total number of turns in the call.
        escalated: Whether escalation was triggered.
    """
    conn = _get_conn()
    try:
        conn.execute(
            """UPDATE calls SET state = ?, ended_at = ?,
               total_turns = ?, escalated = ?
               WHERE call_id = ?""",
            (
                state,
                ended_at.isoformat(),
                total_turns,
                int(escalated),
                call_id,
            ),
        )
        conn.commit()
    finally:
        conn.close()


# ===================================================================
# Public CRUD — Conversation Turns
# ===================================================================


def insert_turn(record: ConversationTurnRecord) -> None:
    """Insert a single conversation turn.

    Args:
        record: The turn to insert. ``turn_id`` must be unique.
    """
    conn = _get_conn()
    try:
        conn.execute(
            """INSERT INTO conversation_turns
               (turn_id, call_id, turn_index, role, text, timestamp,
                severity, domain)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                record.turn_id,
                record.call_id,
                record.turn_index,
                record.role,
                record.text,
                record.timestamp.isoformat(),
                record.severity,
                record.domain,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def insert_turns(records: list[ConversationTurnRecord]) -> None:
    """Insert multiple conversation turns in a single transaction.

    Args:
        records: The turns to insert. Each ``turn_id`` must be unique.
    """
    if not records:
        return
    conn = _get_conn()
    try:
        conn.executemany(
            """INSERT INTO conversation_turns
               (turn_id, call_id, turn_index, role, text, timestamp,
                severity, domain)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    r.turn_id,
                    r.call_id,
                    r.turn_index,
                    r.role,
                    r.text,
                    r.timestamp.isoformat(),
                    r.severity,
                    r.domain,
                )
                for r in records
            ],
        )
        conn.commit()
    finally:
        conn.close()


def get_turns_for_call(call_id: str) -> list[ConversationTurnRecord]:
    """Return all turns for *call_id*, ordered by ``turn_index`` ascending."""
    conn = _get_conn()
    try:
        rows = conn.execute(
            """SELECT * FROM conversation_turns
               WHERE call_id = ?
               ORDER BY turn_index ASC""",
            (call_id,),
        ).fetchall()
        return [_row_to_turn(r) for r in rows]
    finally:
        conn.close()


# ===================================================================
# Public CRUD — Summaries
# ===================================================================


def insert_summary(record: SummaryRecord) -> None:
    """Insert a new summary for a call.

    Only one summary per call is allowed (``call_id`` is UNIQUE).

    Args:
        record: The summary to insert. ``summary_id`` must be unique.
    """
    conn = _get_conn()
    try:
        conn.execute(
            """INSERT INTO summaries
               (summary_id, call_id, created_at, patient_summary,
                procedure_summary, symptoms_summary, decision_summary,
                sources_json, next_steps)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                record.summary_id,
                record.call_id,
                record.created_at.isoformat(),
                record.patient_summary,
                record.procedure_summary,
                record.symptoms_summary,
                record.decision_summary,
                record.sources_json,
                record.next_steps,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def get_summary_for_call(call_id: str) -> SummaryRecord | None:
    """Return the summary for *call_id*, or ``None`` if none exists."""
    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT * FROM summaries WHERE call_id = ?", (call_id,)
        ).fetchone()
        return _row_to_summary(row) if row is not None else None
    finally:
        conn.close()


# ===================================================================
# Public CRUD — Escalation Alerts
# ===================================================================


def insert_escalation_alert(record: EscalationAlertRecord) -> None:
    """Insert a new escalation alert.

    Args:
        record: The alert to insert. ``alert_id`` must be unique.
    """
    conn = _get_conn()
    try:
        conn.execute(
            """INSERT INTO escalation_alerts
               (alert_id, call_id, created_at, severity, reason, domain)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                record.alert_id,
                record.call_id,
                record.created_at.isoformat(),
                record.severity,
                record.reason,
                record.domain,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def get_alerts_for_call(call_id: str) -> list[EscalationAlertRecord]:
    """Return all escalation alerts for *call_id*, newest first."""
    conn = _get_conn()
    try:
        rows = conn.execute(
            """SELECT * FROM escalation_alerts
               WHERE call_id = ?
               ORDER BY created_at DESC""",
            (call_id,),
        ).fetchall()
        return [_row_to_alert(r) for r in rows]
    finally:
        conn.close()
