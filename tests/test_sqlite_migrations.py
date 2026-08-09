"""Regression tests for SQLite schema migrations."""

from __future__ import annotations

import sqlite3

from backend.persistence.sqlite import init_sqlite, _reset_sqlite


def test_existing_documents_database_gets_content_hash_column(tmp_path) -> None:
    db_path = tmp_path / "legacy.db"
    connection = sqlite3.connect(db_path)
    connection.execute(
        """
        CREATE TABLE documents (
            document_id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            status TEXT NOT NULL,
            uploaded_at TEXT NOT NULL,
            size_bytes INTEGER NOT NULL DEFAULT 0,
            error_message TEXT
        )
        """
    )
    connection.execute(
        "INSERT INTO documents(document_id, filename, status, uploaded_at) "
        "VALUES ('legacy-1', 'old.pdf', 'ready', '2026-01-01T00:00:00+00:00')"
    )
    connection.commit()
    connection.close()

    init_sqlite(db_path)
    try:
        connection = sqlite3.connect(db_path)
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(documents)")
        }
        assert "content_hash" in columns
        assert connection.execute(
            "SELECT filename FROM documents WHERE document_id='legacy-1'"
        ).fetchone()[0] == "old.pdf"
        connection.close()
    finally:
        _reset_sqlite()
