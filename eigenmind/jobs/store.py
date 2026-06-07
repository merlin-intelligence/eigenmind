"""SQLite-backed persistence for background ingestion jobs."""
from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime

_SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS jobs (
    id               TEXT PRIMARY KEY,
    status           TEXT NOT NULL DEFAULT 'pending',
    collection       TEXT NOT NULL,
    directories_file TEXT NOT NULL,
    device           TEXT NOT NULL,
    history_file     TEXT,
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL,
    progress_current INTEGER NOT NULL DEFAULT 0,
    progress_total   INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS job_logs (
    rowid   INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id  TEXT    NOT NULL,
    message TEXT    NOT NULL
);
"""


class JobStore:
    """Thread-safe SQLite store for ingestion job state and logs."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init()

    def _init(self) -> None:
        with self._conn() as conn:
            conn.executescript(_SCHEMA)
            # On startup, any job that was 'running' or 'pending' was interrupted
            # by a server restart — mark them failed so they don't hang forever.
            conn.execute(
                "UPDATE jobs SET status='failed', updated_at=? WHERE status IN ('running', 'pending')",
                (datetime.now().isoformat(),),
            )

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def create_job(
        self,
        job_id: str,
        collection: str,
        directories_file: str,
        device: str,
        history_file: str | None,
    ) -> None:
        now = datetime.now().isoformat()
        with self._lock, self._conn() as conn:
            conn.execute(
                """INSERT INTO jobs
                   (id, status, collection, directories_file, device, history_file, created_at, updated_at)
                   VALUES (?, 'pending', ?, ?, ?, ?, ?, ?)""",
                (job_id, collection, directories_file, device, history_file, now, now),
            )

    def set_status(self, job_id: str, status: str) -> None:
        with self._lock, self._conn() as conn:
            conn.execute(
                "UPDATE jobs SET status=?, updated_at=? WHERE id=?",
                (status, datetime.now().isoformat(), job_id),
            )

    def set_progress(self, job_id: str, current: int, total: int) -> None:
        with self._lock, self._conn() as conn:
            conn.execute(
                "UPDATE jobs SET progress_current=?, progress_total=?, updated_at=? WHERE id=?",
                (current, total, datetime.now().isoformat(), job_id),
            )

    def append_log(self, job_id: str, message: str) -> None:
        with self._lock, self._conn() as conn:
            conn.execute(
                "INSERT INTO job_logs (job_id, message) VALUES (?, ?)",
                (job_id, message),
            )

    def get_job(self, job_id: str) -> dict | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            return dict(row) if row else None

    def get_logs(self, job_id: str, after_rowid: int = 0) -> list[tuple[int, str]]:
        """Return new log entries since after_rowid, as (rowid, message) pairs."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT rowid, message FROM job_logs WHERE job_id=? AND rowid>? ORDER BY rowid",
                (job_id, after_rowid),
            ).fetchall()
            return [(r["rowid"], r["message"]) for r in rows]

    def latest_job_for(self, collection: str) -> dict | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE collection=? ORDER BY created_at DESC LIMIT 1",
                (collection,),
            ).fetchone()
            return dict(row) if row else None
