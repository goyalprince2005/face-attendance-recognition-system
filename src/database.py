"""
database.py
============
Persistence layer built on SQLite (file-based, zero-config, ships with
Python's standard library - keeps the project fully offline-runnable
from the command line with no external DB server to install).

Two tables:
    students   -> registered identity records
    attendance -> one row per (student, date) attendance event

All SQL lives in this module only, so the rest of the codebase never
writes raw SQL (single responsibility / modular design).
"""

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

from src import config
from src.utils.logger import get_logger

logger = get_logger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS students (
    student_id   TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    label_id     INTEGER UNIQUE NOT NULL,
    sample_count INTEGER NOT NULL DEFAULT 0,
    created_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS attendance (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id   TEXT NOT NULL,
    date         TEXT NOT NULL,
    time         TEXT NOT NULL,
    confidence   REAL NOT NULL,
    source       TEXT NOT NULL DEFAULT 'image',
    FOREIGN KEY (student_id) REFERENCES students (student_id),
    UNIQUE (student_id, date)
);

CREATE INDEX IF NOT EXISTS idx_attendance_date ON attendance (date);
CREATE INDEX IF NOT EXISTS idx_attendance_student ON attendance (student_id);
"""


class Database:
    """Thin wrapper around sqlite3 exposing the operations the app needs."""

    def __init__(self, db_path: Path = config.DB_PATH):
        self.db_path = db_path
        self._init_schema()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self):
        with self._connect() as conn:
            conn.executescript(SCHEMA)
        logger.debug("Database schema verified at %s", self.db_path)

    # ------------------------------------------------------------------
    # Student management
    # ------------------------------------------------------------------
    def add_student(self, student_id: str, name: str, sample_count: int) -> int:
        """Insert a new student and return the auto-assigned label_id
        used internally by the LBPH recognizer (which only understands
        integer labels, not string IDs)."""
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT MAX(label_id) AS m FROM students"
            ).fetchone()
            next_label = 1 if existing["m"] is None else existing["m"] + 1
            conn.execute(
                """INSERT INTO students (student_id, name, label_id, sample_count, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (student_id, name, next_label, sample_count, datetime.now().isoformat()),
            )
            logger.info("Registered student '%s' (%s) as label_id=%s", name, student_id, next_label)
            return next_label

    def student_exists(self, student_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM students WHERE student_id = ?", (student_id,)
            ).fetchone()
            return row is not None

    def get_student_by_label(self, label_id: int) -> Optional[sqlite3.Row]:
        with self._connect() as conn:
            return conn.execute(
                "SELECT * FROM students WHERE label_id = ?", (label_id,)
            ).fetchone()

    def get_student(self, student_id: str) -> Optional[sqlite3.Row]:
        with self._connect() as conn:
            return conn.execute(
                "SELECT * FROM students WHERE student_id = ?", (student_id,)
            ).fetchone()

    def list_students(self):
        with self._connect() as conn:
            return conn.execute(
                "SELECT * FROM students ORDER BY name"
            ).fetchall()

    def delete_student(self, student_id: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM attendance WHERE student_id = ?", (student_id,)
            )
            cur = conn.execute(
                "DELETE FROM students WHERE student_id = ?", (student_id,)
            )
            deleted = cur.rowcount > 0
            if deleted:
                logger.info("Deleted student '%s' and their attendance history", student_id)
            return deleted

    # ------------------------------------------------------------------
    # Attendance management
    # ------------------------------------------------------------------
    def mark_attendance(self, student_id: str, confidence: float, source: str = "image") -> bool:
        """Insert an attendance row for *today*.

        Returns True if a new row was inserted, False if the student was
        already marked present today (duplicate prevention, controlled by
        config.ALLOW_MULTIPLE_MARKS_PER_DAY).
        """
        now = datetime.now()
        date_str = now.strftime(config.DATE_FORMAT)
        time_str = now.strftime(config.TIME_FORMAT)

        with self._connect() as conn:
            if not config.ALLOW_MULTIPLE_MARKS_PER_DAY:
                existing = conn.execute(
                    "SELECT 1 FROM attendance WHERE student_id = ? AND date = ?",
                    (student_id, date_str),
                ).fetchone()
                if existing:
                    logger.debug("Attendance already marked for %s on %s", student_id, date_str)
                    return False

            conn.execute(
                """INSERT INTO attendance (student_id, date, time, confidence, source)
                   VALUES (?, ?, ?, ?, ?)""",
                (student_id, date_str, time_str, confidence, source),
            )
            logger.info(
                "Marked attendance: %s at %s %s (confidence=%.2f, source=%s)",
                student_id, date_str, time_str, confidence, source,
            )
            return True

    def get_attendance(self, student_id: Optional[str] = None,
                        start_date: Optional[str] = None,
                        end_date: Optional[str] = None):
        query = """
            SELECT a.id, a.student_id, s.name, a.date, a.time, a.confidence, a.source
            FROM attendance a
            JOIN students s ON s.student_id = a.student_id
            WHERE 1 = 1
        """
        params = []
        if student_id:
            query += " AND a.student_id = ?"
            params.append(student_id)
        if start_date:
            query += " AND a.date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND a.date <= ?"
            params.append(end_date)
        query += " ORDER BY a.date DESC, a.time DESC"

        with self._connect() as conn:
            return conn.execute(query, params).fetchall()

    def attendance_summary(self):
        """Per-student total attendance count - powers the reporting module."""
        query = """
            SELECT s.student_id, s.name, COUNT(a.id) AS days_present
            FROM students s
            LEFT JOIN attendance a ON a.student_id = s.student_id
            GROUP BY s.student_id
            ORDER BY days_present DESC, s.name ASC
        """
        with self._connect() as conn:
            return conn.execute(query).fetchall()
