"""
Lightweight SQLite logging for harness runs.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone

DB_PATH = os.getenv(
    "HARNESS_DB_PATH",
    os.path.join(os.path.dirname(__file__), "..", "data", "harness_runs.db"),
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS harness_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    query TEXT NOT NULL,
    answer TEXT NOT NULL,
    route TEXT NOT NULL,
    flagged INTEGER NOT NULL,
    groundedness_score REAL,
    self_consistency_flagged INTEGER,
    confidence_flagged INTEGER,
    label TEXT,
    detail_json TEXT
);
"""


def get_conn() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(SCHEMA)
    return conn


def log_run(
    query: str,
    answer: str,
    route: str,
    flagged: bool,
    groundedness_score: float | None,
    self_consistency_flagged: bool | None,
    confidence_flagged: bool | None,
    detail: dict,
    label: str | None = None,
) -> None:
    conn = get_conn()
    conn.execute(
        """INSERT INTO harness_runs
           (timestamp, query, answer, route, flagged, groundedness_score,
            self_consistency_flagged, confidence_flagged, label, detail_json)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            datetime.now(timezone.utc).isoformat(),
            query, answer, route, int(flagged), groundedness_score,
            int(self_consistency_flagged) if self_consistency_flagged is not None else None,
            int(confidence_flagged) if confidence_flagged is not None else None,
            label,
            json.dumps(detail, default=str),
        ),
    )
    conn.commit()
    conn.close()


def fetch_all_runs() -> list[dict]:
    conn = get_conn()
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM harness_runs ORDER BY timestamp DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]