"""SQLite-backed query metrics — always available, zero external dependencies."""
from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from explorer.config import get_config


class MetricsCollector:
    def __init__(self) -> None:
        db_path = get_config().observability.metrics_db
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self._init_schema()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS query_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    query_hash TEXT NOT NULL,
                    intent TEXT,
                    project_slug TEXT,
                    latency_ms INTEGER,
                    cache_hit INTEGER DEFAULT 0,
                    response_length INTEGER,
                    feedback INTEGER  -- 1=thumbs up, -1=thumbs down, NULL=no feedback
                )
            """)

    def record_query(
        self,
        query: str,
        intent: str,
        project_slug: str | None,
        response: str,
        latency_ms: int = 0,
        cache_hit: bool = False,
    ) -> None:
        import hashlib
        query_hash = hashlib.sha256(query.encode()).hexdigest()[:16]
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO query_log
                   (timestamp, query_hash, intent, project_slug, latency_ms, cache_hit, response_length)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (datetime.utcnow().isoformat(), query_hash, intent, project_slug,
                 latency_ms, int(cache_hit), len(response)),
            )

    def record_feedback(self, query_hash: str, feedback: int) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE query_log SET feedback = ? WHERE query_hash = ? ORDER BY id DESC LIMIT 1",
                (feedback, query_hash),
            )

    def summary(self) -> dict:
        with self._conn() as conn:
            row = conn.execute("""
                SELECT COUNT(*) as total,
                       AVG(latency_ms) as avg_latency,
                       SUM(cache_hit) as cache_hits,
                       AVG(CASE WHEN feedback IS NOT NULL THEN feedback END) as avg_feedback
                FROM query_log
            """).fetchone()
        return dict(row) if row else {}
