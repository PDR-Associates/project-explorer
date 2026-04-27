"""Project Registry — SQLite-backed store for registered GitHub projects."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path


class ProjectStatus(str, Enum):
    ACTIVE = "active"
    INDEXING = "indexing"
    PAUSED = "paused"
    ERROR = "error"


@dataclass
class Project:
    slug: str
    display_name: str
    github_url: str
    description: str = ""
    homepage_url: str = ""
    docs_url: str = ""
    github_token_encrypted: str = ""  # encrypted; "" means use global token
    collections: list[str] = field(default_factory=list)
    status: ProjectStatus = ProjectStatus.ACTIVE
    last_indexed_at: str = ""
    last_stats_fetched_at: str = ""
    last_commit_sha: str = ""
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    error_message: str = ""


class ProjectRegistry:
    def __init__(self, db_path: str = "data/registry.db") -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self._init_schema()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS projects (
                    slug TEXT PRIMARY KEY,
                    display_name TEXT NOT NULL,
                    github_url TEXT NOT NULL,
                    description TEXT DEFAULT '',
                    homepage_url TEXT DEFAULT '',
                    docs_url TEXT DEFAULT '',
                    github_token_encrypted TEXT DEFAULT '',
                    collections TEXT DEFAULT '[]',
                    status TEXT DEFAULT 'active',
                    last_indexed_at TEXT DEFAULT '',
                    last_stats_fetched_at TEXT DEFAULT '',
                    last_commit_sha TEXT DEFAULT '',
                    created_at TEXT NOT NULL,
                    error_message TEXT DEFAULT ''
                )
            """)
            # Migration: add last_commit_sha to existing databases
            existing = {r[1] for r in conn.execute("PRAGMA table_info(projects)")}
            if "last_commit_sha" not in existing:
                conn.execute("ALTER TABLE projects ADD COLUMN last_commit_sha TEXT DEFAULT ''")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS project_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_slug TEXT NOT NULL,
                    fetched_at TEXT NOT NULL,
                    stars INTEGER,
                    forks INTEGER,
                    watchers INTEGER,
                    open_issues INTEGER,
                    contributors_count INTEGER,
                    commits_30d INTEGER,
                    commits_90d INTEGER,
                    releases_count INTEGER,
                    latest_release TEXT,
                    latest_release_at TEXT,
                    avg_release_interval_days INTEGER,
                    lines_of_code INTEGER,
                    file_count INTEGER,
                    repo_size_kb INTEGER,
                    primary_language TEXT,
                    language_breakdown TEXT DEFAULT '{}',
                    license TEXT DEFAULT '',
                    topics TEXT DEFAULT '',
                    repo_created_at TEXT DEFAULT '',
                    last_pushed_at TEXT DEFAULT '',
                    FOREIGN KEY (project_slug) REFERENCES projects(slug)
                )
            """)
            # Migrations: add new columns to existing databases
            existing_stats = {r[1] for r in conn.execute("PRAGMA table_info(project_stats)")}
            for col, defn in [
                ("avg_release_interval_days", "INTEGER DEFAULT 0"),
                ("repo_size_kb", "INTEGER DEFAULT 0"),
                ("license", "TEXT DEFAULT ''"),
                ("topics", "TEXT DEFAULT ''"),
                ("repo_created_at", "TEXT DEFAULT ''"),
                ("last_pushed_at", "TEXT DEFAULT ''"),
            ]:
                if col not in existing_stats:
                    conn.execute(f"ALTER TABLE project_stats ADD COLUMN {col} {defn}")

    def add(self, project: Project) -> None:
        data = {**asdict(project), "collections": json.dumps(project.collections)}
        data["slug"] = self._normalize_slug(data["slug"])
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO projects VALUES (
                    :slug, :display_name, :github_url, :description, :homepage_url,
                    :docs_url, :github_token_encrypted, :collections, :status,
                    :last_indexed_at, :last_stats_fetched_at, :last_commit_sha,
                    :created_at, :error_message
                )""",
                data,
            )

    @staticmethod
    def _normalize_slug(slug: str) -> str:
        return slug.replace("-", "_").lower()

    def get(self, slug: str) -> Project | None:
        normalized = self._normalize_slug(slug)
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM projects WHERE slug = ?", (normalized,)).fetchone()
        return self._row_to_project(row) if row else None

    def list_all(self) -> list[Project]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM projects ORDER BY display_name").fetchall()
        return [self._row_to_project(r) for r in rows]

    def update_status(self, slug: str, status: ProjectStatus, error: str = "") -> None:
        slug = self._normalize_slug(slug)
        with self._conn() as conn:
            conn.execute(
                "UPDATE projects SET status = ?, error_message = ? WHERE slug = ?",
                (status.value, error, slug),
            )

    def update_indexed_at(self, slug: str, collections: list[str]) -> None:
        slug = self._normalize_slug(slug)
        with self._conn() as conn:
            conn.execute(
                "UPDATE projects SET last_indexed_at = ?, collections = ? WHERE slug = ?",
                (datetime.utcnow().isoformat(), json.dumps(collections), slug),
            )

    def update_commit_sha(self, slug: str, sha: str) -> None:
        slug = self._normalize_slug(slug)
        with self._conn() as conn:
            conn.execute(
                "UPDATE projects SET last_commit_sha = ? WHERE slug = ?",
                (sha, slug),
            )

    def remove(self, slug: str) -> None:
        normalized = self._normalize_slug(slug)
        with self._conn() as conn:
            conn.execute("DELETE FROM projects WHERE slug = ?", (normalized,))
            conn.execute("DELETE FROM project_stats WHERE project_slug = ?", (normalized,))

    def exists(self, slug: str) -> bool:
        return self.get(slug) is not None

    def _row_to_project(self, row: sqlite3.Row) -> Project:
        d = dict(row)
        d["collections"] = json.loads(d.get("collections") or "[]")
        d["status"] = ProjectStatus(d["status"])
        return Project(**d)
