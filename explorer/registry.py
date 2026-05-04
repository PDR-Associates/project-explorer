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
    subproject_path: str = ""   # relative subdir to index, e.g. "commands" — "" means full repo
    parent_slug: str = ""       # slug of the parent project when this is a sub-project
    extra_docs_paths: list[str] = field(default_factory=list)  # repo-relative paths outside subproject_path to ingest as docs/examples


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
                    error_message TEXT DEFAULT '',
                    subproject_path TEXT DEFAULT '',
                    parent_slug TEXT DEFAULT '',
                    extra_docs_paths TEXT DEFAULT '[]'
                )
            """)
            # Migrations: add new columns to existing databases
            existing = {r[1] for r in conn.execute("PRAGMA table_info(projects)")}
            for col, defn in [
                ("last_commit_sha", "TEXT DEFAULT ''"),
                ("subproject_path", "TEXT DEFAULT ''"),
                ("parent_slug", "TEXT DEFAULT ''"),
                ("extra_docs_paths", "TEXT DEFAULT '[]'"),
            ]:
                if col not in existing:
                    conn.execute(f"ALTER TABLE projects ADD COLUMN {col} {defn}")
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
                ("ingestion_file_count", "INTEGER DEFAULT NULL"),
                ("ingestion_lines_of_code", "INTEGER DEFAULT NULL"),
                ("commits_365d", "INTEGER DEFAULT NULL"),
            ]:
                if col not in existing_stats:
                    conn.execute(f"ALTER TABLE project_stats ADD COLUMN {col} {defn}")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS project_commits (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_slug TEXT NOT NULL,
                    sha TEXT NOT NULL,
                    message TEXT DEFAULT '',
                    author_name TEXT DEFAULT '',
                    author_email TEXT DEFAULT '',
                    committed_at TEXT NOT NULL,
                    UNIQUE(project_slug, sha),
                    FOREIGN KEY (project_slug) REFERENCES projects(slug)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS project_code_symbols (
                    id             INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_slug   TEXT NOT NULL,
                    file_path      TEXT NOT NULL,
                    language       TEXT NOT NULL,
                    kind           TEXT NOT NULL,
                    name           TEXT NOT NULL,
                    qualified_name TEXT NOT NULL,
                    signature      TEXT DEFAULT '',
                    docstring      TEXT DEFAULT '',
                    summary        TEXT DEFAULT '',
                    start_line     INTEGER DEFAULT 0,
                    end_line       INTEGER DEFAULT 0,
                    UNIQUE(project_slug, file_path, qualified_name),
                    FOREIGN KEY (project_slug) REFERENCES projects(slug)
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_symbols_slug_kind "
                "ON project_code_symbols(project_slug, kind)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_symbols_name "
                "ON project_code_symbols(project_slug, name)"
            )
            conn.execute("""
                CREATE TABLE IF NOT EXISTS project_aliases (
                    alias        TEXT PRIMARY KEY,
                    project_slug TEXT NOT NULL,
                    confirmed_by TEXT DEFAULT 'user',
                    created_at   TEXT NOT NULL,
                    FOREIGN KEY (project_slug) REFERENCES projects(slug)
                )
            """)
            # Migrations: add additions / deletions to existing project_commits rows
            existing_commits = {r[1] for r in conn.execute("PRAGMA table_info(project_commits)")}
            for col, defn in [
                ("additions", "INTEGER DEFAULT NULL"),
                ("deletions", "INTEGER DEFAULT NULL"),
            ]:
                if col not in existing_commits:
                    conn.execute(f"ALTER TABLE project_commits ADD COLUMN {col} {defn}")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS project_contributor_stats (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_slug  TEXT NOT NULL,
                    period_start  TEXT NOT NULL,
                    period_end    TEXT NOT NULL,
                    author_email  TEXT NOT NULL,
                    author_name   TEXT NOT NULL,
                    commits       INTEGER DEFAULT 0,
                    additions     INTEGER DEFAULT 0,
                    deletions     INTEGER DEFAULT 0,
                    tier          TEXT DEFAULT '',
                    UNIQUE(project_slug, period_start, period_end, author_email),
                    FOREIGN KEY (project_slug) REFERENCES projects(slug)
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_contributor_stats_slug "
                "ON project_contributor_stats(project_slug, period_start)"
            )
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversation_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    turn_idx INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    project_slug TEXT DEFAULT '',
                    created_at TEXT NOT NULL DEFAULT (datetime('now'))
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_conv_session "
                "ON conversation_history(session_id, turn_idx)"
            )
            conn.execute("""
                CREATE TABLE IF NOT EXISTS project_dependencies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_slug TEXT NOT NULL,
                    dep_name TEXT NOT NULL,
                    dep_version TEXT DEFAULT '',
                    dep_type TEXT DEFAULT 'runtime',
                    ecosystem TEXT DEFAULT '',
                    source_file TEXT DEFAULT '',
                    indexed_at TEXT NOT NULL,
                    FOREIGN KEY (project_slug) REFERENCES projects(slug)
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_deps_slug "
                "ON project_dependencies(project_slug)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_deps_name "
                "ON project_dependencies(dep_name)"
            )

    def add(self, project: Project) -> None:
        data = {
            **asdict(project),
            "collections": json.dumps(project.collections),
            "extra_docs_paths": json.dumps(project.extra_docs_paths),
        }
        data["slug"] = self._normalize_slug(data["slug"])
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO projects VALUES (
                    :slug, :display_name, :github_url, :description, :homepage_url,
                    :docs_url, :github_token_encrypted, :collections, :status,
                    :last_indexed_at, :last_stats_fetched_at, :last_commit_sha,
                    :created_at, :error_message, :subproject_path, :parent_slug,
                    :extra_docs_paths
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

    def update_extra_docs_paths(self, slug: str, paths: list[str]) -> None:
        slug = self._normalize_slug(slug)
        with self._conn() as conn:
            conn.execute(
                "UPDATE projects SET extra_docs_paths = ? WHERE slug = ?",
                (json.dumps(paths), slug),
            )

    def update_commit_sha(self, slug: str, sha: str) -> None:
        slug = self._normalize_slug(slug)
        with self._conn() as conn:
            conn.execute(
                "UPDATE projects SET last_commit_sha = ? WHERE slug = ?",
                (sha, slug),
            )

    def update_ingestion_stats(self, slug: str, file_count: int, lines_of_code: int) -> None:
        """Update the most recent project_stats row with counts from actual ingestion."""
        slug = self._normalize_slug(slug)
        with self._conn() as conn:
            row = conn.execute(
                "SELECT id FROM project_stats WHERE project_slug = ? ORDER BY id DESC LIMIT 1",
                (slug,),
            ).fetchone()
            if row:
                conn.execute(
                    "UPDATE project_stats SET ingestion_file_count = ?, ingestion_lines_of_code = ? WHERE id = ?",
                    (file_count, lines_of_code, row["id"]),
                )
            else:
                conn.execute(
                    "INSERT INTO project_stats (project_slug, fetched_at, ingestion_file_count, ingestion_lines_of_code) "
                    "VALUES (?, ?, ?, ?)",
                    (slug, datetime.utcnow().isoformat(), file_count, lines_of_code),
                )

    def upsert_code_symbols(self, project_slug: str, symbols: list) -> None:
        """Insert or replace extracted code symbols. symbols is a list of CodeSymbol dataclasses."""
        if not symbols:
            return
        slug = self._normalize_slug(project_slug)
        rows = [
            (slug, s.file_path, s.language, s.kind, s.name,
             s.qualified_name, s.signature, s.docstring, "", s.start_line, s.end_line)
            for s in symbols
        ]
        with self._conn() as conn:
            conn.executemany(
                """INSERT INTO project_code_symbols
                   (project_slug, file_path, language, kind, name, qualified_name,
                    signature, docstring, summary, start_line, end_line)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(project_slug, file_path, qualified_name)
                   DO UPDATE SET
                     language=excluded.language, kind=excluded.kind, name=excluded.name,
                     signature=excluded.signature, docstring=excluded.docstring,
                     start_line=excluded.start_line, end_line=excluded.end_line""",
                rows,
            )

    def clear_code_symbols(self, project_slug: str, language: str | None = None) -> None:
        """Remove symbol entries for a project, optionally filtered by language."""
        slug = self._normalize_slug(project_slug)
        with self._conn() as conn:
            if language:
                conn.execute(
                    "DELETE FROM project_code_symbols WHERE project_slug = ? AND language = ?",
                    (slug, language),
                )
            else:
                conn.execute(
                    "DELETE FROM project_code_symbols WHERE project_slug = ?", (slug,)
                )

    def upsert_contributor_stats(self, project_slug: str, rows: list[dict]) -> None:
        """Insert or replace aggregated per-contributor stats for a time period."""
        if not rows:
            return
        slug = self._normalize_slug(project_slug)
        with self._conn() as conn:
            conn.executemany(
                """INSERT INTO project_contributor_stats
                   (project_slug, period_start, period_end, author_email, author_name,
                    commits, additions, deletions, tier)
                   VALUES (:project_slug, :period_start, :period_end, :author_email, :author_name,
                           :commits, :additions, :deletions, :tier)
                   ON CONFLICT(project_slug, period_start, period_end, author_email)
                   DO UPDATE SET
                     author_name=excluded.author_name,
                     commits=excluded.commits,
                     additions=excluded.additions,
                     deletions=excluded.deletions,
                     tier=excluded.tier""",
                [{**r, "project_slug": slug} for r in rows],
            )

    def remove(self, slug: str) -> None:
        normalized = self._normalize_slug(slug)
        with self._conn() as conn:
            conn.execute("DELETE FROM projects WHERE slug = ?", (normalized,))
            conn.execute("DELETE FROM project_stats WHERE project_slug = ?", (normalized,))
            conn.execute("DELETE FROM project_commits WHERE project_slug = ?", (normalized,))
            conn.execute("DELETE FROM project_code_symbols WHERE project_slug = ?", (normalized,))
            conn.execute("DELETE FROM project_aliases WHERE project_slug = ?", (normalized,))
            conn.execute("DELETE FROM project_contributor_stats WHERE project_slug = ?", (normalized,))

    # ── alias management ──────────────────────────────────────────────────────

    def add_alias(self, alias: str, slug: str, confirmed_by: str = "user") -> None:
        """Store an alias → slug mapping. Alias is normalized (lowercase, spaces→underscores)."""
        normalized = alias.lower().replace(" ", "_").replace("-", "_")
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO project_aliases (alias, project_slug, confirmed_by, created_at) "
                "VALUES (?, ?, ?, ?)",
                (normalized, self._normalize_slug(slug), confirmed_by, datetime.utcnow().isoformat()),
            )

    def remove_alias(self, alias: str) -> bool:
        """Delete an alias. Returns True if a row was deleted."""
        normalized = alias.lower().replace(" ", "_").replace("-", "_")
        with self._conn() as conn:
            cursor = conn.execute("DELETE FROM project_aliases WHERE alias = ?", (normalized,))
            return cursor.rowcount > 0

    def resolve_alias(self, term: str) -> str | None:
        """Look up a term in the alias table; return the project slug or None."""
        normalized = term.lower().replace(" ", "_").replace("-", "_")
        with self._conn() as conn:
            row = conn.execute(
                "SELECT project_slug FROM project_aliases WHERE alias = ?", (normalized,)
            ).fetchone()
        return row["project_slug"] if row else None

    def list_aliases(self, slug: str | None = None) -> list[dict]:
        """Return all aliases, optionally filtered to a single project."""
        with self._conn() as conn:
            if slug:
                rows = conn.execute(
                    "SELECT alias, project_slug, confirmed_by, created_at FROM project_aliases "
                    "WHERE project_slug = ? ORDER BY alias",
                    (self._normalize_slug(slug),),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT alias, project_slug, confirmed_by, created_at FROM project_aliases "
                    "ORDER BY project_slug, alias"
                ).fetchall()
        return [dict(r) for r in rows]

    def fuzzy_candidate(self, query: str) -> tuple[str, str] | None:
        """
        Return (display_term, project_slug) if a phrase in the query fuzzy-matches a project.
        Uses difflib at 0.70 cutoff. Tries 1–4-word ngrams ordered longest-first.
        Returns None when no confident match exists.
        """
        import difflib
        q = query.lower()
        projects = self.list_all()
        candidates: dict[str, str] = {}  # normalized_name → slug
        for p in projects:
            candidates[p.slug.lower()] = p.slug
            if p.display_name:
                key = p.display_name.lower().replace("-", "_").replace(" ", "_")
                candidates[key] = p.slug
        if not candidates:
            return None
        words = q.split()
        for n in range(min(4, len(words)), 0, -1):
            for i in range(len(words) - n + 1):
                term = "_".join(words[i:i + n])
                matches = difflib.get_close_matches(term, list(candidates.keys()), n=1, cutoff=0.70)
                if matches:
                    display_term = " ".join(words[i:i + n])
                    return display_term, candidates[matches[0]]
        return None

    # ── dependency graph ──────────────────────────────────────────────────────

    def upsert_dependencies(self, slug: str, deps: list[dict]) -> None:
        """Store parsed dependency list for a project, replacing any prior data."""
        if not deps:
            return
        slug = self._normalize_slug(slug)
        with self._conn() as conn:
            conn.execute(
                "DELETE FROM project_dependencies WHERE project_slug = ?", (slug,)
            )
            conn.executemany(
                "INSERT INTO project_dependencies "
                "(project_slug, dep_name, dep_version, dep_type, ecosystem, source_file, indexed_at) "
                "VALUES (:project_slug, :dep_name, :dep_version, :dep_type, :ecosystem, :source_file, :indexed_at)",
                [{**d, "project_slug": slug, "indexed_at": datetime.utcnow().isoformat()} for d in deps],
            )

    def query_dependencies(
        self,
        slug: str,
        dep_type: str | None = None,
        ecosystem: str | None = None,
    ) -> list[dict]:
        slug = self._normalize_slug(slug)
        with self._conn() as conn:
            filters = ["project_slug = ?"]
            params: list = [slug]
            if dep_type and dep_type != "all":
                filters.append("dep_type = ?")
                params.append(dep_type)
            if ecosystem:
                filters.append("ecosystem = ?")
                params.append(ecosystem)
            where = " AND ".join(filters)
            rows = conn.execute(
                f"SELECT dep_name, dep_version, dep_type, ecosystem, source_file "  # noqa: S608
                f"FROM project_dependencies WHERE {where} ORDER BY ecosystem, dep_type, dep_name",
                params,
            ).fetchall()
        return [dict(r) for r in rows]

    def query_shared_dependencies(self, slugs: list[str], dep_name: str | None = None) -> list[dict]:
        """Find dependencies shared across the given list of projects."""
        norms = [self._normalize_slug(s) for s in slugs]
        with self._conn() as conn:
            placeholders = ",".join("?" * len(norms))
            rows = conn.execute(
                f"SELECT dep_name, ecosystem, GROUP_CONCAT(project_slug) as projects, COUNT(*) as project_count "  # noqa: S608
                f"FROM project_dependencies WHERE project_slug IN ({placeholders}) "
                f"GROUP BY dep_name, ecosystem HAVING project_count >= 2 ORDER BY project_count DESC, dep_name",
                norms,
            ).fetchall()
        return [dict(r) for r in rows]

    # ── conversation history ──────────────────────────────────────────────────

    def append_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        project_slug: str | None = None,
    ) -> None:
        """Append a single turn (user or assistant) to the conversation log."""
        with self._conn() as conn:
            next_idx = (
                conn.execute(
                    "SELECT COALESCE(MAX(turn_idx), -1) + 1 FROM conversation_history WHERE session_id = ?",
                    (session_id,),
                ).fetchone()[0]
            )
            conn.execute(
                "INSERT INTO conversation_history (session_id, turn_idx, role, content, project_slug, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (session_id, next_idx, role, content, project_slug or "", datetime.utcnow().isoformat()),
            )

    def load_turns(self, session_id: str, limit: int = 40) -> list[dict]:
        """Return up to `limit` most-recent turns for a session, in chronological order."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT role, content, project_slug FROM conversation_history "
                "WHERE session_id = ? ORDER BY turn_idx DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
        return [dict(r) for r in reversed(rows)]

    def list_sessions(self) -> list[dict]:
        """Return a summary of all stored sessions (id, turn count, last activity)."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT session_id, COUNT(*) as turns, MAX(created_at) as last_at "
                "FROM conversation_history GROUP BY session_id ORDER BY last_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_by_github_url(self, url: str) -> "Project | None":
        """Look up a project by GitHub URL, normalizing away .git suffix and case."""
        normalized = url.lower().rstrip("/").removesuffix(".git")
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM projects").fetchall()
        for row in rows:
            stored = row["github_url"].lower().rstrip("/").removesuffix(".git")
            if stored == normalized:
                return self._row_to_project(row)
        return None

    def exists(self, slug: str) -> bool:
        return self.get(slug) is not None

    def _row_to_project(self, row: sqlite3.Row) -> Project:
        import dataclasses
        d = dict(row)
        d["collections"] = json.loads(d.get("collections") or "[]")
        d["extra_docs_paths"] = json.loads(d.get("extra_docs_paths") or "[]")
        d["status"] = ProjectStatus(d["status"])
        # Filter to only known Project fields to stay forward-compatible with schema changes
        known = {f.name for f in dataclasses.fields(Project)}
        return Project(**{k: v for k, v in d.items() if k in known})
