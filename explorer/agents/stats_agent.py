"""Statistics agent — answers quantitative questions from GitHub API + SQLite time-series."""
from __future__ import annotations

import sqlite3

from explorer.agents.base import BaseExplorerAgent
from explorer.prompt_templates import stats_agent_system_prompt
from explorer.registry import ProjectRegistry


class StatsAgent(BaseExplorerAgent):
    """
    Handles statistical queries (commits, stars, contributors, releases, LOC, etc.).
    Sources: SQLite project_stats table (fast, offline) + live GitHub API (optional).
    Renders charts via Plotext (terminal) or Plotly (web/export).

    Does NOT use the vector store — all data is structured, not semantic.
    """

    def system_prompt(self) -> str:
        return stats_agent_system_prompt()

    def tools(self) -> list:
        return []

    def handle(self, query: str, project_slug: str | None = None, **kwargs) -> str:
        stats = self._fetch_stats(project_slug)
        if not stats:
            return "No statistics available. Run 'project-explorer refresh' to fetch project stats."
        from explorer.llm_client import get_llm
        prompt = f"{self.system_prompt()}\n\nStats data:\n{stats}\n\nQuestion: {query}\n\nAnswer:"
        return get_llm().complete(prompt)

    def _fetch_stats(self, project_slug: str | None) -> str:
        if not project_slug:
            return ""

        registry = ProjectRegistry()
        project_slug = registry._normalize_slug(project_slug)
        try:
            conn = sqlite3.connect(registry.db_path)
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM project_stats WHERE project_slug = ? ORDER BY fetched_at DESC LIMIT 1",
                (project_slug,),
            ).fetchone()
            # Fetch up to 4 historical rows for trend context
            history = conn.execute(
                """SELECT fetched_at, stars, commits_30d, forks
                   FROM project_stats WHERE project_slug = ?
                   ORDER BY fetched_at DESC LIMIT 4""",
                (project_slug,),
            ).fetchall()
            conn.close()
        except Exception:
            return ""

        if not row:
            return ""

        d = dict(row)
        lines = [
            f"Project: {project_slug}",
            f"Stars: {d.get('stars', 'N/A')}",
            f"Forks: {d.get('forks', 'N/A')}",
            f"Watchers: {d.get('watchers', 'N/A')}",
            f"Open Issues: {d.get('open_issues', 'N/A')}",
            f"Contributors: {d.get('contributors_count', 'N/A')}",
            f"Commits (last 30 days): {d.get('commits_30d', 'N/A')}",
            f"Commits (last 90 days): {d.get('commits_90d', 'N/A')}",
            f"Total Releases: {d.get('releases_count', 'N/A')}",
            f"Latest Release: {d.get('latest_release', 'N/A')}",
            f"Latest Release Date: {d.get('latest_release_at', 'N/A')}",
            f"File Count: {d.get('file_count', 'N/A')}",
            f"Primary Language: {d.get('primary_language', 'N/A')}",
            f"Language Breakdown (bytes of code): {d.get('language_breakdown', 'N/A')}",
            f"Stats as of: {d.get('fetched_at', 'N/A')}",
        ]

        if len(history) > 1:
            oldest = dict(history[-1])
            newest = dict(history[0])
            star_diff = (newest.get("stars") or 0) - (oldest.get("stars") or 0)
            fork_diff = (newest.get("forks") or 0) - (oldest.get("forks") or 0)
            since = (oldest.get("fetched_at") or "")[:10]
            if star_diff or fork_diff:
                lines.append("")
                lines.append(f"Trends since {since}:")
                if star_diff:
                    lines.append(f"  Stars: {'+' if star_diff > 0 else ''}{star_diff}")
                if fork_diff:
                    lines.append(f"  Forks: {'+' if fork_diff > 0 else ''}{fork_diff}")

        return "\n".join(lines)
