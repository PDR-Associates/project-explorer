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

        def _val(key, suffix=""):
            v = d.get(key)
            if v is None or v == "":
                return "N/A"
            return f"{v}{suffix}"

        def _loc_fmt(n):
            if n is None:
                return "N/A"
            n = int(n)
            if n >= 1_000_000:
                return f"{n / 1_000_000:.1f}M (estimated)"
            if n >= 1_000:
                return f"{n / 1_000:.1f}K (estimated)"
            return f"{n} (estimated)"

        def _size_fmt(kb):
            if kb is None:
                return "N/A"
            kb = int(kb)
            if kb >= 1024:
                return f"{kb / 1024:.1f} MB"
            return f"{kb} KB"

        lines = [
            f"Project: {project_slug}",
            "",
            "── Repository ──────────────────────────",
            f"  License:            {_val('license')}",
            f"  Topics:             {_val('topics') or 'none'}",
            f"  Created:            {_val('repo_created_at', '')[:10] or 'N/A'}",
            f"  Last pushed:        {_val('last_pushed_at', '')[:10] or 'N/A'}",
            f"  Size:               {_size_fmt(d.get('repo_size_kb'))}",
            f"  Primary language:   {_val('primary_language')}",
            "",
            "── Code ────────────────────────────────",
            f"  Files:              {_val('file_count')}",
            f"  Lines of code:      {_loc_fmt(d.get('lines_of_code'))}",
            f"  Language breakdown: {_val('language_breakdown')}",
            "",
            "── Community ───────────────────────────",
            f"  Stars:              {_val('stars')}",
            f"  Forks:              {_val('forks')}",
            f"  Watchers:           {_val('watchers')}",
            f"  Contributors:       {_val('contributors_count')}",
            f"  Open issues:        {_val('open_issues')}",
            "",
            "── Activity ────────────────────────────",
            f"  Commits (30 days):  {_val('commits_30d')}",
            f"  Commits (90 days):  {_val('commits_90d')}",
            "",
            "── Releases ────────────────────────────",
            f"  Total releases:     {_val('releases_count')}",
            f"  Latest release:     {_val('latest_release')}",
            f"  Latest release at:  {_val('latest_release_at', '')[:10] or 'N/A'}",
            f"  Avg interval:       {_val('avg_release_interval_days', ' days')}",
            "",
            f"Stats as of: {_val('fetched_at', '')[:19] or 'N/A'}",
        ]

        if len(history) > 1:
            oldest = dict(history[-1])
            newest = dict(history[0])
            star_diff = (newest.get("stars") or 0) - (oldest.get("stars") or 0)
            fork_diff = (newest.get("forks") or 0) - (oldest.get("forks") or 0)
            since = (oldest.get("fetched_at") or "")[:10]
            if star_diff or fork_diff:
                lines += ["", f"Trends since {since}:"]
                if star_diff:
                    lines.append(f"  Stars: {'+' if star_diff > 0 else ''}{star_diff}")
                if fork_diff:
                    lines.append(f"  Forks: {'+' if fork_diff > 0 else ''}{fork_diff}")

        return "\n".join(lines)
