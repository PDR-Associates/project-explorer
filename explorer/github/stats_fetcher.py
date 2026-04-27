"""Fetches GitHub statistics and writes them to the SQLite project_stats time-series table."""
from __future__ import annotations

from datetime import datetime, timedelta

from explorer.github.client import GitHubClient
from explorer.registry import ProjectRegistry


class StatsFetcher:
    """
    Fetches project statistics from GitHub API and persists them to SQLite.
    Called on initial add and during scheduled refresh.

    Metrics collected:
    - stars, forks, watchers, open_issues
    - contributors_count
    - commits in last 30 and 90 days
    - release count + latest release
    - primary language + language breakdown
    - file count (from tree traversal — optional/expensive)
    """

    def __init__(self) -> None:
        self.client = GitHubClient()
        self.registry = ProjectRegistry()

    def fetch(self, project_slug: str) -> dict:
        import sqlite3

        project = self.registry.get(project_slug)
        if not project:
            raise ValueError(f"Project '{project_slug}' not found")

        repo = self.client.get_repo(project.github_url)
        now = datetime.utcnow()

        stats = {
            "project_slug": project_slug,
            "fetched_at": now.isoformat(),
            "stars": repo.stargazers_count,
            "forks": repo.forks_count,
            "watchers": repo.watchers_count,
            "open_issues": repo.open_issues_count,
            "contributors_count": repo.get_contributors().totalCount,
            "commits_30d": self._count_commits(repo, days=30),
            "commits_90d": self._count_commits(repo, days=90),
            "releases_count": repo.get_releases().totalCount,
            "latest_release": self._latest_release_tag(repo),
            "latest_release_at": self._latest_release_date(repo),
            "primary_language": repo.language or "",
            "language_breakdown": self._language_breakdown(repo),
            "file_count": self._count_files(repo),
        }

        # Write to registry.db where project_stats table is initialized
        conn = sqlite3.connect(self.registry.db_path)
        conn.execute("""
            INSERT INTO project_stats
            (project_slug, fetched_at, stars, forks, watchers, open_issues,
             contributors_count, commits_30d, commits_90d, releases_count,
             latest_release, latest_release_at, file_count, primary_language, language_breakdown)
            VALUES (:project_slug, :fetched_at, :stars, :forks, :watchers, :open_issues,
                    :contributors_count, :commits_30d, :commits_90d, :releases_count,
                    :latest_release, :latest_release_at, :file_count, :primary_language, :language_breakdown)
        """, stats)
        conn.commit()
        conn.close()
        return stats

    def _count_commits(self, repo, days: int) -> int:
        since = datetime.utcnow() - timedelta(days=days)
        return repo.get_commits(since=since).totalCount

    def _latest_release_tag(self, repo) -> str:
        try:
            return repo.get_latest_release().tag_name
        except Exception:
            return ""

    def _latest_release_date(self, repo) -> str:
        try:
            return repo.get_latest_release().published_at.isoformat()
        except Exception:
            return ""

    def _language_breakdown(self, repo) -> str:
        """Returns language breakdown as bytes, clearly labeled to avoid LLM misinterpretation."""
        langs = {k: v for k, v in repo.get_languages().items() if isinstance(v, int)}
        parts = [f"{lang}: {bytes_:,} bytes" for lang, bytes_ in sorted(langs.items(), key=lambda x: -x[1])]
        return "; ".join(parts)

    def _count_files(self, repo) -> int:
        """Count files via the git tree (recursive). Returns 0 on error."""
        try:
            tree = repo.get_git_tree(repo.default_branch, recursive=True)
            return sum(1 for e in tree.tree if e.type == "blob")
        except Exception:
            return 0
