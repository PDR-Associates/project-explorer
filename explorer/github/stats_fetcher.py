"""Fetches GitHub statistics and writes them to the SQLite project_stats time-series table."""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

from explorer.github.client import GitHubClient
from explorer.registry import ProjectRegistry

_COMMIT_LOOKBACK_DAYS = 90

# Rough bytes-per-line by language for LOC estimation
_BYTES_PER_LINE: dict[str, int] = {
    "python": 45, "ruby": 42, "go": 45, "javascript": 40, "typescript": 45,
    "java": 52, "c#": 50, "c++": 50, "c": 48, "rust": 50, "kotlin": 48,
    "swift": 46, "shell": 38, "bash": 38, "scala": 50, "r": 40,
    "jupyter notebook": 80, "html": 55, "css": 35, "yaml": 30, "json": 35,
}


class StatsFetcher:
    """
    Fetches project statistics from GitHub API and persists them to SQLite.
    Called on initial add and during scheduled refresh.

    Metrics collected:
    - stars, forks, watchers, open_issues
    - contributors_count
    - commits in last 30 and 90 days
    - release count + latest release + avg release interval
    - primary language + language breakdown (bytes)
    - lines_of_code (estimated from language bytes)
    - file_count (from git tree traversal)
    - repo_size_kb, license, topics
    - repo_created_at, last_pushed_at
    """

    def __init__(self) -> None:
        self.client = GitHubClient()
        self.registry = ProjectRegistry()

    def fetch(self, project_slug: str) -> dict:
        import sqlite3

        project = self.registry.get(project_slug)
        if not project:
            raise ValueError(f"Project '{project_slug}' not found")

        slug = project.slug  # always use normalized slug for DB writes

        repo = self.client.get_repo(project.github_url)
        now = datetime.utcnow()

        releases = list(repo.get_releases())

        stats = {
            "project_slug": slug,
            "fetched_at": now.isoformat(),
            "stars": repo.stargazers_count,
            "forks": repo.forks_count,
            "watchers": repo.watchers_count,
            "open_issues": repo.open_issues_count,
            "contributors_count": repo.get_contributors().totalCount,
            "commits_30d": self._count_commits(repo, days=30),
            "commits_90d": self._count_commits(repo, days=90),
            "releases_count": len(releases),
            "latest_release": self._latest_release_tag(releases),
            "latest_release_at": self._latest_release_date(releases),
            "avg_release_interval_days": self._avg_release_interval(releases),
            "primary_language": repo.language or "",
            "language_breakdown": self._language_breakdown(repo),
            "lines_of_code": self._estimate_loc(repo),
            "file_count": self._count_files(repo),
            "repo_size_kb": repo.size,
            "license": self._license_name(repo),
            "topics": ",".join(repo.get_topics()),
            "repo_created_at": repo.created_at.isoformat() if repo.created_at else "",
            "last_pushed_at": repo.pushed_at.isoformat() if repo.pushed_at else "",
        }

        conn = sqlite3.connect(self.registry.db_path)
        conn.execute("""
            INSERT INTO project_stats
            (project_slug, fetched_at, stars, forks, watchers, open_issues,
             contributors_count, commits_30d, commits_90d, releases_count,
             latest_release, latest_release_at, avg_release_interval_days,
             primary_language, language_breakdown, lines_of_code, file_count,
             repo_size_kb, license, topics, repo_created_at, last_pushed_at)
            VALUES (:project_slug, :fetched_at, :stars, :forks, :watchers, :open_issues,
                    :contributors_count, :commits_30d, :commits_90d, :releases_count,
                    :latest_release, :latest_release_at, :avg_release_interval_days,
                    :primary_language, :language_breakdown, :lines_of_code, :file_count,
                    :repo_size_kb, :license, :topics, :repo_created_at, :last_pushed_at)
        """, stats)
        conn.commit()
        conn.close()
        try:
            count = self._fetch_commits(slug, repo)
            stats["commits_fetched"] = count
        except Exception as exc:
            stats["commits_fetch_error"] = str(exc)
        return stats

    def _fetch_commits(self, project_slug: str, repo) -> int:
        """Fetch recent commits and upsert into project_commits table. Returns row count inserted."""
        since = datetime.utcnow() - timedelta(days=_COMMIT_LOOKBACK_DAYS)
        commits = repo.get_commits(since=since)  # raises on API failure — caller handles
        rows = []
        for c in commits:
            commit = c.commit
            author = commit.author
            if author and author.date:
                d = author.date
                # Normalize to timezone-naive UTC so all stored values are consistent
                if d.tzinfo is not None:
                    d = d.astimezone(timezone.utc).replace(tzinfo=None)
                committed_at = d.isoformat()
            else:
                committed_at = ""
            if not committed_at:
                continue  # skip commits with no date — they break date-based queries
            rows.append((
                project_slug,
                c.sha,
                (commit.message or "").split("\n")[0][:200],
                author.name if author else "",
                author.email if author else "",
                committed_at,
            ))
        if not rows:
            return 0
        conn = sqlite3.connect(self.registry.db_path)
        conn.executemany(
            """INSERT OR IGNORE INTO project_commits
               (project_slug, sha, message, author_name, author_email, committed_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            rows,
        )
        conn.commit()
        conn.close()
        return len(rows)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _count_commits(self, repo, days: int) -> int:
        since = datetime.utcnow() - timedelta(days=days)
        return repo.get_commits(since=since).totalCount

    def _latest_release_tag(self, releases: list) -> str:
        return releases[0].tag_name if releases else ""

    def _latest_release_date(self, releases: list) -> str:
        if releases and releases[0].published_at:
            return releases[0].published_at.isoformat()
        return ""

    def _avg_release_interval(self, releases: list) -> int:
        """Average days between releases (0 if fewer than 2 releases)."""
        dated = [r.published_at for r in releases if r.published_at]
        if len(dated) < 2:
            return 0
        dated.sort(reverse=True)
        gaps = [(dated[i] - dated[i + 1]).days for i in range(len(dated) - 1)]
        return round(sum(gaps) / len(gaps))

    def _language_breakdown(self, repo) -> str:
        langs = {k: v for k, v in repo.get_languages().items() if isinstance(v, int)}
        parts = [f"{lang}: {bytes_:,} bytes"
                 for lang, bytes_ in sorted(langs.items(), key=lambda x: -x[1])]
        return "; ".join(parts)

    def _estimate_loc(self, repo) -> int:
        """Estimate lines of code from language byte counts."""
        langs = {k: v for k, v in repo.get_languages().items() if isinstance(v, int)}
        total = 0
        for lang, bytes_ in langs.items():
            bpl = _BYTES_PER_LINE.get(lang.lower(), 45)
            total += bytes_ // bpl
        return total

    def _count_files(self, repo) -> int:
        try:
            tree = repo.get_git_tree(repo.default_branch, recursive=True)
            return sum(1 for e in tree.tree if e.type == "blob")
        except Exception:
            return 0

    def _license_name(self, repo) -> str:
        try:
            lic = repo.get_license()
            return lic.license.name if lic and lic.license else ""
        except Exception:
            return ""
