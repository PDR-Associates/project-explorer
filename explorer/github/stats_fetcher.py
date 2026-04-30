"""Fetches GitHub statistics and writes them to the SQLite project_stats time-series table."""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

from explorer.github.client import GitHubClient
from explorer.registry import ProjectRegistry

_COMMIT_LOOKBACK_DAYS = 90


def _percentile(values: list[int], p: int) -> float:
    """Return the p-th percentile of a sorted integer list (linear interpolation)."""
    if not values:
        return 0.0
    sv = sorted(values)
    idx = (len(sv) - 1) * p / 100
    lo, hi = int(idx), min(int(idx) + 1, len(sv) - 1)
    return sv[lo] + (sv[hi] - sv[lo]) * (idx - lo)

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
        """
        Fetch recent commits, store per-commit additions/deletions, and compute contributor stats.
        Returns row count processed. additions/deletions require one extra API call per new commit;
        stops fetching them gracefully if the rate limit is hit.
        """
        since = datetime.utcnow() - timedelta(days=_COMMIT_LOOKBACK_DAYS)
        commits = repo.get_commits(since=since)  # raises on API failure — caller handles

        # SHAs already stored with non-null additions — skip extra API call for these
        conn = sqlite3.connect(self.registry.db_path)
        existing_with_stats = {
            row[0] for row in conn.execute(
                "SELECT sha FROM project_commits WHERE project_slug = ? AND additions IS NOT NULL",
                (project_slug,),
            ).fetchall()
        }
        conn.close()

        rows = []
        fetch_diff_stats = True  # flipped to False on rate limit
        for c in commits:
            commit = c.commit
            author = commit.author
            if author and author.date:
                d = author.date
                if d.tzinfo is not None:
                    d = d.astimezone(timezone.utc).replace(tzinfo=None)
                committed_at = d.isoformat()
            else:
                committed_at = ""
            if not committed_at:
                continue

            additions = deletions = None
            if fetch_diff_stats and c.sha not in existing_with_stats:
                try:
                    additions = c.stats.additions
                    deletions = c.stats.deletions
                except Exception as exc:
                    # Stop fetching diff stats if rate-limited; other errors are per-commit noise
                    if "rate limit" in str(exc).lower():
                        fetch_diff_stats = False

            rows.append((
                project_slug,
                c.sha,
                (commit.message or "").split("\n")[0][:200],
                author.name if author else "",
                author.email if author else "",
                committed_at,
                additions,
                deletions,
            ))

        if not rows:
            return 0

        conn = sqlite3.connect(self.registry.db_path)
        # Use ON CONFLICT DO UPDATE so additions/deletions get backfilled for rows already stored
        conn.executemany(
            """INSERT INTO project_commits
               (project_slug, sha, message, author_name, author_email, committed_at,
                additions, deletions)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(project_slug, sha) DO UPDATE SET
                 additions = COALESCE(excluded.additions, project_commits.additions),
                 deletions = COALESCE(excluded.deletions, project_commits.deletions)""",
            rows,
        )
        conn.commit()
        conn.close()

        self._compute_contributor_stats(project_slug)
        return len(rows)

    def _compute_contributor_stats(self, project_slug: str) -> None:
        """
        Aggregate per-author commits/additions/deletions for 30d and 90d windows
        and classify each contributor into a tier (core / regular / occasional).
        Tiers are relative to the project's own distribution, not a global threshold.
        """
        now = datetime.utcnow()
        for days in (30, 90):
            cutoff = (now - timedelta(days=days)).isoformat()
            period_start = (now - timedelta(days=days)).date().isoformat()
            period_end = now.date().isoformat()

            conn = sqlite3.connect(self.registry.db_path)
            raw = conn.execute(
                """SELECT author_email, author_name,
                          COUNT(*) AS commits,
                          COALESCE(SUM(additions), 0) AS additions,
                          COALESCE(SUM(deletions), 0) AS deletions
                   FROM project_commits
                   WHERE project_slug = ? AND committed_at >= ?
                   GROUP BY author_email
                   ORDER BY commits DESC""",
                (project_slug, cutoff),
            ).fetchall()
            conn.close()

            if not raw:
                continue

            commit_counts = [r[2] for r in raw]
            p75 = _percentile(commit_counts, 75)
            p25 = _percentile(commit_counts, 25)

            stat_rows = []
            for email, name, commits, additions, deletions in raw:
                if commits >= p75:
                    tier = "core"
                elif commits >= p25:
                    tier = "regular"
                else:
                    tier = "occasional"
                stat_rows.append({
                    "period_start": period_start,
                    "period_end": period_end,
                    "author_email": email or "",
                    "author_name": name or "",
                    "commits": commits,
                    "additions": additions,
                    "deletions": deletions,
                    "tier": tier,
                })
            self.registry.upsert_contributor_stats(project_slug, stat_rows)

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
