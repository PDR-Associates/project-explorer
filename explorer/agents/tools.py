"""BeeAI tool definitions — shared across all explorer agents."""
from __future__ import annotations

from beeai_framework.tools import tool


@tool(description=(
    "Search indexed project content (code, docs, API specs) for text relevant to the query. "
    "collection_names is a comma-separated list of fully-qualified collection names, "
    "e.g. 'unitycatalog_python_code,unitycatalog_markdown_docs'. "
    "Call multiple times with different queries or collections to gather more context."
))
def vector_search(query: str, collection_names: str) -> str:
    from explorer.multi_collection_store import MultiCollectionStore
    collections = [c.strip() for c in collection_names.split(",") if c.strip()]
    if not collections:
        return "Error: no collection names provided."
    results = MultiCollectionStore().search(query, collections)
    if not results:
        return "No relevant content found in the specified collections."
    parts = [f"[{r.collection} | score={r.score:.2f}]\n{r.text}" for r in results]
    return "\n\n---\n\n".join(parts)


@tool(description=(
    "Get GitHub statistics for a project: stars, forks, watchers, open issues, contributors, "
    "commits (30d/90d), releases, lines of code, file count, primary language, language breakdown, "
    "license, topics, repo size, creation date, last pushed date."
))
def query_project_stats(project_slug: str) -> str:
    import sqlite3
    from datetime import datetime, timedelta, timezone
    from explorer.registry import ProjectRegistry
    registry = ProjectRegistry()
    slug = registry._normalize_slug(project_slug)
    try:
        conn = sqlite3.connect(registry.db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM project_stats WHERE project_slug = ? ORDER BY fetched_at DESC LIMIT 1",
            (slug,),
        ).fetchone()
        # Derive live commit counts from project_commits for consistency with query_top_committers
        now = datetime.now(timezone.utc)
        cutoff_30 = (now - timedelta(days=30)).isoformat()
        cutoff_90 = (now - timedelta(days=90)).isoformat()
        live_30 = conn.execute(
            "SELECT COUNT(*) FROM project_commits WHERE project_slug = ? AND committed_at >= ?",
            (slug, cutoff_30),
        ).fetchone()[0]
        live_90 = conn.execute(
            "SELECT COUNT(*) FROM project_commits WHERE project_slug = ? AND committed_at >= ?",
            (slug, cutoff_90),
        ).fetchone()[0]
        conn.close()
    except Exception as e:
        return f"Error reading stats: {e}"
    if not row:
        return f"No stats found for '{slug}'. Run: project-explorer refresh {slug}"
    d = dict(row)

    # Prefer live counts from project_commits; fall back to snapshot if no commit data
    commits_30d = live_30 if live_30 or live_90 else (d.get("commits_30d") or 0)
    commits_90d = live_90 if live_30 or live_90 else (d.get("commits_90d") or 0)

    def v(key, suffix=""):
        val = d.get(key)
        return "N/A" if val is None or val == "" else f"{val}{suffix}"

    loc = d.get("lines_of_code")
    if loc:
        loc = int(loc)
        loc_str = f"{loc / 1_000_000:.1f}M (est.)" if loc >= 1_000_000 else f"{loc / 1_000:.1f}K (est.)"
    else:
        loc_str = "N/A"

    kb = d.get("repo_size_kb")
    size_str = f"{int(kb) / 1024:.1f} MB" if kb and int(kb) >= 1024 else (f"{kb} KB" if kb else "N/A")

    return "\n".join([
        f"Project: {slug}  (stats as of {v('fetched_at')[:19]})",
        f"Stars: {v('stars')}  Forks: {v('forks')}  Watchers: {v('watchers')}  Open issues: {v('open_issues')}",
        f"Contributors: {v('contributors_count')}  Commits 30d: {commits_30d}  Commits 90d: {commits_90d}",
        f"Releases: {v('releases_count')}  Latest: {v('latest_release')} ({v('latest_release_at')[:10]})",
        f"Avg release interval: {v('avg_release_interval_days')} days",
        f"Primary language: {v('primary_language')}  LOC: {loc_str}  Files: {v('file_count')}  Size: {size_str}",
        f"Languages: {v('language_breakdown')}",
        f"License: {v('license')}  Topics: {v('topics')}",
        f"Created: {v('repo_created_at')[:10]}  Last pushed: {v('last_pushed_at')[:10]}",
    ])


@tool(description=(
    "Get the top committers to a project by commit count over the last 90 days. "
    "Returns a ranked list with commit counts and email addresses."
))
def query_top_committers(project_slug: str, limit: int = 10) -> str:
    import sqlite3
    from collections import Counter
    from explorer.registry import ProjectRegistry
    registry = ProjectRegistry()
    slug = registry._normalize_slug(project_slug)
    try:
        conn = sqlite3.connect(registry.db_path)
        rows = conn.execute(
            "SELECT author_name, author_email FROM project_commits WHERE project_slug = ?",
            (slug,),
        ).fetchall()
        conn.close()
    except Exception as e:
        return f"Error reading commits: {e}"
    if not rows:
        return f"No commit data for '{slug}'. Run: project-explorer refresh {slug}"
    counter: Counter = Counter()
    for name, email in rows:
        label = name or email or "unknown"
        counter[label] += 1
    top = counter.most_common(limit)
    lines = [f"Top {min(limit, len(top))} committers to {slug} (last 90 days):"]
    for i, (name, count) in enumerate(top, 1):
        lines.append(f"  {i}. {name}: {count} commit{'s' if count != 1 else ''}")
    return "\n".join(lines)


@tool(description=(
    "Get weekly commit activity trends for a project over the last 12 weeks. "
    "Shows commit counts per week with a simple bar chart."
))
def query_commit_activity(project_slug: str) -> str:
    import sqlite3
    from collections import defaultdict
    from datetime import datetime, timedelta
    from explorer.registry import ProjectRegistry
    registry = ProjectRegistry()
    slug = registry._normalize_slug(project_slug)
    try:
        conn = sqlite3.connect(registry.db_path)
        rows = conn.execute(
            "SELECT committed_at FROM project_commits WHERE project_slug = ? ORDER BY committed_at DESC",
            (slug,),
        ).fetchall()
        conn.close()
    except Exception as e:
        return f"Error reading commits: {e}"
    if not rows:
        return f"No commit data for '{slug}'. Run: project-explorer refresh {slug}"
    now = datetime.utcnow()
    week_counts: defaultdict = defaultdict(int)
    for (ts,) in rows:
        try:
            dt = datetime.fromisoformat(ts.replace("Z", ""))
            weeks_ago = (now - dt).days // 7
            if weeks_ago < 12:
                week_counts[weeks_ago] += 1
        except Exception:
            pass
    lines = [f"Weekly commit activity for {slug} (most recent first):"]
    for week in range(12):
        count = week_counts.get(week, 0)
        label = "this week" if week == 0 else f"{week}w ago"
        bar = "█" * min(count, 30)
        lines.append(f"  {label:>10}: {bar} {count}")
    return "\n".join(lines)
