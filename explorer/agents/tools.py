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
    "commits, releases, lines of code, file count, primary language, language breakdown, "
    "license, topics, repo size, creation date, last pushed date. "
    "days controls the commit-count window (default 90; use 30 for 30-day counts, 180 for 6-month, etc.)."
))
def query_project_stats(project_slug: str, days: int = 90) -> str:
    import sqlite3
    from datetime import datetime, timedelta
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
        now = datetime.utcnow()
        cutoff = (now - timedelta(days=days)).isoformat()
        cutoff_30 = (now - timedelta(days=30)).isoformat()
        cutoff_90 = (now - timedelta(days=90)).isoformat()
        live_n = conn.execute(
            "SELECT COUNT(*) FROM project_commits WHERE project_slug = ? AND committed_at >= ?",
            (slug, cutoff),
        ).fetchone()[0]
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
    have_live = bool(live_30 or live_90)
    commits_30d = live_30 if have_live else (d.get("commits_30d") or 0)
    commits_90d = live_90 if have_live else (d.get("commits_90d") or 0)

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

    lines = [
        f"Project: {slug}  (stats as of {v('fetched_at')[:19]})",
        f"Stars: {v('stars')}  Forks: {v('forks')}  Watchers: {v('watchers')}  Open issues: {v('open_issues')}",
        f"Contributors: {v('contributors_count')}  Commits 30d: {commits_30d}  Commits 90d: {commits_90d}",
    ]
    if days not in (30, 90):
        lines.append(f"Commits {days}d: {live_n if have_live else 'N/A (run refresh)'}")
    lines += [
        f"Releases: {v('releases_count')}  Latest: {v('latest_release')} ({v('latest_release_at')[:10]})",
        f"Avg release interval: {v('avg_release_interval_days')} days",
        f"Primary language: {v('primary_language')}  LOC: {loc_str}  Files: {v('file_count')}  Size: {size_str}",
        f"Languages: {v('language_breakdown')}",
        f"License: {v('license')}  Topics: {v('topics')}",
        f"Created: {v('repo_created_at')[:10]}  Last pushed: {v('last_pushed_at')[:10]}",
    ]
    return "\n".join(lines)


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


def _query_code_symbols_raw(
    project_slug: str,
    kind: str = "all",
    pattern: str = "",
    file_path: str = "",
    limit: int = 50,
) -> str:
    import sqlite3
    from explorer.registry import ProjectRegistry
    registry = ProjectRegistry()
    slug = registry._normalize_slug(project_slug)
    try:
        conn = sqlite3.connect(registry.db_path)
        conn.row_factory = sqlite3.Row
        filters = ["project_slug = ?"]
        params: list = [slug]
        if kind and kind != "all":
            filters.append("kind = ?")
            params.append(kind)
        if pattern:
            filters.append("name LIKE ?")
            params.append(f"%{pattern}%")
        if file_path:
            filters.append("file_path LIKE ?")
            params.append(f"{file_path}%")
        where = " AND ".join(filters)
        rows = conn.execute(
            f"SELECT kind, qualified_name, signature, docstring, file_path, start_line "  # noqa: S608
            f"FROM project_code_symbols WHERE {where} "
            f"ORDER BY file_path, start_line LIMIT ?",
            params + [limit],
        ).fetchall()
        total = conn.execute(
            f"SELECT COUNT(*) FROM project_code_symbols WHERE {where}",  # noqa: S608
            params,
        ).fetchone()[0]
        conn.close()
    except Exception as exc:
        return f"Error reading symbol table: {exc}"
    if not rows:
        msg = f"No {kind if kind != 'all' else ''} symbols found in '{slug}'"
        return msg + (f" matching '{pattern}'" if pattern else "") + ". Has this project been indexed?"
    label = kind if kind != "all" else "symbol"
    header = f"{total} {label}(s) in {slug}"
    if pattern:
        header += f" matching '{pattern}'"
    if total > limit:
        header += f" — showing first {limit}"
    lines = [header + ":"]
    for r in rows:
        sig = f"  {r['signature']}" if r["signature"] else ""
        doc = f"  # {r['docstring']}" if r["docstring"] else ""
        lines.append(f"\n[{r['kind']}] {r['qualified_name']}{sig}")
        lines.append(f"  {r['file_path']}:{r['start_line']}{doc}")
    return "\n".join(lines)


@tool(description=(
    "List or count classes, functions, methods, and interfaces defined in a project's source code. "
    "kind filters by symbol type: 'class', 'function', 'method', 'interface', 'enum', or 'all' (default). "
    "pattern filters by name substring (case-insensitive). "
    "file_path restricts to a specific file or directory prefix. "
    "Returns qualified name, signature, and docstring for each match."
))
def query_code_symbols(
    project_slug: str,
    kind: str = "all",
    pattern: str = "",
    file_path: str = "",
    limit: int = 50,
) -> str:
    return _query_code_symbols_raw(project_slug, kind=kind, pattern=pattern,
                                   file_path=file_path, limit=limit)


@tool(description=(
    "Get the full signature, docstring, and purpose summary for a named class or method. "
    "name can be a simple name ('parse') or qualified name ('CodeParser.parse'). "
    "Returns file location, signature, and a one-sentence summary of its purpose."
))
def get_symbol_detail(project_slug: str, name: str) -> str:
    import sqlite3
    from explorer.registry import ProjectRegistry
    registry = ProjectRegistry()
    slug = registry._normalize_slug(project_slug)
    try:
        conn = sqlite3.connect(registry.db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM project_code_symbols "
            "WHERE project_slug = ? AND (qualified_name = ? OR name = ?) "
            "ORDER BY CASE kind WHEN 'class' THEN 0 WHEN 'method' THEN 1 ELSE 2 END "
            "LIMIT 1",
            (slug, name, name),
        ).fetchone()
        conn.close()
    except Exception as exc:
        return f"Error: {exc}"
    if not row:
        return (
            f"Symbol '{name}' not found in '{slug}'. "
            f"Use query_code_symbols to browse available symbols."
        )
    r = dict(row)
    summary = r["summary"] or r["docstring"] or _generate_summary(slug, r, registry.db_path)
    lines = [f"[{r['kind']}] {r['qualified_name']}"]
    lines.append(f"File: {r['file_path']}  lines {r['start_line']}–{r['end_line']}")
    if r["signature"]:
        lines.append(f"Signature: {r['signature']}")
    if summary:
        lines.append(f"Purpose: {summary}")
    return "\n".join(lines)


def _generate_summary(project_slug: str, symbol: dict, db_path: str) -> str:
    """Search for the symbol's source in Milvus and ask the LLM for a one-sentence summary."""
    try:
        from explorer.multi_collection_store import MultiCollectionStore
        from explorer.collection_router import CollectionRouter
        from explorer.llm_client import get_llm
        import sqlite3

        query = symbol["qualified_name"]
        collections = CollectionRouter().select(query, project_slug)
        code_collections = [c for c in collections if "code" in c]
        if not code_collections:
            return ""
        results = MultiCollectionStore().search(query, code_collections)
        if not results:
            return ""
        snippet = results[0].text[:600]
        prompt = (
            f"Summarize what this {symbol['kind']} does in one sentence:\n\n{snippet}"
        )
        summary = get_llm().complete(prompt).strip().split("\n")[0]
        # Persist so the next call is instant
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "UPDATE project_code_symbols SET summary = ? "
                "WHERE project_slug = ? AND file_path = ? AND qualified_name = ?",
                (summary, project_slug, symbol["file_path"], symbol["qualified_name"]),
            )
        return summary
    except Exception:
        return ""


@tool(description=(
    "Get weekly commit activity trends for a project. "
    "weeks controls how many weeks of history to show (default 12). "
    "Shows commit counts per week with a simple bar chart, plus lines added/deleted when available."
))
def query_commit_activity(project_slug: str, weeks: int = 12) -> str:
    import sqlite3
    from collections import defaultdict
    from datetime import datetime, timedelta
    from explorer.registry import ProjectRegistry
    registry = ProjectRegistry()
    slug = registry._normalize_slug(project_slug)
    weeks = max(1, min(weeks, 52))
    try:
        conn = sqlite3.connect(registry.db_path)
        rows = conn.execute(
            "SELECT committed_at, additions, deletions FROM project_commits "
            "WHERE project_slug = ? ORDER BY committed_at DESC",
            (slug,),
        ).fetchall()
        conn.close()
    except Exception as e:
        return f"Error reading commits: {e}"
    if not rows:
        return f"No commit data for '{slug}'. Run: project-explorer refresh {slug}"
    now = datetime.utcnow()
    week_commits: defaultdict = defaultdict(int)
    week_adds: defaultdict = defaultdict(int)
    week_dels: defaultdict = defaultdict(int)
    has_churn = False
    for ts, adds, dels in rows:
        try:
            dt = datetime.fromisoformat(ts[:19])
            w = (now - dt).days // 7
            if w < weeks:
                week_commits[w] += 1
                if adds is not None:
                    week_adds[w] += adds
                    week_dels[w] += dels or 0
                    has_churn = True
        except Exception:
            pass
    lines = [f"Weekly commit activity for {slug} (most recent first):"]
    for week in range(weeks):
        count = week_commits.get(week, 0)
        label = "this week" if week == 0 else f"{week}w ago"
        bar = "█" * min(count, 30)
        line = f"  {label:>10}: {bar} {count}"
        if has_churn:
            a, d = week_adds.get(week, 0), week_dels.get(week, 0)
            line += f"  (+{a:,} / -{d:,})"
        lines.append(line)
    if has_churn:
        total_adds = sum(week_adds.values())
        total_dels = sum(week_dels.values())
        lines.append(f"\nTotal over {weeks} weeks: +{total_adds:,} additions / -{total_dels:,} deletions")
    return "\n".join(lines)


@tool(description=(
    "Get a detailed contribution profile for a specific author: commit count, lines added/deleted, "
    "activity tier (core/regular/occasional relative to the project team), and comparison to project median. "
    "author can be a name or email substring — fuzzy matched against stored commits. "
    "days controls the lookback window (default 90)."
))
def query_contributor_profile(project_slug: str, author: str, days: int = 90) -> str:
    import sqlite3
    from datetime import datetime, timedelta
    from explorer.registry import ProjectRegistry
    registry = ProjectRegistry()
    slug = registry._normalize_slug(project_slug)
    days = max(1, min(days, 365))
    try:
        now = datetime.utcnow()
        cutoff = (now - timedelta(days=days)).isoformat()
        conn = sqlite3.connect(registry.db_path)
        conn.row_factory = sqlite3.Row

        # Find matching contributors by name or email substring
        author_lower = author.lower()
        all_authors = conn.execute(
            "SELECT DISTINCT author_name, author_email FROM project_commits WHERE project_slug = ?",
            (slug,),
        ).fetchall()
        matches = [
            (r["author_name"], r["author_email"])
            for r in all_authors
            if author_lower in (r["author_name"] or "").lower()
            or author_lower in (r["author_email"] or "").lower()
        ]
        if not matches:
            conn.close()
            return (
                f"No contributor matching '{author}' found in '{slug}'. "
                f"Use query_top_committers to see available names."
            )

        name, email = matches[0]
        row = conn.execute(
            """SELECT COUNT(*) AS commits,
                      COALESCE(SUM(additions), 0) AS additions,
                      COALESCE(SUM(deletions), 0) AS deletions
               FROM project_commits
               WHERE project_slug = ? AND committed_at >= ?
               AND (author_name = ? OR author_email = ?)""",
            (slug, cutoff, name, email),
        ).fetchone()

        # Project-wide average commits per contributor in this window
        median_row = conn.execute(
            """SELECT AVG(c) AS avg_commits FROM (
                   SELECT COUNT(*) AS c FROM project_commits
                   WHERE project_slug = ? AND committed_at >= ?
                   GROUP BY author_email
               )""",
            (slug, cutoff),
        ).fetchone()

        # Tier from contributor_stats (most recent period)
        tier_row = conn.execute(
            """SELECT tier FROM project_contributor_stats
               WHERE project_slug = ? AND author_email = ?
               ORDER BY period_start DESC LIMIT 1""",
            (slug, email or ""),
        ).fetchone()
        conn.close()

        commits = row["commits"]
        additions = row["additions"]
        deletions = row["deletions"]
        avg_commits = round(median_row["avg_commits"] or 0, 1)
        rel_pct = round((commits - avg_commits) / max(avg_commits, 1) * 100)
        tier = tier_row["tier"] if tier_row else "unknown (run refresh to compute tiers)"

        lines = [f"Contributor: {name or email} in {slug} (last {days} days)"]
        lines.append(f"Commits: {commits}  (project avg: {avg_commits}; {rel_pct:+d}% relative)")
        if additions or deletions:
            net = additions - deletions
            lines.append(
                f"Lines added: {additions:,}  Lines deleted: {deletions:,}  Net: {net:+,}"
            )
        else:
            lines.append("Lines added/deleted: not yet fetched (run refresh to populate)")
        lines.append(f"Activity tier: {tier}")
        if len(matches) > 1:
            also = ", ".join(n or e for n, e in matches[1:4])
            lines.append(f"Other matches: {also}")
        return "\n".join(lines)
    except Exception as exc:
        return f"Error: {exc}"


def _build_example_context_raw(project_slug: str, topic: str) -> str:
    from explorer.multi_collection_store import MultiCollectionStore
    from explorer.registry import ProjectRegistry

    registry = ProjectRegistry()
    slug = registry._normalize_slug(project_slug)

    collection_types = ["examples", "python_code", "api_reference", "markdown_docs"]
    candidate_collections = [f"{slug}_{ctype}" for ctype in collection_types]

    store = MultiCollectionStore()
    client = store._get_client()
    existing = [c for c in candidate_collections if client.has_collection(c)]
    if not existing:
        return f"No indexed collections found for '{slug}'. Run: project-explorer add/refresh {slug}"

    # Targeted searches: topic itself, import/setup patterns, usage examples
    pkg = slug.replace("_", " ")
    searches = [
        (topic, existing, 3),
        (f"from {slug} import", existing, 2),
        (f"import {slug} setup {pkg}", existing, 2),
        ("usage example how to", [c for c in existing if any(t in c for t in ("examples", "markdown"))], 2),
    ]

    seen: set[str] = set()
    sections: list[str] = []
    for query_text, cols, top_k in searches:
        if not cols:
            continue
        for r in store.search(query_text, cols, top_k=top_k):
            if r.text not in seen and r.score >= 0.3:
                seen.add(r.text)
                sections.append(f"[{r.collection} | score={r.score:.2f}]\n{r.text}")

    if not sections:
        return (
            f"No relevant context found for '{topic}' in '{slug}'. "
            "The project may need re-indexing, or try a broader topic."
        )

    # Extract class names that appear as constructors or type annotations in the
    # retrieved chunks and prepend a concrete import hint.  This compensates for
    # the import chunk rarely scoring high enough to be retrieved on its own.
    import re as _re
    combined = "\n".join(sections)
    class_names = _re.findall(r'\b([A-Z][A-Za-z0-9]+)\s*\(', combined)
    known_classes = sorted({
        n for n in class_names
        if n not in {"True", "False", "None"} and not n[0].isdigit()
    })
    import_hint = ""
    if known_classes:
        import_hint = (
            f"IMPORT HINT: These classes come from the '{slug}' package.\n"
            f"  from {slug} import {', '.join(known_classes[:6])}\n\n"
        )

    header = f"Context for: '{topic}' in {slug}\n{'=' * 60}\n\n{import_hint}"
    return header + "\n\n---\n\n".join(sections)


@tool(description=(
    "Gather context for generating a Python code example about a specific topic. "
    "Searches the examples, python_code, api_reference, and markdown_docs collections "
    "and returns relevant snippets ready for code generation. "
    "topic should be a concrete task: e.g. 'connect to server', 'list metadata assets', "
    "'create a glossary term', 'authenticate with token'. "
    "Call multiple times with different topics to gather broader context."
))
def build_example_context(project_slug: str, topic: str) -> str:
    return _build_example_context_raw(project_slug, topic)


@tool(description=(
    "Query the dependency graph for a project. Returns a markdown table of dependencies. "
    "dep_type can be 'runtime', 'dev', 'optional', 'indirect', 'test', or 'all' (default). "
    "Pass a comma-separated list of project_slugs to find shared dependencies across projects."
))
def query_dependencies(project_slug: str, dep_type: str = "all") -> str:
    from explorer.registry import ProjectRegistry
    registry = ProjectRegistry()
    slugs = [s.strip() for s in project_slug.split(",") if s.strip()]
    if not slugs:
        return "No project slug provided."

    if len(slugs) == 1:
        slug = slugs[0]
        deps = registry.query_dependencies(slug, dep_type=dep_type if dep_type != "all" else None)
        if not deps:
            return (
                f"No dependencies found for '{slug}'. "
                f"Run 'project-explorer refresh {slug}' to index them."
            )
        lines = [
            f"Dependencies for **{slug}** (filter: {dep_type}):", "",
            "| Package | Version | Type | Ecosystem | Source |",
            "|---------|---------|------|-----------|--------|",
        ]
        for d in deps:
            lines.append(
                f"| {d['dep_name']} | {d['dep_version'] or '—'} | "
                f"{d['dep_type']} | {d['ecosystem']} | {d['source_file']} |"
            )
        return "\n".join(lines)

    shared = registry.query_shared_dependencies(slugs)
    if not shared:
        return f"No shared dependencies found across: {', '.join(slugs)}"
    lines = [
        f"Shared dependencies across {', '.join(slugs)}:", "",
        "| Package | Ecosystem | Projects |",
        "|---------|-----------|---------|",
    ]
    for d in shared:
        lines.append(f"| {d['dep_name']} | {d['ecosystem']} | {d['projects']} |")
    return "\n".join(lines)
