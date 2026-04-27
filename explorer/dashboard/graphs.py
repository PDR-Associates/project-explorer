"""Graph builders — Plotext for terminal output, Plotly for web/export."""
from __future__ import annotations

import json
import sqlite3

from explorer.registry import ProjectRegistry


# ── shared data helpers ───────────────────────────────────────────────────────

def _load_history(project_slug: str, limit: int = 12) -> list[dict]:
    """Return up to `limit` project_stats rows ordered oldest → newest."""
    registry = ProjectRegistry()
    try:
        conn = sqlite3.connect(registry.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """SELECT * FROM project_stats
               WHERE project_slug = ?
               ORDER BY fetched_at ASC
               LIMIT ?""",
            (project_slug, limit),
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception:
        return []


def _latest_row(project_slug: str) -> dict:
    registry = ProjectRegistry()
    try:
        conn = sqlite3.connect(registry.db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM project_stats WHERE project_slug = ? ORDER BY fetched_at DESC LIMIT 1",
            (project_slug,),
        ).fetchone()
        conn.close()
        return dict(row) if row else {}
    except Exception:
        return {}


# ── terminal charts (Plotext) ─────────────────────────────────────────────────

def commits_over_time_terminal(project_slug: str) -> None:
    """Print a commit-frequency bar chart to the terminal using Plotext."""
    import plotext as plt

    rows = _load_history(project_slug)
    if not rows:
        print(f"No stats data for '{project_slug}'. Run 'project-explorer refresh' first.")
        return

    dates = [r["fetched_at"][:10] for r in rows]
    counts = [r.get("commits_30d") or 0 for r in rows]

    plt.clf()
    plt.title(f"Commits (30-day window) — {project_slug}")
    plt.xlabel("Snapshot date")
    plt.ylabel("Commits")
    plt.bar(dates, counts)
    plt.show()


def stars_over_time_terminal(project_slug: str) -> None:
    """Print a star-growth line chart to the terminal using Plotext."""
    import plotext as plt

    rows = _load_history(project_slug)
    if not rows:
        print(f"No stats data for '{project_slug}'. Run 'project-explorer refresh' first.")
        return

    dates = [r["fetched_at"][:10] for r in rows]
    stars = [r.get("stars") or 0 for r in rows]

    plt.clf()
    plt.title(f"Star growth — {project_slug}")
    plt.xlabel("Snapshot date")
    plt.ylabel("Stars")
    plt.plot(dates, stars, marker="hd")
    plt.show()


# ── web charts (Plotly) ───────────────────────────────────────────────────────

def stars_over_time_plotly(project_slug: str) -> "plotly.graph_objects.Figure":
    """Return a Plotly figure for star growth over time."""
    import plotly.graph_objects as go

    rows = _load_history(project_slug)
    dates = [r["fetched_at"][:10] for r in rows]
    stars = [r.get("stars") or 0 for r in rows]

    fig = go.Figure(go.Scatter(x=dates, y=stars, mode="lines+markers", name="Stars"))
    fig.update_layout(
        title=f"Stars over time — {project_slug}",
        xaxis_title="Date",
        yaxis_title="Stars",
    )
    return fig


def commits_over_time_plotly(project_slug: str) -> "plotly.graph_objects.Figure":
    """Return a Plotly figure for commit frequency over time."""
    import plotly.graph_objects as go

    rows = _load_history(project_slug)
    dates = [r["fetched_at"][:10] for r in rows]
    counts = [r.get("commits_30d") or 0 for r in rows]

    fig = go.Figure(go.Bar(x=dates, y=counts, name="Commits (30d)"))
    fig.update_layout(
        title=f"Commit activity — {project_slug}",
        xaxis_title="Snapshot date",
        yaxis_title="Commits (30-day window)",
    )
    return fig


def language_breakdown_plotly(project_slug: str) -> "plotly.graph_objects.Figure":
    """Return a Plotly pie chart of language breakdown."""
    import plotly.graph_objects as go

    row = _latest_row(project_slug)
    labels: list[str] = []
    values: list[int] = []

    if row.get("language_breakdown"):
        try:
            raw = row["language_breakdown"]
            # StatsFetcher stores as str(dict); try json first, then eval-safe ast
            try:
                breakdown: dict = json.loads(raw)
            except json.JSONDecodeError:
                import ast
                breakdown = ast.literal_eval(raw)
            labels = list(breakdown.keys())
            values = list(breakdown.values())
        except Exception:
            pass

    fig = go.Figure(go.Pie(labels=labels, values=values))
    fig.update_layout(title=f"Language breakdown — {project_slug}")
    return fig


def health_radar_plotly(project_slug: str) -> "plotly.graph_objects.Figure":
    """Return a Plotly radar chart of project health dimensions."""
    import plotly.graph_objects as go

    row = _latest_row(project_slug)
    if not row:
        fig = go.Figure()
        fig.update_layout(title=f"No data — {project_slug}")
        return fig

    commits_30d = row.get("commits_30d") or 0
    contributors = row.get("contributors_count") or 0
    stars = row.get("stars") or 0
    releases = row.get("releases_count") or 0
    open_issues = row.get("open_issues") or 0

    # Normalize to 0–10 scale (rough heuristics for OSS projects)
    activity = min(commits_30d / 3, 10)
    community = min(contributors / 2, 10)
    popularity = min(stars / 1000, 10)
    release_health = min(releases / 2, 10)
    issue_health = max(10 - open_issues / 20, 0)

    dimensions = ["Activity", "Community", "Popularity", "Releases", "Issue Health"]
    scores = [activity, community, popularity, release_health, issue_health]

    fig = go.Figure(go.Scatterpolar(
        r=scores + [scores[0]],
        theta=dimensions + [dimensions[0]],
        fill="toself",
        name=project_slug,
    ))
    fig.update_layout(
        polar={"radialaxis": {"visible": True, "range": [0, 10]}},
        title=f"Project health — {project_slug}",
    )
    return fig
