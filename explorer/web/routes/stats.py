"""Statistics endpoints — project metrics and chart data."""
from __future__ import annotations

import json
import sqlite3

from fastapi import APIRouter, HTTPException

from explorer.registry import ProjectRegistry

router = APIRouter()

_VALID_METRICS = frozenset({
    "stars", "forks", "watchers", "open_issues",
    "contributors_count", "commits_30d", "commits_90d",
    "releases_count",
})


@router.get("/{slug}")
async def get_stats(slug: str) -> dict:
    registry = ProjectRegistry()
    if not registry.exists(slug):
        raise HTTPException(status_code=404, detail=f"Project '{slug}' not found")

    row = _latest_stats(registry.db_path, slug)
    if not row:
        raise HTTPException(status_code=404, detail=f"No stats for '{slug}' — run refresh first")

    # Parse language_breakdown from stored string
    lang_raw = row.get("language_breakdown") or "{}"
    try:
        try:
            lang = json.loads(lang_raw)
        except json.JSONDecodeError:
            import ast
            lang = ast.literal_eval(lang_raw)
    except Exception:
        lang = {}

    return {
        "slug": slug,
        "fetched_at": row.get("fetched_at"),
        "stats": {
            "stars": row.get("stars"),
            "forks": row.get("forks"),
            "watchers": row.get("watchers"),
            "open_issues": row.get("open_issues"),
            "contributors_count": row.get("contributors_count"),
            "commits_30d": row.get("commits_30d"),
            "commits_90d": row.get("commits_90d"),
            "releases_count": row.get("releases_count"),
            "latest_release": row.get("latest_release"),
            "latest_release_at": row.get("latest_release_at"),
            "primary_language": row.get("primary_language"),
            "language_breakdown": lang,
        },
    }


@router.get("/{slug}/history")
async def get_stats_history(
    slug: str,
    metric: str = "stars",
    limit: int = 30,
) -> dict:
    if metric not in _VALID_METRICS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid metric '{metric}'. Valid: {sorted(_VALID_METRICS)}",
        )

    registry = ProjectRegistry()
    if not registry.exists(slug):
        raise HTTPException(status_code=404, detail=f"Project '{slug}' not found")

    rows = _history(registry.db_path, slug, metric, limit)
    return {
        "slug": slug,
        "metric": metric,
        "data": rows,
    }


@router.get("/{slug}/charts/stars")
async def stars_chart(slug: str) -> dict:
    """Return Plotly figure JSON for the star-growth chart."""
    from explorer.dashboard.graphs import stars_over_time_plotly
    fig = stars_over_time_plotly(slug)
    return json.loads(fig.to_json())


@router.get("/{slug}/charts/commits")
async def commits_chart(slug: str) -> dict:
    """Return Plotly figure JSON for the commit-activity chart."""
    from explorer.dashboard.graphs import commits_over_time_plotly
    fig = commits_over_time_plotly(slug)
    return json.loads(fig.to_json())


@router.get("/{slug}/charts/languages")
async def languages_chart(slug: str) -> dict:
    """Return Plotly figure JSON for the language-breakdown pie chart."""
    from explorer.dashboard.graphs import language_breakdown_plotly
    fig = language_breakdown_plotly(slug)
    return json.loads(fig.to_json())


@router.get("/{slug}/charts/top_committers")
async def top_committers_chart(slug: str) -> dict:
    """Return Plotly figure JSON for the top-committers horizontal bar chart."""
    from explorer.dashboard.graphs import top_committers_plotly
    from fastapi import HTTPException
    fig = top_committers_plotly(slug)
    if fig is None:
        raise HTTPException(
            status_code=404,
            detail=f"No commit data for '{slug}' — run 'project-explorer refresh {slug}' first",
        )
    return json.loads(fig.to_json())


@router.get("/{slug}/charts/weekly_commits")
async def weekly_commits_chart(slug: str) -> dict:
    """Return Plotly figure JSON for the weekly commit-activity bar chart."""
    from explorer.dashboard.graphs import weekly_commits_plotly
    fig = weekly_commits_plotly(slug)
    return json.loads(fig.to_json())


@router.get("/compare/charts/stats")
async def compare_stats_chart(slugs: str) -> dict:
    """Return Plotly grouped bar chart comparing stats across comma-separated project slugs.

    Example: GET /api/stats/compare/charts/stats?slugs=proj_a,proj_b
    """
    from explorer.dashboard.graphs import compare_stats_plotly
    slug_list = [s.strip() for s in slugs.split(",") if s.strip()]
    if len(slug_list) < 2:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="Provide at least two comma-separated slugs")
    fig = compare_stats_plotly(slug_list)
    return json.loads(fig.to_json())


@router.get("/{slug}/charts/health")
async def health_chart(slug: str) -> dict:
    """Return Plotly figure JSON for the project-health radar chart."""
    from explorer.dashboard.graphs import health_radar_plotly
    fig = health_radar_plotly(slug)
    return json.loads(fig.to_json())


# ── helpers ───────────────────────────────────────────────────────────────────

def _latest_stats(db_path: str, slug: str) -> dict:
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM project_stats WHERE project_slug = ? ORDER BY fetched_at DESC LIMIT 1",
            (slug,),
        ).fetchone()
        conn.close()
        return dict(row) if row else {}
    except Exception:
        return {}


def _history(db_path: str, slug: str, metric: str, limit: int) -> list[dict]:
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            f"SELECT fetched_at, {metric} FROM project_stats "  # noqa: S608 — metric validated above
            "WHERE project_slug = ? ORDER BY fetched_at ASC LIMIT ?",
            (slug, min(limit, 365)),
        ).fetchall()
        conn.close()
        return [{"date": r["fetched_at"][:10], "value": r[metric]} for r in rows]
    except Exception:
        return []
