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
            "file_count": row.get("ingestion_file_count") or row.get("file_count"),
            "lines_of_code": row.get("ingestion_lines_of_code") or row.get("lines_of_code"),
            "file_count_exact": row.get("ingestion_file_count") is not None,
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
    """Return Plotly figure JSON for weekly commit activity (last 13 weeks)."""
    from explorer.dashboard.graphs import weekly_commits_plotly
    fig = weekly_commits_plotly(slug)
    return json.loads(fig.to_json())


@router.get("/{slug}/charts/languages")
async def languages_chart(slug: str) -> dict:
    """Return Plotly figure JSON for the language-breakdown pie chart."""
    from explorer.dashboard.graphs import language_breakdown_plotly
    from fastapi import HTTPException
    fig = language_breakdown_plotly(slug)
    fig_dict = json.loads(fig.to_json())
    # Return 404 when the pie has no slices so the UI shows "No data" instead of a blank chart
    if not fig_dict.get("data") or not fig_dict["data"][0].get("labels"):
        raise HTTPException(
            status_code=404,
            detail=f"No language data for '{slug}' — run 'project-explorer refresh {slug}' first",
        )
    return fig_dict


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


@router.get("/{slug}/charts/file_types")
async def file_types_chart(slug: str) -> dict:
    """Return Plotly figure JSON for the file-count-by-extension bar chart."""
    from explorer.dashboard.graphs import file_types_plotly
    fig = file_types_plotly(slug)
    return json.loads(fig.to_json())


@router.get("/{slug}/charts/health")
async def health_chart(slug: str) -> dict:
    """Return Plotly figure JSON for the project-health radar chart."""
    from explorer.dashboard.graphs import health_radar_plotly
    fig = health_radar_plotly(slug)
    return json.loads(fig.to_json())


# ── database charts ───────────────────────────────────────────────────────────

@router.get("/databases/{slug}/schema_distribution")
async def database_schema_distribution(slug: str) -> dict:
    """Return schema size distribution for a database."""
    registry = ProjectRegistry()
    database = registry.get_database(slug)
    if not database:
        raise HTTPException(status_code=404, detail=f"Database '{slug}' not found")
    
    surveys = registry.get_database_surveys(slug)
    if not surveys:
        raise HTTPException(status_code=404, detail=f"No surveys for '{slug}' — run survey first")
    
    latest = surveys[0]
    survey_data = json.loads(latest.get("survey_data", "{}"))
    schemas = survey_data.get("schemas", [])
    
    if not schemas:
        return {"schemas": [], "table_counts": [], "sizes": []}
    
    return {
        "schemas": [s["name"] for s in schemas],
        "table_counts": [s.get("table_count", 0) for s in schemas],
        "column_counts": [s.get("column_count", 0) for s in schemas],
    }


@router.get("/databases/{slug}/table_sizes")
async def database_table_sizes(slug: str, limit: int = 20) -> dict:
    """Return top N tables by row count."""
    registry = ProjectRegistry()
    database = registry.get_database(slug)
    if not database:
        raise HTTPException(status_code=404, detail=f"Database '{slug}' not found")
    
    surveys = registry.get_database_surveys(slug)
    if not surveys:
        raise HTTPException(status_code=404, detail=f"No surveys for '{slug}' — run survey first")
    
    latest = surveys[0]
    survey_data = json.loads(latest.get("survey_data", "{}"))
    
    # Collect all tables from all schemas
    all_tables = []
    for schema in survey_data.get("schemas", []):
        for table in schema.get("tables", []):
            all_tables.append({
                "schema": schema["name"],
                "table": table["name"],
                "rows": table.get("row_count", 0),
                "size_mb": table.get("size_mb", 0),
            })
    
    # Sort by row count and take top N
    all_tables.sort(key=lambda t: t["rows"], reverse=True)
    top_tables = all_tables[:limit]
    
    return {
        "tables": [f"{t['schema']}.{t['table']}" for t in top_tables],
        "row_counts": [t["rows"] for t in top_tables],
        "sizes_mb": [t["size_mb"] for t in top_tables],
    }


@router.get("/databases/{slug}/column_types")
async def database_column_types(slug: str) -> dict:
    """Return distribution of column data types."""
    registry = ProjectRegistry()
    database = registry.get_database(slug)
    if not database:
        raise HTTPException(status_code=404, detail=f"Database '{slug}' not found")
    
    surveys = registry.get_database_surveys(slug)
    if not surveys:
        raise HTTPException(status_code=404, detail=f"No surveys for '{slug}' — run survey first")
    
    latest = surveys[0]
    survey_data = json.loads(latest.get("survey_data", "{}"))
    
    # Count column types across all schemas and tables
    type_counts = {}
    for schema in survey_data.get("schemas", []):
        for table in schema.get("tables", []):
            for column in table.get("columns", []):
                col_type = column.get("data_type", "unknown")
                type_counts[col_type] = type_counts.get(col_type, 0) + 1
    
    # Sort by count
    sorted_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)
    
    return {
        "types": [t[0] for t in sorted_types],
        "counts": [t[1] for t in sorted_types],
    }


@router.get("/databases/{slug}/survey_history")
async def database_survey_history(slug: str, limit: int = 30) -> dict:
    """Return survey history timeline."""
    registry = ProjectRegistry()
    database = registry.get_database(slug)
    if not database:
        raise HTTPException(status_code=404, detail=f"Database '{slug}' not found")
    
    surveys = registry.get_database_surveys(slug)
    if not surveys:
        return {"dates": [], "schema_counts": [], "table_counts": [], "column_counts": []}
    
    # Take most recent N surveys
    recent = surveys[:limit]
    recent.reverse()  # Oldest first for timeline
    
    return {
        "dates": [s.get("surveyed_at", "")[:10] for s in recent],
        "schema_counts": [s.get("schema_count", 0) for s in recent],
        "table_counts": [s.get("table_count", 0) for s in recent],
        "column_counts": [s.get("column_count", 0) for s in recent],
    }


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
