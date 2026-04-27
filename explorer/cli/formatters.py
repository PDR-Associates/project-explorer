"""Rich output helpers for the CLI."""
from __future__ import annotations

from rich.console import Console
from rich.table import Table

from explorer.registry import Project


def print_project_table(projects: list[Project], console: Console) -> None:
    if not projects:
        console.print("[yellow]No projects registered. Run: project-explorer add <github-url>[/yellow]")
        return
    table = Table(title="Registered Projects")
    table.add_column("Slug", style="cyan")
    table.add_column("Name")
    table.add_column("Status")
    table.add_column("Collections", justify="right")
    table.add_column("Last Indexed")
    for p in projects:
        status_color = {"active": "green", "indexing": "yellow", "paused": "dim", "error": "red"}.get(
            p.status.value, "white"
        )
        table.add_row(
            p.slug,
            p.display_name,
            f"[{status_color}]{p.status.value}[/{status_color}]",
            str(len(p.collections)),
            p.last_indexed_at[:10] if p.last_indexed_at else "never",
        )
    console.print(table)
