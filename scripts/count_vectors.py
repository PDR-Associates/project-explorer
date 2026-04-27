"""Print vector counts for all registered project collections."""
from __future__ import annotations

from rich.console import Console
from rich.table import Table

from explorer.multi_collection_store import MultiCollectionStore
from explorer.registry import ProjectRegistry


def main() -> None:
    console = Console()
    registry = ProjectRegistry()
    store = MultiCollectionStore()

    table = Table(title="Vector Counts by Collection")
    table.add_column("Project", style="cyan")
    table.add_column("Collection")
    table.add_column("Vectors", justify="right")

    total = 0
    for project in registry.list_all():
        for i, collection in enumerate(project.collections):
            count = store.count(collection)
            total += count
            table.add_row(project.display_name if i == 0 else "", collection, str(count))

    console.print(table)
    console.print(f"\nTotal vectors: [bold]{total:,}[/bold]")


if __name__ == "__main__":
    main()
