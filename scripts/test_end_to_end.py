"""End-to-end test suite — validates all system components."""
from __future__ import annotations

import argparse
import sys

from rich.console import Console

console = Console()


def test_environment() -> bool:
    console.print("[bold]Checking environment...[/bold]")
    ok = True
    try:
        from explorer.config import get_config
        get_config()
        console.print("  [green]✓[/green] Config loaded")
    except Exception as e:
        console.print(f"  [red]✗[/red] Config: {e}")
        ok = False
    return ok


def test_registry() -> bool:
    console.print("[bold]Testing project registry...[/bold]")
    try:
        from explorer.registry import Project, ProjectRegistry
        r = ProjectRegistry(db_path="/tmp/test_registry.db")
        p = Project(slug="test", display_name="Test", github_url="https://github.com/test/test")
        r.add(p)
        assert r.get("test") is not None
        r.remove("test")
        assert r.get("test") is None
        console.print("  [green]✓[/green] Registry CRUD")
        return True
    except Exception as e:
        console.print(f"  [red]✗[/red] Registry: {e}")
        return False


def test_query_processor() -> bool:
    console.print("[bold]Testing intent classification...[/bold]")
    try:
        from explorer.query_processor import QueryIntent, QueryProcessor
        qp = QueryProcessor()
        assert qp.classify("how many commits last month") == QueryIntent.STATISTICAL
        assert qp.classify("compare project-a and project-b") == QueryIntent.COMPARISON
        assert qp.classify("is this project actively maintained") == QueryIntent.HEALTH
        console.print("  [green]✓[/green] Intent classifier")
        return True
    except Exception as e:
        console.print(f"  [red]✗[/red] QueryProcessor: {e}")
        return False


def test_cache() -> bool:
    console.print("[bold]Testing query cache...[/bold]")
    try:
        from explorer.query_cache import QueryCache
        cache = QueryCache(max_size=10, ttl_seconds=60)
        cache.set("what is X", None, "general", "X is a thing")
        assert cache.get("what is X", None, "general") == "X is a thing"
        assert cache.get("what is Y", None, "general") is None
        console.print("  [green]✓[/green] Query cache")
        return True
    except Exception as e:
        console.print(f"  [red]✗[/red] Cache: {e}")
        return False


def run(quick: bool = False) -> None:
    tests = [test_environment, test_registry, test_query_processor, test_cache]
    results = [t() for t in tests]
    passed = sum(results)
    console.print(f"\n[bold]{passed}/{len(results)} tests passed[/bold]")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    run(quick=args.quick)
