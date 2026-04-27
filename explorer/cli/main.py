"""CLI entry point — all project-explorer commands."""
from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console

app = typer.Typer(
    name="project-explorer",
    help="Multi-agent RAG assistant for exploring GitHub projects.",
    rich_markup_mode="rich",
)
console = Console()


@app.command()
def add(
    github_url: str = typer.Argument(help="GitHub repository URL to add"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Accept all proposed collections without prompting"),
):
    """Add a GitHub project (runs onboarding wizard to detect content and plan ingestion)."""
    from explorer.cli.wizard import OnboardingWizard
    wizard = OnboardingWizard()
    wizard.run(github_url, accept_all=yes)


@app.command()
def remove(
    slug: str = typer.Argument(help="Project slug to remove"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
):
    """Remove a project and drop all its Milvus collections."""
    from explorer.registry import ProjectRegistry
    registry = ProjectRegistry()
    project = registry.get(slug)
    if not project:
        console.print(f"[red]Project '{slug}' not found.[/red]")
        raise typer.Exit(1)
    if not yes:
        typer.confirm(f"Remove '{project.display_name}' and all its collections?", abort=True)
    from explorer.multi_collection_store import MultiCollectionStore
    store = MultiCollectionStore()
    for collection in project.collections:
        store.drop_collection(collection)
    registry.remove(slug)
    console.print(f"[green]Removed {project.display_name}.[/green]")


@app.command(name="list")
def list_projects():
    """List all registered projects and their status."""
    from explorer.cli.formatters import print_project_table
    from explorer.registry import ProjectRegistry
    projects = ProjectRegistry().list_all()
    print_project_table(projects, console)


@app.command()
def ask(
    query: str = typer.Argument(help="Question to ask"),
    project: Optional[str] = typer.Option(None, "--project", "-p", help="Scope to a specific project slug"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Bypass query cache"),
):
    """Ask a one-shot question about a project (or all projects)."""
    import hashlib
    from explorer.rag_system import RAGSystem
    from explorer.observability.feedback_collector import FeedbackCollector
    system = RAGSystem()
    response = system.query(query, project_slug=project)
    console.print(response)
    query_hash = hashlib.sha256(query.encode()).hexdigest()[:16]
    FeedbackCollector().prompt_and_collect(query_hash)


@app.command()
def chat(
    project: Optional[str] = typer.Option(None, "--project", "-p", help="Scope to a specific project slug"),
):
    """Start an interactive multi-turn chat session."""
    from explorer.cli.interactive import InteractiveSession
    InteractiveSession(project_slug=project).run()


@app.command()
def refresh(slug: str = typer.Argument(help="Project slug to re-index")):
    """Incrementally re-index a project (only changed files since last run)."""
    from explorer.ingestion.incremental import IncrementalIndexer
    from explorer.query_cache import QueryCache
    from explorer.registry import ProjectRegistry
    project = ProjectRegistry().get(slug)
    if not project:
        console.print(f"[red]Project '{slug}' not found.[/red]")
        raise typer.Exit(1)
    indexer = IncrementalIndexer()
    indexer.refresh(project)
    dropped = QueryCache().invalidate_project(slug)
    if dropped:
        console.print(f"[dim]Cleared {dropped} cached query result(s) for '{slug}'.[/dim]")


@app.command()
def status():
    """Show environment health: services, projects, collection counts."""
    from explorer.dashboard.terminal_dashboard import print_status
    print_status(console)


@app.command()
def tui():
    """Launch the full-screen Textual TUI (project sidebar + chat + feedback)."""
    from explorer.tui.app import run
    run()


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Bind host"),
    port: int = typer.Option(8080, help="Bind port"),
):
    """Start the AgentStack A2A server (exposes agents to beeai.dev platform)."""
    from explorer.agentstack_server import run as agentstack_run
    console.print(f"[cyan]Starting AgentStack server on {host}:{port}[/cyan]")
    agentstack_run(host=host, port=port)


if __name__ == "__main__":
    app()
