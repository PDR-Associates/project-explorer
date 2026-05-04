"""CLI entry point — all project-explorer commands."""
from __future__ import annotations

import os

# Suppress gRPC fork warnings, HuggingFace progress bars, and tokenizer parallelism
# noise that pollutes normal CLI output. Set DEBUG=1 to restore verbose logging.
if not os.environ.get("DEBUG"):
    os.environ.setdefault("GRPC_VERBOSITY", "NONE")
    os.environ.setdefault("GLOG_minloglevel", "3")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TQDM_DISABLE", "1")

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
    subpath: Optional[str] = typer.Option(None, "--subpath", help="Index only this subdirectory (for monorepos)"),
    name: Optional[str] = typer.Option(None, "--name", help="Project slug override (required when --subpath is used)"),
    extra_docs_path: list[str] = typer.Option(
        [], "--extra-docs-path",
        help="Repo-relative path (file or directory) outside --subpath to ingest as docs/examples. "
             "Repeat to add multiple paths, e.g. --extra-docs-path docs/guide.md --extra-docs-path examples/",
    ),
    from_local: Optional[str] = typer.Option(
        None, "--from-local",
        help="Path to a local clone of the repository. Skips the GitHub download — "
             "useful when registering multiple sub-projects from the same repo.",
    ),
):
    """Add a GitHub project (runs onboarding wizard to detect content and plan ingestion)."""
    if subpath and not name:
        console.print("[red]--name is required when using --subpath[/red]")
        raise typer.Exit(1)
    if extra_docs_path and not subpath:
        console.print("[yellow]--extra-docs-path is only useful together with --subpath; ignoring.[/yellow]")
        extra_docs_path = []
    from explorer.cli.wizard import OnboardingWizard
    wizard = OnboardingWizard()
    wizard.run(github_url, accept_all=yes, subproject_path=subpath, slug_override=name,
               extra_docs_paths=extra_docs_path or None, local_path=from_local)


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
def list_projects(
    details: bool = typer.Option(False, "--details", "-d", help="Show collection names and vector counts"),
):
    """List all registered projects and their status."""
    from explorer.cli.formatters import print_project_table
    from explorer.registry import ProjectRegistry
    projects = ProjectRegistry().list_all()
    print_project_table(projects, console, details=details)


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

    # When no explicit project is given, check for a fuzzy alias match
    if not project:
        project = _maybe_resolve_alias(query)

    system = RAGSystem()
    response = system.query(query, project_slug=project)
    console.print(response)
    query_hash = hashlib.sha256(query.encode()).hexdigest()[:16]
    FeedbackCollector().prompt_and_collect(query_hash)


@app.command()
def chat(
    project: Optional[str] = typer.Option(None, "--project", "-p", help="Scope to a specific project slug"),
    session_id: Optional[str] = typer.Option(None, "--session-id", "-s", help="Resume a previous session by ID"),
):
    """Start an interactive multi-turn chat session."""
    from explorer.cli.interactive import InteractiveSession
    InteractiveSession(project_slug=project, session_id=session_id).run()


@app.command(name="add-docs")
def add_docs(
    slug: str = typer.Argument(help="Project slug"),
    docs_url: Optional[str] = typer.Option(None, "--docs-url", help="Documentation site URL to ingest as web_docs"),
    homepage_url: Optional[str] = typer.Option(None, "--homepage", help="Homepage URL to store on the project"),
):
    """Attach a documentation site or homepage to an already-registered project."""
    from explorer.registry import ProjectRegistry
    from explorer.multi_collection_store import MultiCollectionStore

    registry = ProjectRegistry()
    project = registry.get(slug)
    if not project:
        console.print(f"[red]Project '{slug}' not found.[/red]")
        raise typer.Exit(1)

    if homepage_url:
        # Just update the metadata field — no ingestion needed
        import sqlite3
        conn = sqlite3.connect(registry.db_path)
        conn.execute("UPDATE projects SET homepage_url = ? WHERE slug = ?", (homepage_url, project.slug))
        conn.commit()
        conn.close()
        console.print(f"[green]Homepage URL updated for '{slug}'.[/green]")

    if docs_url:
        # Update stored docs_url
        import sqlite3
        conn = sqlite3.connect(registry.db_path)
        conn.execute("UPDATE projects SET docs_url = ? WHERE slug = ?", (docs_url, project.slug))
        conn.commit()
        conn.close()
        console.print(f"[cyan]Ingesting docs from {docs_url}...[/cyan]")
        _ingest_web_docs(project, docs_url, registry)

    if not docs_url and not homepage_url:
        console.print("[yellow]Provide at least --docs-url or --homepage.[/yellow]")
        raise typer.Exit(1)


def _ingest_web_docs(project, docs_url: str, registry) -> None:
    """Fetch a docs site via Docling and insert into the project's web_docs collection."""
    from explorer.ingestion.doc_parser import DocParser, DocChunk
    from explorer.ingestion.data_prep import DataPrep
    from explorer.multi_collection_store import MultiCollectionStore
    from config.collection_config import COLLECTION_TYPES

    ctype = COLLECTION_TYPES.get("web_docs")
    if not ctype:
        console.print("[red]web_docs collection type not configured.[/red]")
        return

    collection_name = f"{project.slug}_web_docs"
    parser = DocParser(ctype.chunk_size, ctype.chunk_overlap)

    try:
        chunks = parser.parse_url(docs_url, project.slug)
    except Exception as exc:
        console.print(f"[red]Failed to fetch docs:[/red] {exc}")
        return

    chunks = DataPrep().filter(chunks)
    if not chunks:
        console.print("[yellow]No content extracted from docs URL.[/yellow]")
        return

    store = MultiCollectionStore()
    count = store.insert(collection_name, [c.text for c in chunks], [c.metadata for c in chunks])

    collections = list(project.collections)
    if collection_name not in collections:
        collections.append(collection_name)
    registry.update_indexed_at(project.slug, collections)
    console.print(f"[green]Inserted {count} chunks into {collection_name}.[/green]")


@app.command()
def refresh(
    slug: str = typer.Argument(help="Project slug to re-index"),
    no_stats: bool = typer.Option(False, "--no-stats", help="Skip GitHub statistics update"),
    history: int = typer.Option(
        90, "--history", "-H",
        help="Days of commit history to fetch (default: 90). Use 365 for a full year.",
    ),
    symbols: bool = typer.Option(
        False, "--symbols",
        help="Extract (or re-extract) code symbols (classes, methods, functions) from source files. "
             "Use this once after upgrading to populate the symbol index for existing projects.",
    ),
):
    """Incrementally re-index a project and refresh GitHub statistics."""
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
    if symbols:
        console.print("[dim]Extracting code symbols...[/dim]")
        try:
            from explorer.ingestion.pipeline import IngestionPipeline
            count = IngestionPipeline().extract_symbols_only(
                slug, project.github_url, project.collections or []
            )
            if count:
                console.print(f"[green]Extracted {count} symbols.[/green]")
            else:
                console.print("[yellow]No code collections found — nothing extracted.[/yellow]")
        except Exception as exc:
            console.print(f"[yellow]Symbol extraction failed: {exc}[/yellow]")
    if not no_stats:
        console.print("[dim]Refreshing project statistics...[/dim]")
        try:
            from explorer.github.stats_fetcher import StatsFetcher
            result = StatsFetcher().fetch(slug, lookback_days=history)
            if "commits_fetch_error" in result:
                console.print(
                    f"[yellow]Warning:[/yellow] commit history could not be fetched: "
                    f"{result['commits_fetch_error']}"
                )
            else:
                n = result.get("commits_fetched", 0)
                console.print(f"[dim]Stats updated ({n} commits stored, {history}d lookback).[/dim]")
        except Exception as exc:
            console.print(f"[dim]Stats update skipped: {exc}[/dim]")


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
def web(
    host: str = typer.Option("127.0.0.1", help="Bind host"),
    port: int = typer.Option(8000, help="Bind port"),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload on code changes (dev mode)"),
):
    """Start the web UI (FastAPI + HTML frontend with Plotly charts and markdown)."""
    import uvicorn
    console.print(f"[cyan]Starting web UI at http://{host}:{port}[/cyan]")
    uvicorn.run("explorer.web.app:app", host=host, port=port, reload=reload)


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Bind host"),
    port: int = typer.Option(8100, help="Base port"),
    all_agents: bool = typer.Option(False, "--all", help="Start all 6 specialist agents on consecutive ports"),
):
    """Start the AgentStack A2A server (exposes agents to beeai.dev platform).

    Without --all: orchestrator only on PORT (routes by intent, general RAG fallback).

    With --all: starts all 6 agents on consecutive ports:
      PORT+0  orchestrator
      PORT+1  statistics
      PORT+2  code search
      PORT+3  documentation
      PORT+4  health
      PORT+5  compare
    """
    from explorer.agentstack_server import run as agentstack_run
    if all_agents:
        console.print(f"[cyan]Starting all Project Explorer agents (base port {port})...[/cyan]")
    else:
        console.print(f"[cyan]Starting Project Explorer orchestrator on {host}:{port}[/cyan]")
    agentstack_run(host=host, port=port, all_agents=all_agents)


def _maybe_resolve_alias(query: str) -> Optional[str]:
    """
    Check for a fuzzy alias candidate before running a query.
    If found, prompt the user to confirm. If confirmed, saves the alias and returns the slug.
    """
    try:
        from explorer.registry import ProjectRegistry
        registry = ProjectRegistry()
        result = registry.fuzzy_candidate(query)
        if not result:
            return None
        term, slug = result
        project = registry.get(slug)
        display = project.display_name if project else slug
        confirmed = typer.confirm(
            f'Did you mean "{display}"? Remember "{term}" as an alias for {slug}?',
            default=False,
        )
        if confirmed:
            registry.add_alias(term, slug)
            return slug
    except Exception:
        pass
    return None


# ── aliases sub-group ─────────────────────────────────────────────────────────

aliases_app = typer.Typer(name="aliases", help="Manage project name aliases.")
app.add_typer(aliases_app)


@aliases_app.command(name="list")
def aliases_list(
    slug: Optional[str] = typer.Argument(None, help="Project slug (omit to list all)"),
):
    """List stored aliases, optionally filtered to one project."""
    from explorer.registry import ProjectRegistry
    rows = ProjectRegistry().list_aliases(slug)
    if not rows:
        console.print("[dim]No aliases found.[/dim]")
        return
    from rich.table import Table
    tbl = Table("Alias", "Project", "Source", "Created")
    for r in rows:
        tbl.add_row(r["alias"], r["project_slug"], r["confirmed_by"], r["created_at"][:10])
    console.print(tbl)


@aliases_app.command(name="add")
def aliases_add(
    alias: str = typer.Argument(help='Alias phrase, e.g. "Egeria Platform"'),
    slug: str = typer.Argument(help="Project slug to map to"),
):
    """Add a name alias for a project."""
    from explorer.registry import ProjectRegistry
    registry = ProjectRegistry()
    if not registry.exists(slug):
        console.print(f"[red]Project '{slug}' not found.[/red]")
        raise typer.Exit(1)
    registry.add_alias(alias, slug)
    console.print(f'[green]Alias "{alias}" → {slug} saved.[/green]')


@aliases_app.command(name="remove")
def aliases_remove(
    alias: str = typer.Argument(help="Alias to remove"),
):
    """Remove a stored alias."""
    from explorer.registry import ProjectRegistry
    removed = ProjectRegistry().remove_alias(alias)
    if removed:
        console.print(f'[green]Alias "{alias}" removed.[/green]')
    else:
        console.print(f'[yellow]Alias "{alias}" not found.[/yellow]')


if __name__ == "__main__":
    app()
