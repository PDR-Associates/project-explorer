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
    slugs: Optional[list[str]] = typer.Argument(
        default=None,
        help="Project slug(s) to re-index. Omit when using --all.",
    ),
    all_projects: bool = typer.Option(False, "--all", help="Re-index every registered project"),
    top_level: bool = typer.Option(
        False, "--top-level",
        help="With --all: skip sub-projects (those with a parent_slug set)",
    ),
    no_stats: bool = typer.Option(False, "--no-stats", help="Skip GitHub statistics update"),
    history: int = typer.Option(
        90, "--history", "-H",
        help="Days of commit history to fetch (default: 90). Use 365 for a full year.",
    ),
    symbols: bool = typer.Option(
        False, "--symbols",
        help="Extract (or re-extract) code symbols from source files.",
    ),
):
    """Incrementally re-index one or more projects and refresh GitHub statistics.

    Examples:

      project-explorer refresh myproject
      project-explorer refresh proj1 proj2 proj3
      project-explorer refresh --all
      project-explorer refresh --all --top-level   # skip sub-projects
    """
    from explorer.ingestion.incremental import IncrementalIndexer
    from explorer.query_cache import QueryCache
    from explorer.registry import ProjectRegistry

    registry = ProjectRegistry()
    targets = _resolve_slugs(registry, slugs, all_projects, top_level)
    if not targets:
        raise typer.Exit(1)

    failed: list[str] = []
    for project in targets:
        slug = project.slug
        label = f"[bold]{project.display_name}[/bold]" + (
            f" [dim]({slug})[/dim]" if project.display_name != slug else ""
        )
        if project.parent_slug:
            label += f" [dim]← sub-project of {project.parent_slug}[/dim]"
        console.print(f"\n[cyan]Refreshing {label} …[/cyan]")
        try:
            indexer = IncrementalIndexer()
            indexer.refresh(project)
            dropped = QueryCache().invalidate_project(slug)
            if dropped:
                console.print(f"  [dim]Cleared {dropped} cached query result(s).[/dim]")
            if symbols:
                try:
                    from explorer.ingestion.pipeline import IngestionPipeline
                    count = IngestionPipeline().extract_symbols_only(
                        slug, project.github_url, project.collections or []
                    )
                    console.print(
                        f"  [dim]Symbols: {count} extracted.[/dim]" if count
                        else "  [yellow]No code collections — symbols skipped.[/yellow]"
                    )
                except Exception as exc:
                    console.print(f"  [yellow]Symbol extraction failed: {exc}[/yellow]")
            if not no_stats:
                try:
                    from explorer.github.stats_fetcher import StatsFetcher
                    result = StatsFetcher().fetch(slug, lookback_days=history)
                    if "commits_fetch_error" in result:
                        console.print(
                            f"  [yellow]Commit history warning:[/yellow] {result['commits_fetch_error']}"
                        )
                    else:
                        n = result.get("commits_fetched", 0)
                        console.print(f"  [dim]Stats updated ({n} commits, {history}d lookback).[/dim]")
                except Exception as exc:
                    console.print(f"  [dim]Stats update skipped: {exc}[/dim]")
            console.print(f"  [green]✓ Done[/green]")
        except Exception as exc:
            console.print(f"  [red]✗ Failed: {exc}[/red]")
            failed.append(slug)

    _print_batch_summary(targets, failed, "refreshed")


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


def _resolve_slugs(registry, slugs, all_projects: bool, top_level: bool):
    """Return the list of Project objects to act on, validating inputs."""
    from explorer.registry import ProjectRegistry  # already imported at call site, but safe
    if all_projects:
        projects = registry.list_all()
        if top_level:
            projects = [p for p in projects if not p.parent_slug]
        if not projects:
            console.print("[yellow]No projects registered.[/yellow]")
            return []
        return projects

    if not slugs:
        console.print("[red]Provide at least one slug or use --all.[/red]")
        return []

    result = []
    for slug in slugs:
        p = registry.get(slug)
        if p is None:
            console.print(f"[red]Project '{slug}' not found — skipping.[/red]")
        else:
            result.append(p)
    if not result:
        console.print("[red]No valid projects found.[/red]")
    return result


def _print_batch_summary(targets, failed: list[str], verb: str) -> None:
    """Print a one-liner batch completion summary only when >1 project was processed."""
    if len(targets) <= 1:
        return
    ok = len(targets) - len(failed)
    color = "green" if not failed else ("yellow" if ok else "red")
    console.print(
        f"\n[{color}]{verb.title()}: {ok}/{len(targets)} succeeded"
        + (f" — failed: {', '.join(failed)}" if failed else "")
        + f"[/{color}]"
    )


def _print_survey_report(result) -> None:
    """Print a full single-project survey report to the console."""
    from explorer.surveyors.survey_report import AnnotationType
    console.print(f"\n[bold]Survey Report: {result.project_display_name}[/bold]")
    console.print(f"GitHub: {result.github_url}")
    if result.project_slug != result.project_display_name:
        console.print(f"Slug:   {result.project_slug}")
    console.print(f"Surveyed at: {result.surveyed_at.isoformat()}")
    console.print(f"Annotations: {len(result.annotations)}  |  Errors: {len(result.errors)}\n")

    for ann_type in AnnotationType:
        group = result.by_type(ann_type)
        if not group:
            continue
        console.print(f"[bold yellow]{ann_type.value}[/bold yellow] ({len(group)})")
        for ann in group:
            console.print(f"  • {ann.summary}")
            if ann.explanation:
                console.print(f"    [dim]{ann.explanation}[/dim]")
        console.print()

    if result.errors:
        console.print("[bold red]Survey errors:[/bold red]")
        for err in result.errors:
            console.print(f"  [red]• {err}[/red]")
        console.print()


def _print_survey_batch_summary(rows: list[dict], published: bool) -> None:
    """Print a Rich table summarising all projects surveyed in batch mode."""
    from rich.table import Table
    tbl = Table(
        "Project", "Annotations", "Errors",
        *( ["Report GUID"] if published else [] ),
        "Status",
        title="Survey Batch Summary",
        title_style="bold",
    )
    for r in rows:
        guid_short = (r["report_guid"] or "")[:12] + "…" if r["report_guid"] else "—"
        status = "[green]✓[/green]" if r["ok"] else "[red]✗[/red]"
        err_str = str(r["errors"]) if r["errors"] else "[dim]0[/dim]"
        row = [r["slug"], str(r["annotations"]), err_str]
        if published:
            row.append(guid_short)
        row.append(status)
        tbl.add_row(*row)
    console.print(tbl)


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


@app.command()
def survey(
    slugs: Optional[list[str]] = typer.Argument(
        default=None,
        help="Project slug(s) to survey. Omit when using --all.",
    ),
    all_projects: bool = typer.Option(False, "--all", help="Survey every registered project"),
    top_level: bool = typer.Option(
        False, "--top-level",
        help="With --all: skip sub-projects (those with a parent_slug set)",
    ),
    publish: bool = typer.Option(False, "--publish", help="Push the survey report to Egeria"),
    refresh_cache: bool = typer.Option(False, "--refresh", help="Force-refresh the file type cache from Egeria before surveying"),
    platform_url: Optional[str] = typer.Option(None, "--egeria-url", help="Egeria platform URL (overrides EGERIA_PLATFORM_URL env var)"),
    view_server: Optional[str] = typer.Option(None, "--egeria-server", help="Egeria view server name (overrides EGERIA_VIEW_SERVER)"),
    data_path: Optional[str] = typer.Option(None, "--data-path", help="Path to a local clone for CSV/Parquet column-level profiling"),
):
    """Survey one or more projects and produce Egeria-aligned annotation reports.

    Without --publish: prints the survey as a markdown report (no Egeria required).
    With --publish: also pushes to Egeria.  Sub-projects that share a GitHub URL
    with their parent will share the same SourceControlLibrary asset in Egeria.

    Examples:

      project-explorer survey myproject
      project-explorer survey proj1 proj2 proj3
      project-explorer survey --all
      project-explorer survey --all --top-level     # skip sub-projects
      project-explorer survey --all --publish       # survey + publish all
    """
    from explorer.registry import ProjectRegistry
    from explorer.surveyors.survey_orchestrator import SurveyOrchestrator
    from explorer.surveyors.survey_report import AnnotationType

    registry = ProjectRegistry()
    targets = _resolve_slugs(registry, slugs, all_projects, top_level)
    if not targets:
        raise typer.Exit(1)

    batch = len(targets) > 1

    # Build Egeria client once (if needed) and reuse across projects
    pyegeria_client = None
    if refresh_cache or publish:
        pyegeria_client = _try_build_egeria_client(platform_url, view_server)

    publisher = None
    if publish:
        from explorer.surveyors.egeria_publisher import EgeriaConnectionError, EgeriaPublisher
        try:
            publisher = EgeriaPublisher(
                platform_url=platform_url,
                view_server=view_server,
                registry=registry,
            )
        except EgeriaConnectionError as exc:
            console.print(f"[red]Egeria connection failed: {exc}[/red]")
            raise typer.Exit(1)

    orchestrator = SurveyOrchestrator(
        registry=registry,
        pyegeria_client=pyegeria_client,
        force_refresh=refresh_cache,
        data_path=data_path,
    )

    failed: list[str] = []
    survey_rows: list[dict] = []   # for batch summary table

    for project in targets:
        slug = project.slug
        label = f"[bold]{project.display_name}[/bold]"
        if project.parent_slug:
            label += f" [dim](sub-project of {project.parent_slug}, path: {project.subproject_path or '/'})[/dim]"
        console.print(f"\n[cyan]Surveying {label} …[/cyan]")

        try:
            result = orchestrator.run(slug)
        except Exception as exc:
            console.print(f"  [red]✗ Survey failed: {exc}[/red]")
            failed.append(slug)
            survey_rows.append({"slug": slug, "annotations": 0, "errors": 1, "report_guid": None, "ok": False})
            continue

        # ── Per-project report (condensed in batch mode) ──────────────────
        if not batch:
            _print_survey_report(result)
        else:
            ann_counts = {t.value: len(result.by_type(t)) for t in AnnotationType if result.by_type(t)}
            summary = ", ".join(f"{v} {k}" for k, v in ann_counts.items())
            status_str = f"[green]✓[/green]" if not result.errors else f"[yellow]⚠ {len(result.errors)} error(s)[/yellow]"
            console.print(f"  {status_str} {len(result.annotations)} annotations — {summary}")
            if result.errors:
                for err in result.errors[:3]:
                    console.print(f"    [dim red]{err}[/dim red]")

        # ── Egeria publish ────────────────────────────────────────────────
        report_guid = None
        if publisher:
            try:
                console.print(f"  [dim]Publishing to Egeria…[/dim]")
                report_guid = publisher.publish(result)
                console.print(f"  [green]SurveyReport GUID: {report_guid}[/green]")
            except Exception as exc:
                console.print(f"  [red]Egeria publish failed: {exc}[/red]")
                if not batch:
                    raise typer.Exit(1)
                failed.append(f"{slug}(publish)")

        survey_rows.append({
            "slug": slug,
            "annotations": len(result.annotations),
            "errors": len(result.errors),
            "report_guid": report_guid,
            "ok": slug not in failed,
        })

    if batch:
        _print_survey_batch_summary(survey_rows, publish)
    elif not failed and publish and not batch:
        # Single-project governance action prompt (batch skips this)
        catalog = typer.confirm(
            "\nTrigger governance action process to catalog this asset in Egeria?",
            default=False,
        )
        if catalog:
            console.print(
                "[yellow]Governance action integration is not yet implemented — "
                "watch for this in a future release.[/yellow]"
            )

    _print_batch_summary(targets, failed, "surveyed")


@app.command(name="egeria-reports")
def egeria_reports(
    slug: str = typer.Argument(help="Project slug"),
    full: bool = typer.Option(False, "--full", help="Show all annotations for the latest survey"),
    platform_url: Optional[str] = typer.Option(None, "--egeria-url", help="Egeria platform URL (overrides EGERIA_PLATFORM_URL env var)"),
    view_server: Optional[str] = typer.Option(None, "--egeria-server", help="Egeria view server name (overrides EGERIA_VIEW_SERVER)"),
):
    """Show Egeria survey reports published for a project.

    Reads from the local registry by default (no Egeria connection needed).
    Use --full to fetch and display all annotations from Egeria for the latest survey.
    """
    from explorer.registry import ProjectRegistry
    from explorer.surveyors.egeria_reader import EgeriaReader, EgeriaReaderError

    registry = ProjectRegistry()
    project = registry.get(slug)
    if project is None:
        console.print(f"[red]Project '{slug}' not found.[/red]")
        raise typer.Exit(1)

    reader = EgeriaReader(
        platform_url=platform_url,
        view_server=view_server,
        registry=registry,
    )

    # ── Asset registration status ─────────────────────────────────────────────
    asset_guid = registry.get_egeria_asset_guid(slug)
    console.print(f"\n[bold]Egeria surveys for {project.display_name}[/bold]")
    if asset_guid:
        console.print(f"Asset GUID : [cyan]{asset_guid}[/cyan]  (SourceControlLibrary)")
    else:
        console.print("Asset GUID : [dim]not registered[/dim]")
    console.print(f"Platform   : {reader.platform_url}\n")

    # ── Survey history (from local registry) ──────────────────────────────────
    surveys = reader.get_survey_reports_from_registry(slug)

    if not surveys:
        console.print("[yellow]No surveys published to Egeria yet for this project.[/yellow]")
        console.print("Run [bold]project-explorer survey " + slug + " --publish[/bold] to publish one.")
        raise typer.Exit(0)

    # Build table
    from rich.table import Table
    table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
    table.add_column("#", style="dim", width=3)
    table.add_column("Surveyed at", min_width=22)
    table.add_column("Annotations", justify="right", min_width=11)
    table.add_column("Egeria Report GUID")

    for i, row in enumerate(surveys, start=1):
        suffix = "  ← latest" if i == 1 else ""
        count = str(row.get("annotation_count") or "—")
        table.add_row(
            str(i),
            row.get("surveyed_at", "—"),
            count,
            f"[cyan]{row.get('egeria_report_guid', '—')}[/cyan]{suffix}",
        )

    console.print(table)

    if not full:
        console.print("\nRun with [bold]--full[/bold] to display all annotations for the latest survey.")
        raise typer.Exit(0)

    # ── Full annotation display for the latest survey ─────────────────────────
    latest = surveys[0]
    report_guid = latest.get("egeria_report_guid", "")
    surveyed_at = latest.get("surveyed_at", "")

    console.print(f"\n[bold]Annotations for survey: {surveyed_at}[/bold]")
    console.print(f"Report GUID: [cyan]{report_guid}[/cyan]\n")

    try:
        annotations = reader.get_annotations(slug, surveyed_at)
    except EgeriaReaderError as exc:
        console.print(f"[red]Could not connect to Egeria: {exc}[/red]")
        console.print("[dim]Hint: set EGERIA_PLATFORM_URL in .env to enable full annotation display.[/dim]")
        raise typer.Exit(1)

    if not annotations:
        console.print("[yellow]No annotations found in Egeria for this survey.[/yellow]")
        raise typer.Exit(0)

    # Group by annotation_type
    from collections import defaultdict
    groups: dict[str, list] = defaultdict(list)
    for ann in annotations:
        groups[ann.get("annotation_type", "Unknown")].append(ann)

    for ann_type, group in sorted(groups.items()):
        console.print(f"[bold yellow]{ann_type}[/bold yellow] ({len(group)})")
        for ann in group:
            summary = ann.get("summary", "")
            confidence = ann.get("confidence")
            line = f"  • {summary}"
            if confidence is not None:
                line += f"  [dim][confidence: {confidence}][/dim]"
            console.print(line)
            explanation = ann.get("explanation", "")
            if explanation:
                console.print(f"    [dim]{explanation}[/dim]")
            # Show key subtype fields inline
            sub = ann.get("subtype_data", {})
            if sub.get("actionRequested"):
                console.print(f"    [yellow]Action: {sub['actionRequested']}[/yellow]")
            if sub.get("qualityScores"):
                scores = "  ".join(f"{k}={v}" for k, v in sub["qualityScores"].items())
                console.print(f"    [dim]Scores: {scores}[/dim]")
        console.print()


def _try_build_egeria_client(platform_url: Optional[str], view_server: Optional[str]):
    """Attempt to build a pyegeria client for optional cache refresh. Returns None on failure."""
    import os
    url = platform_url or os.getenv("EGERIA_PLATFORM_URL", "")
    server = view_server or os.getenv("EGERIA_VIEW_SERVER", "")
    user = os.getenv("EGERIA_USER", "")
    password = os.getenv("EGERIA_USER_PASSWORD", "")
    if not url:
        return None
    try:
        from pyegeria import ValidMetadataManager
        client = ValidMetadataManager(server, url, user, password)
        client.create_egeria_bearer_token(user, password)
        return client
    except Exception:
        return None


if __name__ == "__main__":
    app()
