"""Onboarding wizard — interactive flow for adding a new GitHub project."""
from __future__ import annotations

import re

from rich.console import Console
from rich.prompt import Confirm, Prompt
from rich.table import Table

from explorer.github.analyzer import RepoAnalyzer
from explorer.registry import Project, ProjectRegistry


class OnboardingWizard:
    """
    Walks the user through adding a new project:
      1. Validate GitHub URL
      2. Fetch repo metadata
      3. Analyze repo → propose collections
      4. User confirms or customizes collection selection
      5. Register project
      6. Trigger ingestion pipeline
    """

    def __init__(self) -> None:
        self.console = Console()
        self.registry = ProjectRegistry()
        self.analyzer = RepoAnalyzer()

    def run(self, github_url: str, accept_all: bool = False) -> None:
        self.console.print(f"\n[bold]Analyzing[/bold] {github_url} ...")

        try:
            plan = self.analyzer.analyze(github_url)
        except ValueError as exc:
            self.console.print(f"[red]Error:[/red] {exc}")
            return
        except Exception as exc:
            # Strip the raw traceback for common API errors (404, rate limit, auth)
            msg = str(exc)
            if "404" in msg or "Not Found" in msg:
                self.console.print(
                    f"[red]Repository not found:[/red] {github_url}\n"
                    "[dim]Check the URL is a public repo (not an org or private repo).[/dim]"
                )
            elif "401" in msg or "403" in msg:
                self.console.print(
                    "[red]Authentication error.[/red] Check GITHUB_TOKEN in .env has repo read scope."
                )
            elif "rate limit" in msg.lower():
                self.console.print(
                    "[red]GitHub API rate limit exceeded.[/red] Wait and retry, or use a token with higher limits."
                )
            else:
                self.console.print(f"[red]Failed to analyze repository:[/red] {exc}")
            return

        slug = self._url_to_slug(github_url)

        if self.registry.exists(slug):
            self.console.print(f"[yellow]Project '{slug}' is already registered.[/yellow]")
            if accept_all or Confirm.ask("Re-index it?"):
                self._trigger_ingestion(slug, plan)
            return

        self._show_plan(plan)

        if accept_all:
            confirmed_types = plan.proposed_collections
            self.console.print("[dim]Accepting all proposed collections (--yes).[/dim]")
        else:
            confirmed_types = self._confirm_collections(plan)
        if not confirmed_types:
            self.console.print("[yellow]No collections selected — aborting.[/yellow]")
            return

        docs_url = "" if accept_all else Prompt.ask(
            "Documentation site URL (optional, press Enter to skip)", default=""
        )

        project = Project(
            slug=slug,
            display_name=plan.display_name,
            github_url=github_url,
            description=plan.description,
            homepage_url=plan.homepage_url,
            docs_url=docs_url,
            collections=[f"{slug}_{ct.name}" for ct in confirmed_types],
        )
        self.registry.add(project)
        self.console.print(f"\n[green]Registered '{plan.display_name}' as '{slug}'.[/green]")
        self._trigger_ingestion(slug, plan, confirmed_types)

    def _show_plan(self, plan) -> None:
        table = Table(title=f"Proposed collections for {plan.display_name}")
        table.add_column("Collection", style="cyan")
        table.add_column("Description")
        table.add_column("Files", justify="right")
        for ct in plan.proposed_collections:
            table.add_row(ct.name, ct.description, str(plan.file_counts.get(ct.name, "?")))
        self.console.print(table)

    def _confirm_collections(self, plan) -> list:
        self.console.print("\nSelect collections to create (press Enter to accept all):")
        selected = []
        for ct in plan.proposed_collections:
            if Confirm.ask(f"  Include [cyan]{ct.name}[/cyan]?", default=True):
                selected.append(ct)
        return selected

    def _trigger_ingestion(self, slug: str, plan, collection_types=None) -> None:
        from explorer.ingestion.pipeline import IngestionPipeline
        self.console.print("\n[bold]Starting ingestion...[/bold]")
        pipeline = IngestionPipeline()
        pipeline.run(slug, plan.github_url, collection_types or plan.proposed_collections)
        self._post_ingestion(slug, plan.github_url)

    def _post_ingestion(self, slug: str, github_url: str) -> None:
        # Store initial commit SHA so incremental refresh has a baseline
        try:
            from explorer.github.client import GitHubClient
            client = GitHubClient()
            repo = client.get_repo(github_url)
            sha = client.get_latest_commit_sha(repo)
            self.registry.update_commit_sha(slug, sha)
        except Exception:
            pass

        # Fetch initial GitHub stats into SQLite
        self.console.print("[dim]Fetching project statistics...[/dim]")
        try:
            from explorer.github.stats_fetcher import StatsFetcher
            result = StatsFetcher().fetch(slug)
            if "commits_fetch_error" in result:
                self.console.print(
                    f"[yellow]Warning:[/yellow] commit history could not be fetched: "
                    f"{result['commits_fetch_error']}\n"
                    f"[dim]Run 'project-explorer refresh {slug}' to retry.[/dim]"
                )
            else:
                n = result.get("commits_fetched", 0)
                self.console.print(f"[dim]Stats fetched ({n} commits stored).[/dim]")
        except Exception as exc:
            self.console.print(f"[dim]Stats fetch skipped: {exc}[/dim]")

    @staticmethod
    def _url_to_slug(url: str) -> str:
        url = url.rstrip("/")
        slug = url.split("/")[-1]
        return re.sub(r"[^a-z0-9_]", "_", slug.lower())
