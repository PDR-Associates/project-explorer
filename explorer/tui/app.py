"""Textual TUI — full-screen interactive interface for Project Explorer."""
from __future__ import annotations

import hashlib
from typing import ClassVar

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import (
    Footer,
    Header,
    Input,
    Label,
    ListItem,
    ListView,
    Static,
)


class ProjectItem(ListItem):
    """A single project row in the sidebar."""

    def __init__(self, slug: str, display_name: str, status: str, collections: int) -> None:
        super().__init__()
        self.slug = slug
        self.display_name = display_name
        self.status = status
        self.collections = collections

    def compose(self) -> ComposeResult:
        status_color = {
            "active": "green",
            "indexing": "yellow",
            "error": "red",
            "paused": "dim",
        }.get(self.status, "white")
        yield Label(
            f"[bold]{self.display_name}[/bold]\n"
            f"[{status_color}]{self.status}[/{status_color}]  "
            f"[dim]{self.collections} collection(s)[/dim]"
        )


class ChatMessage(Static):
    """A single chat bubble — user or assistant."""

    def __init__(self, role: str, text: str) -> None:
        color = "cyan" if role == "You" else "green"
        content = f"[bold {color}]{role}:[/bold {color}]\n{text}"
        super().__init__(content)
        self.add_class(f"msg-{role.lower()}")


class ProjectExplorerApp(App):
    """
    Full-screen TUI for Project Explorer.

    Layout:
      ┌─────────────┬──────────────────────────────────┐
      │  Projects   │  Chat / response area            │
      │  (sidebar)  │                                  │
      │             ├──────────────────────────────────┤
      │             │  Input bar                       │
      └─────────────┴──────────────────────────────────┘

    Keys:
      Tab      — move focus between sidebar and chat
      Enter    — submit query (when input focused)
      f        — feedback thumbs-up/down for last response
      r        — refresh selected project
      Ctrl+C   — quit
    """

    CSS = """
    Screen {
        layout: vertical;
    }
    #main {
        layout: horizontal;
        height: 1fr;
    }
    #sidebar {
        width: 28;
        border-right: solid $accent-darken-2;
        background: $surface-darken-1;
    }
    #sidebar-label {
        background: $accent-darken-2;
        color: $text;
        padding: 0 1;
        text-align: center;
        height: 1;
    }
    #project-list {
        height: 1fr;
    }
    #chat-area {
        layout: vertical;
        width: 1fr;
    }
    #chat-log {
        height: 1fr;
        overflow-y: auto;
        padding: 1 2;
    }
    #status-bar {
        height: 1;
        background: $accent-darken-3;
        color: $text-muted;
        padding: 0 2;
    }
    #input-row {
        height: 3;
        layout: horizontal;
        border-top: solid $accent-darken-2;
        padding: 0 1;
    }
    #query-input {
        width: 1fr;
    }
    ChatMessage {
        margin: 0 0 1 0;
    }
    ProjectItem {
        padding: 0 1;
    }
    ProjectItem:hover {
        background: $accent-darken-1;
    }
    ProjectItem.--highlight {
        background: $accent;
    }
    """

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("ctrl+c", "quit", "Quit"),
        Binding("tab", "focus_next", "Switch pane", show=False),
        Binding("f", "feedback", "Feedback", show=True),
        Binding("r", "refresh_project", "Refresh", show=True),
    ]

    selected_project: reactive[str | None] = reactive(None)
    last_query_hash: reactive[str | None] = reactive(None)
    _awaiting_feedback: bool = False

    def __init__(self, rag_system) -> None:
        super().__init__()
        self._rag = rag_system

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="main"):
            with Vertical(id="sidebar"):
                yield Label("Projects", id="sidebar-label")
                yield ListView(id="project-list")
            with Vertical(id="chat-area"):
                yield Vertical(id="chat-log")
                yield Static("", id="status-bar")
                with Horizontal(id="input-row"):
                    yield Input(
                        placeholder="Ask a question… (Tab to switch pane, f=feedback, r=refresh)",
                        id="query-input",
                    )
        yield Footer()

    def on_mount(self) -> None:
        self._load_projects()
        self.query_one("#query-input").focus()
        self._append_message(
            "Assistant",
            "Welcome to Project Explorer! Select a project in the sidebar (Tab) "
            "or ask a question across all projects.",
        )

    # ── project sidebar ───────────────────────────────────────────────────────

    def _load_projects(self) -> None:
        from explorer.registry import ProjectRegistry
        lv = self.query_one("#project-list", ListView)
        lv.clear()
        projects = ProjectRegistry().list_all()
        for p in projects:
            lv.append(ProjectItem(p.slug, p.display_name, p.status.value, len(p.collections)))
        if not projects:
            lv.append(ListItem(Label("[dim]No projects — run 'project-explorer add'[/dim]")))

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        if isinstance(event.item, ProjectItem):
            self.selected_project = event.item.slug
            self._set_status(f"Scoped to: {event.item.display_name}  [{event.item.slug}]")

    # ── chat ──────────────────────────────────────────────────────────────────

    def on_input_submitted(self, event: Input.Submitted) -> None:
        query = event.value.strip()
        if not query:
            return
        event.input.value = ""
        self._append_message("You", query)
        self.last_query_hash = hashlib.sha256(query.encode()).hexdigest()[:16]
        self._set_status("Thinking…")
        self._run_query(query)

    @work(thread=True)
    def _run_query(self, query: str) -> None:
        try:
            response = self._rag.query(query, project_slug=self.selected_project)
        except Exception as exc:
            response = f"Error: {exc}"
        self.call_from_thread(self._on_response, response)

    def _on_response(self, response: str) -> None:
        self._append_message("Assistant", response)
        self._set_status("f=👍/👎 feedback  |  r=refresh project  |  Tab=switch pane")

    def _append_message(self, role: str, text: str) -> None:
        log = self.query_one("#chat-log")
        log.mount(ChatMessage(role, text))
        log.scroll_end(animate=False)

    def _set_status(self, text: str) -> None:
        self.query_one("#status-bar", Static).update(text)

    # ── actions ─────���─────────────────────────────────────────────────────────

    def action_feedback(self) -> None:
        if not self.last_query_hash:
            self._set_status("No query to give feedback on yet.")
            return
        self._set_status("Feedback: y=👍  n=👎  (press key)")
        self._awaiting_feedback = True

    def on_key(self, event) -> None:
        if self._awaiting_feedback:
            self._awaiting_feedback = False
            from explorer.observability.metrics_collector import MetricsCollector
            if event.key == "y":
                MetricsCollector().record_feedback(self.last_query_hash, 1)
                self._set_status("👍 Feedback recorded — thanks!")
            elif event.key == "n":
                MetricsCollector().record_feedback(self.last_query_hash, -1)
                self._set_status("👎 Feedback recorded.")
            else:
                self._set_status("Feedback cancelled.")
            event.prevent_default()

    def action_refresh_project(self) -> None:
        if not self.selected_project:
            self._set_status("Select a project in the sidebar first (Tab).")
            return
        self._set_status(f"Refreshing {self.selected_project}…")
        self._append_message("Assistant", f"Starting refresh for **{self.selected_project}**…")
        self._run_refresh(self.selected_project)

    @work(thread=True)
    def _run_refresh(self, slug: str) -> None:
        try:
            from explorer.ingestion.incremental import IncrementalIndexer
            from explorer.query_cache import QueryCache
            from explorer.registry import ProjectRegistry
            project = ProjectRegistry().get(slug)
            if project:
                IncrementalIndexer().refresh(project)
                QueryCache().invalidate_project(slug)
                msg = f"Refresh complete for **{slug}**."
            else:
                msg = f"Project **{slug}** not found."
        except Exception as exc:
            msg = f"Refresh failed: {exc}"
        self.call_from_thread(self._on_refresh_done, msg)

    def _on_refresh_done(self, msg: str) -> None:
        self._append_message("Assistant", msg)
        self._load_projects()
        self._set_status("Ready.")


def run() -> None:
    """
    Pre-warm the RAG system (loads embedding model + Milvus connection) in the
    main thread before Textual starts. This prevents gRPC/PyTorch FD conflicts
    that occur when those libraries initialise inside a worker thread.
    """
    from explorer.rag_system import RAGSystem
    from explorer.embeddings import get_embedding_model

    # Force model + Milvus client init now, in the main thread
    get_embedding_model()
    rag = RAGSystem()
    # Touch the store to open the gRPC connection before Textual's event loop
    try:
        from explorer.multi_collection_store import MultiCollectionStore
        MultiCollectionStore()._get_client()
    except Exception:
        pass

    ProjectExplorerApp(rag).run()


if __name__ == "__main__":
    run()
