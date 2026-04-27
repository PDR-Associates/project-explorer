"""Interactive REPL session with multi-turn conversation history."""
from __future__ import annotations

from rich.console import Console
from rich.prompt import Prompt

from explorer.agents.conversation_agent import ConversationAgent


class InteractiveSession:
    def __init__(self, project_slug: str | None = None) -> None:
        self.console = Console()
        self.agent = ConversationAgent(project_slug=project_slug)
        self.project_slug = project_slug

    def run(self) -> None:
        scope = f" [{self.project_slug}]" if self.project_slug else ""
        self.console.print(f"[bold]Project Explorer{scope}[/bold] — type 'exit' to quit\n")
        while True:
            try:
                query = Prompt.ask("[cyan]You[/cyan]")
            except (KeyboardInterrupt, EOFError):
                break
            if query.strip().lower() in ("exit", "quit", "q"):
                break
            if not query.strip():
                continue
            response = self.agent.handle(query, self.project_slug)
            self.console.print(f"\n[green]Assistant:[/green] {response}\n")
            import hashlib
            from explorer.observability.feedback_collector import FeedbackCollector
            query_hash = hashlib.sha256(query.encode()).hexdigest()[:16]
            FeedbackCollector().prompt_and_collect(query_hash)
