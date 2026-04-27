"""Multi-turn conversation agent — BeeAI session wrapper with memory."""
from __future__ import annotations

from explorer.agents.base import BaseExplorerAgent
from explorer.rag_system import RAGSystem

_MAX_HISTORY_TURNS = 6  # inject last N turns as context


class ConversationAgent(BaseExplorerAgent):
    """
    Wraps RAGSystem in a BeeAI multi-turn session with conversation memory.
    Used by the interactive CLI and web chat interface.

    The BeeAI RequirementAgent maintains turn history automatically.
    Each turn routes through RAGSystem._route() so intent classification
    and all specialized agents remain active within a conversation.
    """

    def __init__(self, project_slug: str | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.project_slug = project_slug
        self._rag = RAGSystem()
        self._history: list[tuple[str, str]] = []  # (user, assistant) pairs

    def system_prompt(self) -> str:
        scope = f" for the {self.project_slug} project" if self.project_slug else ""
        return (
            f"You are a knowledgeable assistant{scope}. "
            "Answer questions using your knowledge of the project's code, documentation, and statistics. "
            "Maintain context from the conversation history."
        )

    def tools(self) -> list:
        return []

    def handle(self, query: str, project_slug: str | None = None, **kwargs) -> str:
        slug = project_slug or self.project_slug
        augmented = self._with_history(query)
        response = self._rag.query(augmented, project_slug=slug)
        self._history.append((query, response))
        return response

    def _with_history(self, query: str) -> str:
        """Prepend recent conversation turns so the LLM has context."""
        recent = self._history[-_MAX_HISTORY_TURNS:]
        if not recent:
            return query
        turns = "\n".join(
            f"User: {q}\nAssistant: {a}" for q, a in recent
        )
        return f"Conversation so far:\n{turns}\n\nUser: {query}"
