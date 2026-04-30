"""Multi-turn conversation agent — persistent BeeAI agent with TokenMemory."""
from __future__ import annotations

import asyncio
import concurrent.futures

from explorer.agents.base import BaseExplorerAgent


class ConversationAgent(BaseExplorerAgent):
    """
    Multi-turn chat agent that maintains conversation context across calls.

    Uses a single persistent BeeAI RequirementAgent instance with TokenMemory so
    that prior turns are available to the LLM without manual history injection.
    All explorer tools are wired in: the agent decides which to call based on the
    question, rather than routing through RAGSystem.

    Falls back to RAGSystem.query() if BeeAI is unavailable.
    """

    def __init__(
        self,
        project_slug: str | None = None,
        rag_system=None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.project_slug = project_slug
        self._rag = rag_system  # optional pre-warmed fallback; created lazily if None
        self._agent = None  # lazy-init; kept alive across turns

    def system_prompt(self) -> str:
        scope = f" for the {self.project_slug} project" if self.project_slug else ""
        return (
            f"You are a knowledgeable assistant{scope} for exploring GitHub projects. "
            "Use your available tools to answer questions about code, documentation, "
            "statistics, and community health. "
            "Maintain context from the conversation — refer back to earlier questions "
            "and answers when relevant. "
            "When a question names a project explicitly, pass that slug to the tools. "
            "When it doesn't, infer the project from context or ask the user."
        )

    def tools(self) -> list:
        from explorer.agents.tools import (
            vector_search,
            query_project_stats,
            query_top_committers,
            query_commit_activity,
            query_contributor_profile,
            query_code_symbols,
            get_symbol_detail,
        )
        return [
            vector_search,
            query_project_stats,
            query_top_committers,
            query_commit_activity,
            query_contributor_profile,
            query_code_symbols,
            get_symbol_detail,
        ]

    def _get_agent(self):
        """Return the persistent agent, creating it once on first use."""
        if self._agent is None:
            from beeai_framework.agents.requirement import RequirementAgent
            from beeai_framework.memory.token_memory import TokenMemory
            self._agent = RequirementAgent(
                llm=self._llm_name(),
                tools=self.tools(),
                instructions=self.system_prompt(),
                memory=TokenMemory(max_tokens=8000),
            )
        return self._agent

    def handle(self, query: str, project_slug: str | None = None, **kwargs) -> str:
        slug = project_slug or self.project_slug or self._infer_project_slug(query)

        lines: list[str] = []
        if slug:
            lines.append(f"Project: {slug}")
            # Tell the agent which collection names exist so it can call vector_search correctly
            from explorer.collection_router import CollectionRouter
            collections = CollectionRouter().select(query, slug)
            if collections:
                lines.append(f"Available collections: {', '.join(collections)}")
        lines.append(f"Question: {query}")
        prompt = "\n".join(lines)

        try:
            return self._run_persistent(prompt)
        except Exception:
            if self._rag is None:
                from explorer.rag_system import RAGSystem
                self._rag = RAGSystem()
            return self._rag.query(query, project_slug=slug)

    def _run_persistent(self, prompt: str) -> str:
        """Run the persistent agent, reusing the same instance so TokenMemory accumulates."""
        async def _inner() -> str:
            agent = self._get_agent()
            result = await agent.run(prompt)
            if hasattr(result, "output") and result.output:
                first = result.output[0]
                return first.text if hasattr(first, "text") else str(first)
            return str(result)

        try:
            asyncio.get_running_loop()
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                return ex.submit(lambda: asyncio.run(_inner())).result()
        except RuntimeError:
            return asyncio.run(_inner())
