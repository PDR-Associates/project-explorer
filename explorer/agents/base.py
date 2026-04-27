"""Base agent — BeeAI RequirementAgent wrapper, following the lfai/ML_LLM_Ops pattern."""
from __future__ import annotations

import asyncio
import concurrent.futures
from abc import ABC, abstractmethod
from typing import Any

from explorer.config import ExplorerConfig, get_config


class BaseExplorerAgent(ABC):
    """
    Wraps BeeAI RequirementAgent with standard configuration.

    Each subclass defines its tools and system prompt. The agent loop,
    retry logic, streaming, and Phoenix instrumentation are handled here.

    Pattern (from lfai/ML_LLM_Ops):
        agent = RequirementAgent(
            llm="ollama:llama3.1:8b",
            tools=[VectorStoreSearchTool(...)],
            instructions=self.system_prompt(),
        )
        result = await agent.run(query)
        text = result.output[0].text
    """

    def __init__(self, config: ExplorerConfig | None = None) -> None:
        self.config = config or get_config()

    @abstractmethod
    def system_prompt(self) -> str:
        """Return the system prompt for this agent."""
        ...

    @abstractmethod
    def tools(self) -> list[Any]:
        """Return the list of BeeAI tools this agent has access to."""
        ...

    @abstractmethod
    def handle(self, query: str, project_slug: str | None = None, **kwargs) -> str:
        """Process a query and return a response string."""
        ...

    def _build_agent(self):
        """Construct the BeeAI RequirementAgent."""
        from beeai_framework.agents.requirement import RequirementAgent

        return RequirementAgent(
            llm=self._llm_name(),
            tools=self.tools(),
            instructions=self.system_prompt(),
        )

    def _llm_name(self) -> str:
        """Return the BeeAI ChatModel name string, e.g. 'ollama:llama3.1:8b'."""
        cfg = self.config.llm
        backend = cfg.backend
        if backend == "openai":
            model = cfg.openai.model
        elif backend == "anthropic":
            model = cfg.anthropic.model
        else:
            model = cfg.ollama.model
        return f"{backend}:{model}"

    def _run_agent(self, prompt: str) -> str:
        """Run the BeeAI RequirementAgent synchronously, handling both sync and async callers."""
        async def _inner():
            agent = self._build_agent()
            result = await agent.run(prompt)
            if hasattr(result, "output") and result.output:
                first = result.output[0]
                return first.text if hasattr(first, "text") else str(first)
            return str(result)

        try:
            asyncio.get_running_loop()
            # Called from inside an async context (e.g. FastAPI) — run in a new thread
            def _in_thread():
                return asyncio.run(_inner())
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                return executor.submit(_in_thread).result()
        except RuntimeError:
            # No running event loop — safe to call asyncio.run directly
            return asyncio.run(_inner())
