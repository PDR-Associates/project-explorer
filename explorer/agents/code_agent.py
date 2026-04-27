"""Code search agent — finds methods, classes, and usage examples in source code."""
from __future__ import annotations

from explorer.agents.base import BaseExplorerAgent
from explorer.prompt_templates import code_agent_system_prompt


class CodeAgent(BaseExplorerAgent):
    def system_prompt(self) -> str:
        return code_agent_system_prompt()

    def tools(self) -> list:
        return []

    def handle(self, query: str, project_slug: str | None = None, **kwargs) -> str:
        from explorer.collection_router import CollectionRouter
        from explorer.multi_collection_store import MultiCollectionStore
        from explorer.prompt_templates import build_rag_prompt

        collections = CollectionRouter().select(query, project_slug)
        results = MultiCollectionStore().search(query, collections)
        if not results:
            return "I couldn't find relevant code for that query in the indexed repositories."
        context = "\n\n---\n\n".join(
            f"[{r.collection} | score={r.score:.2f}]\n{r.text}" for r in results
        )
        prompt = build_rag_prompt(query, context, project_slug)
        try:
            return self._run_agent(prompt)
        except Exception:
            from explorer.llm_client import get_llm
            return get_llm().complete(prompt, system=self.system_prompt())
