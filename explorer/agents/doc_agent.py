"""Documentation agent — conceptual Q&A from markdown and web docs."""
from __future__ import annotations

from explorer.agents.base import BaseExplorerAgent
from explorer.prompt_templates import doc_agent_system_prompt


class DocAgent(BaseExplorerAgent):
    def system_prompt(self) -> str:
        return doc_agent_system_prompt()

    def tools(self) -> list:
        return []

    def handle(self, query: str, project_slug: str | None = None, **kwargs) -> str:
        from explorer.collection_router import CollectionRouter
        from explorer.multi_collection_store import MultiCollectionStore
        from explorer.prompt_templates import build_rag_prompt

        collections = CollectionRouter().select(query, project_slug)
        results = MultiCollectionStore().search(query, collections)
        if not results:
            return "I couldn't find relevant documentation for that query."
        context = "\n\n---\n\n".join(r.text for r in results)
        prompt = build_rag_prompt(query, context, project_slug)
        try:
            return self._run_agent(prompt)
        except Exception:
            from explorer.llm_client import get_llm
            return get_llm().complete(prompt, system=self.system_prompt())
