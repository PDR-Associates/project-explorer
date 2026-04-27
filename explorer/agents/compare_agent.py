"""Comparison agent — side-by-side analysis of two or more projects."""
from __future__ import annotations

from explorer.agents.base import BaseExplorerAgent
from explorer.prompt_templates import compare_agent_system_prompt


class CompareAgent(BaseExplorerAgent):
    """
    Extracts project names from the query, retrieves context from each project's
    collections, then asks the LLM to produce a structured comparison.
    """

    def system_prompt(self) -> str:
        return compare_agent_system_prompt()

    def tools(self) -> list:
        return []

    def handle(self, query: str, project_slug: str | None = None, **kwargs) -> str:
        slugs = self._extract_project_slugs(query)
        if not slugs:
            return "Please name the projects you'd like to compare (e.g. 'compare project-a and project-b')."

        from explorer.multi_collection_store import MultiCollectionStore
        from explorer.collection_router import CollectionRouter
        from explorer.llm_client import get_llm

        store = MultiCollectionStore()
        router = CollectionRouter()
        sections = []
        for slug in slugs:
            collections = router.select(query, slug)
            results = store.search(query, collections, top_k=5)
            context = "\n".join(r.text for r in results)
            sections.append(f"## {slug}\n{context}")

        combined = "\n\n".join(sections)
        prompt = f"{self.system_prompt()}\n\nContext:\n{combined}\n\nQuestion: {query}\n\nComparison:"
        return get_llm().complete(prompt)

    def _extract_project_slugs(self, query: str) -> list[str]:
        import re
        from explorer.registry import ProjectRegistry
        known = {p.slug for p in ProjectRegistry().list_all()}
        found = [slug for slug in known if slug in query.lower()]
        return found
