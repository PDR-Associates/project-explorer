"""Main orchestrator — entry point for all queries."""
from __future__ import annotations

import threading
import time

from explorer.collection_router import CollectionRouter
from explorer.config import get_config
from explorer.llm_client import get_llm
from explorer.multi_collection_store import MultiCollectionStore
from explorer.prompt_templates import build_rag_prompt
from explorer.query_cache import QueryCache
from explorer.query_processor import QueryIntent, QueryProcessor
from explorer.registry import ProjectRegistry


class RAGSystem:
    """
    Orchestrates the full query pipeline:
      cache check → intent classify → agent or RAG → LLM → async observability
    """

    def __init__(self) -> None:
        self.config = get_config()
        self.registry = ProjectRegistry()
        self.processor = QueryProcessor()
        self.cache = QueryCache()
        self.store = MultiCollectionStore()
        self.router = CollectionRouter()
        self.llm = get_llm()
        self._init_observability()

    def query(self, query: str, project_slug: str | None = None) -> str:
        intent = self.processor.classify(query)

        cached = self.cache.get(query, project_slug, intent.value)
        if cached:
            threading.Thread(
                target=self._track,
                args=(query, intent, project_slug, cached, 0, True),
                daemon=True,
            ).start()
            return cached

        t0 = time.monotonic()
        response = self._route(query, intent, project_slug)
        latency_ms = int((time.monotonic() - t0) * 1000)

        self.cache.set(query, project_slug, intent.value, response)

        threading.Thread(
            target=self._track,
            args=(query, intent, project_slug, response, latency_ms, False),
            daemon=True,
        ).start()

        return response

    def _route(self, query: str, intent: QueryIntent, project_slug: str | None) -> str:
        if intent == QueryIntent.STATISTICAL:
            from explorer.agents.stats_agent import StatsAgent
            return StatsAgent().handle(query, project_slug)

        if intent == QueryIntent.COMPARISON:
            from explorer.agents.compare_agent import CompareAgent
            return CompareAgent().handle(query, project_slug)

        if intent == QueryIntent.HEALTH:
            from explorer.agents.health_agent import HealthAgent
            return HealthAgent().handle(query, project_slug)

        if intent == QueryIntent.CODE_SEARCH:
            from explorer.agents.code_agent import CodeAgent
            return CodeAgent().handle(query, project_slug)

        if intent == QueryIntent.CONCEPTUAL:
            from explorer.agents.doc_agent import DocAgent
            return DocAgent().handle(query, project_slug)

        return self._rag(query, project_slug)

    def _rag(self, query: str, project_slug: str | None) -> str:
        collections = self.router.select(query, project_slug)
        results = self.store.search(query, collections)
        if not results:
            return "I don't have enough information in the indexed content to answer that."
        context = "\n\n---\n\n".join(r.text for r in results)
        prompt = build_rag_prompt(query, context, project_slug)
        return self.llm.complete(prompt)

    def _init_observability(self) -> None:
        from explorer.observability.phoenix_client import init_phoenix
        from explorer.observability.metrics_collector import MetricsCollector
        init_phoenix()
        self.metrics = MetricsCollector()

    def _track(
        self,
        query: str,
        intent: QueryIntent,
        project_slug: str | None,
        response: str,
        latency_ms: int = 0,
        cache_hit: bool = False,
    ) -> None:
        try:
            self.metrics.record_query(
                query, intent.value, project_slug, response,
                latency_ms=latency_ms, cache_hit=cache_hit,
            )
        except Exception:
            pass
