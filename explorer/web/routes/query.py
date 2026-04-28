"""Query endpoint — POST a question, get a response."""
from __future__ import annotations

import hashlib
import json
import time
from collections.abc import AsyncIterator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

router = APIRouter()

# ── session store ──────────────────────────────────────────────────────────────
# Maps session_id → (ConversationAgent, last_used_timestamp)
# Capped at 50 sessions; idle sessions expire after 30 minutes.
_SESSION_TTL = 1800  # seconds
_SESSION_MAX = 50
_sessions: dict[str, tuple] = {}


def _get_or_create_session(session_id: str, project_slug: str | None):
    """Return a ConversationAgent for this session, creating one if needed."""
    from explorer.agents.conversation_agent import ConversationAgent

    now = time.monotonic()

    # Evict expired sessions
    expired = [sid for sid, (_, ts) in _sessions.items() if now - ts > _SESSION_TTL]
    for sid in expired:
        del _sessions[sid]

    # Evict oldest when at capacity
    if len(_sessions) >= _SESSION_MAX and session_id not in _sessions:
        oldest = min(_sessions, key=lambda sid: _sessions[sid][1])
        del _sessions[oldest]

    if session_id not in _sessions:
        agent = ConversationAgent(project_slug=project_slug)
        _sessions[session_id] = (agent, now)
    else:
        agent, _ = _sessions[session_id]
        _sessions[session_id] = (agent, now)
        if project_slug:
            agent.project_slug = project_slug

    return agent


class QueryRequest(BaseModel):
    query: str
    project_slug: str | None = None
    session_id: str | None = None  # browser-generated UUID for cross-turn memory


class QueryResponse(BaseModel):
    response: str
    intent: str
    query_hash: str
    cached: bool = False
    chart: dict | None = None  # Plotly figure JSON when intent warrants a chart


class FeedbackRequest(BaseModel):
    query_hash: str
    vote: int  # +1 or -1


def _pick_chart(query: str, intent: str, project_slug: str) -> dict | None:
    """Return a Plotly figure as a JSON-serialisable dict, or None."""
    if not project_slug:
        return None
    if intent not in ("statistical", "health", "comparison"):
        return None
    try:
        from explorer.dashboard import graphs
        q = query.lower()

        if intent == "comparison":
            # Extract slugs from the query for a cross-project stats chart
            from explorer.agents.compare_agent import CompareAgent
            slugs = CompareAgent()._extract_project_slugs(query)
            if len(slugs) >= 2:
                fig = graphs.compare_stats_plotly(slugs)
            else:
                return None

        elif intent == "health":
            fig = graphs.health_radar_plotly(project_slug)

        elif any(w in q for w in ("committer", "contributor", "who commit", "who contribut", "top commit")):
            fig = graphs.top_committers_plotly(project_slug)
            if fig is None:
                return None  # no data — don't show an empty chart

        elif any(w in q for w in ("star", "popular", "growth")):
            fig = graphs.stars_over_time_plotly(project_slug)

        elif any(w in q for w in ("language", "breakdown", "written in", "code in")):
            fig = graphs.language_breakdown_plotly(project_slug)

        elif any(w in q for w in ("week", "weekly", "per week", "week-by-week")):
            fig = graphs.weekly_commits_plotly(project_slug)

        else:
            fig = graphs.commits_over_time_plotly(project_slug)

        return json.loads(fig.to_json())
    except Exception:
        return None


@router.post("/", response_model=QueryResponse)
async def ask(request: QueryRequest) -> QueryResponse:
    from explorer.query_processor import QueryProcessor
    intent = QueryProcessor().classify(request.query)
    query_hash = hashlib.sha256(request.query.encode()).hexdigest()[:16]

    if request.session_id:
        agent = _get_or_create_session(request.session_id, request.project_slug)
        response = agent.handle(request.query, project_slug=request.project_slug)
    else:
        from explorer.rag_system import RAGSystem
        response = RAGSystem().query(request.query, project_slug=request.project_slug)

    chart = _pick_chart(request.query, intent.value, request.project_slug or "")
    return QueryResponse(
        response=response,
        intent=intent.value,
        query_hash=query_hash,
        chart=chart,
    )


@router.post("/stream")
async def stream(request: QueryRequest) -> StreamingResponse:
    """
    SSE streaming variant of /api/query/.

    Yields newline-delimited JSON events:
      {"t":"chunk","v":"<text>"}   — one or more text chunks
      {"t":"done","intent":"...","hash":"...","chart":...}  — terminal event
    """
    import asyncio

    def _sse(obj: dict) -> str:
        return f"data: {json.dumps(obj)}\n\n"

    async def _generate() -> AsyncIterator[str]:
        loop = asyncio.get_event_loop()

        queue: asyncio.Queue = asyncio.Queue()

        def _producer() -> None:
            try:
                if request.session_id:
                    from explorer.query_processor import QueryProcessor
                    intent = QueryProcessor().classify(request.query).value
                    agent = _get_or_create_session(request.session_id, request.project_slug)
                    text = agent.handle(request.query, project_slug=request.project_slug)
                    loop.call_soon_threadsafe(queue.put_nowait, text)
                    loop.call_soon_threadsafe(queue.put_nowait, {"_done": True, "intent": intent, "hash": hashlib.sha256(request.query.encode()).hexdigest()[:16], "cached": False})
                else:
                    from explorer.rag_system import RAGSystem
                    rag = RAGSystem()
                    for item in rag.stream(request.query, project_slug=request.project_slug):
                        loop.call_soon_threadsafe(queue.put_nowait, item)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel

        import threading
        threading.Thread(target=_producer, daemon=True).start()

        while True:
            item = await queue.get()
            if item is None:
                break
            if isinstance(item, dict) and item.get("_done"):
                # Build the done event, attach chart if relevant
                chart = _pick_chart(
                    request.query, item.get("intent", ""), request.project_slug or ""
                )
                yield _sse({
                    "t": "done",
                    "intent": item.get("intent", ""),
                    "hash": item.get("hash", ""),
                    "cached": item.get("cached", False),
                    "chart": chart,
                })
            else:
                yield _sse({"t": "chunk", "v": str(item)})

    return StreamingResponse(_generate(), media_type="text/event-stream")


@router.post("/feedback")
async def feedback(request: FeedbackRequest) -> dict:
    from explorer.observability.metrics_collector import MetricsCollector
    MetricsCollector().record_feedback(request.query_hash, request.vote)
    return {"recorded": True}
