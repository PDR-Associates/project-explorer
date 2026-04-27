"""Query endpoint — POST a question, get a response."""
from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class QueryRequest(BaseModel):
    query: str
    project_slug: str | None = None


class QueryResponse(BaseModel):
    response: str
    intent: str
    cached: bool = False


@router.post("/", response_model=QueryResponse)
async def ask(request: QueryRequest) -> QueryResponse:
    from explorer.rag_system import RAGSystem
    from explorer.query_processor import QueryProcessor
    system = RAGSystem()
    intent = QueryProcessor().classify(request.query)
    response = system.query(request.query, project_slug=request.project_slug)
    return QueryResponse(response=response, intent=intent.value)
