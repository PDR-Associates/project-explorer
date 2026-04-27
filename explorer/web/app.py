"""FastAPI application — query and project management web interface."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from explorer.web.routes import projects, query, stats

app = FastAPI(
    title="Project Explorer",
    description="Multi-agent RAG assistant for GitHub projects",
    version="0.1.0",
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

app.include_router(query.router, prefix="/api/query", tags=["query"])
app.include_router(projects.router, prefix="/api/projects", tags=["projects"])
app.include_router(stats.router, prefix="/api/stats", tags=["stats"])


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
