"""Project management endpoints — list, get, remove, refresh."""
from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

router = APIRouter()


class ProjectSummary(BaseModel):
    slug: str
    display_name: str
    github_url: str
    description: str
    status: str
    collections: list[str]
    last_indexed_at: str
    last_commit_sha: str


def _to_summary(p) -> ProjectSummary:
    return ProjectSummary(
        slug=p.slug,
        display_name=p.display_name,
        github_url=p.github_url,
        description=p.description,
        status=p.status.value,
        collections=p.collections,
        last_indexed_at=p.last_indexed_at,
        last_commit_sha=p.last_commit_sha,
    )


@router.get("/", response_model=list[ProjectSummary])
async def list_projects() -> list[ProjectSummary]:
    from explorer.registry import ProjectRegistry
    return [_to_summary(p) for p in ProjectRegistry().list_all()]


@router.get("/{slug}", response_model=ProjectSummary)
async def get_project(slug: str) -> ProjectSummary:
    from explorer.registry import ProjectRegistry
    project = ProjectRegistry().get(slug)
    if not project:
        raise HTTPException(status_code=404, detail=f"Project '{slug}' not found")
    return _to_summary(project)


@router.post("/{slug}/refresh")
async def refresh_project(slug: str, background_tasks: BackgroundTasks) -> dict:
    """Trigger an incremental re-index in the background."""
    from explorer.registry import ProjectRegistry
    project = ProjectRegistry().get(slug)
    if not project:
        raise HTTPException(status_code=404, detail=f"Project '{slug}' not found")

    def _do_refresh():
        from explorer.ingestion.incremental import IncrementalIndexer
        from explorer.query_cache import QueryCache
        IncrementalIndexer().refresh(project)
        QueryCache().invalidate_project(slug)

    background_tasks.add_task(_do_refresh)
    return {"status": "refresh_started", "slug": slug}


@router.delete("/{slug}")
async def remove_project(slug: str) -> dict:
    from explorer.registry import ProjectRegistry
    from explorer.multi_collection_store import MultiCollectionStore
    registry = ProjectRegistry()
    project = registry.get(slug)
    if not project:
        raise HTTPException(status_code=404, detail=f"Project '{slug}' not found")
    store = MultiCollectionStore()
    for collection in project.collections:
        store.drop_collection(collection)
    registry.remove(slug)
    return {"removed": slug}
