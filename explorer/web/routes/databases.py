"""Database management endpoints — list, get, register, survey, remove."""
from __future__ import annotations

import asyncio
import os
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()


class DatabaseSummary(BaseModel):
    """Summary information for a database entity."""
    slug: str
    display_name: str
    db_type: str
    host: str
    port: int
    database_name: str
    description: str
    status: str
    last_surveyed_at: str
    schema_count: int | None
    table_count: int | None
    column_count: int | None


class DatabaseRegistration(BaseModel):
    """Request body for registering a new database."""
    slug: str
    display_name: str
    db_type: str
    host: str
    port: int
    database_name: str
    connection_ref: str  # Reference to connection config (e.g., env var name or secrets path)
    description: str = ""


class SurveyRequest(BaseModel):
    """Request body for triggering a database survey."""
    username: str  # Database username for this survey
    password: str  # Database password for this survey
    use_egeria: bool = False
    force_custom: bool = False
    egeria_url: str | None = None
    egeria_server: str | None = None
    secrets_path: str | None = None


class SurveyResult(BaseModel):
    """Result of a database survey operation."""
    status: str  # "ok" | "error" | "pending"
    slug: str
    message: str = ""
    error: str | None = None
    source: str | None = None  # "egeria" | "custom"
    schema_count: int | None = None
    table_count: int | None = None
    column_count: int | None = None


def _to_summary(db) -> DatabaseSummary:
    """Convert DatabaseEntity to DatabaseSummary."""
    # Get latest survey data if available
    from explorer.registry import ProjectRegistry
    registry = ProjectRegistry()
    surveys = registry.get_database_surveys(db.slug)
    latest = surveys[0] if surveys else None
    
    return DatabaseSummary(
        slug=db.slug,
        display_name=db.display_name,
        db_type=db.db_type,
        host=db.host,
        port=db.port,
        database_name=db.database_name,
        description=db.description,
        status=db.status.value,
        last_surveyed_at=db.last_surveyed_at or "",
        schema_count=latest.get("schema_count") if latest else None,
        table_count=latest.get("table_count") if latest else None,
        column_count=latest.get("column_count") if latest else None,
    )


@router.get("/", response_model=list[DatabaseSummary])
async def list_databases(db_type: str | None = None) -> list[DatabaseSummary]:
    """List all registered databases, optionally filtered by type."""
    from explorer.registry import ProjectRegistry
    registry = ProjectRegistry()
    databases = registry.list_databases(db_type=db_type)
    return [_to_summary(db) for db in databases]


@router.get("/{slug}", response_model=DatabaseSummary)
async def get_database(slug: str) -> DatabaseSummary:
    """Get details for a specific database."""
    from explorer.registry import ProjectRegistry
    registry = ProjectRegistry()
    database = registry.get_database(slug)
    if not database:
        raise HTTPException(status_code=404, detail=f"Database '{slug}' not found")
    return _to_summary(database)


@router.post("/register", response_model=DatabaseSummary)
async def register_database(req: DatabaseRegistration) -> DatabaseSummary:
    """Register a new database."""
    from explorer.registry import DatabaseEntity, ProjectRegistry, ProjectStatus
    
    registry = ProjectRegistry()
    
    # Check if slug already exists
    existing = registry.get_database(req.slug)
    if existing:
        raise HTTPException(status_code=400, detail=f"Database '{req.slug}' already exists")
    
    # Create database entity
    database = DatabaseEntity(
        slug=req.slug,
        display_name=req.display_name,
        db_type=req.db_type,
        host=req.host,
        port=req.port,
        database_name=req.database_name,
        connection_ref=req.connection_ref,
        description=req.description,
        status=ProjectStatus.ACTIVE,
        last_surveyed_at="",
    )
    
    # Register in registry
    registry.register_database(database)
    
    return _to_summary(database)


@router.post("/{slug}/survey", response_model=SurveyResult)
async def survey_database(slug: str, req: SurveyRequest) -> SurveyResult:
    """Trigger a database survey (hybrid mode by default)."""
    from explorer.registry import ProjectRegistry, ProjectStatus
    
    registry = ProjectRegistry()
    database = registry.get_database(slug)
    if not database:
        raise HTTPException(status_code=404, detail=f"Database '{slug}' not found")
    
    # Update status to surveying
    registry.update_database_status(slug, ProjectStatus.INDEXING, "")
    
    # Prepare credentials dict
    credentials = {
        "user": req.username,
        "password": req.password,
    }
    
    def _do_survey() -> dict[str, Any]:
        """Run survey in thread."""
        try:
            if req.force_custom:
                # Use custom surveyor only
                from explorer.surveyors.database.database_surveyor import run_database_survey
                result = run_database_survey(
                    db_slug=slug,
                    credentials=credentials,
                    registry=registry,
                )
                return {
                    "source": "custom",
                    "schema_count": result.get("schema_count"),
                    "table_count": result.get("table_count"),
                    "column_count": result.get("column_count"),
                }
            elif req.use_egeria:
                # Use hybrid approach (Egeria first, custom fallback)
                from explorer.surveyors.database.hybrid_database_surveyor import run_hybrid_survey
                result = run_hybrid_survey(
                    db_slug=slug,
                    credentials=credentials,
                    registry=registry,
                    force_custom=False,
                    platform_url=req.egeria_url,
                    view_server=req.egeria_server,
                    secrets_path=req.secrets_path,
                )
                return {
                    "source": result.get("source", "unknown"),
                    "schema_count": result.get("schema_count"),
                    "table_count": result.get("table_count"),
                    "column_count": result.get("column_count"),
                }
            else:
                # Default: custom surveyor
                from explorer.surveyors.database.database_surveyor import run_database_survey
                result = run_database_survey(
                    db_slug=slug,
                    credentials=credentials,
                    registry=registry,
                )
                return {
                    "source": "custom",
                    "schema_count": result.get("schema_count"),
                    "table_count": result.get("table_count"),
                    "column_count": result.get("column_count"),
                }
        except Exception as e:
            registry.update_database_status(slug, ProjectStatus.ERROR, str(e))
            raise
    
    try:
        result = await asyncio.to_thread(_do_survey)
        registry.update_database_status(slug, ProjectStatus.ACTIVE, "")
        
        return SurveyResult(
            status="ok",
            slug=slug,
            message=f"Survey completed using {result['source']} surveyor",
            source=result["source"],
            schema_count=result.get("schema_count"),
            table_count=result.get("table_count"),
            column_count=result.get("column_count"),
        )
    except Exception as exc:
        return SurveyResult(
            status="error",
            slug=slug,
            error=str(exc),
        )


@router.delete("/{slug}")
async def remove_database(slug: str) -> dict:
    """Remove a database from the registry."""
    from explorer.registry import ProjectRegistry
    
    registry = ProjectRegistry()
    database = registry.get_database(slug)
    if not database:
        raise HTTPException(status_code=404, detail=f"Database '{slug}' not found")
    
    registry.remove_database(slug)
    return {"removed": slug}


@router.get("/{slug}/surveys")
async def get_database_surveys(slug: str) -> list[dict]:
    """Get survey history for a database."""
    from explorer.registry import ProjectRegistry
    
    registry = ProjectRegistry()
    database = registry.get_database(slug)
    if not database:
        raise HTTPException(status_code=404, detail=f"Database '{slug}' not found")
    
    surveys = registry.get_database_surveys(slug)
    return surveys

# Made with Bob
