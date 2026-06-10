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
    server_slug: str = ""       # FK to db_servers; empty for standalone databases
    egeria_asset_guid: str = "" # DB element GUID in Egeria; "" = not yet cataloged
    last_survey_source: str = ""# "local" | "egeria" | "egeria-published" from latest survey
    db_user: str = ""           # stored username (no password exposed)
    egeria_host: str = ""
    egeria_url: str = ""
    egeria_server: str = ""
    egeria_user: str = ""


class DatabaseRegistration(BaseModel):
    """Request body for registering a new database."""
    slug: str
    display_name: str
    db_type: str
    host: str
    port: int
    database_name: str
    connection_ref: str = ""
    description: str = ""
    # Optional stored credentials
    db_user: str = ""
    db_password: str = ""
    # Egeria-visible hostname for the DB (e.g. host.docker.internal when DB is in Docker)
    egeria_host: str = ""
    # Optional stored Egeria connection details
    egeria_url: str = ""
    egeria_server: str = ""
    egeria_user: str = ""
    egeria_password: str = ""


class SurveyRequest(BaseModel):
    """Request body for triggering a database survey."""
    username: str = ""  # DB username — falls back to stored db_user if blank
    password: str = ""  # DB password — falls back to stored db_password if blank
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
        server_slug=getattr(db, "server_slug", "") or "",
        egeria_asset_guid=getattr(db, "egeria_asset_guid", "") or "",
        last_survey_source=latest.get("source", "") if latest else "",
        db_user=db.db_user or "",
        egeria_host=db.egeria_host or "",
        egeria_url=db.egeria_url or "",
        egeria_server=db.egeria_server or "",
        egeria_user=db.egeria_user or "",
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
        db_user=req.db_user,
        db_password=req.db_password,
        egeria_host=req.egeria_host,
        egeria_url=req.egeria_url,
        egeria_server=req.egeria_server,
        egeria_user=req.egeria_user,
        egeria_password=req.egeria_password,
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
    
    # Resolve credentials — request overrides stored values
    resolved_user = req.username or database.db_user
    resolved_pwd  = req.password or database.db_password
    if not resolved_user or not resolved_pwd:
        raise HTTPException(
            status_code=400,
            detail="Database credentials are required. Either supply username/password in the request or store them at registration.",
        )
    credentials = {"user": resolved_user, "password": resolved_pwd}

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

    return registry.get_database_surveys(slug)


class PublishRequest(BaseModel):
    """Request body for publishing a database survey to Egeria.

    All fields are optional — stored values on the DatabaseEntity are used as defaults.
    Pass fields only to override stored values for a single call.
    """
    egeria_url: str | None = None
    egeria_server: str | None = None
    egeria_user: str | None = None
    egeria_password: str | None = None
    db_user: str = ""
    db_pwd: str = ""


class PublishResult(BaseModel):
    """Result of publishing a database survey to Egeria."""
    status: str
    slug: str
    server_guid: str | None = None     # Egeria PostgreSQL server element GUID
    asset_guid: str | None = None      # Egeria database element GUID
    report_guid: str | None = None     # Egeria survey action GUID
    annotation_count: int | None = None
    server_display_name: str | None = None
    database_display_name: str | None = None
    error: str | None = None


@router.post("/{slug}/publish", response_model=PublishResult)
async def publish_database_survey(slug: str, req: PublishRequest = PublishRequest()) -> PublishResult:
    """Publish the latest local database survey to Egeria."""
    from explorer.registry import ProjectRegistry

    registry = ProjectRegistry()
    database = registry.get_database(slug)
    if not database:
        raise HTTPException(status_code=404, detail=f"Database '{slug}' not found")

    surveys = registry.get_database_surveys(slug)
    if not surveys:
        raise HTTPException(status_code=404, detail=f"No survey data for '{slug}' — run a survey first")

    def _do_publish() -> dict[str, Any]:
        import json as _json
        from explorer.surveyors.database.egeria_database_surveyor import EgeriaDatabaseSurveyor

        # Resolve: request overrides > stored values > env vars (inside EgeriaDatabaseSurveyor)
        resolved_db_user = req.db_user or database.db_user
        resolved_db_pwd  = req.db_pwd  or database.db_password

        if not resolved_db_user or not resolved_db_pwd:
            raise ValueError(
                "Database credentials are required to catalog in Egeria. "
                "Store them at registration or supply db_user/db_pwd in the request."
            )

        surveyor = EgeriaDatabaseSurveyor(
            platform_url=req.egeria_url or database.egeria_url or None,
            view_server=req.egeria_server or database.egeria_server or None,
            user_id=req.egeria_user or database.egeria_user or None,
            user_password=req.egeria_password or database.egeria_password or None,
        )

        latest = surveys[0]
        survey_data = _json.loads(latest.get("survey_data", "{}"))
        schema_info = survey_data.get("schema_info", {})

        result = surveyor.publish_local_survey(
            db_entity=database,
            schema_info=schema_info,
            schema_count=latest.get("schema_count", 0),
            table_count=latest.get("table_count", 0),
            column_count=latest.get("column_count", 0),
            surveyed_at=latest.get("surveyed_at", ""),
            registry=registry,
            db_user=resolved_db_user,
            db_pwd=resolved_db_pwd,
        )
        return result

    try:
        result = await asyncio.to_thread(_do_publish)
        egeria_host = database.egeria_host or database.host
        return PublishResult(
            status="ok",
            slug=slug,
            server_guid=result.get("server_guid"),
            asset_guid=result.get("asset_guid"),
            report_guid=result.get("report_guid"),
            annotation_count=result.get("annotation_count"),
            server_display_name=f"{egeria_host}:{database.port}",
            database_display_name=database.database_name,
        )
    except Exception as exc:
        return PublishResult(status="error", slug=slug, error=str(exc))
