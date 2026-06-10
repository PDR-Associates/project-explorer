"""Egeria-based database surveyor for triggering native PostgreSQL surveys."""
from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from explorer.registry import DatabaseEntity

log = logging.getLogger(__name__)


class EgeriaDatabaseSurveyorError(RuntimeError):
    """Raised when Egeria survey operations fail."""


class EgeriaDatabaseSurveyor:
    """Trigger and retrieve PostgreSQL surveys using Egeria's native capabilities.
    
    This surveyor uses Egeria's pre-built governance processes to survey databases,
    leveraging Egeria's expertise and standardized metadata.
    
    Parameters
    ----------
    platform_url : Egeria platform URL (falls back to EGERIA_PLATFORM_URL env var)
    view_server : Egeria view server name
    user_id : Egeria user id
    user_password : Egeria user password
    """

    def __init__(
        self,
        platform_url: str | None = None,
        view_server: str | None = None,
        user_id: str | None = None,
        user_password: str | None = None,
    ) -> None:
        self.platform_url = platform_url or os.getenv("EGERIA_PLATFORM_URL", "")
        self.view_server = view_server or os.getenv("EGERIA_VIEW_SERVER", "qs-view-server")
        self.user_id = user_id or os.getenv("EGERIA_USER", "erinoverview")
        self.user_password = user_password or os.getenv("EGERIA_USER_PASSWORD", "secret")
        self._automated_curation = None
        self._asset_maker = None
        self._discovery = None

    def connect(self) -> None:
        """Establish pyegeria client connections."""
        if self._automated_curation is not None:
            return
        
        if not self.platform_url:
            raise EgeriaDatabaseSurveyorError(
                "EGERIA_PLATFORM_URL is not set. "
                "Set it in .env or pass platform_url= to EgeriaDatabaseSurveyor."
            )
        
        try:
            from pyegeria import AutomatedCuration, AssetMaker
            from pyegeria.omvs.data_discovery import DataDiscovery

            # AutomatedCuration for triggering surveys
            self._automated_curation = AutomatedCuration(
                self.view_server, self.platform_url, self.user_id, self.user_password
            )
            self._automated_curation.create_egeria_bearer_token(self.user_id, self.user_password)

            # AssetMaker for finding survey reports
            self._asset_maker = AssetMaker(
                self.view_server, self.platform_url, self.user_id, self.user_password
            )
            self._asset_maker.create_egeria_bearer_token(self.user_id, self.user_password)

            # DataDiscovery for retrieving annotations
            self._discovery = DataDiscovery(
                self.view_server, self.platform_url, self.user_id, self.user_password
            )
            self._discovery.create_egeria_bearer_token(self.user_id, self.user_password)

        except ImportError as exc:
            raise EgeriaDatabaseSurveyorError("pyegeria is not installed.") from exc
        except Exception as exc:
            raise EgeriaDatabaseSurveyorError(
                f"Could not connect to Egeria at {self.platform_url}: {exc}"
            ) from exc

    def catalog_and_survey(
        self,
        db_entity: "DatabaseEntity",
        db_user: str,
        db_pwd: str,
        registry=None,
        survey_after_catalog: bool = True,
    ) -> dict:
        """Catalog the PostgreSQL server + database in Egeria, then optionally initiate a native survey.

        Egeria's template-based creation stores the connection details (including
        credentials) so that subsequent surveys can be initiated without supplying
        credentials again.

        Returns dict with keys: server_guid, database_guid, survey_action_guid.
        """
        self.connect()

        # Use egeria_host if set (e.g. host.docker.internal when Egeria runs in Docker);
        # fall back to the locally-visible host.
        egeria_host = getattr(db_entity, "egeria_host", "") or db_entity.host
        server_name = f"{egeria_host}:{db_entity.port}"

        # ── 1. Create / find PostgreSQL Server element ─────────────────────────
        server_guid = db_entity.egeria_asset_guid  # reuse if already stored as server
        if not server_guid:
            server_guid = self._find_element_guid(server_name)

        if not server_guid:
            try:
                server_guid = self._automated_curation.create_postgres_server_element_from_template(
                    postgres_server=server_name,
                    host_name=egeria_host,
                    port=str(db_entity.port),
                    db_user=db_user,
                    db_pwd=db_pwd,
                    description=db_entity.description or f"PostgreSQL server at {egeria_host}:{db_entity.port}",
                )
                log.info(f"Created PostgreSQL server element: {server_guid}")
            except Exception as exc:
                raise EgeriaDatabaseSurveyorError(
                    f"Could not create PostgreSQL server element for {server_name}: {exc}"
                ) from exc

        # ── 2. Create / find PostgreSQL Database element ───────────────────────
        db_guid = self._find_element_guid(db_entity.database_name)

        if not db_guid:
            try:
                db_guid = self._automated_curation.create_postgres_database_element_from_template(
                    postgres_database=db_entity.database_name,
                    server_name=server_name,
                    host_identifier=egeria_host,
                    port=str(db_entity.port),
                    db_user=db_user,
                    db_pwd=db_pwd,
                    description=db_entity.description or f"PostgreSQL database {db_entity.database_name}",
                )
                log.info(f"Created PostgreSQL database element: {db_guid}")
            except Exception as exc:
                raise EgeriaDatabaseSurveyorError(
                    f"Could not create PostgreSQL database element for {db_entity.database_name}: {exc}"
                ) from exc

        # Persist database GUID to registry
        if registry and db_guid:
            registry.set_database_egeria_guid(db_entity.slug, db_guid)

        # ── 3. Optionally initiate Egeria's native surveys ────────────────────
        server_survey_guid = ""
        survey_action_guid = ""
        if survey_after_catalog:
            # Survey the server first (captures connection info, database list, server config)
            if server_guid:
                try:
                    server_survey_guid = self._automated_curation.initiate_postgres_server_survey(server_guid)
                    log.info(f"Egeria server survey initiated: {server_survey_guid}")
                except AttributeError:
                    log.debug("initiate_postgres_server_survey not available in this pyegeria version")
                except Exception as exc:
                    log.warning(f"Server survey initiation failed (non-fatal): {exc}")

            # Survey the database (captures schemas, tables, columns, relationships)
            if db_guid:
                try:
                    survey_action_guid = self._automated_curation.initiate_postgres_database_survey(db_guid)
                    log.info(f"Egeria database survey initiated: {survey_action_guid}")
                except Exception as exc:
                    log.warning(f"Cataloged OK but Egeria database survey initiation failed: {exc}")

        return {
            "server_guid": server_guid,
            "server_survey_guid": server_survey_guid,
            "database_guid": db_guid,
            "survey_action_guid": survey_action_guid,
        }

    def trigger_survey_by_guid(self, db_guid: str) -> str:
        """Initiate Egeria's native PostgreSQL database survey using a stored GUID.

        Use this when the database is already cataloged in Egeria (db_guid known).
        Egeria uses its stored connection details — no credentials required here.
        """
        self.connect()
        try:
            action_guid = self._automated_curation.initiate_postgres_database_survey(db_guid)
            log.info(f"Egeria database survey initiated: {action_guid}")
            return action_guid
        except Exception as exc:
            raise EgeriaDatabaseSurveyorError(
                f"Failed to initiate Egeria survey for db_guid={db_guid}: {exc}"
            ) from exc

    def _find_element_guid(self, name: str) -> str:
        """Return the GUID of an existing Egeria element by display/qualified name, or '' if not found."""
        try:
            result = self._automated_curation.get_guid_for_name(name)
            if isinstance(result, list) and result:
                return result[0] if isinstance(result[0], str) else result[0].get("guid", "")
            if isinstance(result, str) and len(result) > 10:
                return result
        except Exception:
            pass
        return ""

    def check_survey_exists(self, db_slug: str) -> bool:
        """Check if any survey reports exist for this database in Egeria.
        
        Args:
            db_slug: Database slug to check
            
        Returns:
            True if survey reports exist, False otherwise
        """
        self.connect()
        
        search_prefix = f"SurveyReport::PostgreSQL::{db_slug}::"
        
        try:
            results = self._asset_maker.find_assets(
                search_string=search_prefix,
                starts_with=True,
                ignore_case=False,
                output_format="JSON",
            )
            return isinstance(results, list) and len(results) > 0
        except Exception as exc:
            log.debug(f"check_survey_exists failed for {db_slug}: {exc}")
            return False

    def get_survey_reports(self, db_slug: str) -> list[dict]:
        """Retrieve all survey reports for a database from Egeria.
        
        Args:
            db_slug: Database slug
            
        Returns:
            List of survey report dicts with keys:
            - guid: Survey report GUID
            - qualified_name: Full qualified name
            - display_name: Display name
            - surveyed_at: Timestamp from qualified name
            - annotation_count: Number of annotations
            - description: Report description
        """
        self.connect()
        
        search_prefix = f"SurveyReport::PostgreSQL::{db_slug}::"
        reports = []
        
        try:
            results = self._asset_maker.find_assets(
                search_string=search_prefix,
                starts_with=True,
                ignore_case=False,
                output_format="JSON",
            )
            
            if not isinstance(results, list):
                return []
            
            for el in results:
                header = el.get("elementHeader", {})
                props = el.get("properties", {})
                additional = props.get("additionalProperties", {})
                qn = props.get("qualifiedName", "")
                
                # Extract timestamp from qualifiedName: SurveyReport::PostgreSQL::{slug}::{ts}
                ts = qn.split("::")[-1] if "::" in qn else ""
                
                reports.append({
                    "guid": header.get("guid", ""),
                    "qualified_name": qn,
                    "display_name": props.get("displayName", ""),
                    "surveyed_at": ts,
                    "annotation_count": self._safe_int(additional.get("annotation_count")),
                    "description": props.get("description", ""),
                })
        except Exception as exc:
            log.debug(f"get_survey_reports failed for {db_slug}: {exc}")
        
        # Sort newest first
        reports.sort(key=lambda r: r.get("surveyed_at", ""), reverse=True)
        return reports

    def get_annotations(self, db_slug: str, surveyed_at: str) -> list[dict]:
        """Retrieve all annotations for a specific survey from Egeria.
        
        Args:
            db_slug: Database slug
            surveyed_at: Survey timestamp
            
        Returns:
            List of annotation dicts with keys:
            - guid: Annotation GUID
            - annotation_type: Type of annotation
            - summary: Summary text
            - confidence: Confidence level (0-100)
            - analysis_step: Analysis step name
            - explanation: Detailed explanation
            - json_properties: Additional properties as JSON
        """
        self.connect()
        
        search_prefix = f"Annotation::PostgreSQL::{db_slug}::{surveyed_at}::"
        annotations = []
        
        try:
            results = self._discovery.find_annotations(
                search_string=search_prefix,
                starts_with=True,
                ignore_case=False,
                output_format="JSON",
                page_size=500,
            )
            
            if not isinstance(results, list):
                return []
            
            for el in results:
                header = el.get("elementHeader", {})
                props = el.get("properties", {})
                
                annotations.append({
                    "guid": header.get("guid", ""),
                    "annotation_type": props.get("annotationType", ""),
                    "summary": props.get("summary", ""),
                    "confidence": self._safe_int(props.get("confidence", 100)),
                    "analysis_step": props.get("analysisStep", ""),
                    "explanation": props.get("explanation", ""),
                    "expression": props.get("expression", ""),
                    "json_properties": props.get("jsonProperties", {}),
                })
        except Exception as exc:
            log.debug(f"get_annotations failed for {db_slug}/{surveyed_at}: {exc}")
        
        return annotations

    def get_latest_survey(self, db_slug: str) -> dict | None:
        """Get the most recent survey report for a database.
        
        Args:
            db_slug: Database slug
            
        Returns:
            Survey report dict or None if no surveys exist
        """
        reports = self.get_survey_reports(db_slug)
        return reports[0] if reports else None

    def publish_local_survey(
        self,
        db_entity: "DatabaseEntity",
        schema_info: dict,
        schema_count: int,
        table_count: int,
        column_count: int,
        surveyed_at: str,
        registry=None,
        db_user: str = "",
        db_pwd: str = "",
    ) -> dict:
        """Catalog the database in Egeria and initiate a native PostgreSQL survey.

        Uses AutomatedCuration.create_postgres_server_element_from_template and
        create_postgres_database_element_from_template to register the server and
        database in Egeria (with stored credentials), then optionally initiates
        Egeria's native PostgreSQL survey.

        Returns dict with keys: asset_guid (db element GUID), report_guid (survey action
        GUID or ""), annotation_count (0 — Egeria populates annotations asynchronously).
        """
        result = self.catalog_and_survey(
            db_entity=db_entity,
            db_user=db_user,
            db_pwd=db_pwd,
            registry=registry,
            survey_after_catalog=True,
        )

        server_guid        = result.get("server_guid", "")
        db_guid            = result.get("database_guid", "")
        survey_action_guid = result.get("survey_action_guid", "")
        server_survey_guid = result.get("server_survey_guid", "")

        if registry and db_guid:
            registry.record_database_survey(
                slug=db_entity.slug,
                schema_count=schema_count,
                table_count=table_count,
                column_count=column_count,
                survey_data={"schema_info": schema_info},
                egeria_report_guid=survey_action_guid,
                source="egeria-published",
            )

        return {
            "server_guid":        server_guid,
            "asset_guid":         db_guid,
            "report_guid":        survey_action_guid,
            "server_survey_guid": server_survey_guid,
            "annotation_count":   0,
        }

    @staticmethod
    def _safe_int(value, default: int = 0) -> int:
        """Safely convert value to int."""
        try:
            return int(value) if value is not None else default
        except (ValueError, TypeError):
            return default


def can_use_egeria() -> bool:
    """Check if Egeria is available and configured.
    
    Returns:
        True if EGERIA_PLATFORM_URL is set and pyegeria is installed
    """
    if not os.getenv("EGERIA_PLATFORM_URL"):
        return False
    
    try:
        import pyegeria  # noqa: F401
        return True
    except ImportError:
        return False

# Made with Bob
