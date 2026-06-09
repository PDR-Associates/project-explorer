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

    def trigger_postgresql_survey(
        self,
        db_entity: DatabaseEntity,
        secrets_path: str | None = None,
    ) -> str:
        """Trigger a PostgreSQL survey using Egeria's governance process.
        
        Args:
            db_entity: DatabaseEntity with connection details
            secrets_path: Path to Egeria secrets file (optional)
            
        Returns:
            Engine action GUID for monitoring the survey
            
        Raises:
            EgeriaDatabaseSurveyorError: If survey trigger fails
        """
        self.connect()
        
        # Prepare request parameters for Egeria
        request_parameters = {
            "serverName": db_entity.slug,
            "hostIdentifier": db_entity.host,
            "portNumber": str(db_entity.port),
            "databaseName": db_entity.database_name,
            "versionIdentifier": "1.0",
            "description": db_entity.description or f"PostgreSQL database: {db_entity.display_name}",
        }
        
        # Add secrets path if provided
        if secrets_path:
            request_parameters["secretsStorePathName"] = secrets_path
            request_parameters["secretsCollectionName"] = f"PostgreSQL Server:{db_entity.slug}"
        
        # Trigger the governance process
        process_name = "PostgreSQLServer:CreateAndSurveyGovernanceActionProcess"
        
        try:
            log.info(f"Triggering Egeria survey for {db_entity.slug} using process: {process_name}")
            engine_action_guid = self._automated_curation.initiate_gov_action_process(
                process_name,
                None,  # request_source_guids
                None,  # action_targets
                None,  # start_time
                request_parameters,
                None,  # originator_service_name
                None,  # originator_engine_name
            )
            
            log.info(f"Survey triggered successfully. Engine action GUID: {engine_action_guid}")
            return engine_action_guid
            
        except Exception as exc:
            raise EgeriaDatabaseSurveyorError(
                f"Failed to trigger Egeria survey for {db_entity.slug}: {exc}"
            ) from exc

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
