"""
Publishes a SurveyResult to Egeria via pyegeria.

Asset type: SourceControlLibrary (placeholder — full pyegeria support pending).
See design doc Q2 for context.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime

from explorer.surveyors.survey_report import AnnotationType, SurveyResult

log = logging.getLogger(__name__)

# ── Egeria connection defaults (standard pyegeria env vars) ──────────────────
_DEFAULT_PLATFORM_URL = "https://localhost:9443"
_DEFAULT_VIEW_SERVER = "qs-view-server"
_DEFAULT_USER = "erinoverview"
_DEFAULT_PASSWORD = "secret"
_DEFAULT_TIMEOUT = 30


class EgeriaConnectionError(RuntimeError):
    """Raised when Egeria credentials are absent or the platform is unreachable."""


class EgeriaPublisher:
    """
    Converts a SurveyResult into Egeria API calls:
      1. Find or create the SourceControlLibrary asset  (placeholder)
      2. Create a SurveyReport linked via ReportSubject
      3. Create one Annotation per SurveyResult.annotation

    All Egeria coupling is contained here — sub-surveyors and SurveyResult
    have no pyegeria dependency.
    """

    def __init__(
        self,
        platform_url: str | None = None,
        view_server: str | None = None,
        user_id: str | None = None,
        user_password: str | None = None,
        timeout: int | None = None,
    ) -> None:
        self.platform_url = platform_url or os.getenv("EGERIA_PLATFORM_URL", _DEFAULT_PLATFORM_URL)
        self.view_server = view_server or os.getenv("EGERIA_VIEW_SERVER", _DEFAULT_VIEW_SERVER)
        self.user_id = user_id or os.getenv("EGERIA_USER", _DEFAULT_USER)
        self.user_password = user_password or os.getenv("EGERIA_USER_PASSWORD", _DEFAULT_PASSWORD)
        self.timeout = timeout or int(os.getenv("PYEGERIA_TIMEOUT_SECONDS", str(_DEFAULT_TIMEOUT)))
        self._asset_maker = None
        self._discovery = None

    # ── public entry point ────────────────────────────────────────────────────

    def publish(self, result: SurveyResult) -> str:
        """
        Push the full SurveyResult to Egeria.
        Returns the GUID of the created SurveyReport.
        Raises EgeriaConnectionError if credentials are missing or platform unreachable.
        """
        self._connect()
        asset_guid = self._find_or_create_asset(result)
        report_guid = self._create_survey_report(result, asset_guid)
        self._create_annotations(result, report_guid)
        log.info(
            "Published survey for %s → SurveyReport GUID %s (%d annotations)",
            result.project_slug,
            report_guid,
            len(result.annotations),
        )
        return report_guid

    # ── connection ────────────────────────────────────────────────────────────

    def _connect(self) -> None:
        if not self.platform_url:
            raise EgeriaConnectionError(
                "EGERIA_PLATFORM_URL is not set. "
                "Add it to your .env file or pass platform_url= to EgeriaPublisher."
            )
        try:
            from pyegeria import AssetMaker
            from pyegeria.omvs.data_discovery import DataDiscovery

            self._asset_maker = AssetMaker(
                self.view_server, self.platform_url, self.user_id, self.user_password
            )
            self._asset_maker.create_egeria_bearer_token(self.user_id, self.user_password)

            self._discovery = DataDiscovery(
                self.view_server, self.platform_url, self.user_id, self.user_password
            )
            self._discovery.create_egeria_bearer_token(self.user_id, self.user_password)
        except ImportError as exc:
            raise EgeriaConnectionError(
                "pyegeria is not installed. Add it to your dependencies."
            ) from exc
        except Exception as exc:
            raise EgeriaConnectionError(
                f"Could not connect to Egeria at {self.platform_url}: {exc}"
            ) from exc

    # ── asset registration (placeholder) ─────────────────────────────────────

    def _find_or_create_asset(self, result: SurveyResult) -> str:
        """
        Return the GUID of a SourceControlLibrary asset for the GitHub repo.

        PLACEHOLDER: SourceControlLibrary support is pending in pyegeria.
        Until the new AssetMaker methods land, we search for an existing asset
        by GitHub URL (qualifiedName) and create a bare SoftwareServer as a
        stand-in.  This will be replaced once pyegeria exposes
        create_source_control_library().
        """
        log.warning(
            "EgeriaPublisher: SourceControlLibrary not yet supported by pyegeria — "
            "using qualifiedName search + SoftwareServer placeholder."
        )
        qualified_name = f"GitHubRepo::{result.github_url}"

        # Try to find existing
        try:
            existing = self._asset_maker.find_assets(
                search_string=qualified_name,
                output_format="DICT",
            )
            if isinstance(existing, list) and existing:
                guid = existing[0].get("guid")
                if guid:
                    log.info("Found existing asset GUID %s for %s", guid, result.project_slug)
                    return guid
        except Exception as exc:
            log.debug("Asset search failed (will create): %s", exc)

        # Create placeholder
        body = {
            "class": "NewElementRequestBody",
            "properties": {
                "class": "SoftwareServerProperties",
                "qualifiedName": qualified_name,
                "displayName": result.project_display_name,
                "description": f"GitHub repository: {result.github_url}",
                "additionalProperties": {
                    "github_url": result.github_url,
                    "project_slug": result.project_slug,
                    "placeholder": "true",
                    "replace_with": "SourceControlLibrary",
                },
            },
        }
        guid = self._asset_maker.create_asset(body=body)
        log.info("Created placeholder asset GUID %s for %s", guid, result.project_slug)
        return guid

    # ── survey report ─────────────────────────────────────────────────────────

    def _create_survey_report(self, result: SurveyResult, asset_guid: str) -> str:
        qualified_name = (
            f"SurveyReport::GitHubRepo::{result.project_slug}::{result.surveyed_at.isoformat()}"
        )
        body = {
            "class": "NewElementRequestBody",
            "parentGUID": asset_guid,
            "parentRelationshipTypeName": "ReportSubject",
            "properties": {
                "class": "SurveyReportProperties",
                "qualifiedName": qualified_name,
                "displayName": f"Survey: {result.project_display_name}",
                "description": (
                    f"Automated survey of {result.github_url} "
                    f"run at {result.surveyed_at.isoformat()} by project-explorer."
                ),
                "additionalProperties": {
                    "annotation_count": str(len(result.annotations)),
                    "error_count": str(len(result.errors)),
                },
            },
        }
        return self._asset_maker.create_asset(body=body)

    # ── annotations ───────────────────────────────────────────────────────────

    def _create_annotations(self, result: SurveyResult, report_guid: str) -> None:
        for i, ann in enumerate(result.annotations):
            qualified_name = (
                f"Annotation::{result.project_slug}::{result.surveyed_at.isoformat()}::{i}"
            )
            props: dict = {
                "class": "AnnotationProperties",
                "qualifiedName": qualified_name,
                "annotationType": ann.annotation_type.value,
                "summary": ann.summary,
                "analysisStep": ann.analysis_step,
                "confidence": ann.confidence,
            }
            if ann.explanation:
                props["explanation"] = ann.explanation
            if ann.expression:
                props["expression"] = ann.expression
            if ann.json_properties:
                props["jsonProperties"] = json.dumps(ann.json_properties)

            # Carry subtype-specific fields into additionalProperties
            extra: dict = {}
            if ann.annotation_type == AnnotationType.RESOURCE_MEASURE:
                extra["resourceProperties"] = json.dumps(getattr(ann, "resource_properties", {}))
            elif ann.annotation_type == AnnotationType.CLASSIFICATION:
                extra["candidateClassifications"] = json.dumps(
                    getattr(ann, "candidate_classifications", [])
                )
            elif ann.annotation_type == AnnotationType.QUALITY_SCORE:
                extra["qualityScores"] = json.dumps(getattr(ann, "quality_scores", {}))
            elif ann.annotation_type == AnnotationType.DATA_CLASS:
                extra["candidateDataClassNames"] = json.dumps(
                    getattr(ann, "candidate_data_class_names", [])
                )
            elif ann.annotation_type == AnnotationType.REQUEST_FOR_ACTION:
                extra["actionRequested"] = getattr(ann, "action_requested", "")
                extra["actionTargetName"] = getattr(ann, "action_target_name", "")
            elif ann.annotation_type == AnnotationType.SCHEMA_ANALYSIS:
                extra["schemaName"] = getattr(ann, "schema_name", "")
                extra["schemaType"] = getattr(ann, "schema_type", "")

            if extra:
                props["additionalProperties"] = extra

            body = {
                "class": "NewElementRequestBody",
                "parentGUID": report_guid,
                "parentRelationshipTypeName": "ReportedAnnotation",
                "properties": props,
            }
            try:
                self._discovery.create_annotation(body=body)
            except Exception as exc:
                log.warning("Failed to create annotation %d (%s): %s", i, ann.annotation_type.value, exc)
