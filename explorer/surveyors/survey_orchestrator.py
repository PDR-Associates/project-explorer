"""Runs all sub-surveyors for a project and assembles a SurveyResult."""
from __future__ import annotations

import logging
from datetime import datetime

from explorer.registry import ProjectRegistry
from explorer.surveyors.file_classifier.file_classifier_surveyor import FileClassifierSurveyor
from explorer.surveyors.sub_surveyors import (
    ApiStructureSurveyor,
    DependencySurveyor,
    DocumentationSurveyor,
    FileStructureSurveyor,
    HealthSurveyor,
    LanguageSurveyor,
    SecuritySurveyor,
)
from explorer.surveyors.survey_report import SurveyResult

log = logging.getLogger(__name__)


class SurveyOrchestrator:
    """
    Runs all sub-surveyors in sequence and returns a SurveyResult.

    Parameters
    ----------
    registry        : open ProjectRegistry
    pyegeria_client : optional — passed to FileClassifierSurveyor for cache refresh
    force_refresh   : force FileTypeCache refresh even if not stale
    """

    def __init__(
        self,
        registry: ProjectRegistry,
        pyegeria_client=None,
        force_refresh: bool = False,
    ) -> None:
        self._registry = registry
        self._pyegeria_client = pyegeria_client
        self._force_refresh = force_refresh

    def run(self, project_slug: str) -> SurveyResult:
        """Survey a single project and return the assembled SurveyResult."""
        project = self._registry.get(project_slug)
        if project is None:
            raise ValueError(f"Project '{project_slug}' not found in registry.")

        result = SurveyResult(
            project_slug=project.slug,
            project_display_name=project.display_name,
            github_url=project.github_url,
            surveyed_at=datetime.utcnow(),
        )

        surveyors = [
            FileClassifierSurveyor(
                project,
                self._registry,
                pyegeria_client=self._pyegeria_client,
                force_refresh=self._force_refresh,
            ),
            FileStructureSurveyor(project, self._registry),
            LanguageSurveyor(project, self._registry),
            HealthSurveyor(project, self._registry),
            DependencySurveyor(project, self._registry),
            DocumentationSurveyor(project, self._registry),
            SecuritySurveyor(project, self._registry),
            ApiStructureSurveyor(project, self._registry),
        ]

        for surveyor in surveyors:
            log.info("Running %s for %s …", surveyor.step_name, project.slug)
            try:
                annotations = surveyor.run()
                for ann in annotations:
                    result.add(ann)
                log.info("  → %d annotation(s)", len(annotations))
            except Exception as exc:
                msg = f"{surveyor.step_name} raised unexpectedly: {exc}"
                log.exception(msg)
                result.add_error(msg)

        log.info(
            "Survey complete for %s: %d annotation(s), %d error(s)",
            project.slug,
            len(result.annotations),
            len(result.errors),
        )
        return result
