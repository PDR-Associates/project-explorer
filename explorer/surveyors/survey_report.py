"""
Egeria-aligned survey result dataclasses.

All types are plain Python — no pyegeria dependency here.
EgeriaPublisher (egeria_publisher.py) converts these to API calls.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class AnnotationType(str, Enum):
    RESOURCE_MEASURE = "ResourceMeasureAnnotation"
    CLASSIFICATION = "ClassificationAnnotation"
    SCHEMA_ANALYSIS = "SchemaAnalysis"
    DATA_CLASS = "DataClassAnnotation"
    QUALITY_SCORE = "QualityScoreAnnotation"
    RELATIONSHIP = "RelationshipAnnotation"
    REQUEST_FOR_ACTION = "RequestForAction"


@dataclass
class Annotation:
    """Base annotation — mirrors Egeria Area 6 base Annotation entity."""
    annotation_type: AnnotationType
    summary: str
    analysis_step: str
    confidence: int = 100                   # 0–100
    expression: str = ""                    # relationship detail to the asset
    explanation: str = ""
    json_properties: dict[str, Any] = field(default_factory=dict)
    additional_properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class ResourceMeasureAnnotation(Annotation):
    """File/size/language counts for a project or sub-scope."""
    annotation_type: AnnotationType = field(default=AnnotationType.RESOURCE_MEASURE, init=False)
    resource_properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class ClassificationAnnotation(Annotation):
    """Category or label assignments (language, project type, doc presence, etc.)."""
    annotation_type: AnnotationType = field(default=AnnotationType.CLASSIFICATION, init=False)
    candidate_classifications: list[str] = field(default_factory=list)


@dataclass
class SchemaAnalysisAnnotation(Annotation):
    """Module/API structure — public functions, classes, endpoints."""
    annotation_type: AnnotationType = field(default=AnnotationType.SCHEMA_ANALYSIS, init=False)
    schema_name: str = ""
    schema_type: str = ""


@dataclass
class DataClassAnnotation(Annotation):
    """Dependency classification — package name, version, ecosystem."""
    annotation_type: AnnotationType = field(default=AnnotationType.DATA_CLASS, init=False)
    candidate_data_class_names: list[str] = field(default_factory=list)


@dataclass
class QualityScoreAnnotation(Annotation):
    """Health and quality scores derived from GitHub stats."""
    annotation_type: AnnotationType = field(default=AnnotationType.QUALITY_SCORE, init=False)
    quality_scores: dict[str, float] = field(default_factory=dict)


@dataclass
class RelationshipAnnotation(Annotation):
    """Discovered relationship between two components."""
    annotation_type: AnnotationType = field(default=AnnotationType.RELATIONSHIP, init=False)
    related_entity_name: str = ""
    relationship_type_name: str = ""


@dataclass
class RequestForActionAnnotation(Annotation):
    """Flag for human review — missing artifact, security gap, stale dep, etc."""
    annotation_type: AnnotationType = field(default=AnnotationType.REQUEST_FOR_ACTION, init=False)
    action_requested: str = ""
    action_target_name: str = ""


@dataclass
class SurveyResult:
    """
    Complete output of one survey run against a project.

    Consumed by:
      - SurveyOrchestrator (assembles it from sub-surveyor outputs)
      - EgeriaPublisher    (converts to pyegeria API calls)
      - CLI survey command (renders as markdown without Egeria)
    """
    project_slug: str
    project_display_name: str
    github_url: str
    surveyed_at: datetime = field(default_factory=datetime.utcnow)
    annotations: list[Annotation] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)   # non-fatal issues during survey

    def add(self, annotation: Annotation) -> None:
        self.annotations.append(annotation)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)

    def by_type(self, annotation_type: AnnotationType) -> list[Annotation]:
        return [a for a in self.annotations if a.annotation_type == annotation_type]
