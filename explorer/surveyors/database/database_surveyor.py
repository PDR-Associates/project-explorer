"""Database surveyor for custom database surveys."""
from __future__ import annotations

from datetime import datetime

from explorer.registry import DatabaseEntity, ProjectRegistry
from explorer.surveyors.survey_report import (
    AnnotationType,
    ResourceMeasureAnnotation,
    SchemaAnalysisAnnotation,
)

from .connection import database_connection


class DatabaseSurveyor:
    """Custom surveyor for databases when Egeria can't access them directly."""

    def __init__(
        self,
        db_entity: DatabaseEntity,
        credentials: dict,
        registry: ProjectRegistry,
    ) -> None:
        """Initialize database surveyor.
        
        Args:
            db_entity: DatabaseEntity with connection details
            credentials: Dict with 'user' and 'password' keys
            registry: ProjectRegistry for storing results
        """
        self.db_entity = db_entity
        self.credentials = credentials
        self.registry = registry

    def survey(self) -> dict:
        """Run comprehensive database survey.
        
        Returns:
            Dict with survey results including annotations and statistics
        """
        results = {
            "database_slug": self.db_entity.slug,
            "surveyed_at": datetime.utcnow().isoformat(),
            "annotations": [],
            "schema_info": {},
            "statistics": {},
            "errors": [],
        }

        try:
            with database_connection(self.db_entity, self.credentials) as conn:
                # Survey schema
                schema_info = self._survey_schema(conn)
                results["schema_info"] = schema_info
                results["annotations"].extend(
                    self._create_schema_annotations(schema_info)
                )

                # Survey statistics
                stats_info = self._survey_statistics(conn)
                results["statistics"] = stats_info
                results["annotations"].extend(
                    self._create_statistics_annotations(stats_info)
                )

        except Exception as e:
            error_msg = f"Survey failed: {str(e)}"
            results["errors"].append(error_msg)
            # Update database status to error
            from explorer.registry import ProjectStatus
            self.registry.update_database_status(
                self.db_entity.slug, ProjectStatus.ERROR, error_msg
            )

        # Store results in registry
        if not results["errors"]:
            self._store_results(results)

        return results

    def _survey_schema(self, conn) -> dict:
        """Survey database schema structure."""
        return conn.get_schema_info()

    def _survey_statistics(self, conn) -> dict:
        """Survey database statistics."""
        return conn.get_statistics()

    def _create_schema_annotations(self, schema_info: dict) -> list:
        """Create annotations from schema information."""
        annotations = []

        # Overall schema summary
        total_schemas = len(schema_info.get("schemas", []))
        total_tables = schema_info.get("total_tables", 0)
        total_columns = schema_info.get("total_columns", 0)

        annotations.append(
            SchemaAnalysisAnnotation(
                summary=f"Database contains {total_schemas} schemas, {total_tables} tables, {total_columns} columns",
                analysis_step="DatabaseSchemaSurvey",
                schema_name=self.db_entity.database_name,
                schema_type=self.db_entity.db_type,
                confidence=100,
            )
        )

        # Per-schema annotations
        for schema in schema_info.get("schemas", []):
            schema_name = schema["name"]
            table_count = len(schema["tables"])
            
            annotations.append(
                SchemaAnalysisAnnotation(
                    summary=f"Schema '{schema_name}' contains {table_count} tables",
                    analysis_step="DatabaseSchemaSurvey",
                    schema_name=schema_name,
                    schema_type="schema",
                    confidence=100,
                )
            )

            # Per-table annotations (using SchemaAnalysisAnnotation for tables too)
            for table in schema["tables"]:
                table_name = table["name"]
                column_count = len(table["columns"])
                
                annotations.append(
                    SchemaAnalysisAnnotation(
                        summary=f"Table '{schema_name}.{table_name}' has {column_count} columns",
                        analysis_step="DatabaseSchemaSurvey",
                        schema_name=f"{schema_name}.{table_name}",
                        schema_type="table",
                        confidence=100,
                    )
                )

        return annotations

    def _create_statistics_annotations(self, stats_info: dict) -> list:
        """Create annotations from statistics information."""
        annotations = []

        # Database size annotation
        db_size = stats_info.get("database_size", {})
        if db_size:
            size_bytes = db_size.get("size_bytes", 0)
            size_pretty = db_size.get("size_pretty", "unknown")
            annotations.append(
                ResourceMeasureAnnotation(
                    summary=f"Database size: {size_pretty}",
                    analysis_step="DatabaseStatistics",
                    confidence=100,
                    resource_properties={"size_bytes": size_bytes, "size_pretty": size_pretty},
                )
            )

        # Table statistics
        table_stats = stats_info.get("table_stats", [])
        if table_stats:
            largest_tables = table_stats[:5]  # Top 5 largest
            for table in largest_tables:
                schema = table.get("schemaname", "")
                table_name = table.get("tablename", "")
                size = table.get("total_size", "")
                size_bytes = table.get("total_bytes", 0)
                annotations.append(
                    ResourceMeasureAnnotation(
                        summary=f"Table {schema}.{table_name} size: {size}",
                        analysis_step="DatabaseStatistics",
                        confidence=100,
                        resource_properties={
                            "schema": schema,
                            "table": table_name,
                            "size_bytes": size_bytes,
                            "size_pretty": size,
                        },
                    )
                )

        return annotations

    def _store_results(self, results: dict) -> None:
        """Store survey results in the registry."""
        schema_info = results["schema_info"]
        statistics  = results.get("statistics", {})

        # Enrich each table with row count + activity timestamps from pg_stat_user_tables
        row_lookup: dict[tuple, dict] = {
            (rs["schemaname"], rs["tablename"]): rs
            for rs in statistics.get("row_stats", [])
        }
        for schema in schema_info.get("schemas", []):
            for table in schema["tables"]:
                rs = row_lookup.get((schema["name"], table["name"]), {})
                table["row_count"]      = rs.get("row_count", 0)
                table["last_analyzed"]  = rs.get("last_analyzed", "")
                table["last_vacuumed"]  = rs.get("last_vacuumed", "")
                table["pending_changes"] = rs.get("pending_changes", 0)

        # Also enrich size data from table_stats
        size_lookup: dict[tuple, dict] = {
            (ts["schemaname"], ts["tablename"]): ts
            for ts in statistics.get("table_stats", [])
        }
        for schema in schema_info.get("schemas", []):
            for table in schema["tables"]:
                ts = size_lookup.get((schema["name"], table["name"]), {})
                table["size_bytes"] = ts.get("total_bytes", 0) or 0
                table["size_pretty"] = ts.get("total_size", "")

        self.registry.record_database_survey(
            slug=self.db_entity.slug,
            schema_count=len(schema_info.get("schemas", [])),
            table_count=schema_info.get("total_tables", 0),
            column_count=schema_info.get("total_columns", 0),
            survey_data={
                "schema_info": schema_info,
                "statistics": statistics,
                "annotation_count": len(results["annotations"]),
            },
        )


def run_database_survey(
    db_slug: str,
    credentials: dict,
    registry: ProjectRegistry | None = None,
) -> dict:
    """Convenience function to run a database survey.
    
    Args:
        db_slug: Database slug to survey
        credentials: Dict with 'user' and 'password' keys
        registry: Optional ProjectRegistry (creates new one if not provided)
        
    Returns:
        Survey results dict
        
    Example:
        results = run_database_survey(
            "my-postgres",
            {"user": "admin", "password": "secret"}
        )
    """
    if registry is None:
        registry = ProjectRegistry()

    db_entity = registry.get_database(db_slug)
    if not db_entity:
        raise ValueError(f"Database '{db_slug}' not found in registry")

    surveyor = DatabaseSurveyor(db_entity, credentials, registry)
    return surveyor.survey()

# Made with Bob
