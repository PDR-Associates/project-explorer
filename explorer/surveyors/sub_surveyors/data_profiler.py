"""Sub-surveyor: Data File Profiling → ResourceMeasure + SchemaAnalysis.

Two-tier operation:

  Tier 1 — inventory only (always runs):
    Detects data files (CSV, XLSX, Parquet, etc.) from project_file_inventory,
    reports counts and sizes per format, and identifies which directories
    contain data files.

  Tier 2 — stored profiles (runs when project_data_profiles rows exist):
    During add/refresh the ingestion pipeline profiles each readable data file
    while the repo is on disk and stores results in project_data_profiles.
    The surveyor reads these pre-computed profiles to emit one
    SchemaAnalysisAnnotation per file with column names, dtypes, row count,
    and null-rate summary — no local clone required at survey time.

    Tier 2 also activates when local_path= is passed to the constructor
    (e.g. via --data-path on the CLI) to force a fresh profile without a
    full re-ingest.
"""
from __future__ import annotations

import json as _json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

from explorer.registry import Project, ProjectRegistry
from explorer.surveyors.base_surveyor import BaseSurveyor
from explorer.surveyors.survey_report import (
    Annotation,
    RequestForActionAnnotation,
    ResourceMeasureAnnotation,
    SchemaAnalysisAnnotation,
)

log = logging.getLogger(__name__)

STEP = "DataProfiling"

# Extensions treated as data files for Tier 1 reporting
_DATA_EXTENSIONS: dict[str, str] = {
    # Tabular text
    "csv":     "CSV",
    "tsv":     "CSV",
    "tab":     "CSV",
    "psv":     "CSV",
    # Excel
    "xlsx":    "Excel",
    "xls":     "Excel",
    "xlsm":    "Excel",
    "xlsb":    "Excel",
    "ods":     "Excel",
    # Columnar / binary
    "parquet": "Parquet",
    "avro":    "Avro",
    "orc":     "ORC",
    "arrow":   "Arrow",
    "feather": "Arrow",
    # HDF5 / NumPy
    "h5":      "HDF5",
    "hdf5":    "HDF5",
    "hdf":     "HDF5",
    "npy":     "NumPy",
    "npz":     "NumPy",
    # JSON variants
    "jsonl":   "JSON Lines",
    "ndjson":  "JSON Lines",
    # SQLite
    "db":      "SQLite",
    "sqlite":  "SQLite",
    "sqlite3": "SQLite",
    "duckdb":  "DuckDB",
    # Pickle
    "pkl":     "Pickle",
    "pickle":  "Pickle",
}

# Extensions that pandas can profile
_PANDAS_READABLE: set[str] = {"csv", "tsv", "tab", "psv", "xlsx", "xls", "parquet", "feather", "arrow"}

# Skip profiling files larger than this (to avoid memory issues)
_MAX_PROFILE_SIZE_MB = 50


def _fmt_size(size_bytes: int) -> str:
    if size_bytes >= 1_048_576:
        return f"{size_bytes / 1_048_576:.1f} MB"
    if size_bytes >= 1_024:
        return f"{size_bytes / 1_024:.1f} KB"
    return f"{size_bytes} B"


class DataProfilerSurveyor(BaseSurveyor):
    """
    Detects and profiles data files in the project.

    Parameters
    ----------
    project    : Project from registry
    registry   : open ProjectRegistry
    local_path : path to a local clone of the repo root.  When set, forces
                 fresh pandas profiling for each readable data file regardless
                 of whether stored profiles already exist.
    """

    def __init__(
        self,
        project: Project,
        registry: ProjectRegistry,
        local_path: str | Path | None = None,
    ) -> None:
        super().__init__(project, registry)
        self.local_path: Path | None = Path(local_path) if local_path else None

    @property
    def step_name(self) -> str:
        return STEP

    # ── public entry point ────────────────────────────────────────────────────

    def run(self) -> list[Annotation]:
        results: list[Annotation] = []
        try:
            rows = self.registry.get_file_inventory_with_sizes(self.project.slug)
            if not rows:
                log.debug("DataProfilerSurveyor: no inventory for %s", self.project.slug)
                return results

            data_files = self._filter_data_files(rows)
            if not data_files:
                log.debug("DataProfilerSurveyor: no data files in %s", self.project.slug)
                return results

            # Tier 1: inventory-level format summary
            self._tier1_summary(data_files, results)

            # Tier 2a: fresh profiling from local clone (explicit --data-path)
            if self.local_path:
                self._tier2_local(data_files, results)
                return results

            # Tier 2b: read pre-computed profiles stored during ingestion
            stored = self.registry.get_data_profiles(self.project.slug)
            if stored:
                self._tier2_stored(stored, results)
            else:
                profilable = [
                    f for f in data_files
                    if Path(f["file_path"]).suffix.lstrip(".").lower() in _PANDAS_READABLE
                ]
                if profilable:
                    results.append(
                        RequestForActionAnnotation(
                            summary=(
                                f"{len(profilable)} data file(s) could be profiled — "
                                "re-ingest to capture column schemas"
                            ),
                            analysis_step=STEP,
                            confidence=70,
                            explanation=(
                                "No stored profiles found. Column-level schema profiling "
                                "runs automatically during add/refresh while the repo is "
                                "on disk. Run 'project-explorer refresh <slug>' to populate."
                            ),
                            action_requested="Run refresh to populate data profiles",
                            action_target_name=self.project.slug,
                        )
                    )

        except Exception as exc:
            log.exception("DataProfilerSurveyor failed for %s", self.project.slug)
            self._warn(results, str(exc))

        return results

    # ── tier 1 ───────────────────────────────────────────────────────────────

    def _filter_data_files(self, rows: list[dict]) -> list[dict]:
        out = []
        for r in rows:
            ext = Path(r["file_path"]).suffix.lstrip(".").lower()
            if ext in _DATA_EXTENSIONS:
                out.append({**r, "_format": _DATA_EXTENSIONS[ext], "_ext": ext})
        return out

    def _tier1_summary(self, data_files: list[dict], results: list[Annotation]) -> None:
        count_by_fmt: dict[str, int] = defaultdict(int)
        size_by_fmt: dict[str, int] = defaultdict(int)
        dirs_by_fmt: dict[str, set] = defaultdict(set)
        total_bytes = 0

        for f in data_files:
            fmt = f["_format"]
            count_by_fmt[fmt] += 1
            size_by_fmt[fmt] += f["file_size_bytes"]
            total_bytes += f["file_size_bytes"]
            parts = Path(f["file_path"]).parts
            dirs_by_fmt[fmt].add(parts[0] if len(parts) > 1 else "(root)")

        fmt_summary = {
            fmt: {
                "count": count_by_fmt[fmt],
                "total_size": _fmt_size(size_by_fmt[fmt]),
                "directories": sorted(dirs_by_fmt[fmt]),
            }
            for fmt in sorted(count_by_fmt, key=lambda k: -count_by_fmt[k])
        }

        results.append(
            ResourceMeasureAnnotation(
                summary=(
                    f"{len(data_files)} data file(s) across "
                    f"{len(count_by_fmt)} format(s), "
                    f"total {_fmt_size(total_bytes)}"
                ),
                analysis_step=STEP,
                confidence=95,
                resource_properties={
                    "total_data_files": len(data_files),
                    "total_data_size": _fmt_size(total_bytes),
                    "total_data_size_bytes": total_bytes,
                    "formats": fmt_summary,
                },
                json_properties={"source": "project_file_inventory"},
            )
        )

    # ── tier 2b: stored profiles ──────────────────────────────────────────────

    def _tier2_stored(self, stored: list[dict], results: list[Annotation]) -> None:
        profiled = [p for p in stored if p.get("row_count") is not None]
        unprofiled = [p for p in stored if p.get("row_count") is None]

        for p in profiled:
            cols: list[dict] = []
            if p.get("schema_json"):
                try:
                    cols = _json.loads(p["schema_json"])
                except Exception:
                    pass

            results.append(
                SchemaAnalysisAnnotation(
                    summary=(
                        f"{p['file_path']}: "
                        f"{p['row_count']:,} rows × {p['col_count']} columns"
                    ),
                    analysis_step=STEP,
                    confidence=95,
                    schema_name=p["file_path"],
                    schema_type=p["format"],
                    explanation=p.get("null_summary", ""),
                    json_properties={
                        "row_count": p["row_count"],
                        "col_count": p["col_count"],
                        "columns": cols,
                        "file_size": _fmt_size(p["file_size_bytes"]),
                        "profiled_at": p["profiled_at"],
                    },
                )
            )

        if unprofiled:
            log.debug(
                "DataProfilerSurveyor: %d data file(s) stored but not profiled "
                "(too large or unsupported format)", len(unprofiled),
            )

    # ── tier 2a: fresh local profiling ───────────────────────────────────────

    def _tier2_local(self, data_files: list[dict], results: list[Annotation]) -> None:
        try:
            import pandas as pd
        except ImportError:
            self._warn(
                results,
                "pandas not installed — install with: uv add pandas openpyxl pyarrow",
            )
            return

        limit_bytes = _MAX_PROFILE_SIZE_MB * 1_048_576
        for f in data_files:
            if f["_ext"] not in _PANDAS_READABLE:
                continue
            if f["file_size_bytes"] > limit_bytes:
                results.append(
                    RequestForActionAnnotation(
                        summary=f"Data file too large to profile: {f['file_path']} ({_fmt_size(f['file_size_bytes'])})",
                        analysis_step=STEP,
                        confidence=90,
                        explanation=f"Exceeds the {_MAX_PROFILE_SIZE_MB} MB profiling limit.",
                        action_requested="Profile large data file manually or reduce size",
                        action_target_name=f["file_path"],
                    )
                )
                continue

            local_file = self.local_path / f["file_path"]  # type: ignore[operator]
            if not local_file.exists():
                continue
            try:
                profile = self._profile_file(local_file, f["_ext"], pd)
                if profile:
                    results.append(
                        SchemaAnalysisAnnotation(
                            summary=f"{f['file_path']}: {profile['row_count']:,} rows × {profile['col_count']} columns",
                            analysis_step=STEP,
                            confidence=95,
                            schema_name=f["file_path"],
                            schema_type=f["_format"],
                            explanation=profile.get("null_summary", ""),
                            json_properties=profile,
                        )
                    )
            except Exception as exc:
                log.warning("DataProfilerSurveyor: could not profile %s: %s", f["file_path"], exc)

    # ── shared profiling logic ────────────────────────────────────────────────

    @staticmethod
    def _profile_file(path: Path, ext: str, pd: Any) -> dict | None:
        if ext in {"csv", "tsv", "tab", "psv"}:
            sep = "\t" if ext in {"tsv", "tab"} else (";" if ext == "psv" else ",")
            try:
                df = pd.read_csv(path, sep=sep, nrows=100_000, low_memory=False)
            except Exception:
                df = pd.read_csv(path, sep=sep, nrows=10_000, encoding="latin-1", low_memory=False)
        elif ext in {"xlsx", "xls"}:
            df = pd.read_excel(path, nrows=100_000)
        elif ext == "parquet":
            df = pd.read_parquet(path)
        elif ext in {"feather", "arrow"}:
            df = pd.read_feather(path)
        else:
            return None

        row_count = len(df)
        col_count = len(df.columns)
        columns: list[dict] = []
        high_null_cols: list[str] = []
        for col in df.columns:
            null_rate = df[col].isna().mean()
            columns.append({
                "name": str(col),
                "dtype": str(df[col].dtype),
                "null_pct": round(null_rate * 100, 1),
            })
            if null_rate > 0.5:
                high_null_cols.append(str(col))

        null_summary = ""
        if high_null_cols:
            null_summary = (
                f"{len(high_null_cols)} column(s) >50% null: "
                + ", ".join(high_null_cols[:5])
            )

        return {
            "row_count": row_count,
            "col_count": col_count,
            "columns": columns[:50],
            "null_summary": null_summary,
            "file_size": _fmt_size(path.stat().st_size),
        }
