# Project Explorer — Surveyor Reference Guide

The survey framework produces an **Egeria-aligned annotation report** for any indexed project.
Surveys run entirely from SQLite (no Egeria connection needed); publishing to Egeria is optional.

---

## Quick Start

```bash
# Survey a single project
project-explorer survey <slug>

# Survey multiple projects
project-explorer survey proj1 proj2 proj3

# Survey all registered projects
project-explorer survey --all

# Survey all top-level projects only (skip sub-projects of monorepos)
project-explorer survey --all --top-level

# Survey and publish all to Egeria
project-explorer survey --all --publish

# Survey with column-level profiling of CSV/Parquet files
project-explorer survey <slug> --data-path /path/to/local/clone

# Refresh (re-index) multiple or all projects
project-explorer refresh proj1 proj2
project-explorer refresh --all
project-explorer refresh --all --top-level --no-stats

# View published survey history (no Egeria needed)
project-explorer egeria-reports <slug>

# View published surveys + full annotation detail from Egeria
project-explorer egeria-reports <slug> --full
```

The **web UI** shows the survey report under the **"📊 Survey Report"** tab when a project is
selected. File type counts and health metrics are displayed as Plotly charts. Individual file
types can be selected and cataloged as Egeria `DataSet` assets from the same tab.

---

## Survey Pipeline

`SurveyOrchestrator.run(slug)` executes surveyors in this order:

| # | Surveyor | Primary Source | Output |
|---|---|---|---|
| 1 | `FileClassifierSurveyor` | `project_file_inventory` | `ClassificationAnnotation` per file type group |
| 2 | `FileStructureSurveyor` | `project_stats`, `project_code_symbols` | `ResourceMeasureAnnotation` — counts, LOC, dir tree |
| 3 | `FileSizeSurveyor` | `project_file_inventory` | `ResourceMeasureAnnotation` — size-by-type, top-10 largest; `RequestForAction` for files >50 MB |
| 4 | `DataProfilerSurveyor` | `project_file_inventory`, `project_data_profiles` | `ResourceMeasureAnnotation` — data format summary; `SchemaAnalysisAnnotation` per profiled file |
| 5 | `LanguageSurveyor` | `project_stats`, `project_code_symbols` | `ClassificationAnnotation` — primary/secondary language, project type |
| 6 | `HealthSurveyor` | `project_stats`, `project_commits` | `QualityScoreAnnotation` — activity, community, release cadence, freshness |
| 7 | `DependencySurveyor` | `project_dependencies` | `DataClassAnnotation` per ecosystem + totals |
| 8 | `DocumentationSurveyor` | Milvus collections, `project_file_inventory` | `ClassificationAnnotation` — doc presence, hygiene files |
| 9 | `SecuritySurveyor` | `project_file_inventory` | `RequestForAction` for missing SECURITY.md, CI config, license |
| 10 | `ApiStructureSurveyor` | `project_code_symbols` | `SchemaAnalysisAnnotation` per language — module tree, public symbols |

Each surveyor is independent and failures are non-fatal — a failed surveyor adds an error to
`SurveyResult.errors` and the rest of the pipeline continues.

---

## Data Profiling

Data file profiling runs automatically during `add` and `refresh` while the repo is on disk.
Results are stored in `project_data_profiles` and read at survey time — no local clone is needed
when you run `survey`.

### How It Works

**At ingestion time** (`add` / `refresh`):
```
IngestionPipeline._profile_data_files()
  → walks code_root for _DATA_EXTENSIONS
  → for each readable file ≤50 MB: pandas.read_csv/read_excel/read_parquet
  → stores row_count, col_count, column schemas, null rates → project_data_profiles
```

**At survey time** (`survey`):
```
DataProfilerSurveyor.run()
  → Tier 1: reads project_file_inventory → counts/sizes per format
  → Tier 2: reads project_data_profiles → SchemaAnalysisAnnotation per file
```

### Supported Formats

| Format | Extensions | Profiled (row/col schema) |
|---|---|---|
| CSV | csv, tsv, tab, psv | Yes |
| Excel | xlsx, xls | Yes |
| Parquet | parquet | Yes |
| Arrow / Feather | arrow, feather | Yes |
| Avro | avro | No (size/count only) |
| ORC | orc | No (size/count only) |
| HDF5 | h5, hdf5, hdf | No (size/count only) |
| NumPy | npy, npz | No (size/count only) |
| JSON Lines | jsonl, ndjson | No (size/count only) |
| SQLite | db, sqlite, sqlite3 | No (size/count only) |
| DuckDB | duckdb | No (size/count only) |
| Pickle | pkl, pickle | No (size/count only) |

Files larger than **50 MB** are not profiled (memory safety). A `RequestForAction` annotation
is emitted for these files suggesting external storage or Git LFS.

### Pandas Dependency

Profiling requires `pandas` plus optional readers:

```bash
uv add pandas openpyxl pyarrow   # Excel + Parquet support
```

If `pandas` is not installed, Tier 1 (inventory counts) still runs; Tier 2 is silently skipped
during ingestion and a note is printed.

### Re-profiling Without Full Re-ingest

```bash
# Force fresh profiling from a local clone (skips full download)
project-explorer survey <slug> --data-path /path/to/local/clone
```

This runs `DataProfilerSurveyor` Tier 2 directly from the local path and updates
`project_data_profiles`.

---

## File Type Classification

All file type lookups go through `FileTypeCache` with this priority:

1. Egeria `ValidMetadataValues` by filename (highest priority)
2. Egeria `ValidMetadataValues` by extension
3. Built-in defaults by filename (e.g., `Dockerfile`, `pyproject.toml`)
4. Built-in defaults by extension (lowest priority)

Unrecognized files land in the **"Other"** bucket. The extension breakdown is stored in
`details_json` and shown as a hover tooltip in the web UI File Types chart.

### Built-in Extension Coverage

**Source code:** py, pyi, ipynb, js, mjs, ts, tsx, jsx, html, css, java, kt, scala, go, rs, c, cpp, h, hpp

**Data — tabular:** csv, tsv, tab, psv, xlsx, xls, xlsm, xlsb, ods

**Data — columnar/binary:** parquet, avro, orc, arrow, feather, h5, hdf5, npy, npz

**Data — serialization:** pkl, pickle, joblib, msgpack, jsonl, ndjson, geojson

**Data — databases:** db, sqlite, sqlite3, duckdb

**Config:** toml, yaml, yml, json, xml, cfg, ini, env, lock

**Docs:** md, mdx, rst, txt, pdf

**Scripts:** sh, bash, zsh, bat, ps1

**SQL:** sql

**Archives:** gz, bz2, xz, zst, tar, tgz, zip, rar, 7z

**ML models:** pt, pth, ckpt, safetensors, onnx, pb, mlmodel, bin

**Images:** png, jpg, jpeg, gif, svg, ico

**Well-known filenames:** Dockerfile, Makefile, LICENSE, .gitignore, requirements.txt, pyproject.toml, package.json, go.mod, Cargo.toml, pom.xml, and more

To add a new type, append to `_BUILTIN_BY_EXTENSION` in
`explorer/surveyors/file_classifier/type_cache.py` **and** to `_DATA_EXTENSIONS` in
`explorer/surveyors/sub_surveyors/data_profiler.py` if it is a data format.

---

## File Size Analysis

`FileSizeSurveyor` reads `project_file_inventory.file_size_bytes` to produce:

- **Total repo disk footprint** with size-by-type breakdown and average file size
- **Top-10 largest files** — useful for spotting accidentally committed binaries
- **`RequestForAction`** for any file exceeding 50 MB recommending Git LFS or external storage

The threshold constants are in `file_size.py`:

```python
_LARGE_FILE_THRESHOLD_MB = 50
_VERY_LARGE_FILE_THRESHOLD_MB = 200
```

---

## Annotation Types

All annotations inherit from `Annotation` and are defined in `survey_report.py`:

| Type | Egeria class | Produced by |
|---|---|---|
| `ClassificationAnnotation` | `ClassificationAnnotationProperties` | FileClassifier, Language, Documentation |
| `ResourceMeasureAnnotation` | `ResourceMeasureAnnotationProperties` | FileStructure, FileSize, DataProfiler, Dependency |
| `QualityScoreAnnotation` | `QualityAnnotationProperties` | Health |
| `DataClassAnnotation` | `DataClassAnnotationProperties` | Dependency |
| `SchemaAnalysisAnnotation` | `SchemaAnalysisAnnotationProperties` | ApiStructure, DataProfiler |
| `RelationshipAnnotation` | `RelationshipAdviceAnnotationProperties` | (reserved) |
| `RequestForActionAnnotation` | `RequestForActionProperties` | Security, FileSize, DataProfiler, any via `_warn()` |

`RequestForActionProperties` has **no "Annotation" suffix** — this is the correct Egeria class name.

---

## Publishing to Egeria

`project-explorer survey <slug> --publish` runs the full survey then:

1. Finds or creates a `SourceControlLibrary` asset for the GitHub URL
2. Creates a `SurveyReport` asset linked via `ReportSubject`
3. Creates one `Annotation` per finding using the correct Egeria subtype class names
4. Persists the asset GUID and report GUID to SQLite for future runs

The asset GUID is cached in `projects.egeria_asset_guid` so repeated publishes skip the
`find_software_capabilities` search call.

### Egeria Environment Variables

```bash
EGERIA_PLATFORM_URL        # default: https://localhost:9443
EGERIA_VIEW_SERVER         # default: qs-view-server
EGERIA_USER                # default: erinoverview
EGERIA_USER_PASSWORD       # default: secret
PYEGERIA_TIMEOUT_SECONDS   # default: 30
```

---

## Cataloging File Types as DataSet Assets

From the web UI **Survey Report** tab, check any file type rows and click **"Catalog selected →"**.
This calls `POST /api/egeria/{slug}/catalog-elements` which:

1. Verifies the project is registered in Egeria (GUID must be cached)
2. Creates a `DataSet` asset per selected type with `qualifiedName = DataSet::{slug}::{label}`
3. Links each `DataSet` to the `SourceControlLibrary` via a `CapabilityAssetUse` (useType: GOVERNS) relationship

---

## SQLite Tables Reference

| Table | Populated by | Read by |
|---|---|---|
| `project_file_inventory` | `IngestionPipeline._store_file_inventory()` | FileClassifier, FileSize, DataProfiler surveyors |
| `project_data_profiles` | `IngestionPipeline._profile_data_files()` | DataProfilerSurveyor (Tier 2) |
| `project_file_type_counts` | `FileClassifierSurveyor` (per survey run) | Web File Types chart, StatsAgent |
| `project_egeria_surveys` | `EgeriaPublisher._create_survey_report()` | EgeriaReader, web Egeria tab |
| `projects.egeria_asset_guid` | `EgeriaPublisher._find_or_create_asset()` | EgeriaPublisher (cache), web status |

---

## Adding a New Sub-Surveyor

1. Create `explorer/surveyors/sub_surveyors/my_surveyor.py` extending `BaseSurveyor`
2. Implement `step_name` property and `run() → list[Annotation]`
3. Use `self._warn(results, msg)` for non-fatal issues (creates a `RequestForAction`)
4. Export from `sub_surveyors/__init__.py`
5. Add to the `surveyors` list in `SurveyOrchestrator.run()`

```python
class MySurveyor(BaseSurveyor):
    @property
    def step_name(self) -> str:
        return "MyStep"

    def run(self) -> list[Annotation]:
        results = []
        try:
            # read from self.registry, self.project
            results.append(ResourceMeasureAnnotation(...))
        except Exception as exc:
            self._warn(results, str(exc))
        return results
```

---

## Batch Operations

Both `survey` and `refresh` accept multiple slugs or `--all`:

```
project-explorer survey proj1 proj2 proj3        # named list
project-explorer survey --all                    # every registered project
project-explorer survey --all --top-level        # top-level projects only
project-explorer refresh --all --no-stats        # re-index all, skip GitHub API
```

**Batch survey output** is condensed — one line per project showing annotation count and any
errors, followed by a summary table. Full per-annotation listings only appear for single-project
runs. `--publish` in batch mode creates a shared `EgeriaPublisher` instance (one Egeria
connection) and publishes each project's result in sequence; the governance action prompt is
suppressed.

---

## Sub-Projects and Monorepos

When a monorepo contains multiple independently indexable sub-projects, each is registered
with `--subpath`:

```bash
project-explorer add https://github.com/owner/monorepo --subpath services/api --name api-service
project-explorer add https://github.com/owner/monorepo --subpath services/worker --name worker-service
```

### File inventory and data profiling scope

Each sub-project's inventory and data profiles are scoped to its `code_root`:

```
full_root = /tmp/extracted/monorepo
code_root = full_root / subproject_path   # e.g. full_root/services/api
```

File paths stored in `project_file_inventory` and `project_data_profiles` are **relative to
`code_root`**, not to the repo root. This means:

- `api-service` inventory: `main.py`, `routes/auth.py`, `data/users.csv` (within `services/api/`)
- `worker-service` inventory: `worker.py`, `data/jobs.csv` (within `services/worker/`)
- CSV files in one sub-project's directory do not appear in another's inventory or profiles

To see the full monorepo including files that belong to sub-projects, add the repo without
`--subpath`:
```bash
project-explorer add https://github.com/owner/monorepo --name monorepo-full
```

### Egeria asset sharing

All sub-projects from the same GitHub URL share one **`SourceControlLibrary`** asset in Egeria
(identified by `SourceControlLibrary::{github_url}`). The first sub-project to publish creates
the asset; subsequent ones find and reuse it via the cached `egeria_asset_guid`.

Each sub-project still gets its **own `SurveyReport`** linked to the shared asset via
`ReportSubject`, so survey histories are independent per sub-project.

| Egeria object | Scope |
|---|---|
| `SourceControlLibrary` asset | Shared across all sub-projects of the same URL |
| `SurveyReport` | One per sub-project per survey run |
| `Annotation` objects | Scoped to the sub-project's survey results |
| `egeria_asset_guid` (SQLite) | Stored per sub-project slug; set from first publish |

### Surveying sub-projects

```bash
# Survey only one sub-project
project-explorer survey api-service

# Survey all sub-projects of a monorepo (they share the SourceControlLibrary asset)
project-explorer survey api-service worker-service

# Survey everything EXCEPT sub-projects
project-explorer survey --all --top-level

# Survey absolutely everything (parent + sub-projects)
project-explorer survey --all
```

`--top-level` filters to projects where `parent_slug` is empty — i.e., projects added without
`--subpath`, or monorepo roots registered as standalone projects.
