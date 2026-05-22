# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

**Project Explorer** is a complete, production-quality reference implementation of a multi-agent RAG system built on open-source components. It ingests GitHub repositories and their documentation, then provides a natural-language interface for exploring and understanding those projects.

**Target users:** Technical users and product managers evaluating GitHub projects. No AI/ML expertise required.

**Reference implementation:** Inspired by [egeria-advisor](https://github.com/odpi/egeria-advisor) and the [lfai/ML_LLM_Ops](https://github.com/lfai/ML_LLM_Ops) reference stack.

## Tech Stack

| Component | Package | Notes |
|---|---|---|
| Agent framework | `beeai-framework[rag]` | `RequirementAgent` with `@tool`-decorated functions |
| Agent runtime | `agentstack-sdk` | A2A server, one `Server` instance per agent |
| Vector store | `pymilvus` | Multi-tenant via collection namespacing |
| Document parsing | `docling` | PDF, web, DOCX, Markdown |
| Embeddings | `sentence-transformers` | `all-MiniLM-L6-v2`, 384-dim, MPS on Apple Silicon |
| LLM default | `ollama` | Metal GPU on Apple Silicon; pluggable |
| LLM tracing | `openinference-instrumentation-beeai` | → Arize Phoenix at localhost:6006 |
| Experiment tracking | `mlflow` | Background thread, non-blocking |
| CLI | `typer` + `rich` | |
| Web UI | `fastapi` + `uvicorn` + Tailwind + Plotly.js | Single-page HTML frontend |
| TUI | `textual` | Full-screen terminal UI |

## Setup

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Install with dev + phoenix dependencies
uv sync --extra dev --extra phoenix

# Copy and configure environment
cp .env.example .env
# Edit .env: set GITHUB_TOKEN, MILVUS_URI, LLM_BACKEND, etc.
```

External services required:
- **Milvus** at `localhost:19530` (or Milvus Cloud URI in `.env`)
- **Ollama** at `localhost:11434` — run `ollama pull llama3.1:8b`
- **Arize Phoenix** (optional) — `python -m phoenix.server.main` → `localhost:6006`
- **MLflow** (optional) — `mlflow server --port 5025` → `localhost:5025`

## Commands

```bash
# Add a GitHub project (triggers onboarding wizard)
project-explorer add https://github.com/owner/repo

# Add a monorepo sub-project (index only one subdirectory)
project-explorer add https://github.com/owner/monorepo \
    --subpath subdir --name myproject

# Add a sub-project and include docs/examples that live outside the subpath
project-explorer add https://github.com/owner/monorepo \
    --subpath subdir --name myproject \
    --extra-docs-path docs/guide.md \
    --extra-docs-path examples/

# Use a local clone to avoid downloading the same large repo for each sub-project
# (GitHub URL is still stored for refresh and stats; --from-local only skips the initial download)
project-explorer add https://github.com/owner/monorepo \
    --subpath subdir --name myproject \
    --extra-docs-path docs/guide.md \
    --from-local /path/to/local/clone

# List registered projects (shows collections and vector counts)
project-explorer list
project-explorer list --details   # full per-collection breakdown

# Ask a question (one-shot)
project-explorer ask "How does authentication work in project X?"

# Ask within a specific project
project-explorer ask --project myproject "What are the main CLI commands?"

# Interactive multi-turn session
project-explorer chat

# Interactive session scoped to a project
project-explorer chat --project myproject

# Refresh a project's index (incremental) and update GitHub stats/commits
project-explorer refresh myproject
project-explorer refresh myproject --no-stats   # skip GitHub API calls
project-explorer refresh proj1 proj2 proj3      # multiple slugs
project-explorer refresh --all                  # every registered project
project-explorer refresh --all --top-level      # skip sub-projects

# Show environment health
project-explorer status

# Remove a project (drops all collections)
project-explorer remove myproject

# Launch full-screen TUI
project-explorer tui

# Survey a project (Egeria-aligned annotation report — no Egeria required by default)
project-explorer survey myproject
project-explorer survey myproject --publish          # also push SurveyReport + Annotations to Egeria
project-explorer survey myproject --refresh          # force-refresh FileTypeCache from Egeria first
project-explorer survey myproject --data-path /path/to/clone  # force re-profile data files from local clone
project-explorer survey proj1 proj2 proj3            # multiple slugs (condensed output per project)
project-explorer survey --all                        # every registered project
project-explorer survey --all --top-level            # skip sub-projects
project-explorer survey --all --publish              # survey + publish all

# Show Egeria survey history for a project (reads local registry — no Egeria needed)
project-explorer egeria-reports myproject
project-explorer egeria-reports myproject --full     # also fetch + display all annotations from Egeria

# Launch browser-based web UI (Plotly charts + markdown rendering)
project-explorer web
project-explorer web --host 0.0.0.0 --port 8080 --reload

# Start AgentStack A2A orchestrator (port 8080)
project-explorer serve

# Start all 6 specialist agents on consecutive ports (8080–8085)
project-explorer serve --all

# Terminal dashboard
python -m explorer.dashboard.terminal_dashboard
```

## Architecture

### Query Flow

```
User Query
  → CLI / Web UI
  → QueryCache                    ← cache hit → return immediately
  → QueryProcessor                ← classifies intent
      ├── statistical  → StatsAgent (GitHub API + SQLite time-series)
      ├── comparison   → CompareAgent (multi-project RAG + structured diff)
      ├── examples     → ExamplesAgent (generates runnable Python code)
      ├── code_search  → CodeAgent (code collections)
      ├── conceptual   → DocAgent (markdown + web docs)
      ├── health       → HealthAgent (community metrics)
      └── general      → RAG (CollectionRouter → Milvus → LLM)
  → LLM generation (Ollama or API backend)
  → Response formatting
  → Async: MLflow + Phoenix tracing, metrics write, cache store
```

### Agent Pattern (BeeAI RequirementAgent + @tool)

All agents follow the pattern validated in lfai/ML_LLM_Ops:
- `RequirementAgent` with `max_iterations=20`, `total_max_retries=10`
- Tools defined as `@tool`-decorated functions in `explorer/agents/tools.py`
- BeeAI uses the function docstring as description and the signature to generate a Pydantic schema
- Middleware captures request/response/error per tool call → Phoenix

Tools in `agents/tools.py`:
- `vector_search(query, collection_names)` — used by Code, Doc, Compare agents
- `query_project_stats(project_slug)` — used by Stats, Health, Compare agents
- `query_top_committers(project_slug, limit)` — used by Stats, Health agents
- `query_commit_activity(project_slug)` — used by Stats agent
- `query_code_symbols(project_slug, kind, pattern)` — used by CodeInventory, Examples agents
- `get_symbol_detail(project_slug, name)` — used by CodeInventory, Examples agents
- `build_example_context(project_slug, topic)` — used by ExamplesAgent; searches examples, python_code, api_reference, and markdown_docs collections in one call and returns formatted context for code generation; prepends IMPORT HINT and CONSTRUCTOR PATTERNS extracted from the retrieved chunks so small models don't invent argument names
- `query_dependencies(project_slug, dep_type)` — used by Dependency agent

`BaseExplorerAgent` also provides:
- `_infer_project_slug(query)` — infers project from query text against registry
- `_clarification_response(query)` — returns a natural-language question listing available projects

See `explorer/agents/base.py` for the shared base class.

### A2A Endpoints (AgentStack)

`agentstack_server.py` exposes six independently discoverable agents:

| Port offset | Agent | Notes |
|---|---|---|
| +0 | Orchestrator | Routes by intent; general RAG fallback |
| +1 | Statistics | `input_required` if no project inferred |
| +2 | Code Search | |
| +3 | Documentation | |
| +4 | Health | `input_required` if no project inferred |
| +5 | Compare | |

Stats and Health use async generators with `yield TaskStatus(state=TaskState.input_required)` to pause and ask for a project name, then resume when the user replies.

### Web UI

`web/static/index.html` is a single-page app served by FastAPI:
- Tailwind CSS (CDN), marked.js (CDN), Plotly.js (CDN)
- Left sidebar: project list with status badges; click to scope queries. Hover reveals three action buttons per project:
  - 🔄 "Refresh & profile" — calls `POST /api/projects/{slug}/refresh` (synchronous), spins, shows ✓/✗
  - 📊 "Run survey" — calls `POST /api/egeria/{slug}/survey`, switches to report tab on success
  - ↗ "Open on GitHub" — `<a>` linking `project.github_url`
  - Buttons use CSS `data-tip` tooltips (`.proj-action-btn[data-tip]::after`) — instant, no browser delay
- Chat area: markdown-rendered responses, 👍/👎 feedback on each message
- Charts: Stars, Commits, Languages, Health, **File Types**, **Egeria** — tab strip per selected project
  - Plotly charts fetched from `/api/stats/{slug}/charts/{type}`; File Types prefers surveyor data from `project_file_type_counts`
  - **Egeria tab**: reads `/api/egeria/{slug}/status` (local registry, no Egeria needed); shows registration badge, asset GUID, survey history table; "▶ View" per row lazily fetches annotations from Egeria and renders grouped by type with inline quality radar; "Publish survey →" button POSTs to `/api/egeria/{slug}/publish` (full survey + push) and shows success/error feedback
- Chart auto-selection (`_pick_chart` in `web/routes/query.py`): keywords in the query select the relevant chart; unrecognised statistical queries return no chart rather than a default commit graph
- Clarification flow: detects "Which project are you asking about?" prefix; sidebar click or typed name re-runs original query

### Egeria Surveyor

`project-explorer survey <slug>` runs the survey framework and prints a markdown report. `--publish` additionally pushes to Egeria. `project-explorer egeria-reports <slug>` displays local survey history (and with `--full`, fetches annotation detail from Egeria).

Survey flow:
```
SurveyOrchestrator.run(slug)
  → FileClassifierSurveyor   (FileTypeCache → ClassificationAnnotation per type group)
  → FileStructureSurveyor    (project_stats + code_symbols → ResourceMeasureAnnotation)
  → FileSizeSurveyor         (project_file_inventory → size-by-type, top-10 largest, RFA for >50 MB files)
  → DataProfilerSurveyor     (project_file_inventory + project_data_profiles → data file summary +
                               SchemaAnalysisAnnotation per profiled CSV/XLSX/Parquet)
  → LanguageSurveyor         (language_breakdown → ClassificationAnnotation)
  → HealthSurveyor           (stars/commits/releases → QualityScoreAnnotation)
  → DependencySurveyor       (project_dependencies → DataClassAnnotation per ecosystem)
  → DocumentationSurveyor    (collection presence + hygiene files → ClassificationAnnotation)
  → SecuritySurveyor         (SECURITY.md / CI / license → RequestForAction for gaps)
  → ApiStructureSurveyor     (project_code_symbols → SchemaAnalysisAnnotation per language)
  → SurveyResult (plain dataclasses, no pyegeria dependency)

  [--publish] → EgeriaPublisher
    → check registry cache for egeria_asset_guid (avoids repeat search on re-publish)
    → find_software_capabilities(search_string="SourceControlLibrary::{github_url}") or
      create_software_capability(typeName="SourceControlLibrary", libraryType="GitHub Repository")
    → create SurveyReport asset linked to asset via parentRelationshipTypeName="ReportSubject"
    → create one Annotation per finding using correct Egeria subtype class names:
        ResourceMeasureAnnotationProperties / ClassificationAnnotationProperties /
        QualityAnnotationProperties / DataClassAnnotationProperties /
        RequestForActionProperties (no "Annotation" suffix) /
        SchemaAnalysisAnnotationProperties / RelationshipAdviceAnnotationProperties
    → persist egeria_asset_guid to projects.egeria_asset_guid (SQLite)
    → persist report GUID + annotation_count to project_egeria_surveys (SQLite)
    → optional: prompt to trigger governance action process
```

`EgeriaReader` (`surveyors/egeria_reader.py`) — read-only pull path:
- `find_asset_guid(github_url)` — searches `SourceControlLibrary` by qualifiedName prefix
- `get_survey_reports_from_registry(slug)` — fast, no Egeria needed; reads `project_egeria_surveys`
- `get_survey_reports_from_egeria(slug)` — fallback that searches Egeria `find_assets` for `SurveyReport::GitHubRepo::{slug}::`
- `get_annotations(slug, surveyed_at)` — calls `discovery.find_annotations` by qualifiedName prefix; parses all subtype-specific fields
- `get_full_report(slug, surveyed_at, report_guid)` — combines report + annotations

`FileTypeCache` (`surveyors/file_classifier/type_cache.py`):
- Persists to `data/file_type_cache.json`; refreshed from Egeria `ValidMetadataValues` when credentials present
- Works fully offline — built-in defaults cover 100+ extensions across code, data, config, images, archives, and ML model formats
- Data file extensions covered: csv, tsv, xlsx/xls, parquet, avro, orc, arrow/feather, h5/hdf5, npy/npz, jsonl/ndjson, pkl, db/sqlite/duckdb, gz/zip/tar, pt/pth/onnx/safetensors and more
- Four-level lookup priority: Egeria by name → Egeria by extension → built-in by name → built-in by extension
- Module-level singleton shared across all surveyors in a process

`project_file_type_counts` SQLite table:
- Populated by `FileClassifierSurveyor` on every survey run (appended, not replaced)
- Columns: `project_slug`, `surveyed_at`, `type_label`, `file_count`, `source` (`"egeria"` or `"extension"`), `details_json` (extension breakdown for "Other" group)
- Read by `file_types_plotly()` for the web File Types chart; `query_file_type_history()` returns one row per run for trending
- Unrecognized file types are consolidated into a single "Other" entry; `details_json` records the breakdown by extension and is shown as a hover tooltip on the chart

`project_file_inventory` SQLite table:
- Populated by `IngestionPipeline._store_file_inventory()` during every `add` and `refresh`
- Stores every file path (relative to repo root) and its size in bytes
- Primary source for `FileClassifierSurveyor._collect_file_paths()` — gives the surveyor a complete view of all file types including YAML, TOML, shell scripts, and anything not vectorised into Milvus
- Also read by `FileSizeSurveyor` (size-by-type breakdown) and `DataProfilerSurveyor` (data file detection)
- Projects indexed before this table was added fall back to the three-source approach (project_code_symbols + Milvus + project_dependencies); run `refresh` to populate

`project_data_profiles` SQLite table:
- Populated by `IngestionPipeline._profile_data_files()` during every `add` and `refresh` while the repo is still on disk
- Columns: `project_slug`, `file_path`, `profiled_at`, `format` (CSV/Excel/Parquet/etc.), `row_count`, `col_count`, `schema_json` (JSON array of {name, dtype, null_pct}), `null_summary`, `file_size_bytes`; UNIQUE on `(project_slug, file_path)`
- Pandas profiling runs for files ≤50 MB in formats: csv, tsv, xlsx/xls, parquet, feather, arrow; silently skipped if pandas not installed
- Read by `DataProfilerSurveyor` to emit `SchemaAnalysisAnnotation` per file at survey time — no local clone needed
- Accessed via `registry.store_data_profiles()` / `registry.get_data_profiles()`
- Force re-profile without full re-ingest: `project-explorer survey <slug> --data-path /path/to/clone`

`projects.egeria_asset_guid` SQLite column:
- `TEXT DEFAULT NULL` column on the `projects` table; migrated automatically on first run
- Cached after the first `--publish` so subsequent runs skip the `find_software_capabilities` search call
- Read/written via `registry.get_egeria_asset_guid(slug)` / `registry.set_egeria_asset_guid(slug, guid)`

`project_egeria_surveys` SQLite table:
- `(project_slug, surveyed_at, egeria_report_guid, published_at, annotation_count)`; UNIQUE on `(project_slug, surveyed_at)`
- Written by `EgeriaPublisher._create_survey_report()` after each `--publish`; UPSERT so re-publishing the same survey updates the GUID
- Read by `EgeriaReader.get_survey_reports_from_registry()` and the web `/api/egeria/{slug}/status` route (no Egeria connection needed)
- Accessed via `registry.record_egeria_survey()`, `get_egeria_surveys()`, `get_latest_egeria_survey()`

Egeria connection (all optional, standard pyegeria env vars):
```
EGERIA_PLATFORM_URL, EGERIA_VIEW_SERVER, EGERIA_USER, EGERIA_USER_PASSWORD, PYEGERIA_TIMEOUT_SECONDS
```

Web API routes:
```
# web/routes/projects.py  (prefix /api/projects)
GET    /api/projects/                     → list[ProjectSummary]
GET    /api/projects/{slug}               → ProjectSummary
POST   /api/projects/{slug}/refresh       → RefreshResult {status, slug, message, error}
                                            synchronous (asyncio.to_thread); detects empty profiles
                                            and downloads repo even when no new commits exist
DELETE /api/projects/{slug}               → {removed: slug}

# web/routes/egeria.py  (prefix /api/egeria)
GET  /api/egeria/{slug}/status            → {asset_guid, is_registered, platform_url, surveys[]}
GET  /api/egeria/{slug}/annotations       → annotations[] from Egeria (requires live Egeria connection)
GET  /api/egeria/{slug}/survey-report     → SurveyReportData from SQLite including data_profiles[]
POST /api/egeria/{slug}/survey            → SurveyOnlyResult {status, annotation_count, surveyed_at, errors}
POST /api/egeria/{slug}/publish           → PublishResult {status, report_guid, annotation_count, surveyed_at}
POST /api/egeria/{slug}/catalog-elements  → CatalogResult; AssetMaker(view_server, platform_url, …) — note arg order
```

Web UI — Survey Report tab (`web/static/index.html`):
- Shown when a project is selected; click "📊 Survey Report" in the main nav bar
- Fetches from `GET /api/egeria/{slug}/survey-report` (SQLite only, no Egeria needed)
- Shows: health metric cards, Plotly donut chart for file types, dependency bar chart, **Data Files** section
  - Data Files: per-file cards with path, format, size, row×col count, null warnings, column type pills
  - When `data_profiles` is empty but `file_types` includes data-format labels (detected via substring match against "csv", "excel", "parquet", etc.): shows a "run refresh" hint with the exact command
- File type rows have checkboxes — select any subset then "Catalog selected →" to create DataSet assets in Egeria
- "💬 Ask about this" button pre-fills the chat input with a summary question and switches to Chat tab
- Resizable sidebar: drag the 4px handle between sidebar and main panel; width persists in localStorage

### TUI (Textual)

`tui/app.py` full-screen Textual app with clarification handling:
- `_pending_clarification` state set when agent returns clarification response
- Sidebar project selection auto-re-runs the pending query
- Typed input treated as a project slug when clarification is pending

### Collection Naming

Each project gets namespaced collections: `{project_slug}_{collection_type}`

| Collection Type | Content | Chunk Size |
|---|---|---|
| `python_code` | .py source | 512 tokens, overlap 64 |
| `javascript_code` | .js/.ts source | 512 tokens, overlap 64 |
| `java_code` | .java source | 512 tokens, overlap 64 |
| `go_code` | .go source | 512 tokens, overlap 64 |
| `markdown_docs` | READMEs, guides | 384 tokens, overlap 48 |
| `web_docs` | MkDocs, Sphinx, Docusaurus | 384 tokens, overlap 48 |
| `api_reference` | OpenAPI specs, docstrings | 256 tokens, overlap 32 |
| `examples` | Code samples, notebooks | 1024 tokens, overlap 128 |
| `pdfs` | PDFs via Docling | 512 tokens, overlap 64 |
| `release_notes` | Changelogs, release bodies | 256 tokens, overlap 32 |

Not every project gets every collection — `RepoAnalyzer` inspects the repo and proposes which to create.

### Key Design Rules (from egeria-advisor learnings)

1. Classify intent before touching the vector store — statistical queries never hit Milvus
2. Min retrieval score = 0.30 — below this, say "I don't have enough information"
3. Query cache is the highest-ROI latency win — implement before optimizing retrieval
4. Observability (MLflow, Phoenix) runs in background threads — never block the response
5. Incremental indexing is not optional for live repos — commit-diff based
6. Chunk size is content-specific — code ≠ prose ≠ examples
7. `refresh` always updates stats and commit history unless `--no-stats` — agents need SQLite data to answer contributor/trend queries
8. Use single-quoted YAML strings for regex patterns containing backslashes (`\w`, `\d`, etc.) — YAML double-quote mode treats `\` as escape and `\w` is invalid
9. A2A `Server` supports exactly one agent per instance — run one server per agent, gather with `asyncio.gather()`
10. GitHub's `get_git_tree(recursive=True)` is truncated for repos with >100k total nodes (files + directories combined) — when `tree.truncated` is set, the returned list is cut off mid-traversal and incomplete; fix by fetching the root **non-recursively** first (never truncated) then walking each top-level subtree individually
11. Fetching per-commit `additions/deletions` costs one REST call per new commit — pre-check the rate limit before the diff-stats loop and re-check every 50 calls; disable diff stat fetching when fewer than 100 calls remain rather than hitting the wall mid-loop
12. `--extra-docs-path` only has effect when `--subpath` is also set — without a subpath the full repo is already downloaded, so all paths are already covered; when both are set, the pipeline downloads the full repo and uses `code_root = full_root / subpath` for code collections while also walking the extra paths for doc/example collections
13. `--from-local` skips the GitHub zipball download for the initial `add`; the GitHub URL is still stored in the registry and used for stats, incremental refresh, and webhook events — `--from-local` has no effect on `refresh`, which always re-downloads from GitHub
14. BeeAI `FunctionTool` objects (produced by `@tool`) have no `.func` attribute — calling `my_tool.func(...)` raises `AttributeError`. To call tool logic outside the agent loop (e.g., in a `_fallback()` method), extract the implementation into a `_<name>_raw()` plain function and have the `@tool` wrapper delegate to it; the fallback imports and calls the raw function directly. See `_build_example_context_raw` and `_query_code_symbols_raw` in `agents/tools.py`.
15. `ExamplesAgent._fallback()` is the reliable path for example generation with small models — BeeAI + llama3.1:8b frequently completes without calling any tools and returns a plain-text method list. Gate on `"```python" in response` (not just `"```"`) so inline-backtick responses still fall through to the fallback. Index the project's functional/scenario tests as extra-docs-paths to give the fallback retrieval concrete, correct constructor signatures to work from.
16. `routing.yaml` `examples` patterns must appear before `code_search` patterns that would otherwise absorb "show me an example of X" and "how do I use X". Use single-quoted strings for all patterns containing `\w` or other regex metacharacters.
17. Milvus `VARCHAR` `max_length` is a **UTF-8 byte limit**, not a Python character limit — `text[:65535]` can still exceed 65535 bytes when the text contains multi-byte Unicode. Always truncate via `encoded[:max_bytes].decode('utf-8', errors='ignore')`. See `MultiCollectionStore._trunc()` in `multi_collection_store.py`.
18. `StatsAgent` file count prefers the survey total from `project_file_type_counts` (all file types) over `COUNT(DISTINCT file_path)` from `project_code_symbols` (code files only). If no survey has been run, the code-only count is used with a note. The system prompt must instruct the LLM to report the `Files:` field and not substitute symbol counts.
19. `_pick_chart()` in `web/routes/query.py` must return `None` for statistical queries that don't match a specific keyword — the previous catch-all `else` always returned the commit chart, misleading users asking about files, contributors, etc. Add keyword branches for each chart type and let the default be no chart.
20. Surveyor results (`project_file_type_counts`) are appended, not replaced — each survey run adds a new dated batch so file-type trends can be tracked over time. `query_file_type_counts()` returns only the latest run; `query_file_type_history()` returns one row per run for trending.
21. `FileClassifierSurveyor` uses `project_file_inventory` as its primary file path source (populated by the ingestion pipeline during `add`/`refresh`). For pre-inventory projects it falls back to project_code_symbols + Milvus + project_dependencies. Unrecognized types are consolidated into "Other" with extension breakdown in `details_json` — never create one label per unknown extension.
22. `Milvus client.query()` requires a non-empty filter expression — `filter=""` raises an exception that is silently caught, returning zero rows. Always use `filter="id >= 0"` or another valid expression when intending to scan all records.
23. `EgeriaPublisher` annotation class names: `RequestForActionProperties` (no "Annotation" suffix), `QualityAnnotationProperties` (not "QualityScore"), `ResourceMeasureAnnotationProperties.resourceProperties` is a typed dict (not a JSON string). These names come from the Egeria type archive, not from the survey_report.py AnnotationType enum values.
24. `EgeriaReader` annotation search uses the qualifiedName prefix `Annotation::{slug}::{surveyed_at}::` passed to `discovery.find_annotations(starts_with=True)`. The slug and surveyed_at come from the `project_egeria_surveys` registry table (written at publish time), not from Egeria.
25. `EgeriaPublisher` and `EgeriaReader` both accept an optional `registry` parameter. When provided, the publisher reads/writes `egeria_asset_guid` (cached per project) and writes `project_egeria_surveys` records. The reader reads `project_egeria_surveys` as its primary survey list source (no Egeria call needed). Always pass `registry=registry` from the CLI and web routes.
26. Web `/api/egeria/{slug}/publish` runs survey + publish synchronously (can take 15–60 s). It returns a structured `{status, report_guid, annotation_count, surveyed_at}` on success, or `{status: "error", error, stage}` where `stage` is `"survey"` or `"publish"` to tell the UI which step failed. The UI shows an inline feedback message and auto-refreshes the Egeria panel on success.
27. `find_software_capabilities` with `output_format="JSON"` returns a list of elements where the GUID is at `element["elementHeader"]["guid"]`. With `output_format="DICT"` the structure differs. Always use `"JSON"` when extracting GUIDs programmatically.
28. Data file types (csv, xlsx, parquet, avro, orc, arrow, h5, npy, pkl, jsonl, sqlite, duckdb, gz, zip, pt, onnx, safetensors, etc.) must be in `_BUILTIN_BY_EXTENSION` in `type_cache.py` to appear in the file type survey. Without a matching entry they fall into the "Other" bucket. Adding a new format means adding it in both `type_cache.py` and `_DATA_EXTENSIONS` in `data_profiler.py`.
29. `IngestionPipeline._profile_data_files()` runs after `_store_file_inventory()` while the repo is still in the temp directory. It profiles CSV/XLSX/Parquet files ≤50 MB using pandas and stores results in `project_data_profiles`. Because the temp dir is deleted after `run()` returns, this is the only point where local file content is accessible. Do not try to read data file content from the surveyor at survey time — read from `project_data_profiles` instead.
30. `DataProfilerSurveyor` has two tiers: Tier 1 reads `project_file_inventory` for counts/sizes (always runs); Tier 2 reads `project_data_profiles` for column-level schema annotations (runs when stored profiles exist). The `--data-path` CLI flag on `survey` forces fresh local profiling by passing `local_path=` to the surveyor constructor, bypassing the stored profiles. Use this when you want updated profiles without a full `refresh`.
31. Web UI "Survey Report" tab: fetches `GET /api/egeria/{slug}/survey-report` (SQLite only, no Egeria). Includes `data_profiles[]` from `project_data_profiles`. Shows a "Data Files" section with per-file cards (path, format, size, row×col, column type pills). When profiles are empty but data-format file types are detected, shows a hint with the `refresh` command. `hasDataFiles` detects data labels via substring match (e.g., "csv", "excel", "parquet") because type_cache labels are "CSV Data File", "Excel Spreadsheet", "Parquet Data File" — not bare format names. Catalog checkboxes POST to `/api/egeria/{slug}/catalog-elements`. Sidebar resizable via 4px `#resize-handle`; width in `localStorage` key `pe_sidebar_w`.
32. `survey` and `refresh` accept multiple slugs (`survey proj1 proj2`) or `--all` (all registered projects) with optional `--top-level` to skip sub-projects. Batch mode prints condensed per-project output and a Rich table summary at the end. `survey --all` creates one `SurveyOrchestrator` shared across all projects (single Egeria connection). The governance action prompt is suppressed in batch mode.
33. Sub-projects (`parent_slug` set) share the parent repo's GitHub URL. `EgeriaPublisher` uses `SourceControlLibrary::{github_url}` as the qualifiedName, so sub-projects from the same repo share one Egeria asset — the first publish creates it, subsequent ones reuse it via the cached `egeria_asset_guid`. Each sub-project still gets its own `SurveyReport` linked to that shared asset. File inventory and data profiles are scoped to `code_root = full_root / subproject_path`, so each sub-project only sees files within its subdirectory. `--top-level` on `survey --all` or `refresh --all` skips sub-projects; omit it to include them.
34. `IncrementalIndexer.refresh()` always calls `_store_file_inventory()` and `_profile_data_files()` when the repo is downloaded (any path that has file-based collections to re-index). When no new commits are found (`last_sha == latest_sha`) but `project_data_profiles` is empty, it calls `_run_profile_only()` which downloads the repo just for profiling. This means `project-explorer refresh <slug>` always populates profiles — never rely on telling users to re-add a project just to get profiles.
35. `AssetMaker` constructor argument order is `(view_server, platform_url, user_id, user_password)` — note that `view_server` comes first, unlike some pyegeria examples. Swapping to `(platform_url, view_server, …)` passes the view server name as the URL and produces `VALIDATION_ERROR_1 → Invalid URL`. Verify against `EgeriaPublisher._connect()` as the canonical reference.
36. `POST /api/projects/{slug}/refresh` is synchronous (uses `asyncio.to_thread`), not a background task. It captures stdout from `IncrementalIndexer.refresh()` and returns `RefreshResult {status, slug, message, error}`. The web sidebar 🔄 button spins until it completes and shows ✓/✗. Do not revert to `BackgroundTasks` — the UI needs the result to know when to reload the report tab.


## Module Map

```
explorer/
├── config.py              # Pydantic settings (ExplorerConfig)
├── registry.py            # Project Registry (SQLite: projects [+egeria_asset_guid], project_stats, project_commits, project_code_symbols, project_dependencies, project_file_type_counts, project_file_inventory, project_data_profiles, project_egeria_surveys); Project dataclass includes subproject_path, parent_slug, extra_docs_paths, egeria_asset_guid
├── rag_system.py          # Main orchestrator — entry point for all queries
├── query_processor.py     # Intent classifier + agent router
├── collection_router.py   # Selects relevant collections per query
├── query_cache.py         # LRU cache with optional Redis backend
├── llm_client.py          # LLMBackend protocol + Ollama/OpenAI/Anthropic impls
├── embeddings.py          # SentenceTransformer wrapper (MPS-aware)
├── multi_collection_store.py  # Milvus multi-tenant operations + feedback reranking
├── prompt_templates.py    # Per-agent prompt templates
├── agentstack_server.py   # AgentStack A2A server (6 agents on ports 8080-8085)
├── github/
├── ingestion/
├── agents/
│   ├── base.py            # BaseExplorerAgent (_infer_project_slug, _clarification_response)
│   ├── tools.py           # BeeAI @tool functions (vector_search, build_example_context, ...); _raw helpers for fallback use
│   ├── stats_agent.py     # GitHub stats + commit trends (uses stats tools)
│   ├── examples_agent.py  # Generates complete runnable Python examples (EXAMPLES intent)
│   └── conversation_agent.py  # Multi-turn BeeAI session wrapper
├── cli/
│   └── main.py            # Typer app (add, list, ask, chat, refresh, survey, egeria-reports, web, serve, tui, ...)
├── web/
│   ├── app.py             # FastAPI application
│   ├── static/index.html  # Single-page UI (Tailwind, Plotly.js, marked.js); Egeria tab with publish + annotation drill-down
│   └── routes/
│       ├── query.py       # POST /api/query/ (_pick_chart keyword routing)
│       ├── projects.py    # GET /api/projects/
│       ├── stats.py       # GET /api/stats/{slug}/charts/{type}
│       └── egeria.py      # GET /api/egeria/{slug}/status|annotations · POST /api/egeria/{slug}/publish
├── tui/
│   └── app.py             # Textual full-screen TUI (clarification-aware)
├── dashboard/
│   └── graphs.py          # Plotly figure builders incl. file_types_plotly (prefers surveyor data)
├── surveyors/             # Egeria-aligned survey framework (no pyegeria dependency except publisher/reader)
│   ├── survey_report.py   # SurveyResult + Annotation dataclasses (AnnotationType enum, 7 subtypes)
│   ├── base_surveyor.py   # Abstract BaseSurveyor; _warn() records errors as RequestForAction
│   ├── survey_orchestrator.py  # Runs all sub-surveyors → SurveyResult; accepts data_path= for Tier 2 profiling
│   ├── egeria_publisher.py     # SurveyResult → pyegeria API; real SourceControlLibrary creation; persists GUIDs to registry; correct annotation subtype class names
│   ├── egeria_reader.py        # Pull path: find_asset_guid, get_survey_reports_from_registry/egeria, get_annotations, get_full_report
│   ├── file_classifier/
│   │   ├── file_classificaiton.py   # FileClassification dataclass
│   │   ├── file_classifier.py       # FileClassifier (filesystem attrs + Egeria ValidMetadataValues lookup)
│   │   ├── file_classifier_surveyor.py  # Sub-surveyor: reads project_file_inventory (primary) or 3-source fallback; consolidates unknowns into "Other" with extension breakdown; persists to project_file_type_counts
│   │   └── type_cache.py            # FileTypeCache — 100+ built-in extensions (code, data, archives, ML models) + optional Egeria ValidMetadataValues; 4-level lookup priority; works fully offline
│   └── sub_surveyors/
│       ├── file_structure.py  # → ResourceMeasureAnnotation (file counts, sizes, LOC, dir breakdown from project_stats)
│       ├── file_size.py       # → ResourceMeasureAnnotation (precise size-by-type, top-10 largest from inventory) + RFA for files >50 MB
│       ├── data_profiler.py   # → ResourceMeasureAnnotation (data file counts/sizes by format) + SchemaAnalysisAnnotation per profiled CSV/XLSX/Parquet; reads project_data_profiles (stored at ingest time); local_path= enables fresh profiling
│       ├── language.py        # → ClassificationAnnotation (primary/secondary language, project type)
│       ├── dependency.py      # → DataClassAnnotation per ecosystem + ResourceMeasureAnnotation totals
│       ├── api_structure.py   # → SchemaAnalysisAnnotation per language (module tree, public symbols)
│       ├── health.py          # → QualityScoreAnnotation (activity, community, release cadence, freshness)
│       ├── documentation.py   # → ClassificationAnnotation (collection presence, hygiene files, quality label)
│       └── security.py        # → RequestForActionAnnotation (missing SECURITY.md, CI, license)
└── observability/
```

## Testing

```bash
# Run all tests
uv run pytest tests/ -v

# With coverage
uv run pytest --cov=explorer --cov-report=html

# End-to-end
uv run python scripts/test_end_to_end.py --quick
uv run python scripts/test_end_to_end.py --full

# Check vector counts per collection
uv run python scripts/count_vectors.py
```

## Code Style

```bash
uv run black explorer/
uv run ruff check explorer/
uv run mypy explorer/
```
