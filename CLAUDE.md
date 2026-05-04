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

# Show environment health
project-explorer status

# Remove a project (drops all collections)
project-explorer remove myproject

# Launch full-screen TUI
project-explorer tui

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
- `build_example_context(project_slug, topic)` — used by ExamplesAgent; searches examples, python_code, api_reference, and markdown_docs collections in one call and returns formatted context for code generation
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
- Left sidebar: project list with status badges; click to scope queries
- Chat area: markdown-rendered responses, 👍/👎 feedback on each message
- Charts: Stars, Commits, Languages, Health — fetched as Plotly JSON from `/api/stats/{slug}/charts/{type}`
- Clarification flow: detects "Which project are you asking about?" prefix; sidebar click or typed name re-runs original query

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


## Module Map

```
explorer/
├── config.py              # Pydantic settings (ExplorerConfig)
├── registry.py            # Project Registry (SQLite: projects, project_stats, project_commits); Project dataclass includes subproject_path, parent_slug, extra_docs_paths
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
│   └── main.py            # Typer app (add, list, ask, chat, refresh, web, serve, tui, ...)
├── web/
│   ├── app.py             # FastAPI application
│   ├── static/index.html  # Single-page UI (Tailwind, Plotly.js, marked.js)
│   └── routes/            # query.py (feedback endpoint), projects.py, stats.py
├── tui/
│   └── app.py             # Textual full-screen TUI (clarification-aware)
├── dashboard/
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
