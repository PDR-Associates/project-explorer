# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

**Project Explorer** is a complete, production-quality reference implementation of a multi-agent RAG system built on open-source components. It ingests GitHub repositories and their documentation, then provides a natural-language interface for exploring and understanding those projects.

**Target users:** Technical users and product managers evaluating GitHub projects. No AI/ML expertise required.

**Reference implementation:** Inspired by [egeria-advisor](https://github.com/odpi/egeria-advisor) and the [lfai/ML_LLM_Ops](https://github.com/lfai/ML_LLM_Ops) reference stack.

## Tech Stack

| Component | Package | Notes |
|---|---|---|
| Agent framework | `beeai-framework[rag]` | `RequirementAgent` pattern |
| Agent runtime | `agentstack-sdk` | Runtime + UI platform |
| Vector store | `langchain-milvus` + `pymilvus` | Multi-tenant via collection namespacing |
| Document parsing | `docling` | PDF, web, DOCX, Markdown |
| Embeddings | `sentence-transformers` | `all-MiniLM-L6-v2`, 384-dim, MPS on Apple Silicon |
| LLM default | `ollama` | Metal GPU on Apple Silicon; pluggable |
| LLM tracing | `openinference-instrumentation-beeai` | → Arize Phoenix at localhost:6006 |
| Experiment tracking | `mlflow` | Background thread, non-blocking |
| CLI | `typer` + `rich` | |
| Web UI | `fastapi` + `uvicorn` | |

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

# List registered projects
project-explorer list

# Ask a question (one-shot)
project-explorer ask "How does authentication work in project X?"

# Ask within a specific project
project-explorer ask --project myproject "What are the main CLI commands?"

# Interactive multi-turn session
project-explorer chat

# Interactive session scoped to a project
project-explorer chat --project myproject

# Refresh a project's index (incremental)
project-explorer refresh myproject

# Show environment health
project-explorer status

# Remove a project (drops all collections)
project-explorer remove myproject

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
      ├── code_search  → CodeAgent (code collections)
      ├── conceptual   → DocAgent (markdown + web docs)
      ├── health       → HealthAgent (community metrics)
      └── general      → RAG (CollectionRouter → Milvus → LLM)
  → LLM generation (Ollama or API backend)
  → Response formatting
  → Async: MLflow + Phoenix tracing, metrics write, cache store
```

### Agent Pattern (BeeAI RequirementAgent)

All agents follow the pattern validated in lfai/ML_LLM_Ops:
- `RequirementAgent` with `max_iterations=20`, `total_max_retries=10`
- Tools (e.g. `VectorStoreSearchTool`) registered at init
- Streaming via `context.yield_async()`
- Middleware captures request/response/error per tool call → Phoenix

See `explorer/agents/base.py` for the shared base class.

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

## Module Map

```
explorer/
├── config.py              # Pydantic settings (ExplorerConfig)
├── registry.py            # Project Registry (SQLite CRUD)
├── rag_system.py          # Main orchestrator — entry point for all queries
├── query_processor.py     # Intent classifier + agent router
├── collection_router.py   # Selects relevant collections per query
├── query_cache.py         # LRU cache with optional Redis backend
├── llm_client.py          # LLMBackend protocol + Ollama/OpenAI/Anthropic impls
├── embeddings.py          # SentenceTransformer wrapper (MPS-aware)
├── multi_collection_store.py  # Milvus multi-tenant operations
├── prompt_templates.py    # Per-agent prompt templates
├── github/
│   ├── client.py          # PyGitHub + GraphQL wrapper (rate-limit aware)
│   ├── analyzer.py        # RepoAnalyzer — detects content types, proposes collections
│   └── stats_fetcher.py   # Fetches GitHub stats → SQLite time-series
├── ingestion/
│   ├── pipeline.py        # Orchestrates full ingestion for a project
│   ├── incremental.py     # Commit-diff / hash-based update detection
│   ├── code_parser.py     # Python/JS/Java/Go/Rust parsers
│   ├── doc_parser.py      # Markdown + Docling (PDF, web, DOCX)
│   ├── notebook_parser.py # Jupyter .ipynb via nbconvert
│   ├── api_parser.py      # OpenAPI/Swagger structured parsing
│   └── data_prep.py       # Quality filtering, dedup, PII detection
├── agents/
│   ├── base.py            # BaseExplorerAgent (RequirementAgent wrapper)
│   ├── code_agent.py      # Code search, method lookup
│   ├── doc_agent.py       # Conceptual Q&A from docs
│   ├── stats_agent.py     # GitHub stats + Plotext/Plotly charts
│   ├── compare_agent.py   # Multi-project side-by-side comparison
│   ├── health_agent.py    # Community health scoring
│   └── conversation_agent.py  # Multi-turn BeeAI session wrapper
├── cli/
│   ├── main.py            # Typer app — all CLI entry points
│   ├── interactive.py     # REPL loop
│   ├��─ wizard.py          # Onboarding wizard (add project flow)
│   └── formatters.py      # Rich output helpers
├── web/
│   ├── app.py             # FastAPI application
│   └── routes/            # query.py, projects.py, stats.py
├── dashboard/
│   ├── terminal_dashboard.py  # Rich Live dashboard
│   └── graphs.py          # Plotext (terminal) + Plotly (web) graph builders
└── observability/
    ├── metrics_collector.py   # SQLite query metrics
    ├── phoenix_client.py      # Arize Phoenix / OpenTelemetry wrapper
    ├── mlflow_tracking.py     # Non-blocking MLflow experiment logging
    └── feedback_collector.py  # User feedback (thumbs up/down)
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
