# Project Explorer — User Guide

Project Explorer is a multi-agent RAG system that lets you ask natural-language questions about GitHub projects. It indexes a project's code, docs, API specs, and release history into a local vector database, then routes your questions to the right specialized agent.

---

## Quick Start

### 1. Prerequisites

| Service | Default URL | Install |
|---|---|---|
| Milvus (vector store) | `localhost:19530` | `docker run -p 19530:19530 milvusdb/milvus:latest standalone` |
| Ollama (local LLM) | `localhost:11434` | [ollama.ai/download](https://ollama.ai/download) |

```bash
# Pull the default model
ollama pull llama3.1:8b
```

### 2. Install

```bash
git clone https://github.com/your-org/project-explorer
cd project-explorer

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install all dependencies
uv sync --extra dev
```

### 3. Configure

```bash
cp .env.example .env
```

The only required setting is a GitHub token:

```bash
# .env
GITHUB_TOKEN=ghp_yourtoken
```

Get a token at github.com/settings/tokens — read-only public repos scope is sufficient.

### 4. Add a Project

```bash
project-explorer add https://github.com/lfai/ML_LLM_Ops
```

This launches an onboarding wizard that:
1. Analyzes the repo to detect which content types exist (code, docs, notebooks, etc.)
2. Proposes which collections to build
3. Asks you to confirm or customize
4. Runs full ingestion (fetches files, chunks, embeds, stores in Milvus)
5. Fetches GitHub stats (stars, forks, contributors, commits)

Typical ingestion time: 2–10 minutes depending on repo size.

---

## Asking Questions

### One-Shot

```bash
# Ask across all indexed projects
project-explorer ask "What is the architecture of this system?"

# Scope to a specific project
project-explorer ask --project ml-llm-ops "What MLflow tracking patterns are used?"
```

### Interactive Chat

```bash
# Multi-turn session
project-explorer chat

# Scoped to a project
project-explorer chat --project ml-llm-ops
```

In chat mode, conversation history is maintained across turns. Type `exit` or `quit` to leave.

### Web API

```bash
# Start the web server
uvicorn explorer.web.app:app --port 8000
```

```bash
# Query via HTTP
curl -X POST http://localhost:8000/api/query/ \
  -H "Content-Type: application/json" \
  -d '{"query": "How does the pipeline work?", "project_slug": "ml-llm-ops"}'
```

---

## How Queries Are Routed

The system classifies your query's intent and routes to the right agent:

| Intent | Example Queries | Agent |
|---|---|---|
| `statistical` | "How many stars does this have?" "Show contributor trends" | StatsAgent — GitHub data + charts |
| `health` | "Is this project actively maintained?" "What's the bus factor?" | HealthAgent — community health scoring |
| `code_search` | "How is authentication implemented?" "Where is the retry logic?" | CodeAgent — code collection search |
| `conceptual` | "How does the routing work?" "Explain the architecture" | DocAgent — docs + markdown |
| `comparison` | "Compare project A vs B" | CompareAgent — multi-project diff |
| `general` | Everything else | RAG — all collections |

---

## Managing Projects

```bash
# List all registered projects
project-explorer list

# Check environment health (Milvus, Ollama, GitHub, MLflow)
project-explorer status

# Refresh a project's index (incremental — only re-indexes changed files)
project-explorer refresh ml-llm-ops

# Remove a project (drops all Milvus collections)
project-explorer remove ml-llm-ops
```

### Incremental Refresh

`refresh` compares the latest commit SHA against the last-indexed SHA. Only files changed in the diff are re-indexed, one collection at a time. Typically completes in under a minute for small changes.

---

## Configuration Reference

All settings can be overridden with environment variables using `__` as the delimiter.

### LLM Backend

```bash
# Use Ollama (default)
LLM__BACKEND=ollama
LLM__OLLAMA__MODEL=llama3.1:8b
LLM__OLLAMA__BASE_URL=http://localhost:11434

# Use OpenAI
LLM__BACKEND=openai
OPENAI_API_KEY=sk-...
LLM__OPENAI__MODEL=gpt-4o-mini

# Use Anthropic
LLM__BACKEND=anthropic
ANTHROPIC_API_KEY=sk-ant-...
LLM__ANTHROPIC__MODEL=claude-haiku-4-5-20251001
```

### Milvus

```bash
# Local Milvus
MILVUS__URI=http://localhost:19530

# Milvus Cloud / Zilliz
MILVUS__URI=https://your-cluster.zillizcloud.com
MILVUS__TOKEN=your_api_key
```

### Cache

```bash
# In-memory LRU (default)
CACHE__BACKEND=memory
CACHE__MAX_SIZE=1000
CACHE__TTL_SECONDS=3600

# Redis (requires: uv sync --extra redis)
CACHE__BACKEND=redis
CACHE__REDIS_URL=redis://localhost:6379/0
```

### Observability (Optional)

```bash
# MLflow experiment tracking
OBSERVABILITY__MLFLOW__ENABLED=true
OBSERVABILITY__MLFLOW__TRACKING_URI=http://localhost:5025

# Arize Phoenix tracing (requires: uv sync --extra phoenix)
OBSERVABILITY__PHOENIX__ENABLED=true
OBSERVABILITY__PHOENIX__COLLECTOR_ENDPOINT=http://localhost:6006/v1/traces
```

---

## Terminal Dashboard

```bash
python -m explorer.dashboard.terminal_dashboard
```

Shows a live view of registered projects, recent query metrics, and collection sizes.

---

## Smoke Test Walkthrough

This walkthrough uses `lfai/ML_LLM_Ops` — a small, well-documented repo.

```bash
# 1. Verify services are running
project-explorer status

# 2. Index the project (accept defaults in the wizard)
project-explorer add https://github.com/lfai/ML_LLM_Ops

# 3. Verify ingestion
project-explorer list

# 4. Test each intent type
project-explorer ask --project ml-llm-ops "How many stars does this project have?"
project-explorer ask --project ml-llm-ops "Is this project actively maintained?"
project-explorer ask --project ml-llm-ops "How does the ML pipeline work?"
project-explorer ask --project ml-llm-ops "Where is the MLflow tracking configured?"

# 5. Interactive session
project-explorer chat --project ml-llm-ops
```

Expected behavior:
- Stars/health questions return data from the stats database
- Architecture/pipeline questions retrieve from markdown collections
- Code questions retrieve from Python code collections
- All responses cite source files and scores

---

## Troubleshooting

**"No collections found for this project"**
The ingestion may have found no files matching a collection's extensions. Run `refresh` or re-add with different collection selections.

**"I don't have enough information..."**
The retrieval score fell below 0.30 (the minimum). The query may be outside the indexed content, or the project needs a refresh.

**Milvus connection refused**
Start Milvus: `docker run -p 19530:19530 milvusdb/milvus:latest standalone`

**Ollama model not found**
Pull the model: `ollama pull llama3.1:8b`

**GitHub rate limit exceeded**
Add a GitHub token to `.env` and re-run. Authenticated requests have a 5000/hour limit vs 60/hour unauthenticated.

---

## Architecture Reference

See [CLAUDE.md](../CLAUDE.md) for the full architecture diagram, module map, and design decisions.
