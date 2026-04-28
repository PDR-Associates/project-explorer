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
5. Fetches GitHub stats (stars, forks, contributors, commits, 90-day commit history)

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

### Interactive Chat (CLI)

```bash
# Multi-turn session
project-explorer chat

# Scoped to a project
project-explorer chat --project ml-llm-ops
```

Conversation history is maintained across turns via BeeAI `TokenMemory` — each turn is aware of prior questions and answers. Type `exit` or `quit` to leave.

### TUI (Full-Screen Terminal)

```bash
project-explorer tui
```

Two-panel layout with a project sidebar and chat area. Responses stream token-by-token into a live bubble, and the conversation agent maintains memory across turns within a session.

| Key | Action |
|---|---|
| `Tab` | Switch focus between sidebar and input |
| `Enter` | Submit query |
| `f` | Open feedback prompt for last response (`y`=👍 / `n`=👎) |
| `r` | Refresh the selected project's index |
| `Ctrl+C` | Quit |

**Clarification in TUI** — if an agent cannot determine which project you mean, it asks in the chat area and the status bar prompts you to select from the sidebar or type a name. Either action re-runs your original question automatically.

### Web UI (Browser)

```bash
project-explorer web
# Opens http://127.0.0.1:8000
```

Options:
```bash
project-explorer web --host 0.0.0.0 --port 8080
project-explorer web --reload   # auto-reload on code changes (dev)
```

The web UI provides:
- **Project sidebar** — click any project to scope all queries to it; status badges show active/indexing/error
- **Streaming responses** — assistant text appears token-by-token via server-sent events (SSE); no waiting for the full answer
- **Conversation memory** — a UUID session ID is stored in `localStorage` and sent with every request; the server maintains a persistent `ConversationAgent` per session (30-minute idle timeout), giving the web UI the same cross-turn memory as the TUI and CLI
- **Inline charts** — when a statistical or health query warrants a chart, a Plotly figure appears directly in the chat response alongside the text
- **Sidebar charts** — Plotly interactive charts (Stars, Commits, Weekly Commits, Languages, Health) per selected project; click chart tabs to switch
- **👍/👎 feedback buttons** — on each assistant message; keyboard `f` key also opens feedback
- **Clarification flow** — when the agent needs a project name, a prompt appears; click a project in the sidebar or type its name to re-run your original question

### Web API

```bash
# Start the server
uvicorn explorer.web.app:app --port 8000
# or
project-explorer web
```

**Streaming endpoint (recommended):**
```bash
curl -X POST http://localhost:8000/api/query/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "Who are the top committers?", "project_slug": "ml-llm-ops", "session_id": "my-uuid-here"}'
```

Yields newline-delimited SSE events:
```
data: {"t": "chunk", "v": "The top committers..."}
data: {"t": "done", "intent": "statistical", "hash": "abc123", "chart": {...}}
```

The `session_id` field is optional but recommended — when provided, the server maintains a persistent conversation agent keyed to that ID, enabling cross-turn memory. The browser generates a UUID automatically; API clients should generate a UUID per user session and reuse it across requests.

**Non-streaming endpoint:**
```bash
curl -X POST http://localhost:8000/api/query/ \
  -H "Content-Type: application/json" \
  -d '{"query": "How does the pipeline work?", "project_slug": "ml-llm-ops", "session_id": "my-uuid-here"}'
```

**Other endpoints:**
```bash
# Submit feedback
curl -X POST http://localhost:8000/api/query/feedback \
  -H "Content-Type: application/json" \
  -d '{"query_hash": "abc123", "vote": 1}'

# List projects
curl http://localhost:8000/api/projects/

# Get a chart (returns Plotly JSON)
curl http://localhost:8000/api/stats/ml-llm-ops/charts/stars
curl http://localhost:8000/api/stats/ml-llm-ops/charts/commits
curl http://localhost:8000/api/stats/ml-llm-ops/charts/weekly_commits
curl http://localhost:8000/api/stats/ml-llm-ops/charts/languages
curl http://localhost:8000/api/stats/ml-llm-ops/charts/health
```

---

## How Queries Are Routed

The system classifies your query's intent and routes to the right agent:

| Intent | Example Queries | Agent |
|---|---|---|
| `statistical` | "How many stars does this have?" "Who are the top committers?" "Show commit trends" "Graph commits per week" | StatsAgent — SQLite + GitHub stats + commit history |
| `health` | "Is this project actively maintained?" "What's the bus factor?" | HealthAgent — community health scoring |
| `code_search` | "How is authentication implemented?" "Where is the retry logic?" | CodeAgent — code collection search |
| `conceptual` | "How does the routing work?" "Explain the architecture" | DocAgent — docs + markdown |
| `comparison` | "Compare project A vs B" | CompareAgent — multi-project diff |
| `general` | Everything else | RAG — all collections |

Statistical and health queries never touch Milvus — they read directly from the SQLite metrics store. Commit counts are read from the `project_commits` table (per-commit live data) rather than the snapshot columns, so counts are accurate and consistent across all queries.

**Project inference** — you don't always need to specify `--project`. If you mention a known project name in your question (e.g., "Who are the top committers to Unity Catalog?"), the agent infers the project automatically. If it can't, it asks you to clarify.

---

## Contributor and Commit Queries

After indexing, you can ask contributor-specific questions:

```
"Who are the top committers to this project?"
"Who has contributed the most in the last 90 days?"
"Show me the commit activity trend"
"Graph commits per week"
"When was the last commit?"
```

Commit counts are read from the `project_commits` table populated during `add` and `refresh`. This table holds the full per-commit history for the last 90 days and is the authoritative source for all count and trend queries. The "graph commits per week" query will automatically render a bar chart showing weekly activity for the last 13 weeks.

---

## Managing Projects

```bash
# List all registered projects (shows collections and vector counts)
project-explorer list

# Show full per-collection detail
project-explorer list --details

# Check environment health (Milvus, Ollama, GitHub, MLflow)
project-explorer status

# Refresh a project's index (incremental) and update stats
project-explorer refresh ml-llm-ops

# Refresh without updating GitHub statistics
project-explorer refresh ml-llm-ops --no-stats

# Remove a project (drops all Milvus collections)
project-explorer remove ml-llm-ops
```

### Incremental Refresh

`refresh` does two things:
1. Compares the latest commit SHA against the last-indexed SHA. Only files changed in the diff are re-indexed, one collection at a time. Typically completes in under a minute for small changes.
2. Fetches updated GitHub statistics and the latest 90 days of commit history into SQLite.

Use `--no-stats` to skip step 2 (e.g., if you've hit a GitHub rate limit).

### Attaching a Documentation Site

```bash
project-explorer add-docs myproject --docs-url https://docs.myproject.io
project-explorer add-docs myproject --homepage https://myproject.io
```

Fetches the docs URL via Docling and stores chunks in the `web_docs` collection.

---

## A2A Agent Endpoints

`project-explorer serve` exposes agents to the [beeai.dev](https://beeai.dev) platform and other A2A-compatible clients:

```bash
# Start orchestrator only (port 8100)
project-explorer serve

# Start all 6 specialist agents on consecutive ports
project-explorer serve --all

# Custom host/port
project-explorer serve --host 0.0.0.0 --port 9000 --all
```

| Port | Agent | Skills |
|---|---|---|
| 8100 | Orchestrator | Classifies intent and delegates; fallback to general RAG |
| 8101 | Statistics | project_stats, top_committers, commit_activity |
| 8102 | Code Search | code_search, usage_examples |
| 8103 | Documentation | conceptual_qa, api_reference |
| 8104 | Health | health_score, pr_metrics |
| 8105 | Compare | project_comparison |

Stats and Health agents use the A2A `input_required` pattern — they pause and ask the user for a project name if it can't be inferred, then resume automatically when the user replies.

You can prefix any query with `project:<slug>` to bypass inference:

```
project:unitycatalog Who are the top committers?
```

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

# 3. Verify ingestion (shows collections and vector counts)
project-explorer list

# 4. Test each intent type
project-explorer ask --project ml-llm-ops "How many stars does this project have?"
project-explorer ask --project ml-llm-ops "Who are the top committers in the last 90 days?"
project-explorer ask --project ml-llm-ops "Is this project actively maintained?"
project-explorer ask --project ml-llm-ops "How does the ML pipeline work?"
project-explorer ask --project ml-llm-ops "Where is the MLflow tracking configured?"

# 5. Test project inference (no --project flag)
project-explorer ask "How many stars does ML LLM Ops have?"

# 6. Interactive CLI session (with cross-turn memory)
project-explorer chat --project ml-llm-ops

# 7. Full-screen TUI (streaming + memory)
project-explorer tui

# 8. Web UI (streaming + session memory + inline charts)
project-explorer web
# Then open http://127.0.0.1:8000 and try the same questions
# Ask "graph commits per week for ml-llm-ops" to see an inline chart in the response
```

Expected behavior:
- Stars/contributor questions return data from the stats database without Milvus lookup
- Architecture/pipeline questions retrieve from markdown collections
- Code questions retrieve from Python code collections
- Project inference works when the project name appears in the query
- When no project can be inferred, the agent asks for clarification
- In the web UI, follow-up questions remember prior context (e.g., "tell me more about the top committer" after asking who they are)
- Commit/activity chart queries render a Plotly bar chart inline in the chat response

---

## Troubleshooting

**"No collections found for this project"**
The ingestion may have found no files matching a collection's extensions. Run `refresh` or re-add with different collection selections.

**"I don't have enough information..."**
The retrieval score fell below 0.30 (the minimum). The query may be outside the indexed content, or the project needs a refresh.

**Agent asks "Which project are you asking about?"**
The query didn't mention a known project name. Either specify `--project <slug>`, click the project in the sidebar (web/TUI), or rephrase with the project name.

**Milvus connection refused**
Start Milvus: `docker run -p 19530:19530 milvusdb/milvus:latest standalone`

**Ollama model not found**
Pull the model: `ollama pull llama3.1:8b`

**GitHub rate limit exceeded**
Add a GitHub token to `.env` and re-run. Authenticated requests have a 5000/hour limit vs 60/hour unauthenticated. Use `--no-stats` on refresh to skip GitHub API calls.

**Web UI charts show "No data — run refresh first"**
The chart endpoint needs stats in SQLite. Run `project-explorer refresh <slug>` to populate them.

**Stats say "0 commits" but committers are listed**
Run `project-explorer refresh <slug>` — the live commit counts come from the `project_commits` table, which is populated during refresh. If the table is empty, both counts and committer lists will be empty.

**Web UI doesn't remember prior questions**
Session memory is keyed to a UUID stored in `localStorage`. If you cleared browser storage or opened a new private window, a new session starts. The server expires idle sessions after 30 minutes.

---

## Architecture Reference

See [Architecture.md](Architecture.md) for the full architecture diagrams, module map, agent class hierarchy, BeeAI tools reference, and extension points.
