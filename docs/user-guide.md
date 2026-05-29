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
- **Project sidebar** — click any project to scope all queries to it; status badges show active/indexing/error. Hover over a project row to reveal three action buttons:
  - **🔄 Refresh & profile** — re-indexes the repo and populates data profiles; runs synchronously and returns ✓ or ✗ when done; auto-reloads the Survey Report tab if it was open
  - **📊 Survey** — runs the survey pipeline, switches to the Survey Report tab, and reloads it
  - **↗** — opens the project's GitHub URL in a new tab
  - Tooltips appear immediately on hover (CSS-based, no browser delay)
- **Multi-project comparison** — Shift+click a second (or third) project to enter compare mode; the scope badge updates to show all selected projects and queries are automatically prefixed to trigger `CompareAgent` or `IntegrationAgent`
- **Streaming responses** — assistant text appears token-by-token via server-sent events (SSE); no waiting for the full answer
- **Conversation memory** — a UUID session ID is stored in `localStorage` and sent with every request; the server maintains a persistent `ConversationAgent` per session (30-minute idle timeout), giving the web UI the same cross-turn memory as the TUI and CLI
- **Inline charts** — when a statistical or health query warrants a chart, a Plotly figure appears directly in the chat response alongside the text
- **Inline symbol tables** — when a code inventory query ("how many classes?", "list all functions") is answered, a sortable, searchable table of symbols appears below the response; type in the filter box to narrow by name or file; click column headers to sort
- **API surface comparison** — when a comparison query mentions classes, methods, or API surface ("compare the public API of A vs B"), a side-by-side panel shows per-kind symbol counts and top symbols for each project
- **Survey Report tab** — click "📊 Survey Report" in the top nav when a project is selected. Shows health metric cards, a file type donut chart, a dependency bar chart, and a **Data Files** section with column schemas, row counts, and null rates for profiled CSV/Excel/Parquet files. When data files are detected but no profiles exist yet, a hint appears with the exact `refresh` command to run. Select file type rows and click **"Catalog selected →"** to create Egeria `DataSet` assets.
- **Sidebar charts** — Plotly interactive charts (Stars, Commits, Languages, Health) per selected project; click chart tabs to switch. The Commits tab shows the last 13 weeks of actual commit activity from the `project_commits` table, not snapshot aggregates
- **👍/👎 feedback buttons** — on each assistant message; keyboard `f` key also opens feedback
- **Alias suggestions** — when the agent finds a fuzzy match for an unrecognized project name, a banner appears with Yes/No confirmation; confirmed aliases are stored and resolve automatically
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

# Refresh and profile a project (synchronous — returns when done)
curl -X POST http://localhost:8000/api/projects/ml-llm-ops/refresh
# Returns: {"status": "ok", "slug": "ml-llm-ops", "message": "..."}

# Get a chart (returns Plotly JSON)
curl http://localhost:8000/api/stats/ml-llm-ops/charts/stars
curl http://localhost:8000/api/stats/ml-llm-ops/charts/commits
curl http://localhost:8000/api/stats/ml-llm-ops/charts/weekly_commits
curl http://localhost:8000/api/stats/ml-llm-ops/charts/languages
curl http://localhost:8000/api/stats/ml-llm-ops/charts/health

# Survey a project (no publish — returns annotation count and timestamp)
curl -X POST http://localhost:8000/api/egeria/ml-llm-ops/survey
# Returns: {"status": "ok", "annotation_count": 18, "surveyed_at": "2026-05-22T..."}

# Get survey report from SQLite (no Egeria needed)
curl http://localhost:8000/api/egeria/ml-llm-ops/survey-report

# Egeria registration status
curl http://localhost:8000/api/egeria/ml-llm-ops/status
```

---

## How Queries Are Routed

The system classifies your query's intent and routes to the right agent:

| Intent | Example Queries | Agent |
|---|---|---|
| `statistical` | "How many stars does this have?" "Who are the top committers?" "Show commit trends" "Graph commits per week" | StatsAgent — SQLite + GitHub stats + commit history |
| `health` | "Is this project actively maintained?" "What's the bus factor?" | HealthAgent — community health scoring |
| `code_inventory` | "How many classes does egeria have?" "List all functions in the ingestion module" "What's the signature of parse?" | CodeAgent — SQLite symbol table (no vector search) |
| `code_search` | "How is authentication implemented?" "Where is the retry logic?" | CodeAgent — code collection search |
| `conceptual` | "How does the routing work?" "Explain the architecture" | DocAgent — docs + markdown |
| `comparison` | "Compare project A vs B" "Which has more stars, A or B?" | CompareAgent — multi-project diff |
| `integration` | "Can I use egeria with agentstack?" "How do A and B work together?" | IntegrationAgent — ecosystem fit across projects |
| `general` | Everything else | RAG — all collections |

Statistical, health, and code-inventory queries never touch Milvus — they read directly from the SQLite metrics store. Commit counts are read from the `project_commits` table (per-commit live data) rather than the snapshot columns, so counts are accurate and consistent across all queries.

**Project inference** — you don't always need to specify `--project`. If you mention a known project name in your question (e.g., "Who are the top committers to Unity Catalog?"), the agent infers the project automatically. If it can't, it asks you to clarify.

**Alias inference** — if your query mentions a name the system doesn't recognize exactly (e.g., "Egeria Platform" instead of "egeria"), the agent checks a stored alias table first, then tries a fuzzy match. If a close candidate is found you'll be asked to confirm it as an alias — once confirmed, the name resolves automatically in all future queries.

---

## Question Reference

This section catalogs the kinds of questions the system can answer, organized by the type of information you're looking for. Each type maps to a specialized agent that knows exactly where to look.

You don't need to phrase questions exactly as shown — the intent classifier uses regex patterns, so natural variations work. The examples below are meant to illustrate the range, not to be memorized.

---

### GitHub Statistics and Activity

**Agent:** StatsAgent — reads from SQLite (`project_stats`, `project_commits`). Never touches Milvus.

```
"How many stars does this project have?"
"What's the fork count for egeria?"
"How many contributors does Unity Catalog have?"
"How many commits were made in the last 30 days?"
"How many commits in the last 90 days?"
"Show me commit activity over time"
"Graph commits per week"
"Show the weekly commit trend for the last 13 weeks"
"When was the last commit?"
"How much code was added vs deleted this month?"
"What's the release cadence?"
"How many open issues are there?"
```

The "graph commits per week" query automatically renders a bar chart inline in the chat response (web UI) or prints an ASCII sparkline (CLI). The chart shows the last 13 weeks of actual commit timestamps — not an aggregate.

---

### Contributors and Committers

**Agent:** StatsAgent — reads from `project_commits` and `project_contributor_stats`.

```
"Who are the top committers to this project?"
"Who has contributed the most in the last 90 days?"
"Show me the top 5 contributors"
"Tell me about Alice's contributions"
"What is Bob's commit tier?"
"Who is driving the most code change?"
"Which contributors are in the core tier?"
"Show me the committer breakdown"
```

After `refresh`, per-commit additions and deletions are stored alongside commit metadata. Contributor profiles include:
- Commit count and comparison to the project average
- Lines added and deleted
- Activity tier: `core` (top 25% by commits), `regular` (25th–75th percentile), or `occasional` (below 25th)

The weekly activity chart includes `(+adds / -dels)` annotations per week when churn data is available. Run `refresh` to populate churn data for projects indexed before this feature was added.

---

### Project Health and Maintenance

**Agent:** HealthAgent — scores from `project_stats`; no Milvus.

```
"Is this project actively maintained?"
"What's the health score for beeai_framework?"
"What's the bus factor?"
"How active is the community?"
"Is this a risky dependency?"
"When was the last release?"
"How regular are the releases?"
"Is the commit activity declining?"
"How responsive are the maintainers?"
```

Health queries return a structured score across four dimensions: **Activity** (commit frequency), **Community** (contributor count and spread), **Release Cadence** (regularity of tagged releases), and **Freshness** (how recently the project was updated). A radar chart is rendered inline in the web UI.

---

### Code Structure and Inventory

**Agent:** CodeAgent — reads from `project_code_symbols` SQLite table. No vector search.

This intent handles structural questions that have precise, countable answers. The agent queries the symbol table extracted at ingest time — it does not guess or approximate.

```
"How many classes does egeria have?"
"How many functions are in the ingestion module?"
"List all classes in explorer/agents/"
"What methods does CodeParser have?"
"Show me all public interfaces"
"Show me the signature of the parse method"
"What does the query_project_stats function return?"
"List all enums in the codebase"
"How many methods does the IngestionPipeline class have?"
"Show the API surface for the registry module"
"Which classes are defined in registry.py?"
"What's the signature of BaseExplorerAgent.__init__?"
```

In the web UI, code inventory answers include a **sortable, searchable symbol table** below the response. Click any column header to sort; type in the filter box to narrow results by name, kind, or file. For comparison queries that mention API surface (e.g., "compare the classes in project A vs B"), a side-by-side panel shows per-kind counts for both projects.

Symbol data is populated automatically during `add` and `refresh`. For projects indexed before code intelligence was added, run a one-time backfill:

```bash
project-explorer refresh <slug> --symbols --no-stats
```

---

### Code Search and Implementation

**Agent:** CodeAgent — vector search over code collections (`python_code`, `javascript_code`, `java_code`, `go_code`).

This intent handles semantic questions about *how* something is implemented — where to find a pattern, what a piece of code does, how a module is structured.

```
"How is authentication implemented?"
"Where is the retry logic?"
"How does the ingestion pipeline handle large files?"
"Show me the database connection setup"
"Where are environment variables read?"
"How does the caching layer work?"
"Find all places that call the GitHub API"
"What does the DataPrep class do?"
"How is error handling done in the web routes?"
"Show me the embedding logic"
"Where is rate limiting enforced?"
"How is the LLM backend selected?"
```

---

### Runnable Code Examples

**Agent:** ExamplesAgent — retrieves from `examples`, `python_code`, `api_reference`, and `markdown_docs` collections, then generates a complete, runnable Python snippet.

```
"Show me an example of adding a project"
"Give me a Python example of querying the vector store"
"How do I use the BeeAI @tool decorator?"
"Write an example that indexes a repo and asks a question"
"Show me how to use the LLM client"
"Give me an example of creating a custom agent"
"Show me example code for connecting to Milvus"
"How do I call the streaming endpoint from Python?"
```

Examples are generated from actual indexed code, not hallucinated. Constructor signatures and import paths are extracted from the `api_reference` and `examples` collections so the generated code is accurate for the indexed version of the project.

---

### Documentation and Architecture

**Agent:** DocAgent — vector search over `markdown_docs`, `web_docs`, `api_reference`, `pdfs`, `release_notes` collections.

```
"How does the routing work?"
"Explain the architecture"
"What is the query flow?"
"How does incremental indexing work?"
"What's documented in the README?"
"How is the TUI structured?"
"What are the configuration options?"
"Explain the collection namespace model"
"What changed in the last release?"
"What was added in version 2.0?"
"How do I configure Redis caching?"
"What does the onboarding wizard do?"
"What data formats does the data profiler support?"
```

---

### Data Files and Profiling

**Agent:** General RAG / DocAgent — reads from survey data stored in SQLite (`project_data_profiles`, `project_file_type_counts`).

These questions work after running `survey` or `refresh` (which profiles data files during ingest).

```
"What data files are in this project?"
"How many rows does train.csv have?"
"What columns are in the main dataset?"
"Which columns have high null rates?"
"What's the schema of the Parquet files?"
"How large is the data directory?"
"What data formats does this project use?"
"Are there any CSV files larger than 50 MB?"
"Summarize the dataset structure"
```

The **Survey Report tab** in the web UI also shows this information visually — per-file cards with format, size, row × column count, column type pills, and null rate warnings. You can catalog selected file types as Egeria `DataSet` assets directly from the tab.

---

### Project Comparison

**Agent:** CompareAgent — multi-project vector search plus `project_stats` for both projects.

In the web UI, Shift+click a second project to enter compare mode — queries are automatically sent to CompareAgent. In the CLI, mention both project names in your question.

```
"Compare project A and project B"
"Which has more stars, egeria or beeai_framework?"
"How does the architecture of A differ from B?"
"Compare the documentation quality of these two projects"
"Which project is more actively maintained?"
"Show me side-by-side commit activity for A and B"
"What's the difference in contributor count?"
"Compare the dependency footprint of A vs B"
"How similar are the APIs of A and B?"
"Compare the classes in egeria vs unity-catalog"
"Which project has more public methods?"
"Show me the API surface diff between these two"
```

When a comparison question mentions code structure (classes, methods, API surface, public symbols), the web UI renders a **side-by-side API surface panel** below the response — showing per-kind symbol counts and top symbols for each project.

---

### Integration Questions

**Agent:** IntegrationAgent — asks how two or more projects work together; combines context from both.

```
"Can I use egeria with agentstack?"
"How does beeai_framework integrate with Milvus?"
"Can I swap out Ollama for OpenAI in this stack?"
"How would I use project A as a backend for project B?"
"What would I need to connect these two systems?"
"Is there overlap between A and B?"
"How do A and B complement each other?"
"What's the migration path from A to B?"
```

---

### Multi-Turn Follow-Up Questions

The conversation agent maintains memory across turns in all interfaces — CLI `chat`, TUI, and web UI. You can ask follow-up questions that refer back to previous answers without repeating context.

```
# First question
"Who are the top committers to egeria?"

# Follow-ups — the agent remembers the project and the answer
"Tell me more about the top one"
"How does their contribution compare to the team average?"
"When did they join the project?"

# Or pivot topic without re-specifying the project
"Now show me the commit trend"
"What's the health score?"
"How many classes are in the codebase?"
```

In the web UI, session memory persists for 30 minutes of idle time. If you close the tab and return within that window, conversation context is restored from the server.

---

### Tips for Better Answers

**Be specific about what you want:** "How many Python classes are in the ingestion module?" gets a faster, more accurate answer than "Tell me about the code" because it routes directly to the symbol table rather than triggering a vector search.

**Scope your question when you know the project:** Add the project name or use `--project <slug>` to skip project inference. This is especially important when multiple indexed projects have overlapping terminology.

**Ask for charts explicitly in the web UI:** Phrases like "graph commits per week", "show me a chart of stars over time", or "visualize the language breakdown" trigger inline Plotly charts alongside the text response.

**Use natural follow-ups:** After any answer, you can ask "Can you show that as a chart?", "Which file is that in?", "Give me an example of using that function", or "What changed in the latest release?" — the agent keeps context across turns.

**If the answer seems wrong:** Run `project-explorer refresh <slug>` to update the index, then re-ask. Stats and code-inventory answers are only as fresh as the last refresh.

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

# Extract code symbols (classes, methods, functions) — run once per project after upgrading
project-explorer refresh ml-llm-ops --symbols --no-stats

# Remove a project (drops all Milvus collections)
project-explorer remove ml-llm-ops
```

### Incremental Refresh

`refresh` does three things:
1. Compares the latest commit SHA against the last-indexed SHA. Only files changed in the diff are re-indexed, one collection at a time. Typically completes in under a minute for small changes.
2. Always re-runs file inventory (`project_file_inventory`) and data profiling (`project_data_profiles`) when the repo is downloaded — so profiles are kept current whenever collections are re-indexed.
3. Fetches updated GitHub statistics and the latest 90 days of commit history into SQLite.

**If no commits are detected but `project_data_profiles` is empty**, refresh automatically downloads the repo just to run profiling — so a plain `project-explorer refresh <slug>` is always enough to populate profiles, even if the code hasn't changed. This covers projects that were indexed before data profiling was added.

Use `--no-stats` to skip the GitHub statistics step (e.g., if you've hit a rate limit).

### Code Symbol Index

The system maintains a `project_code_symbols` table in SQLite with every class, method, function, and interface extracted from the project's source code, along with signatures and docstrings. This powers structural queries like "how many classes does egeria have?" and "show me the signature of the parse method."

**For projects indexed before code intelligence was added**, run a one-time backfill:

```bash
# Fast — downloads repo, extracts symbols only, does not re-embed anything
project-explorer refresh <slug> --symbols --no-stats
```

For projects added or refreshed after this version, symbol extraction is automatic — it runs as part of every code collection ingestion.

### Managing Aliases

The system learns project name aliases when you use an unrecognized name. You can also manage them directly:

```bash
# List all aliases (or aliases for one project)
project-explorer aliases list
project-explorer aliases list egeria

# Add an alias manually
project-explorer aliases add "Egeria Platform" egeria
project-explorer aliases add "AIMD" ai_model_deployment

# Remove an alias
project-explorer aliases remove "Egeria Platform"
```

Aliases are normalized (lowercase, spaces → underscores) and resolve in all future queries without prompting. The web UI shows a confirmation banner when a fuzzy match is detected.

### Attaching a Documentation Site

```bash
project-explorer add-docs myproject --docs-url https://docs.myproject.io
project-explorer add-docs myproject --homepage https://myproject.io
```

Fetches the docs URL via Docling and stores chunks in the `web_docs` collection.

---

## Surveying Projects

`project-explorer survey` produces an **Egeria-aligned annotation report** for any indexed
project. It reads entirely from SQLite — no local clone, no Egeria connection needed. Publishing
to Egeria is optional.

```bash
# Survey one project
project-explorer survey egeria

# Survey several at once
project-explorer survey egeria beeai_framework ml_llm_ops

# Survey all registered projects
project-explorer survey --all

# Survey and publish to Egeria in one step
project-explorer survey egeria --publish
```

### What the survey covers

| Section | What it tells you |
|---|---|
| **File Classification** | Every file type in the repo (Python, Markdown, YAML, CSV, Parquet, …) with counts |
| **File Structure** | Total file count, repo size, lines of code, per-directory breakdown |
| **File Size** | Disk footprint by type, top-10 largest files; flags files >50 MB for Git LFS |
| **Data Profiling** | Counts and sizes per data format; column schemas for CSV, Excel, Parquet, Arrow |
| **Language** | Primary and secondary languages, inferred project type (Library / CLI / Service) |
| **Health** | Activity, community, release cadence, and freshness scores (0–100) |
| **Dependencies** | All dependencies grouped by ecosystem (PyPI, npm, Maven, …) |
| **Documentation** | Which collection types are indexed, which hygiene files exist |
| **Security** | Flags for missing SECURITY.md, no CI configuration, no license |
| **API Structure** | Public classes, functions, and module tree per language |

### Data profiling

Profiling runs automatically during `add` and `refresh` while the repo is on disk. When you
run `survey` later, the stored profiles are read from SQLite — no re-download needed.

```
project-explorer survey house_prices_global
```

Example data profiling output:

```
DataProfiling (4)
  • 6 data file(s) across 1 format(s), total 18.3 MB

  • train.csv: 1,460 rows × 81 columns  [CSV]
    5 column(s) >50% null: PoolQC, MiscFeature, Alley, Fence, FireplaceQu
    Columns: Id (int64), MSSubClass (int64), MSZoning (object), LotFrontage (float64), ... +77 more

  • test.csv: 1,459 rows × 80 columns  [CSV]

  • ⚠ large_backup.csv (62 MB) — exceeds 50 MB profiling limit for text formats
    Run: project-explorer survey house_prices_global --data-path ~/repos/house-prices
```

**If you see "No file inventory found — run refresh"**, the project was indexed before the file
inventory feature was added. Fix it with:

```bash
project-explorer refresh <slug>
```

Profiling supports CSV, Excel, Parquet, and Arrow/Feather. Parquet and Arrow files are profiled
from metadata only — no size limit. CSV and Excel are skipped above 50 MB. See
[docs/surveyor-reference.md](surveyor-reference.md) for the full format table.

### Viewing results

**CLI** — `survey` prints the full annotation report to the terminal.

**Web UI** — select a project and click **"📊 Survey Report"** in the top nav. Shows health
metric cards, a file type donut chart, dependency bar chart, and a **Data Files** section with
column schemas, row counts, and null rates. If data files are detected but no profiles exist, a
hint displays the `refresh` command. Select file type rows and click **"Catalog selected →"** to
create `DataSet` assets in Egeria. You can also trigger refresh or survey directly from the
sidebar hover buttons (🔄 and 📊).

**Chat** — ask naturally:

```
project-explorer ask --project house_prices_global "What data files are in this project?"
project-explorer ask --project house_prices_global "How many rows does train.csv have?"
project-explorer ask --project house_prices_global "Which columns have high null rates?"
```

### Publishing to Egeria

`--publish` runs the survey and pushes all annotations to Egeria as a linked `SurveyReport`:

```bash
project-explorer survey egeria --publish
```

The project's Egeria asset GUID is cached locally after the first publish, so subsequent runs
skip the discovery search. Each survey run creates a new `SurveyReport` — history is preserved.

View published survey history (no Egeria connection needed):

```bash
project-explorer egeria-reports egeria
project-explorer egeria-reports egeria --full   # fetch full annotation detail from Egeria
```

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

### Debug / Verbose Logging

By default the CLI suppresses noisy log output from gRPC (Milvus), HuggingFace `transformers`, and `tqdm` progress bars. To restore verbose output:

```bash
DEBUG=1 project-explorer ask "..."
DEBUG=1 project-explorer chat
DEBUG=1 project-explorer tui
```

When `DEBUG=1` is set, the following are restored: gRPC fork warnings, `transformers` info/warning logs, tokenizer parallelism messages, and `tqdm` progress bars (including the "Loading weights" bar from sentence-transformers on first model load).

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

### Choosing a Model

The system uses multi-tool agents (BeeAI `RequirementAgent`) that must follow precise instructions about which tool to call. **Tool-use compliance varies significantly by model size and family** — this is the most important factor in answer quality.

#### Ollama (local, no API cost)

| Model | Size | Tool-use | Notes |
|---|---|---|---|
| `llama3.1:8b` | 4.9 GB | Fair | Default; occasionally ignores tool-selection rules on complex queries |
| `qwen2.5-coder:latest` | 4.7 GB | Good | Code-specialized; better instruction following for code queries |
| `granite3.3:8b` | 4.9 GB | Good | IBM model; reliable instruction following |
| `mistral:7b` | 4.4 GB | Fair | General purpose; similar to llama3.1:8b |
| `codellama:13b` | 7.4 GB | Fair | Better at code retrieval but slower |

To switch models, change one line in `.env`:

```bash
LLM__OLLAMA__MODEL=qwen2.5-coder:latest
```

#### API backends (best quality, usage cost)

API-hosted models are trained specifically for tool use and follow system prompt instructions much more reliably than local 8B models. If you see the agent picking the wrong tool, calling `query_code_symbols` for implementation questions, or producing hallucinated summaries ("listed above" with nothing listed), switching to an API backend will resolve it.

```bash
# Anthropic — best tool-use compliance; Haiku is fast and cheap
LLM__BACKEND=anthropic
ANTHROPIC_API_KEY=sk-ant-...
LLM__ANTHROPIC__MODEL=claude-haiku-4-5-20251001   # fast, low cost
# LLM__ANTHROPIC__MODEL=claude-sonnet-4-6          # higher quality

# OpenAI
LLM__BACKEND=openai
OPENAI_API_KEY=sk-...
LLM__OPENAI__MODEL=gpt-4o-mini    # fast, low cost
# LLM__OPENAI__MODEL=gpt-4o       # higher quality
```

#### Symptoms of poor tool-use compliance

If you see these, try a different model:
- Agent answers "The top N symbols are listed above" but shows nothing — model called `query_code_symbols` for a semantic question and couldn't fit the result in context
- "How is X implemented?" returns a symbol count instead of explaining the code — model ignored tool-selection rules
- Agent asks for clarification on every question even when the project is selected — model isn't reading the system prompt reliably
- Responses trail off mid-sentence or repeat themselves — context window filling up from oversized tool results

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
project-explorer ask --project ml-llm-ops "How many Python classes are in this project?"
project-explorer ask --project ml-llm-ops "What's the signature of the main pipeline function?"
project-explorer ask --project ml-llm-ops "How does the ML pipeline work?"
project-explorer ask --project ml-llm-ops "Where is the MLflow tracking configured?"
project-explorer ask --project ml-llm-ops "Show me an example of running the pipeline"

# 5. Test project inference (no --project flag)
project-explorer ask "How many stars does ML LLM Ops have?"

# 6. Test multi-turn memory (CLI)
project-explorer chat --project ml-llm-ops
# Inside the session:
# > Who are the top committers?
# > Tell me more about the top one
# > What's the commit trend for the last 13 weeks?
# > Now show me the architecture

# 7. Run survey (generates annotation report from SQLite — no Egeria needed)
project-explorer survey ml-llm-ops

# 8. Full-screen TUI (streaming + memory)
project-explorer tui

# 9. Web UI (streaming + session memory + inline charts + symbol tables)
project-explorer web
# Open http://127.0.0.1:8000 and try:
# - "graph commits per week for ml-llm-ops"   → inline Plotly bar chart
# - "how many classes does ml-llm-ops have?"  → inline sortable/searchable symbol table
# - "who are the top committers?"             → inline bar chart (after selecting project)
# - Select the project, click "📊 Survey Report" to see file types and data profiles
```

Expected behavior:
- Stars/contributor questions return data from the stats database without Milvus lookup
- Code inventory questions ("how many classes?") return structured counts from the symbol table, with a searchable table in the web UI
- Architecture/pipeline questions retrieve from markdown collections
- Code search questions retrieve from Python code collections
- Project inference works when the project name appears in the query
- When no project can be inferred, the agent asks for clarification
- In the web UI, follow-up questions remember prior context (e.g., "tell me more about the top committer" after asking who they are)
- Commit/activity chart queries render a Plotly bar chart inline in the chat response

---

## Troubleshooting

**"No collections found for this project"**
The ingestion may have found no files matching a collection's extensions. Run `refresh` or re-add with different collection selections.

**Agent returns "The top N symbols are listed above" but shows no symbols, or answers "How is X implemented?" with a symbol count**
The agent called `query_code_symbols` (a structural listing tool) instead of `vector_search` (semantic code retrieval). This is a tool-use compliance failure in the underlying LLM. Workaround: rephrase the question to be more explicit — "Show me the code that handles X" or "Where is the X logic implemented?" tends to route correctly. Permanent fix: switch to a model with better instruction following — see [Choosing a Model](#choosing-a-model).

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

**Survey Report tab shows "Data file inventory detected but no column profiles yet"**
The project was indexed before data profiling was added. Run `project-explorer refresh <slug>` — it will detect empty profiles and download + profile automatically, even if no new commits exist.

**"Catalog selected →" returns a VALIDATION_ERROR or "Invalid URL"**
Egeria must be running and reachable at `EGERIA_PLATFORM_URL`. Verify the platform URL is set correctly and the server is up, then try again. The project must also have been published to Egeria first (run `project-explorer survey <slug> --publish`).

**Web UI commit chart appears flat or shows only a single bar**
The Commits chart reads from the `project_commits` table (per-commit history, last 13 weeks). If the table is empty, all bars are zero. Run `project-explorer refresh <slug>` to fetch commit history. With only one week of data the chart will still display correctly because the y-axis always starts from zero.

**Console shows gRPC fork warnings or "Loading weights" progress bars**
These are suppressed by default. If you are seeing them, check that `DEBUG` is not set in your environment. If they appear in normal operation, file a bug — the suppression is applied at startup via environment variables before any heavy imports.

**Stats say "0 commits" but committers are listed**
Run `project-explorer refresh <slug>` — the live commit counts come from the `project_commits` table, which is populated during refresh. If the table is empty, both counts and committer lists will be empty.

**Web UI doesn't remember prior questions**
Session memory is keyed to a UUID stored in `localStorage`. If you cleared browser storage or opened a new private window, a new session starts. The server expires idle sessions after 30 minutes.

---

## Architecture Reference

See [Architecture.md](Architecture.md) for the full architecture diagrams, module map, agent class hierarchy, BeeAI tools reference, and extension points.
