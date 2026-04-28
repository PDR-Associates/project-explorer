# Building Project Explorer: An Incremental Design Tutorial

This tutorial walks through how Project Explorer was designed and built in stages, explaining the reasoning behind each decision and the open-source tools chosen at each step. The goal is not just to describe what was built, but to show *why* each piece was added when it was, what problem it solved, and what you would have been missing without it.

The project is a multi-agent RAG system that lets you ask natural-language questions about GitHub repositories. By the end of this tutorial you will understand how to build something similar from scratch using the same open-source stack.

---

## The Starting Point: What Are We Actually Building?

Before writing a line of code, it helps to define the problem clearly. We want to answer questions like:

- "How does the authentication module work in this project?"
- "Who are the top contributors in the last 90 days?"
- "Is this project actively maintained?"
- "Compare the architecture of project A versus project B."

These questions span fundamentally different data sources and require different reasoning strategies. A single RAG pipeline cannot handle all of them well — some need vector search over code and docs, some need live database lookups, some need structured comparison logic. This observation shapes the entire design.

The reference implementation we drew from was [lfai/ML_LLM_Ops](https://github.com/lfai/ML_LLM_Ops), which validated the BeeAI + Milvus + Ollama pattern for production-grade agent workflows.

---

## Phase 1: The Core RAG Pipeline

### The Minimum Viable Query

The first thing to get right is the basic retrieval-augmented generation loop:

1. Embed a user query into a vector
2. Search a vector store for similar chunks
3. Pass the chunks as context to an LLM
4. Return the generated answer

This is the foundation everything else builds on. Every other phase either improves this loop or routes around it entirely.

### Choosing the Vector Store: Milvus

We chose [Milvus](https://milvus.io/) (`pymilvus`) over alternatives like Chroma or Qdrant for three reasons:

- **Milvus Lite** runs as a local `.db` file — zero infrastructure for development, identical API for production
- **MilvusClient** provides a simple high-level interface that works against both Lite and standalone
- **COSINE similarity** over 384-dim vectors is fast even with `FLAT` index type on Milvus Lite

The collection schema is intentionally minimal — four fields, no dynamic fields, no complex metadata indexes:

```python
schema = client.create_schema(auto_id=True, enable_dynamic_field=False)
schema.add_field("id",            DataType.INT64,         is_primary=True)
schema.add_field("vector",        DataType.FLOAT_VECTOR,  dim=384)
schema.add_field("text",          DataType.VARCHAR,        max_length=65_535)
schema.add_field("metadata_json", DataType.VARCHAR,        max_length=65_535)
```

Metadata (file path, language, chunk index, etc.) is stored as a JSON string in a single VARCHAR field. This avoids schema migrations when metadata shape changes, at the cost of in-memory filtering when needed. For a read-heavy workload like this, that tradeoff is acceptable.

The `FLAT` index type is critical for Milvus Lite compatibility. When deploying to Milvus standalone at scale, swap it for `HNSW` in `_ensure_collection()`.

### Choosing Embeddings: sentence-transformers

We used [sentence-transformers](https://www.sbert.net/) (`sentence-transformers` package) with the `all-MiniLM-L6-v2` model:

- 384-dimensional vectors — small enough that FLAT search is fast, large enough to capture semantic similarity well
- Runs locally on CPU or MPS (Apple Silicon Metal GPU)
- No API key or rate limits
- Familiar in the LF AI community from prior projects

The model is loaded once per process and reused:

```python
from sentence_transformers import SentenceTransformer

_model: SentenceTransformer | None = None

def get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        _model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    return _model
```

This singleton pattern matters: loading a transformer model takes 2–4 seconds. Re-loading it per query would make the system unusable.

### Choosing the LLM: Ollama

[Ollama](https://ollama.ai/) runs large language models locally with Metal GPU acceleration on Apple Silicon. For development, this means no API costs and no data leaving your machine. For production, the same code paths work with OpenAI or Anthropic backends.

The `LLMBackend` protocol defines two methods:

```python
class LLMBackend(Protocol):
    def complete(self, prompt: str, system: str = "", **kwargs) -> str: ...
    def stream(self, prompt: str, system: str = "", **kwargs) -> Iterator[str]: ...
```

The `stream()` method was added later when we needed token-by-token output for the TUI and web UI. Having a protocol from the start made this a safe addition — the `complete()` callers were unaffected.

The Ollama implementation uses `httpx` to call the local API:

```python
response = httpx.post(
    f"{self.base_url}/api/generate",
    json={"model": self.model, "prompt": prompt, "stream": False},
    timeout=120.0,
)
return response.json()["response"]
```

For streaming, the same endpoint is called with `"stream": True` and each newline-delimited JSON chunk is yielded as it arrives.

### The RAG Loop

With these three pieces in place, the first working query looked like:

```python
# Embed the query
q_vec = embed_one(query)

# Search Milvus
results = client.search(
    collection_name=collection,
    data=[q_vec],
    limit=5,
    output_fields=["text", "metadata_json"],
)

# Build context and call LLM
context = "\n\n---\n\n".join(r["entity"]["text"] for r in results[0])
prompt = build_rag_prompt(query, context)
return llm.complete(prompt)
```

This worked. But it had two immediate problems: it searched only one collection, and it had no way to answer questions about commit counts or project health. These drove the next two phases.

---

## Phase 2: Content-Aware Ingestion

### The Problem With a Single Collection

A single flat collection means code chunks compete with documentation chunks in the same vector space. A query about "how authentication is implemented" might return a README paragraph about authentication instead of the actual implementation code. Worse, different content types benefit from very different chunk sizes:

| Content Type | Ideal Chunk Size | Why |
|---|---|---|
| Source code | 512 tokens, overlap 64 | Functions fit; overlap catches split signatures |
| Markdown docs | 384 tokens, overlap 48 | Paragraphs and sections; smaller than code |
| API specs | 256 tokens, overlap 32 | Dense; small chunks are more precise |
| Notebooks | 1024 tokens, overlap 128 | Code + output pairs need to stay together |

The solution is **namespaced collections** per project per content type: `{project_slug}_{collection_type}`. Unity Catalog gets `unitycatalog_python_code`, `unitycatalog_markdown_docs`, `unitycatalog_api_reference`, and so on. Each collection is configured independently with its own chunk size and file extension list.

### Docling for Document Parsing

For PDF, web pages, and structured documents, we used [Docling](https://github.com/DS4SD/docling) — an open-source document understanding library from IBM Research. Docling extracts clean structured text from PDFs (including tables and figures) and HTML documentation sites, feeding it into the same chunking pipeline as everything else.

Without Docling, PDF ingestion requires Tesseract OCR or ad-hoc `pdfminer` hacks. Docling handles layout analysis, table extraction, and reading order detection out of the box.

### The Onboarding Wizard

Rather than requiring users to configure collections manually, we built `RepoAnalyzer` to inspect a repository and propose a collection plan:

```python
def _build_plan(self, repo) -> list[CollectionType]:
    """Inspect file extensions and propose which collections to create."""
    extensions = self._sample_extensions(repo, max_files=500)
    plan = []
    if any(e in extensions for e in [".py"]):
        plan.append(CollectionType.PYTHON_CODE)
    if any(e in extensions for e in [".js", ".ts"]):
        plan.append(CollectionType.JAVASCRIPT_CODE)
    if any(e in extensions for e in [".md", ".rst"]):
        plan.append(CollectionType.MARKDOWN_DOCS)
    # ... etc
    return plan
```

The `OnboardingWizard` presents this plan to the user and lets them confirm or customize before ingestion begins. This pattern — propose and confirm — is more robust than either fully manual or fully automatic configuration. Users who accept the defaults get a sensible setup; users who know their repo can tune it.

### Zipball Download Instead of Per-File API Calls

The first ingestion implementation used the GitHub API to fetch each file individually. This hit rate limits on large repos (Egeria has thousands of files). The fix was to download the entire repo as a single zipball — one API call regardless of repo size:

```python
def download_zipball(self, repo, tmp_dir: Path) -> Path:
    url = repo.get_archive_link("zipball")
    response = requests.get(url, stream=True)
    zip_path = tmp_dir / "repo.zip"
    with open(zip_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(tmp_dir)
    return next(tmp_dir.iterdir())  # the extracted root dir
```

This is a case where the obvious implementation (per-file API) worked fine in testing but failed in production on large repos. Download-once-parse-locally is more robust and much faster.

---

## Phase 3: Intent Classification Before Retrieval

### The Problem With RAG-for-Everything

After Phase 2, queries about code and documentation worked well. But "How many stars does this project have?" returned hallucinated numbers — the vector store has no star count, and the LLM guessed.

The root cause is that some queries should never touch the vector store at all. Statistical queries have precise answers in structured databases. Sending them through RAG is not just ineffective — it actively degrades quality because the LLM gets permission to fill in gaps with made-up numbers.

The fix is to classify intent *before* touching Milvus.

### QueryProcessor: Regex Patterns Over YAML

We chose a simple regex-based classifier over an LLM-based classifier for two reasons:

1. **Speed** — no LLM call, no embeddings. Classification is a few microseconds.
2. **Predictability** — patterns can be audited, tested, and edited without touching code.

Patterns live in `config/routing.yaml`:

```yaml
intent_patterns:
  statistical:
    priority: CRITICAL
    patterns:
      - "how many (commits|contributors|stars|forks|releases)"
      - "who (are the|have been)? (contribut\w+|committ?\w+)"
      - "show (me )?(a )?(graph|chart|plot|trend)"
      - "(stars|forks|watchers|issues|releases|contributors)"

  health:
    priority: HIGH
    patterns:
      - "(is|how).*(actively|well) maintained"
      - "bus factor"
      - "abandoned|archived|dead"

  code_search:
    priority: NORMAL
    patterns:
      - "how (do I|to|can I) (use|call|implement)"
      - "how is .+ implemented"
```

The classifier tries intents in priority order, first match wins:

```python
def classify(self, query: str) -> QueryIntent:
    q = query.lower()
    priority_order = ["statistical", "comparison", "health", "code_search", "conceptual"]
    for intent_name in priority_order:
        rule = self._rules.get(intent_name, {})
        for pattern in rule.get("patterns", []):
            if re.search(pattern, q, re.IGNORECASE):
                return QueryIntent(intent_name)
    return QueryIntent.GENERAL
```

One important YAML gotcha: regex patterns containing backslashes (`\w`, `\d`) must use single-quoted YAML strings. Double-quoted strings treat `\` as an escape character, making `\w` invalid. This caused silent failures in early testing that were hard to diagnose.

### The RAGSystem Orchestrator

With classification in place, `RAGSystem._route()` became a simple dispatch table:

```python
def _route(self, query, intent, project_slug):
    if intent == QueryIntent.STATISTICAL:
        return StatsAgent().handle(query, project_slug), []
    if intent == QueryIntent.HEALTH:
        return HealthAgent().handle(query, project_slug), []
    if intent == QueryIntent.CODE_SEARCH:
        return CodeAgent().handle(query, project_slug), []
    if intent == QueryIntent.CONCEPTUAL:
        return DocAgent().handle(query, project_slug), []
    if intent == QueryIntent.COMPARISON:
        return CompareAgent().handle(query, project_slug), []
    return self._rag(query, project_slug)  # GENERAL fallback
```

Statistical and health agents read from SQLite. They never touch Milvus. This separation is load-bearing: it's what prevents hallucinated commit counts.

---

## Phase 4: Tool-Using Agents with BeeAI

### Why Agents Instead of Direct Function Calls

After Phase 3, each agent was just a function: `stats_agent(query, project_slug) -> str`. This worked for simple queries but failed on compound ones:

- "Show me the top committers and the commit trend for the last 3 months" — requires two database lookups plus synthesis
- "How does the authentication work and what are the test coverage stats?" — spans code search and a stats lookup

The solution is agents that can call multiple tools iteratively. [BeeAI Framework](https://beeai.dev) provides `RequirementAgent` — a ReAct-style agent that loops between "think, call a tool, observe result, think again" until it has enough information to answer.

### The @tool Pattern

BeeAI tools are ordinary Python functions decorated with `@tool`. The decorator uses the docstring as the tool description (what the LLM sees to decide when to call it) and the function signature to generate a Pydantic schema for argument validation:

```python
from beeai_framework.tools import tool

@tool(description=(
    "Search indexed project content (code, docs, API specs) for text relevant to the query. "
    "collection_names is a comma-separated list of fully-qualified collection names, "
    "e.g. 'unitycatalog_python_code,unitycatalog_markdown_docs'. "
    "Call multiple times with different queries or collections to gather more context."
))
def vector_search(query: str, collection_names: str) -> str:
    collections = [c.strip() for c in collection_names.split(",") if c.strip()]
    results = MultiCollectionStore().search(query, collections)
    if not results:
        return "No relevant content found in the specified collections."
    parts = [f"[{r.collection} | score={r.score:.2f}]\n{r.text}" for r in results]
    return "\n\n---\n\n".join(parts)
```

The description is critical — it is the only thing the LLM uses to decide when and how to call the tool. Vague descriptions lead to incorrect calls; specific ones (including the format of arguments) lead to reliable behavior.

We defined four shared tools:

| Tool | What it does |
|---|---|
| `vector_search(query, collection_names)` | Milvus similarity search across named collections |
| `query_project_stats(project_slug)` | Returns formatted stats from SQLite |
| `query_top_committers(project_slug, limit)` | Ranked contributor list from `project_commits` |
| `query_commit_activity(project_slug)` | Weekly commit bar chart from `project_commits` |

Tools are shared across agents — a tool defined once can appear in the `tools()` list of any agent that needs it. This is cleaner than duplicating retrieval logic per agent.

### BaseExplorerAgent

All agents share a common base class that handles BeeAI setup and the async/sync bridge:

```python
class BaseExplorerAgent(ABC):
    @abstractmethod
    def system_prompt(self) -> str: ...

    @abstractmethod
    def tools(self) -> list: ...

    def _build_agent(self):
        return RequirementAgent(
            llm=self._llm_name(),      # e.g. "ollama:llama3.1:8b"
            tools=self.tools(),
            instructions=self.system_prompt(),
        )

    def _run_agent(self, prompt: str) -> str:
        async def _inner():
            agent = self._build_agent()
            result = await agent.run(prompt)
            return result.output[0].text

        try:
            asyncio.get_running_loop()
            # Inside FastAPI or Textual — use a thread to avoid event loop conflicts
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                return ex.submit(lambda: asyncio.run(_inner())).result()
        except RuntimeError:
            # No running loop — call asyncio.run directly
            return asyncio.run(_inner())
```

The async/sync bridge (`asyncio.get_running_loop()` → ThreadPoolExecutor) is boilerplate that every BeeAI integration in a synchronous codebase needs. Centralizing it in `BaseExplorerAgent` means individual agents never think about it.

### Defining an Agent

With the base class in place, a new agent is about 15 lines:

```python
class StatsAgent(BaseExplorerAgent):
    def system_prompt(self) -> str:
        return """You are a data analyst for GitHub project statistics.
Always call the appropriate tool first, then present the results.
Never write code, never describe how to fetch data, never use hypothetical numbers."""

    def tools(self) -> list:
        from explorer.agents.tools import (
            query_project_stats, query_top_committers, query_commit_activity
        )
        return [query_project_stats, query_top_committers, query_commit_activity]

    def handle(self, query: str, project_slug: str | None = None, **kwargs) -> str:
        slug = project_slug or self._infer_project_slug(query)
        if not slug:
            return self._clarification_response(query)
        return self._run_agent(f"Project: {slug}\n\nQuestion: {query}")
```

Notice the system prompt explicitly says "never write code" — this was added after the agent started generating Python tutorials in response to "graph commits per week". LLMs default to describing *how* to do things rather than *doing* them via tools. The system prompt must explicitly prohibit this.

### Project Inference

A key usability feature: you should not have to specify a project slug on every query. `_infer_project_slug()` scans the registry for a project whose name or slug appears in the query text:

```python
def _infer_project_slug(self, query: str) -> str | None:
    q = query.lower()
    for project in ProjectRegistry().list_all():
        if project.slug.lower() in q:
            return project.slug
        # Also match on display name words ("Unity Catalog" → slug "unitycatalog")
        words = project.display_name.lower().split()
        if all(w in q for w in words):
            return project.slug
    return None
```

When inference fails, `_clarification_response()` returns a message that starts with the exact string "Which project are you asking about?" — every UI layer detects this prefix and enters a clarification flow (sidebar selection, typed name, or `project:` prefix).

---

## Phase 5: Multiple Interfaces

With the backend solid, we built three interfaces: a CLI for scripting and automation, a TUI for power users, and a web UI for broader access. All three share the same `RAGSystem` backend.

### CLI with Typer and Rich

[Typer](https://typer.tiangolo.com/) generates a fully typed CLI from Python function signatures. [Rich](https://rich.readthedocs.io/) handles formatting. Together they give you a polished CLI with almost no boilerplate:

```python
import typer
from rich.console import Console

app = typer.Typer()
console = Console()

@app.command()
def ask(
    query: str = typer.Argument(..., help="Question to ask"),
    project: str | None = typer.Option(None, "--project", "-p"),
):
    """Ask a one-shot question."""
    rag = RAGSystem()
    response = rag.query(query, project_slug=project)
    console.print(response)
```

Typer automatically generates `--help`, handles type coercion, and validates arguments. The `chat` command extends this with a REPL loop backed by `ConversationAgent`, giving users an interactive session with cross-turn memory.

### TUI with Textual

[Textual](https://textual.textualize.io/) is a Python framework for building full-screen terminal UIs. It provides a widget system, reactive state, and CSS-like layout — significantly higher-level than curses.

The TUI architecture has three key challenges:

**1. Streaming into a live widget.** Textual runs its own event loop. BeeAI and the LLM run blocking operations. The solution is `@work(thread=True)` — Textual's worker decorator that runs a function in a background thread, with `call_from_thread()` to safely update the UI:

```python
@work(thread=True)
def _run_query(self, query: str) -> None:
    bubble = ChatMessage("Assistant")
    self.call_from_thread(log.mount, bubble)

    for chunk in self._rag.stream(query, project_slug=self.selected_project):
        if isinstance(chunk, dict) and chunk.get("_done"):
            break
        self.call_from_thread(bubble.append_text, str(chunk))
        self.call_from_thread(log.scroll_end, False)
```

**2. Conversation memory across turns.** Each query runs in a fresh worker thread. BeeAI's `TokenMemory` is not thread-safe across `asyncio.run()` calls in separate threads. The solution: stream the response via `RAGSystem`, then feed the completed turn back into the `ConversationAgent`'s BeeAI memory *after* the stream finishes:

```python
async def _add_to_memory():
    mem = self._conv._get_agent().memory
    await mem.add(UserMessage(query))
    await mem.add(AssistantMessage(accumulated))

with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
    ex.submit(lambda: asyncio.run(_add_to_memory())).result(timeout=5)
```

**3. Pre-warming to avoid threading conflicts.** PyTorch and gRPC initialize file descriptors in the main thread. If they initialize inside a Textual worker thread, FD conflicts occur. The `run()` entry point forces model and client initialization in the main thread before Textual starts:

```python
def run() -> None:
    get_embedding_model()   # force SentenceTransformer init in main thread
    rag = RAGSystem()
    try:
        MultiCollectionStore()._get_client()  # force Milvus client init
    except Exception:
        pass
    conv = ConversationAgent(rag_system=rag)
    ProjectExplorerApp(conv, rag).run()
```

### Web UI with FastAPI and Plotly

[FastAPI](https://fastapi.tiangolo.com/) serves both the API and the single-page HTML frontend. The frontend uses Tailwind CSS, marked.js for markdown rendering, and Plotly.js for charts — all from CDN, no build step.

**Server-Sent Events for streaming.** FastAPI's `StreamingResponse` with `media_type="text/event-stream"` streams tokens to the browser as they are generated. The synchronous `RAGSystem.stream()` generator runs in a daemon thread, bridging to the async FastAPI handler via an `asyncio.Queue`:

```python
@router.post("/stream")
async def stream(request: QueryRequest) -> StreamingResponse:
    async def _generate():
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_event_loop()

        def _producer():
            for item in rag.stream(request.query, project_slug=request.project_slug):
                loop.call_soon_threadsafe(queue.put_nowait, item)
            loop.call_soon_threadsafe(queue.put_nowait, None)

        threading.Thread(target=_producer, daemon=True).start()

        while True:
            item = await queue.get()
            if item is None:
                break
            if isinstance(item, dict) and item.get("_done"):
                yield f"data: {json.dumps({'t': 'done', 'chart': pick_chart(...)})}\n\n"
            else:
                yield f"data: {json.dumps({'t': 'chunk', 'v': str(item)})}\n\n"

    return StreamingResponse(_generate(), media_type="text/event-stream")
```

The `_done` sentinel dict carries metadata — intent, query hash, and an optional Plotly figure — alongside the final text event. The browser renders markdown and injects the chart after the stream finishes.

**Plotly charts as JSON.** Plotly figures are serialized with `fig.to_json()` and sent to the browser as plain JSON. The browser calls `Plotly.newPlot()` with the deserialized figure. This means chart logic lives entirely on the server (Python), and the browser is just a renderer. Plotly has a rich Python API for constructing figures; there is no need to duplicate chart logic in JavaScript.

---

## Phase 6: Statistics from Live Data

### The Snapshot vs. Live Data Problem

After ingestion, stats were stored as a snapshot in `project_stats`: `commits_30d`, `commits_90d`, etc. When a user asked "how many commits were there in the last 30 days?", the agent read from this snapshot. When they asked "who are the top committers?", the agent read from `project_commits` (the per-commit table). These two paths could return inconsistent numbers — the snapshot said 0 commits (stale), while the per-commit table had current data.

The fix: treat `project_commits` as the authoritative source for all commit counts. The stats tool now queries `project_commits` directly:

```python
now = datetime.now(timezone.utc)
cutoff_30 = (now - timedelta(days=30)).isoformat()
live_30 = conn.execute(
    "SELECT COUNT(*) FROM project_commits "
    "WHERE project_slug = ? AND committed_at >= ?",
    (slug, cutoff_30),
).fetchone()[0]

# Prefer live count; fall back to snapshot only if project_commits is empty
commits_30d = live_30 if live_30 or live_90 else (d.get("commits_30d") or 0)
```

`project_commits` uses `UNIQUE(project_slug, sha)` with `INSERT OR IGNORE` for idempotent fetches, so refreshes are safe to run repeatedly.

### Weekly Commit Charts

The sidebar "Commits" chart showed snapshots (the 30-day window recorded at each refresh). A user asking "graph commits per week" expects per-week granularity from actual commit data, not a bar chart of snapshot windows.

`weekly_commits_plotly()` reads directly from `project_commits`, buckets commits into 13 weekly bins, and builds a Plotly bar chart:

```python
week_counts: defaultdict = defaultdict(int)
for (ts,) in rows:
    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    weeks_ago = (now - dt).days // 7
    if 0 <= weeks_ago < 13:
        week_counts[weeks_ago] += 1

week_offsets = list(range(12, -1, -1))   # oldest → newest left-to-right
dates = [(now - timedelta(weeks=w)).strftime("%Y-%m-%d") for w in week_offsets]
counts = [week_counts.get(w, 0) for w in week_offsets]
```

The `_pick_chart()` function selects this chart when the query contains "week", "weekly", or "per week":

```python
elif any(w in q for w in ("week", "weekly", "per week", "week-by-week")):
    fig = graphs.weekly_commits_plotly(project_slug)
```

The chart appears inline in the chat response — not just in the sidebar — so the answer and its visualization arrive together.

---

## Phase 7: Observability

### Why Observability Matters for AI Systems

For conventional software, logging and metrics are for debugging. For AI systems, they serve an additional purpose: understanding *why* users got the answer they got. Without observability, you cannot answer questions like:

- What fraction of queries hit the cache versus the LLM?
- Which intents have the highest latency?
- Where did retrieval fail (low scores, empty results)?
- What did users give negative feedback on?

We wired up two observability backends: MLflow for experiment tracking and Arize Phoenix for LLM tracing.

### Arize Phoenix for LLM Tracing

[Arize Phoenix](https://phoenix.arize.com/) is an open-source LLM observability platform that captures traces of every LLM call — including the prompt, the retrieved context, the generated output, and latency. It runs locally at `localhost:6006`.

Phoenix integrates with BeeAI via OpenTelemetry instrumentation:

```python
from openinference.instrumentation.beeai import BeeAIInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

provider = TracerProvider()
provider.add_span_processor(
    BatchSpanProcessor(OTLPSpanExporter(endpoint="http://localhost:6006/v1/traces"))
)
BeeAIInstrumentor().instrument(tracer_provider=provider)
```

After this, every BeeAI agent call — including each tool invocation — appears as a span in the Phoenix UI. You can see which tools were called, in what order, and what they returned.

### MLflow for Experiment Tracking

[MLflow](https://mlflow.org/) tracks query-level metrics: intent, latency, cache hit rate, chunk references, and feedback scores. This is logged asynchronously so it never adds latency to the response path:

```python
def _track(self, query, intent, project_slug, response, latency_ms, cache_hit, chunk_refs):
    try:
        self.metrics.record_query(
            query, intent.value, project_slug, response,
            latency_ms=latency_ms, cache_hit=cache_hit, chunk_refs=chunk_refs,
        )
    except Exception:
        pass  # observability must never crash the main path
```

The `threading.Thread(target=self._track, ..., daemon=True).start()` pattern in `RAGSystem.query()` fires-and-forgets the tracking call. The `daemon=True` flag ensures the thread does not prevent process exit.

The `try/except: pass` in `_track` is intentional and important: observability infrastructure going down must not take down the query path.

---

## Phase 8: Production Hardening

### Query Caching

The highest-ROI performance optimization is caching repeated queries. `QueryCache` sits at the front of `RAGSystem.query()`, before classification and before any database or LLM calls:

```python
def query(self, query: str, project_slug: str | None = None) -> str:
    intent = self.processor.classify(query)
    cached = self.cache.get(query, project_slug, intent.value)
    if cached:
        threading.Thread(target=self._track, ..., daemon=True).start()
        return cached                # ← returns in microseconds
    # ... full pipeline
```

The default backend is an in-memory LRU (`max_size=1000`, `ttl=3600s`). For multi-process deployments (multiple uvicorn workers, CLI + web running simultaneously), switch to the Redis backend:

```bash
CACHE__BACKEND=redis
CACHE__REDIS_URL=redis://localhost:6379/0
```

The Redis backend uses sets keyed as `pe:project:{slug}:keys` to track all cache entries for a project, enabling efficient per-project cache invalidation on `refresh`.

Cache invalidation is one of two hard problems in computer science, and here it is straightforward: when a project's index is refreshed, invalidate all cache entries tagged to that project slug. Everything else expires by TTL.

### Incremental Indexing

Re-ingesting an entire project on every code change is too slow. `IncrementalIndexer` uses the GitHub commit API to find what changed:

```python
def refresh(self, project: Project) -> None:
    repo = self.client.get_repo(project.github_url)
    latest_sha = self.client.get_latest_commit_sha(repo)
    last_sha = self._get_last_sha(project.slug)

    if last_sha == latest_sha:
        print(f"No changes since last index ({latest_sha[:8]})")
        return

    changed_files = self._get_changed_files(repo, last_sha, latest_sha)
    # Map changed file extensions → affected collection types
    affected_types = self._affected_collections(changed_files)
    # Drop and re-ingest only the affected collections
    for ctype in affected_types:
        self.store.drop_collection(f"{project.slug}_{ctype.name}")
    IngestionPipeline().run(project.slug, project.github_url, affected_types)
```

Re-indexing happens at **collection granularity**, not per-file. Milvus does not support efficient delete-by-metadata-filter on this schema, and per-file upsert would require tracking chunk-to-file mappings across schema versions. Dropping and re-ingesting a single collection (typically a few thousand chunks) completes in under 30 seconds — fast enough for a manual `refresh` command.

### Session Memory for the Web UI

The CLI and TUI naturally maintain state across a session (they are long-running processes). The web UI is stateless by default — each HTTP request creates a fresh `RAGSystem` with no memory of prior turns.

The fix is a server-side session store keyed to a browser-generated UUID:

**Frontend (index.html):**
```javascript
// Generate once, persist across page reloads
let sessionId = localStorage.getItem('pe_session_id');
if (!sessionId) {
    sessionId = crypto.randomUUID();
    localStorage.setItem('pe_session_id', sessionId);
}

// Send with every request
fetch('/api/query/stream', {
    method: 'POST',
    body: JSON.stringify({ query, project_slug, session_id: sessionId }),
});
```

**Backend (query.py):**
```python
_SESSION_TTL = 1800   # 30 minutes
_SESSION_MAX = 50     # max concurrent sessions
_sessions: dict[str, tuple] = {}

def _get_or_create_session(session_id: str, project_slug: str | None):
    now = time.monotonic()
    # Evict expired
    expired = [sid for sid, (_, ts) in _sessions.items() if now - ts > _SESSION_TTL]
    for sid in expired:
        del _sessions[sid]
    # Evict oldest when at capacity
    if len(_sessions) >= _SESSION_MAX and session_id not in _sessions:
        oldest = min(_sessions, key=lambda sid: _sessions[sid][1])
        del _sessions[oldest]
    # Create or retrieve
    if session_id not in _sessions:
        _sessions[session_id] = (ConversationAgent(project_slug=project_slug), now)
    else:
        agent, _ = _sessions[session_id]
        _sessions[session_id] = (agent, now)
    return _sessions[session_id][0]
```

`ConversationAgent` wraps a **single persistent `RequirementAgent`** with `TokenMemory(max_tokens=8000)`. The same BeeAI agent object is reused across all requests in a session — BeeAI's `TokenMemory` accumulates the full conversation history automatically.

No login, no cookies, no database for sessions. The UUID in `localStorage` is the session key. This is deliberately simple: the session store lives in a single dict in the FastAPI process. For a multi-worker deployment, move it to Redis (same approach as the query cache).

---

## Phase 9: A2A Agent Interoperability with AgentStack

### What A2A Adds

The interfaces built so far (CLI, TUI, web) are user-facing. A2A (Agent-to-Agent) exposes the same agents as programmatic services that other agents and orchestrators can discover and call via a standard protocol. This is the foundation of multi-agent systems where agents delegate to specialists.

[AgentStack SDK](https://agentstack.sh) provides the A2A `Server` abstraction. Each specialist agent runs as its own server:

```python
from agentstack.sdk import Server

def make_stats_server(port: int) -> Server:
    server = Server(
        name="Project Explorer — Statistics",
        description="GitHub project stats, contributor rankings, commit trends",
        port=port,
    )

    @server.skill(name="project_stats", description="...")
    async def project_stats_skill(input: str, context) -> AsyncIterator[str]:
        agent = StatsAgent()
        slug = agent._infer_project_slug(input)
        if not slug:
            # Pause and ask the client for a project name
            yield TaskStatus(state=TaskState.input_required,
                             message="Which project? Available: " + list_slugs())
            # Resume when client replies
            reply = await context.wait_for_input()
            slug = reply.strip()
        yield agent.handle(input, project_slug=slug)

    return server
```

The `input_required` pattern is the A2A equivalent of the TUI's clarification flow: the agent pauses execution, sends a question to the client, and resumes with the client's reply. This lets the Stats and Health agents maintain a proper request/response cycle without requiring the project slug upfront.

### One Server Per Agent

AgentStack's `Server` supports exactly one agent per instance. Running all six agents requires starting six servers and `asyncio.gather()`-ing them:

```python
async def serve_all(base_port: int = 8100) -> None:
    servers = [
        make_orchestrator_server(base_port),
        make_stats_server(base_port + 1),
        make_code_server(base_port + 2),
        make_doc_server(base_port + 3),
        make_health_server(base_port + 4),
        make_compare_server(base_port + 5),
    ]
    await asyncio.gather(*[s.start() for s in servers])
```

We defaulted to port 8100 rather than 8080 because 8080 is a common default for other local services (Docker, HTTP proxies, other development servers). Shifting to 8100 avoids surprise port conflicts in typical development environments.

---

## Lessons Learned

### Classify Intent Before Retrieving

The most impactful architectural decision was classifying query intent before touching Milvus. Statistical queries with precise answers should never go through a vector similarity search — doing so opens the door to hallucination. The regex-over-YAML classifier is fast, auditable, and tunable without code changes.

### Cache Is the Highest-ROI Optimization

Retrieval + LLM generation takes 2–10 seconds. A cache hit takes microseconds. Implement caching before optimizing retrieval or generation. The key design choice is *where* to put the cache — before classification, so even the classifier is skipped on a hit.

### Data Consistency Requires a Single Source of Truth

The bug where "0 commits" and "5 committers" coexisted in the same response was caused by reading from two different tables. Designate one table as authoritative for each fact. `project_commits` is the source of truth for all commit counts; `project_stats` is a snapshot used only for fields that are not in `project_commits`.

### System Prompts Must Be Explicit About What Not to Do

LLMs will describe how to accomplish a task if they don't have a tool to accomplish it directly. Adding "never write code, never describe how to fetch data, always call the appropriate tool first" to the StatsAgent system prompt was what stopped the agent from generating Python tutorials in response to "graph commits per week."

### Incremental at Collection Granularity, Not File Granularity

Per-file incremental updates require tracking chunk-to-source-file mappings and supporting efficient delete-by-metadata in the vector store. Collection-level invalidation is simpler: detect which collection types were touched by changed files, drop those collections, re-ingest. For typical repos and change frequencies, collection-level granularity is fast enough.

### Pre-Warm Heavy Dependencies in the Main Thread

PyTorch, gRPC (Milvus), and BeeAI all initialize global state (file descriptors, thread pools, event loops) when first used. If they initialize inside a framework thread (Textual worker, FastAPI startup), race conditions and FD conflicts occur. Force initialization in the main thread before starting any framework.

### Observability Must Be Non-Blocking and Failure-Tolerant

Any synchronous call to an observability backend in the response path is a latency risk. Any unhandled exception from an observability call is a reliability risk. Dispatch observability to daemon threads; wrap `_track()` bodies in `try/except: pass`. The system must continue working if MLflow or Phoenix is unreachable.

### Streaming Requires Two Different Bridges

Synchronous code (LLM token generator) needs to reach asynchronous consumers (FastAPI SSE, Textual worker) through two different bridges:
- **FastAPI**: `asyncio.Queue` + `loop.call_soon_threadsafe()` — push from thread, pull in async generator
- **Textual**: `call_from_thread()` — call UI methods safely from a `@work(thread=True)` worker

Both patterns are short (5–10 lines each) but easy to get wrong. Test them early.

---

## The Full Stack, Summarized

| Problem | Tool | Why This Choice |
|---|---|---|
| Vector search | Milvus (pymilvus) | Lite mode for development, same API for production; COSINE similarity with FLAT index |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) | Local, no API key, MPS-accelerated, 384-dim is a good size/quality tradeoff |
| LLM inference | Ollama | Local, Metal GPU, no API costs, same interface as cloud |
| Cloud LLM | OpenAI / Anthropic | Pluggable via LLMBackend protocol |
| Agent framework | BeeAI Framework | RequirementAgent + @tool = iterative tool-calling with minimal boilerplate |
| A2A protocol | AgentStack SDK | Standard inter-agent discovery and communication |
| Document parsing | Docling | PDF, HTML, structured docs with layout analysis |
| CLI | Typer + Rich | Type-safe commands, zero boilerplate, beautiful output |
| TUI | Textual | Full-screen terminal UI, widget system, CSS layout |
| Web backend | FastAPI | Async, SSE streaming, typed request/response models |
| Web frontend | Tailwind + Plotly.js + marked.js | No build step, dark theme, interactive charts, markdown rendering |
| LLM tracing | Arize Phoenix | Per-call traces via OpenTelemetry, open source, runs locally |
| Experiment tracking | MLflow | Query metrics, latency, feedback logging, open source |
| Caching | LRU (in-process) / Redis | LRU for single-process; Redis for multi-process deployments |

Every component in this stack is open source. The only proprietary dependency is a GitHub token for API access — and unauthenticated access works for public repos at 60 requests/hour.

---

## Where to Go From Here

The natural next extensions, roughly in order of impact:

1. **Evaluation harness** — build a test set of (query, expected intent, expected answer) triples and measure retrieval recall, answer correctness, and latency regression automatically
2. **Feedback-driven reranking** — the `MultiCollectionStore` already records feedback scores; train a lightweight reranker on thumbs-up/down signals to boost relevant chunks
3. **More collection types** — GitHub Issues, PR bodies, and discussion threads are high-signal content not yet indexed
4. **Cross-project ConversationAgent** — today the conversation agent scopes to one project; a true cross-project agent would call `vector_search` across all indexed projects simultaneously
5. **Async ingestion** — the `add` command blocks until ingestion completes; move it to a background job with progress polling via the web API
6. **Milvus HNSW index** — when collection sizes exceed ~100k chunks, switch from FLAT to HNSW for sub-linear search time

The architecture is designed to support all of these without structural changes. New collection types follow the five-step extension process. New agents extend `BaseExplorerAgent`. New LLM backends implement the two-method `LLMBackend` protocol. New routing rules go in `routing.yaml`.
