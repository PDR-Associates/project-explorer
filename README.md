# Project Explorer

Multi-agent RAG reference implementation for exploring GitHub projects.

## Quick Start

```bash
uv sync
cp .env.example .env
# Edit .env: set GITHUB_TOKEN, adjust LLM_BACKEND if needed

# Add a project
project-explorer add https://github.com/owner/repo

# Ask a question
project-explorer ask "How does authentication work?"

# Interactive chat
project-explorer chat
```

## Requirements

- Milvus at `localhost:19530`
- Ollama at `localhost:11434` (`ollama pull llama3.1:8b`)
- Optional: Arize Phoenix at `localhost:6006`, MLflow at `localhost:5025`

See [CLAUDE.md](CLAUDE.md) for full architecture, setup, and usage documentation.
