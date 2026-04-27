"""AgentStack A2A server — exposes Project Explorer agents via the BeeAI platform protocol."""
from __future__ import annotations

from agentstack_sdk.server import AgentServer
from agentstack_sdk.server.agent import agent as agentstack_agent

from explorer.rag_system import RAGSystem


_rag = RAGSystem()


@agentstack_agent(
    name="Project Explorer",
    description=(
        "Multi-agent RAG assistant for exploring GitHub projects. "
        "Routes questions to specialised agents (code, docs, stats, health, compare) "
        "based on query intent. Optionally scope to a project with 'project:<slug>' prefix."
    ),
)
def project_explorer_agent(query: str) -> str:
    """
    Unified entry point. Supports an optional project scope prefix:
      'project:ml_llm_ops How does the pipeline work?'
    """
    project_slug = None
    if query.lower().startswith("project:"):
        parts = query.split(None, 1)
        project_slug = parts[0].split(":", 1)[1].strip()
        query = parts[1] if len(parts) > 1 else ""

    return _rag.query(query, project_slug=project_slug)


def run(host: str = "0.0.0.0", port: int = 8080) -> None:
    server = AgentServer()
    server.register(project_explorer_agent)
    server.run(host=host, port=port)


if __name__ == "__main__":
    run()
