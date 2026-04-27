"""Pydantic settings — loaded from config/explorer.yaml + .env overrides."""
from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class MilvusConfig(BaseSettings):
    uri: str = "http://localhost:19530"
    token: str = ""


class OllamaConfig(BaseSettings):
    base_url: str = "http://localhost:11434"
    model: str = "llama3.1:8b"
    temperature: float = 0.1


class OpenAIConfig(BaseSettings):
    model: str = "gpt-4o-mini"
    temperature: float = 0.1


class AnthropicConfig(BaseSettings):
    model: str = "claude-haiku-4-5-20251001"
    temperature: float = 0.1


class LLMConfig(BaseSettings):
    backend: str = "ollama"  # ollama | openai | anthropic
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)
    anthropic: AnthropicConfig = Field(default_factory=AnthropicConfig)


class EmbeddingsConfig(BaseSettings):
    model: str = "all-MiniLM-L6-v2"
    device: str = "auto"  # auto | mps | cuda | cpu
    dimension: int = 384


class RAGConfig(BaseSettings):
    top_k: int = 10
    min_score: float = 0.15
    max_collections_per_query: int = 3


class CacheConfig(BaseSettings):
    ttl_seconds: int = 3600
    max_size: int = 1000
    backend: str = "memory"  # memory | redis
    redis_url: str = ""


class GitHubConfig(BaseSettings):
    token: str = Field(default="", alias="GITHUB_TOKEN")
    requests_per_hour: int = 5000
    clone_timeout_seconds: int = 300

    model_config = SettingsConfigDict(populate_by_name=True)


class MLflowConfig(BaseSettings):
    enabled: bool = True
    tracking_uri: str = "http://localhost:5025"
    experiment_name: str = "project-explorer"


class PhoenixConfig(BaseSettings):
    enabled: bool = True
    collector_endpoint: str = "http://localhost:6006/v1/traces"


class ObservabilityConfig(BaseSettings):
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)
    phoenix: PhoenixConfig = Field(default_factory=PhoenixConfig)
    metrics_db: str = "data/metrics.db"


class AgentsConfig(BaseSettings):
    max_iterations: int = 20
    max_retries: int = 10
    stream_responses: bool = True


class ExplorerConfig(BaseSettings):
    milvus: MilvusConfig = Field(default_factory=MilvusConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    github: GitHubConfig = Field(default_factory=GitHubConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    agents: AgentsConfig = Field(default_factory=AgentsConfig)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )


_config: ExplorerConfig | None = None


def get_config() -> ExplorerConfig:
    global _config
    if _config is None:
        _config = ExplorerConfig()
    return _config
