"""Milvus multi-tenant vector store — one namespace per project."""
from __future__ import annotations

import json
from dataclasses import dataclass

from explorer.config import get_config


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    collection: str


class MultiCollectionStore:
    """
    Manages Milvus collections namespaced as {project_slug}_{collection_type}.

    Uses MilvusClient directly (compatible with both Milvus Lite and standalone).
    Each collection uses COSINE similarity with 384-dim sentence-transformer embeddings.
    Schema: auto_id int pk + "vector" float array + "text" varchar + "metadata_json" varchar.
    """

    _TEXT_MAX_LEN = 65_535
    _META_MAX_LEN = 65_535

    def __init__(self) -> None:
        self._cfg = get_config()
        self._client = None  # lazy-initialized on first use

    def _get_client(self):
        if self._client is None:
            from pymilvus import MilvusClient
            self._client = MilvusClient(
                uri=self._cfg.milvus.uri,
                token=self._cfg.milvus.token or None,
            )
        return self._client

    def collection_name(self, project_slug: str, collection_type: str) -> str:
        return f"{project_slug}_{collection_type}"

    def _ensure_collection(self, collection: str) -> None:
        client = self._get_client()
        if client.has_collection(collection):
            return
        from pymilvus import DataType
        # Simple API: auto_id pk + "vector" field created automatically
        schema = client.create_schema(auto_id=True, enable_dynamic_field=False)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=self._cfg.embeddings.dimension)
        schema.add_field("text", DataType.VARCHAR, max_length=self._TEXT_MAX_LEN)
        schema.add_field("metadata_json", DataType.VARCHAR, max_length=self._META_MAX_LEN)
        # FLAT index: compatible with both Milvus Lite and standalone
        index_params = client.prepare_index_params()
        index_params.add_index(field_name="vector", metric_type="COSINE", index_type="FLAT")
        client.create_collection(
            collection_name=collection,
            schema=schema,
            index_params=index_params,
        )
        client.load_collection(collection)

    def search(
        self,
        query: str,
        collections: list[str],
        top_k: int | None = None,
        min_score: float | None = None,
    ) -> list[SearchResult]:
        from explorer.embeddings import embed_one
        k = top_k or self._cfg.rag.top_k
        threshold = min_score or self._cfg.rag.min_score
        client = self._get_client()
        q_vec = embed_one(query)
        results: list[SearchResult] = []
        for collection in collections:
            if not client.has_collection(collection):
                continue
            hits = client.search(
                collection_name=collection,
                data=[q_vec],
                limit=k,
                output_fields=["text", "metadata_json"],
                search_params={"metric_type": "COSINE"},
            )
            for hit in hits[0]:
                score = hit.get("distance", 0.0)
                if score < threshold:
                    continue
                entity = hit.get("entity", {})
                text = entity.get("text", "")
                try:
                    metadata = json.loads(entity.get("metadata_json", "{}"))
                except Exception:
                    metadata = {}
                results.append(SearchResult(text=text, score=score,
                                            metadata=metadata, collection=collection))
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:k]

    def insert(self, collection: str, texts: list[str], metadatas: list[dict]) -> int:
        from explorer.embeddings import embed_texts
        self._ensure_collection(collection)
        client = self._get_client()
        vectors = embed_texts(texts)
        data = [
            {
                "vector": vec,
                "text": text[: self._TEXT_MAX_LEN],
                "metadata_json": json.dumps(meta)[: self._META_MAX_LEN],
            }
            for vec, text, meta in zip(vectors, texts, metadatas)
        ]
        result = client.insert(collection_name=collection, data=data)
        client.flush(collection)
        return result.get("insert_count", len(data))

    def drop_collection(self, collection: str) -> None:
        client = self._get_client()
        if client.has_collection(collection):
            client.drop_collection(collection)

    def count(self, collection: str) -> int:
        client = self._get_client()
        if not client.has_collection(collection):
            return 0
        stats = client.get_collection_stats(collection)
        return int(stats.get("row_count", 0))
