"""Orchestrates full ingestion for a project across all selected collection types."""
from __future__ import annotations

import base64
import os
import tempfile
from pathlib import PurePath

from rich.console import Console
from rich.progress import Progress

from config.collection_config import CollectionType
from explorer.multi_collection_store import MultiCollectionStore
from explorer.registry import ProjectRegistry, ProjectStatus


class IngestionPipeline:
    """
    Full ingestion flow for a project:
      1. Fetch repo file tree via GitHub API
      2. For each CollectionType: parse files → chunk → filter → embed → insert into Milvus
      3. Update registry with last_indexed_at and active collections

    See IncrementalIndexer for the partial-update (changed files only) path.
    """

    def __init__(self) -> None:
        self.registry = ProjectRegistry()
        self.store = MultiCollectionStore()
        self.console = Console()

    def run(
        self,
        project_slug: str,
        github_url: str,
        collection_types: list[CollectionType],
    ) -> None:
        self.registry.update_status(project_slug, ProjectStatus.INDEXING)
        active_collections = []

        try:
            with Progress(console=self.console) as progress:
                task = progress.add_task("Ingesting...", total=len(collection_types))
                for ctype in collection_types:
                    collection_name = f"{project_slug}_{ctype.name}"
                    count = self._ingest_collection(github_url, project_slug, collection_name, ctype)
                    if count > 0:
                        active_collections.append(collection_name)
                    progress.advance(task)

            self.registry.update_indexed_at(project_slug, active_collections)
            self.registry.update_status(project_slug, ProjectStatus.ACTIVE)
            self.console.print(
                f"[green]Ingestion complete. {len(active_collections)} collections populated.[/green]"
            )

        except Exception as e:
            self.registry.update_status(project_slug, ProjectStatus.ERROR, str(e))
            raise

    def _ingest_collection(
        self,
        github_url: str,
        project_slug: str,
        collection_name: str,
        ctype: CollectionType,
    ) -> int:
        _file_dispatch = {
            "python_code": self._ingest_code,
            "javascript_code": self._ingest_code,
            "java_code": self._ingest_code,
            "go_code": self._ingest_code,
            "markdown_docs": self._ingest_markdown,
            "web_docs": self._ingest_web_docs,
            "api_reference": self._ingest_api_specs,
            "examples": self._ingest_examples,
            "pdfs": self._ingest_pdfs,
        }

        # Unknown collection type — bail before any GitHub API call
        if ctype.name != "release_notes" and ctype.name not in _file_dispatch:
            return 0

        from explorer.github.client import GitHubClient
        from explorer.ingestion.data_prep import DataPrep

        client = GitHubClient()
        repo = client.get_repo(github_url)
        data_prep = DataPrep()

        # release_notes come from the GitHub releases API, not the file tree
        if ctype.name == "release_notes":
            chunks = self._ingest_releases(repo, project_slug, ctype)
        else:
            chunks = _file_dispatch[ctype.name](repo, client, project_slug, ctype)

        chunks = data_prep.filter(chunks)
        if not chunks:
            return 0

        texts = [c.text for c in chunks]
        metadatas = [c.metadata for c in chunks]
        return self.store.insert(collection_name, texts, metadatas)

    # ── per-collection-type helpers ───────────────────────────────────────────

    def _repo_files(self, repo, client, extensions: list[str]) -> list[tuple[str, str]]:
        """Return (path, content) pairs for all repo files matching extensions."""
        result = []
        for path in client.list_files(repo):
            if PurePath(path).suffix.lower() in extensions:
                content = client.get_file_content(repo, path)
                if content:
                    result.append((path, content))
        return result

    def _ingest_code(self, repo, client, project_slug: str, ctype: CollectionType) -> list:
        from explorer.ingestion.code_parser import CodeParser
        parser = CodeParser(ctype.chunk_size, ctype.chunk_overlap)
        chunks = []
        for path, content in self._repo_files(repo, client, ctype.file_extensions):
            chunks.extend(parser.parse(path, content, project_slug))
        return chunks

    def _ingest_markdown(self, repo, client, project_slug: str, ctype: CollectionType) -> list:
        from explorer.ingestion.doc_parser import DocParser
        parser = DocParser(ctype.chunk_size, ctype.chunk_overlap)
        chunks = []
        for path, content in self._repo_files(repo, client, ctype.file_extensions):
            chunks.extend(parser.parse_markdown(content, path, project_slug))
        return chunks

    def _ingest_web_docs(self, repo, client, project_slug: str, ctype: CollectionType) -> list:
        import re
        from explorer.ingestion.doc_parser import DocChunk, DocParser
        parser = DocParser(ctype.chunk_size, ctype.chunk_overlap)
        chunks = []
        for path, content in self._repo_files(repo, client, ctype.file_extensions):
            text = re.sub(r"<[^>]+>", " ", content)
            text = re.sub(r"\s+", " ", text).strip()
            for chunk_text in parser._fixed_window(text):
                chunks.append(DocChunk(
                    text=chunk_text,
                    metadata={"file_path": path, "project_slug": project_slug, "type": "web"},
                ))
        return chunks

    def _ingest_api_specs(self, repo, client, project_slug: str, ctype: CollectionType) -> list:
        from explorer.ingestion.api_parser import APIParser
        parser = APIParser()
        chunks = []
        for path, content in self._repo_files(repo, client, ctype.file_extensions):
            parsed = parser.parse(path, content, project_slug)
            if parsed:  # non-empty → actual OpenAPI spec
                chunks.extend(parsed)
        return chunks

    def _ingest_examples(self, repo, client, project_slug: str, ctype: CollectionType) -> list:
        from explorer.ingestion.code_parser import CodeParser
        from explorer.ingestion.notebook_parser import NotebookParser
        code_parser = CodeParser(ctype.chunk_size, ctype.chunk_overlap)
        nb_parser = NotebookParser()
        chunks = []
        for path, content in self._repo_files(repo, client, ctype.file_extensions):
            if path.endswith(".ipynb"):
                with tempfile.NamedTemporaryFile(mode="w", suffix=".ipynb", delete=False) as f:
                    f.write(content)
                    tmp = f.name
                try:
                    chunks.extend(nb_parser.parse(tmp, project_slug))
                finally:
                    os.unlink(tmp)
            else:
                chunks.extend(code_parser.parse(path, content, project_slug))
        return chunks

    def _ingest_pdfs(self, repo, client, project_slug: str, ctype: CollectionType) -> list:
        from explorer.ingestion.doc_parser import DocParser
        parser = DocParser(ctype.chunk_size, ctype.chunk_overlap)
        chunks = []
        for path in client.list_files(repo):
            if PurePath(path).suffix.lower() != ".pdf":
                continue
            try:
                item = repo.get_contents(path)
                pdf_bytes = base64.b64decode(item.content)
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
                    f.write(pdf_bytes)
                    tmp = f.name
                try:
                    chunks.extend(parser.parse_pdf(tmp, project_slug))
                finally:
                    os.unlink(tmp)
            except Exception:
                pass
        return chunks

    def _ingest_releases(self, repo, project_slug: str, ctype: CollectionType) -> list:
        from explorer.ingestion.doc_parser import DocChunk, DocParser
        parser = DocParser(ctype.chunk_size, ctype.chunk_overlap)
        chunks = []
        try:
            for release in repo.get_releases():
                body = release.body or ""
                if not body.strip():
                    continue
                date_str = (
                    release.published_at.strftime("%Y-%m-%d")
                    if release.published_at
                    else ""
                )
                header = f"## {release.tag_name}" + (f" ({date_str})" if date_str else "")
                text = f"{header}\n\n{body}"
                for chunk_text in parser._fixed_window(text):
                    chunks.append(DocChunk(
                        text=chunk_text,
                        metadata={
                            "project_slug": project_slug,
                            "tag": release.tag_name,
                            "published_at": release.published_at.isoformat() if release.published_at else "",
                            "type": "release_notes",
                        },
                    ))
        except Exception:
            pass
        return chunks
