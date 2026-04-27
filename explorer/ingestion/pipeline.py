"""Orchestrates full ingestion for a project across all selected collection types."""
from __future__ import annotations

import tempfile
from pathlib import Path

from rich.console import Console
from rich.progress import Progress

from config.collection_config import CollectionType
from explorer.multi_collection_store import MultiCollectionStore
from explorer.registry import ProjectRegistry, ProjectStatus


class IngestionPipeline:
    """
    Full ingestion flow for a project:
      1. Download repo as a single zipball (1 GitHub API call)
      2. For each CollectionType: parse files from disk → chunk → filter → embed → insert
      3. Update registry with last_indexed_at and active collections

    Downloading via zipball avoids per-file API calls, which hit rate limits on
    large repos (e.g. ODPI/egeria has thousands of files).

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
        from explorer.github.client import GitHubClient

        self.registry.update_status(project_slug, ProjectStatus.INDEXING)
        active_collections = []

        client = GitHubClient()
        repo = client.get_repo(github_url)

        try:
            with tempfile.TemporaryDirectory() as tmp:
                self.console.print("[cyan]Downloading repository...[/cyan]")
                local_root = client.download_zipball(repo, Path(tmp))
                self.console.print(f"[cyan]Extracted to {local_root.name}[/cyan]")

                with Progress(console=self.console) as progress:
                    task = progress.add_task("Ingesting...", total=len(collection_types))
                    for ctype in collection_types:
                        collection_name = f"{project_slug}_{ctype.name}"
                        count = self._ingest_collection(
                            repo, project_slug, collection_name, ctype, local_root
                        )
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
        repo,
        project_slug: str,
        collection_name: str,
        ctype: CollectionType,
        local_root: Path | None = None,
    ) -> int:
        """
        Ingest one collection type. local_root is the extracted repo directory;
        if None (incremental single-collection re-index), downloads on the spot.
        """
        from explorer.ingestion.data_prep import DataPrep

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

        if ctype.name != "release_notes" and ctype.name not in _file_dispatch:
            return 0

        # release_notes use the GitHub releases API, not file content
        if ctype.name == "release_notes":
            chunks = self._ingest_releases(repo, project_slug, ctype)
        else:
            if local_root is None:
                # Fallback for incremental single-collection calls
                from explorer.github.client import GitHubClient
                client = GitHubClient()
                with tempfile.TemporaryDirectory() as tmp:
                    local_root = client.download_zipball(repo, Path(tmp))
                    chunks = _file_dispatch[ctype.name](local_root, project_slug, ctype)
            else:
                chunks = _file_dispatch[ctype.name](local_root, project_slug, ctype)

        chunks = DataPrep().filter(chunks)
        if not chunks:
            return 0

        return self.store.insert(collection_name, [c.text for c in chunks],
                                 [c.metadata for c in chunks])

    # ── local file helpers ────────────────────────────────────────────────────

    def _local_files(self, local_root: Path, extensions: list[str]) -> list[tuple[str, str]]:
        """Walk the extracted repo and return (relative_path, content) for matching files."""
        results = []
        for p in local_root.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() not in extensions:
                continue
            try:
                content = p.read_text(encoding="utf-8", errors="ignore")
                results.append((str(p.relative_to(local_root)), content))
            except Exception:
                pass
        return results

    # ── per-collection-type ingestors ─────────────────────────────────────────

    def _ingest_code(self, local_root: Path, project_slug: str, ctype: CollectionType) -> list:
        from explorer.ingestion.code_parser import CodeParser
        parser = CodeParser(ctype.chunk_size, ctype.chunk_overlap)
        chunks = []
        for path, content in self._local_files(local_root, ctype.file_extensions):
            chunks.extend(parser.parse(path, content, project_slug))
        return chunks

    def _ingest_markdown(self, local_root: Path, project_slug: str, ctype: CollectionType) -> list:
        from explorer.ingestion.doc_parser import DocParser
        parser = DocParser(ctype.chunk_size, ctype.chunk_overlap)
        chunks = []
        for path, content in self._local_files(local_root, ctype.file_extensions):
            chunks.extend(parser.parse_markdown(content, path, project_slug))
        return chunks

    def _ingest_web_docs(self, local_root: Path, project_slug: str, ctype: CollectionType) -> list:
        import re
        from explorer.ingestion.doc_parser import DocChunk, DocParser
        parser = DocParser(ctype.chunk_size, ctype.chunk_overlap)
        chunks = []
        for path, content in self._local_files(local_root, ctype.file_extensions):
            text = re.sub(r"<[^>]+>", " ", content)
            text = re.sub(r"\s+", " ", text).strip()
            for chunk_text in parser._fixed_window(text):
                chunks.append(DocChunk(
                    text=chunk_text,
                    metadata={"file_path": path, "project_slug": project_slug, "type": "web"},
                ))
        return chunks

    def _ingest_api_specs(self, local_root: Path, project_slug: str, ctype: CollectionType) -> list:
        from explorer.ingestion.api_parser import APIParser
        parser = APIParser()
        chunks = []
        for path, content in self._local_files(local_root, ctype.file_extensions):
            parsed = parser.parse(path, content, project_slug)
            if parsed:
                chunks.extend(parsed)
        return chunks

    def _ingest_examples(self, local_root: Path, project_slug: str, ctype: CollectionType) -> list:
        from explorer.ingestion.code_parser import CodeParser
        from explorer.ingestion.notebook_parser import NotebookParser
        code_parser = CodeParser(ctype.chunk_size, ctype.chunk_overlap)
        nb_parser = NotebookParser()
        chunks = []
        for path, content in self._local_files(local_root, ctype.file_extensions):
            if path.endswith(".ipynb"):
                # NotebookParser needs a file path; write to a temp file
                import tempfile, os
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

    def _ingest_pdfs(self, local_root: Path, project_slug: str, ctype: CollectionType) -> list:
        from explorer.ingestion.doc_parser import DocParser
        parser = DocParser(ctype.chunk_size, ctype.chunk_overlap)
        chunks = []
        for path in local_root.rglob("*.pdf"):
            try:
                chunks.extend(parser.parse_pdf(str(path), project_slug))
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
                    if release.published_at else ""
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
