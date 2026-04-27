"""GitHub API wrapper — PyGitHub for REST, httpx for GraphQL batch queries."""
from __future__ import annotations

from functools import cached_property

from github import Github, GithubException
from github.Repository import Repository

from explorer.config import get_config


class GitHubClient:
    """
    Rate-limit-aware GitHub API client.

    Uses PyGitHub (REST) for standard metadata and file tree operations.
    GraphQL available via query() for complex stats queries that would
    otherwise require many REST calls (e.g. commit counts per contributor).
    """

    def __init__(self) -> None:
        cfg = get_config().github
        self._gh = Github(cfg.token or None, per_page=100)

    def get_repo(self, github_url: str) -> Repository:
        slug = self._url_to_slug(github_url)
        if "/" not in slug:
            raise ValueError(
                f"'{github_url}' looks like an organization or user URL, not a repository. "
                f"Please provide a full repo URL, e.g. https://github.com/{slug}/{slug}"
            )
        return self._gh.get_repo(slug)

    def download_zipball(self, repo: Repository, dest_dir: "Path") -> "Path":
        """
        Download entire repo as a single zipball (1 API call) and extract it.
        Returns the extracted repo root directory inside dest_dir.
        Far more rate-limit-friendly than fetching files individually.
        """
        import io
        import zipfile
        from pathlib import Path
        import requests

        branch = repo.default_branch
        url = f"https://api.github.com/repos/{repo.full_name}/zipball/{branch}"
        token = get_config().github.token
        headers = {"Authorization": f"token {token}"} if token else {}
        resp = requests.get(url, headers=headers, stream=True, timeout=300)
        resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            zf.extractall(dest_dir)
        # GitHub zips have a single top-level dir named "owner-repo-sha"
        subdirs = [d for d in Path(dest_dir).iterdir() if d.is_dir()]
        return subdirs[0] if subdirs else Path(dest_dir)

    def list_files(self, repo: Repository, path: str = "", recursive: bool = True) -> list[str]:
        """Return all file paths via git tree (1 API call). Kept for stats/incremental use."""
        try:
            tree = repo.get_git_tree(repo.default_branch, recursive=True)
            return [e.path for e in tree.tree if e.type == "blob"]
        except Exception:
            return []

    def get_file_content(self, repo: Repository, path: str) -> str | None:
        try:
            return repo.get_contents(path).decoded_content.decode("utf-8", errors="ignore")
        except GithubException:
            return None

    def get_default_branch(self, repo: Repository) -> str:
        return repo.default_branch

    def get_latest_commit_sha(self, repo: Repository) -> str:
        return repo.get_commits()[0].sha

    def check_rate_limit(self) -> dict:
        rate = self._gh.get_rate_limit()
        core = getattr(rate, "core", None) or rate.rate
        return {
            "remaining": core.remaining,
            "limit": core.limit,
            "reset_at": core.reset.isoformat(),
        }

    @staticmethod
    def _url_to_slug(url: str) -> str:
        url = url.rstrip("/")
        if url.endswith(".git"):
            url = url[:-4]
        if "github.com/" in url:
            return url.split("github.com/")[-1]
        return url
