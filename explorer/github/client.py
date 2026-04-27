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
        return self._gh.get_repo(slug)

    def list_files(self, repo: Repository, path: str = "", recursive: bool = True) -> list[str]:
        """Return all file paths in the repo (shallow or recursive)."""
        contents = repo.get_contents(path)
        files = []
        while contents:
            item = contents.pop(0)
            if item.type == "dir":
                if recursive:
                    contents.extend(repo.get_contents(item.path))
            else:
                files.append(item.path)
        return files

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
