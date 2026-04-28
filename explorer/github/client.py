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
        Retries up to 3 times on transient network errors.
        """
        import time
        import zipfile
        from pathlib import Path
        import requests
        from requests.exceptions import ConnectionError, SSLError, Timeout

        cfg = get_config().github
        branch = repo.default_branch
        url = f"https://api.github.com/repos/{repo.full_name}/zipball/{branch}"
        headers = {"Authorization": f"token {cfg.token}"} if cfg.token else {}
        zip_path = Path(dest_dir) / "_repo.zip"

        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                resp = requests.get(
                    url, headers=headers, stream=True,
                    timeout=cfg.clone_timeout_seconds,
                    verify=cfg.ssl_verify,
                )
                resp.raise_for_status()
                with open(zip_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        f.write(chunk)
                last_exc = None
                break
            except SSLError as exc:
                raise RuntimeError(
                    f"SSL error downloading repo — {exc}\n"
                    "Fixes:\n"
                    "  • pip install --upgrade certifi\n"
                    "  • set REQUESTS_CA_BUNDLE=/path/to/your-ca-bundle.pem in .env\n"
                    "  • set GITHUB__SSL_VERIFY=false in .env to skip verification (insecure)"
                ) from exc
            except (ConnectionError, Timeout) as exc:
                last_exc = exc
                if attempt < 2:
                    time.sleep(2 ** attempt)

        if last_exc:
            raise RuntimeError(
                f"Network error downloading repo after 3 attempts — {last_exc}"
            ) from last_exc

        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(dest_dir)
        zip_path.unlink(missing_ok=True)

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
