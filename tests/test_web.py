"""Tests for FastAPI web routes — projects, stats, query endpoints."""
from __future__ import annotations

import importlib.util
import json
import sqlite3
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

_pygithub_available = pytest.mark.skipif(
    importlib.util.find_spec("github") is None, reason="PyGitHub not installed"
)

from explorer.registry import Project, ProjectRegistry


# ── test app setup ─────────────────────────────────────────────────────────────

@pytest.fixture
def registry(tmp_path):
    r = ProjectRegistry(db_path=str(tmp_path / "test.db"))
    r.add(Project(
        slug="myproj",
        display_name="My Project",
        github_url="https://github.com/test/myproj",
        description="A test project",
        collections=["myproj_python_code", "myproj_markdown_docs"],
    ))
    return r


@pytest.fixture
def client(registry, monkeypatch):
    monkeypatch.setattr("explorer.registry.ProjectRegistry.__init__",
                        lambda self, db_path=None: setattr(self, "__dict__", registry.__dict__) or None)
    from explorer.web.app import app
    return TestClient(app)


# ── /health ───────────────────────────────────────────────────────────────────

class TestHealth:
    def test_health_returns_ok(self):
        from explorer.web.app import app
        c = TestClient(app)
        resp = c.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


# ── /api/projects ─────────────────────────────────────────────────────────────

class TestProjectsRouter:
    def test_list_projects(self, client):
        resp = client.get("/api/projects/")
        assert resp.status_code == 200
        projects = resp.json()
        assert len(projects) == 1
        assert projects[0]["slug"] == "myproj"
        assert projects[0]["display_name"] == "My Project"

    def test_list_projects_includes_required_fields(self, client):
        resp = client.get("/api/projects/")
        p = resp.json()[0]
        assert "slug" in p
        assert "display_name" in p
        assert "github_url" in p
        assert "status" in p
        assert "collections" in p
        assert "last_indexed_at" in p

    def test_get_project_found(self, client):
        resp = client.get("/api/projects/myproj")
        assert resp.status_code == 200
        assert resp.json()["slug"] == "myproj"

    def test_get_project_not_found(self, client):
        resp = client.get("/api/projects/ghost")
        assert resp.status_code == 404

    def test_delete_project(self, client):
        with patch("explorer.multi_collection_store.MultiCollectionStore") as mock_store:
            mock_store.return_value.drop_collection = MagicMock()
            resp = client.delete("/api/projects/myproj")
        assert resp.status_code == 200
        assert resp.json()["removed"] == "myproj"

    def test_delete_project_not_found(self, client):
        resp = client.delete("/api/projects/ghost")
        assert resp.status_code == 404

    @_pygithub_available
    def test_refresh_project_returns_started(self, client):
        with patch("explorer.ingestion.incremental.IncrementalIndexer") as mock_idx, \
             patch("explorer.query_cache.QueryCache") as mock_cache:
            mock_idx.return_value.refresh = MagicMock()
            mock_cache.return_value.invalidate_project = MagicMock(return_value=0)
            resp = client.post("/api/projects/myproj/refresh")
        assert resp.status_code == 200
        assert resp.json()["status"] == "refresh_started"

    def test_refresh_project_not_found(self, client):
        resp = client.post("/api/projects/ghost/refresh")
        assert resp.status_code == 404


# ── /api/stats ────────────────────────────────────────────────────────────────

def _insert_stats(db_path, slug):
    conn = sqlite3.connect(db_path)
    conn.execute("""
        INSERT INTO project_stats
        (project_slug, fetched_at, stars, forks, watchers, open_issues,
         contributors_count, commits_30d, commits_90d, releases_count,
         latest_release, latest_release_at, primary_language, language_breakdown)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (slug, "2024-06-01T12:00:00", 1200, 150, 1200, 30,
          25, 15, 48, 10, "v2.0.0", "2024-05-01T00:00:00",
          "Python", json.dumps({"Python": 50000})))
    conn.commit()
    conn.close()


class TestStatsRouter:
    def test_get_stats_not_found_project(self, client):
        resp = client.get("/api/stats/ghost")
        assert resp.status_code == 404

    def test_get_stats_no_data_returns_404(self, client):
        resp = client.get("/api/stats/myproj")
        assert resp.status_code == 404

    def test_get_stats_with_data(self, client, registry):
        _insert_stats(registry.db_path, "myproj")
        resp = client.get("/api/stats/myproj")
        assert resp.status_code == 200
        body = resp.json()
        assert body["slug"] == "myproj"
        assert body["stats"]["stars"] == 1200
        assert body["stats"]["primary_language"] == "Python"
        assert isinstance(body["stats"]["language_breakdown"], dict)

    def test_get_history_valid_metric(self, client, registry):
        _insert_stats(registry.db_path, "myproj")
        resp = client.get("/api/stats/myproj/history?metric=stars")
        assert resp.status_code == 200
        body = resp.json()
        assert body["metric"] == "stars"
        assert len(body["data"]) == 1
        assert body["data"][0]["value"] == 1200

    def test_get_history_invalid_metric(self, client):
        resp = client.get("/api/stats/myproj/history?metric=banana")
        assert resp.status_code == 400

    def test_get_history_not_found_project(self, client):
        resp = client.get("/api/stats/ghost/history")
        assert resp.status_code == 404


# ── /api/query ────────────────────────────────────────────────────────────────

class TestQueryRouter:
    def test_query_endpoint(self, client):
        with patch("explorer.rag_system.RAGSystem") as mock_rag_cls:
            mock_rag_cls.return_value.query.return_value = "mocked response"
            resp = client.post("/api/query/", json={"query": "what is this project?"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["response"] == "mocked response"
        assert "intent" in body

    def test_query_with_project_scope(self, client):
        with patch("explorer.rag_system.RAGSystem") as mock_rag_cls:
            rag_mock = mock_rag_cls.return_value
            rag_mock.query.return_value = "scoped response"
            resp = client.post("/api/query/", json={
                "query": "what is this project?",
                "project_slug": "myproj",
            })
        assert resp.status_code == 200
        rag_mock.query.assert_called_once_with("what is this project?", project_slug="myproj")
