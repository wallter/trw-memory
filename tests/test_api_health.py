"""Tests for the health check endpoint."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from trw_memory._version import __version__
from trw_memory.api.app import create_app
from trw_memory.api.deps import get_backend
from trw_memory.storage.sqlite_backend import SQLiteBackend


@pytest.fixture()
def _tmp_backend(tmp_path: Path) -> SQLiteBackend:
    """Create a thread-safe temporary SQLiteBackend for testing."""
    db_path = tmp_path / "test.db"
    backend = SQLiteBackend(db_path)
    backend._conn.close()
    backend._conn = sqlite3.connect(str(db_path), check_same_thread=False)
    return backend


@pytest.fixture()
def client(_tmp_backend: SQLiteBackend) -> TestClient:
    """Create a TestClient with a temporary backend."""
    app = create_app()
    app.dependency_overrides[get_backend] = lambda: _tmp_backend
    return TestClient(app, root_path="/v1")


class TestHealthEndpoint:
    """GET /health returns system health status."""

    def test_health_returns_200(self, client: TestClient) -> None:
        """Health endpoint responds with 200."""
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_response_structure(self, client: TestClient) -> None:
        """Health response contains status and components."""
        resp = client.get("/health")
        data = resp.json()
        assert data["status"] == "healthy"
        assert "components" in data
        assert data["components"]["database"] == "ok"
        assert data["components"]["version"] == __version__

    def test_health_includes_version(self, client: TestClient) -> None:
        """Health check includes the package version."""
        resp = client.get("/health")
        version = resp.json()["components"]["version"]
        assert version == __version__
        # Version is semver-like
        parts = version.split(".")
        assert len(parts) >= 2
