"""Tests for X-API-Key authentication middleware."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

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


class TestApiKeyAuth:
    """X-API-Key middleware enforces authentication when configured."""

    def test_no_api_key_env_var_allows_all_requests(
        self, client: TestClient
    ) -> None:
        """Without MEMORY_API_KEY set, all requests pass through."""
        with patch.dict(os.environ, {}, clear=False):
            # Ensure key is not set
            os.environ.pop("MEMORY_API_KEY", None)
            resp = client.get("/memories/nonexistent")
        assert resp.status_code == 404  # Not 401

    def test_missing_key_returns_401(self, client: TestClient) -> None:
        """Request without header when MEMORY_API_KEY is set returns 401."""
        with patch.dict(os.environ, {"MEMORY_API_KEY": "secret-key-123"}):
            resp = client.get("/memories/nonexistent")
        assert resp.status_code == 401
        assert "Invalid or missing API key" in resp.json()["detail"]

    def test_wrong_key_returns_401(self, client: TestClient) -> None:
        """Request with wrong key returns 401."""
        with patch.dict(os.environ, {"MEMORY_API_KEY": "secret-key-123"}):
            resp = client.get(
                "/memories/nonexistent", headers={"X-API-Key": "wrong-key"}
            )
        assert resp.status_code == 401

    def test_correct_key_passes_through(self, client: TestClient) -> None:
        """Request with correct key passes through to handler."""
        with patch.dict(os.environ, {"MEMORY_API_KEY": "secret-key-123"}):
            resp = client.get(
                "/memories/nonexistent",
                headers={"X-API-Key": "secret-key-123"},
            )
        # Should get 404 (not found), not 401 (auth)
        assert resp.status_code == 404

    def test_health_bypasses_auth(self, client: TestClient) -> None:
        """Health endpoint is always accessible even with API key set."""
        with patch.dict(os.environ, {"MEMORY_API_KEY": "secret-key-123"}):
            resp = client.get("/health")
        assert resp.status_code == 200

    def test_empty_key_returns_401(self, client: TestClient) -> None:
        """Empty key header returns 401 when MEMORY_API_KEY is set."""
        with patch.dict(os.environ, {"MEMORY_API_KEY": "secret-key-123"}):
            resp = client.get(
                "/memories/nonexistent", headers={"X-API-Key": ""}
            )
        assert resp.status_code == 401
