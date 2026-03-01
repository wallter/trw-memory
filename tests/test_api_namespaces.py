"""Tests for namespace management endpoints."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from trw_memory.api.app import create_app
from trw_memory.api.deps import get_backend
from trw_memory.models.memory import MemoryEntry
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


def _seed_entry(
    backend: SQLiteBackend,
    entry_id: str,
    namespace: str = "default",
    content: str = "test",
) -> None:
    """Insert a memory entry directly into the backend."""
    now = datetime.now(timezone.utc)
    entry = MemoryEntry(
        id=entry_id,
        content=content,
        namespace=namespace,
        created_at=now,
        updated_at=now,
    )
    backend.store(entry)


class TestCreateNamespace:
    """POST /namespaces registers a namespace."""

    def test_create_valid_namespace(self, client: TestClient) -> None:
        """Valid namespace returns 201 with info."""
        resp = client.post(
            "/namespaces", json={"namespace": "project:test-repo"}
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["namespace"] == "project:test-repo"
        assert data["entry_count"] == 0

    def test_create_invalid_namespace_returns_422(
        self, client: TestClient
    ) -> None:
        """Invalid namespace pattern returns 422."""
        resp = client.post(
            "/namespaces", json={"namespace": "bad-format"}
        )
        assert resp.status_code == 422

    def test_create_idempotent(self, client: TestClient) -> None:
        """Registering the same namespace twice is OK."""
        client.post("/namespaces", json={"namespace": "global"})
        resp = client.post("/namespaces", json={"namespace": "global"})
        assert resp.status_code == 201


class TestListNamespaces:
    """GET /namespaces lists all namespaces with entries."""

    def test_list_empty(self, client: TestClient) -> None:
        """Empty store returns empty namespace list."""
        resp = client.get("/namespaces")
        assert resp.status_code == 200
        assert resp.json()["namespaces"] == []

    def test_list_after_entries_created(
        self, client: TestClient, _tmp_backend: SQLiteBackend
    ) -> None:
        """Namespaces with entries appear in the list."""
        _seed_entry(_tmp_backend, "M-001", namespace="project:repo-a")
        _seed_entry(_tmp_backend, "M-002", namespace="project:repo-b")

        resp = client.get("/namespaces")
        data = resp.json()
        ns_names = [n["namespace"] for n in data["namespaces"]]
        assert "project:repo-a" in ns_names
        assert "project:repo-b" in ns_names

    def test_list_includes_entry_count(
        self, client: TestClient, _tmp_backend: SQLiteBackend
    ) -> None:
        """Each namespace includes its entry count."""
        _seed_entry(_tmp_backend, "M-001", namespace="default")
        _seed_entry(_tmp_backend, "M-002", namespace="default")
        _seed_entry(_tmp_backend, "M-003", namespace="project:solo")

        resp = client.get("/namespaces")
        data = resp.json()
        ns_map = {n["namespace"]: n["entry_count"] for n in data["namespaces"]}
        assert ns_map["default"] == 2
        assert ns_map["project:solo"] == 1


class TestGetNamespace:
    """GET /namespaces/{ns} returns namespace details."""

    def test_get_existing_namespace(
        self, client: TestClient, _tmp_backend: SQLiteBackend
    ) -> None:
        """Get namespace details for one with entries."""
        _seed_entry(_tmp_backend, "M-001", namespace="default")
        resp = client.get("/namespaces/default")
        assert resp.status_code == 200
        assert resp.json()["namespace"] == "default"
        assert resp.json()["entry_count"] == 1

    def test_get_empty_namespace_returns_404(
        self, client: TestClient
    ) -> None:
        """A namespace with no entries returns 404."""
        resp = client.get("/namespaces/default")
        assert resp.status_code == 404

    def test_get_invalid_namespace_returns_422(
        self, client: TestClient
    ) -> None:
        """Invalid namespace pattern returns 422."""
        resp = client.get("/namespaces/bad-format")
        assert resp.status_code == 422


class TestDeleteNamespace:
    """DELETE /namespaces/{ns} removes all entries in a namespace."""

    def test_delete_removes_entries(
        self, client: TestClient, _tmp_backend: SQLiteBackend
    ) -> None:
        """Deleting a namespace removes all its entries."""
        _seed_entry(_tmp_backend, "M-001", namespace="project:to-delete")
        _seed_entry(_tmp_backend, "M-002", namespace="project:to-delete")

        resp = client.delete("/namespaces/project:to-delete")
        assert resp.status_code == 204

        # Verify entries are gone
        assert _tmp_backend.count(namespace="project:to-delete") == 0

    def test_delete_does_not_affect_other_namespaces(
        self, client: TestClient, _tmp_backend: SQLiteBackend
    ) -> None:
        """Deleting one namespace leaves others intact."""
        _seed_entry(_tmp_backend, "M-001", namespace="project:keep")
        _seed_entry(_tmp_backend, "M-002", namespace="project:remove")

        client.delete("/namespaces/project:remove")
        assert _tmp_backend.count(namespace="project:keep") == 1

    def test_delete_invalid_namespace_returns_422(
        self, client: TestClient
    ) -> None:
        """Invalid namespace returns 422."""
        resp = client.delete("/namespaces/bad-ns")
        assert resp.status_code == 422
