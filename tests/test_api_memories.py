"""Tests for memory CRUD endpoints."""

from __future__ import annotations

import sqlite3
from pathlib import Path

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


def _create_memory(
    client: TestClient,
    content: str = "test memory",
    namespace: str = "default",
    **kwargs: object,
) -> dict[str, object]:
    """Helper to create a memory entry and return the response dict."""
    body: dict[str, object] = {
        "content": content,
        "namespace": namespace,
        **kwargs,
    }
    resp = client.post("/memories", json=body)
    assert resp.status_code == 201, resp.text
    return resp.json()


class TestCreateMemory:
    """POST /memories creates a new entry."""

    def test_create_returns_201(self, client: TestClient) -> None:
        """Successful creation returns 201."""
        resp = client.post(
            "/memories", json={"content": "Test learning"}
        )
        assert resp.status_code == 201

    def test_create_returns_id(self, client: TestClient) -> None:
        """Created memory has an M- prefixed ID."""
        data = _create_memory(client, content="New memory")
        assert data["id"].startswith("M-")
        assert len(data["id"]) > 3

    def test_create_stores_content(self, client: TestClient) -> None:
        """Content is persisted correctly."""
        data = _create_memory(client, content="Important discovery")
        assert data["content"] == "Important discovery"

    def test_create_with_all_fields(self, client: TestClient) -> None:
        """All optional fields are stored."""
        data = _create_memory(
            client,
            content="Full entry",
            detail="Extended detail text",
            tags=["testing", "gotcha"],
            evidence=["file.py:42"],
            importance=0.8,
            namespace="default",
            source="api",
            metadata={"key": "value"},
        )
        assert data["detail"] == "Extended detail text"
        assert data["tags"] == ["testing", "gotcha"]
        assert data["evidence"] == ["file.py:42"]
        assert data["importance"] == 0.8
        assert data["source"] == "api"
        assert data["metadata"] == {"key": "value"}

    def test_create_default_namespace(self, client: TestClient) -> None:
        """Default namespace is 'default' when not specified."""
        data = _create_memory(client, content="Namespaced")
        assert data["namespace"] == "default"

    def test_create_with_custom_namespace(self, client: TestClient) -> None:
        """Custom valid namespace is accepted."""
        data = _create_memory(
            client, content="Project scoped", namespace="project:my-repo"
        )
        assert data["namespace"] == "project:my-repo"

    def test_create_invalid_namespace_returns_422(
        self, client: TestClient
    ) -> None:
        """Invalid namespace returns 422."""
        resp = client.post(
            "/memories",
            json={"content": "Bad ns", "namespace": "invalid-ns"},
        )
        assert resp.status_code == 422

    def test_create_timestamps_set(self, client: TestClient) -> None:
        """Created entry has timestamps."""
        data = _create_memory(client, content="Timestamped")
        assert data["created_at"] is not None
        assert data["updated_at"] is not None


class TestGetMemory:
    """GET /memories/{id} retrieves an entry."""

    def test_get_existing_entry(self, client: TestClient) -> None:
        """Retrieving an existing entry returns 200."""
        created = _create_memory(client, content="To be retrieved")
        resp = client.get(f"/memories/{created['id']}")
        assert resp.status_code == 200
        assert resp.json()["content"] == "To be retrieved"

    def test_get_unknown_returns_404(self, client: TestClient) -> None:
        """Requesting a non-existent ID returns 404."""
        resp = client.get("/memories/M-nonexistent99")
        assert resp.status_code == 404

    def test_get_returns_all_fields(self, client: TestClient) -> None:
        """Retrieved entry includes all expected fields."""
        created = _create_memory(
            client,
            content="Full fields",
            detail="Some detail",
            tags=["tag1"],
        )
        resp = client.get(f"/memories/{created['id']}")
        data = resp.json()
        assert data["id"] == created["id"]
        assert data["content"] == "Full fields"
        assert data["detail"] == "Some detail"
        assert data["tags"] == ["tag1"]
        assert "status" in data
        assert "importance" in data


class TestUpdateMemory:
    """PATCH /memories/{id} updates fields."""

    def test_update_content(self, client: TestClient) -> None:
        """Updating content returns the new value."""
        created = _create_memory(client, content="Original")
        resp = client.patch(
            f"/memories/{created['id']}",
            json={"content": "Updated content"},
        )
        assert resp.status_code == 200
        assert resp.json()["content"] == "Updated content"

    def test_update_importance(self, client: TestClient) -> None:
        """Importance can be updated via PATCH."""
        created = _create_memory(client, content="Scored")
        resp = client.patch(
            f"/memories/{created['id']}", json={"importance": 0.9}
        )
        assert resp.status_code == 200
        assert resp.json()["importance"] == 0.9

    def test_update_status(self, client: TestClient) -> None:
        """Status can be changed to resolved."""
        created = _create_memory(client, content="To resolve")
        resp = client.patch(
            f"/memories/{created['id']}", json={"status": "resolved"}
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "resolved"

    def test_update_tags(self, client: TestClient) -> None:
        """Tags can be replaced."""
        created = _create_memory(client, content="Tagged", tags=["old"])
        resp = client.patch(
            f"/memories/{created['id']}", json={"tags": ["new", "tags"]}
        )
        assert resp.status_code == 200
        assert resp.json()["tags"] == ["new", "tags"]

    def test_update_nonexistent_returns_404(
        self, client: TestClient
    ) -> None:
        """Updating a non-existent entry returns 404."""
        resp = client.patch(
            "/memories/M-nonexistent99", json={"content": "Nope"}
        )
        assert resp.status_code == 404

    def test_update_preserves_unchanged_fields(
        self, client: TestClient
    ) -> None:
        """Fields not in the PATCH body are preserved."""
        created = _create_memory(
            client, content="Keep this", detail="Also keep"
        )
        resp = client.patch(
            f"/memories/{created['id']}", json={"importance": 0.7}
        )
        data = resp.json()
        assert data["content"] == "Keep this"
        assert data["detail"] == "Also keep"


class TestDeleteMemory:
    """DELETE /memories/{id} removes an entry."""

    def test_delete_existing_entry(self, client: TestClient) -> None:
        """Deleting an existing entry returns 204."""
        created = _create_memory(client, content="To delete")
        resp = client.delete(f"/memories/{created['id']}")
        assert resp.status_code == 204

    def test_delete_makes_entry_gone(self, client: TestClient) -> None:
        """After deletion, GET returns 404."""
        created = _create_memory(client, content="Gone soon")
        client.delete(f"/memories/{created['id']}")
        resp = client.get(f"/memories/{created['id']}")
        assert resp.status_code == 404

    def test_delete_nonexistent_returns_404(
        self, client: TestClient
    ) -> None:
        """Deleting a non-existent entry returns 404."""
        resp = client.delete("/memories/M-nonexistent99")
        assert resp.status_code == 404


class TestSearchMemories:
    """POST /memories/search returns matching entries."""

    def test_search_empty_store(self, client: TestClient) -> None:
        """Searching an empty store returns empty list."""
        resp = client.post("/memories/search", json={"query": ""})
        assert resp.status_code == 200
        assert resp.json() == []

    def test_search_by_content(self, client: TestClient) -> None:
        """Search finds entries matching content."""
        _create_memory(client, content="pytest is great for testing")
        _create_memory(client, content="unrelated entry")
        resp = client.post(
            "/memories/search", json={"query": "pytest"}
        )
        results = resp.json()
        assert len(results) >= 1
        assert any("pytest" in r["content"] for r in results)

    def test_search_by_namespace(self, client: TestClient) -> None:
        """Search can filter by namespace."""
        _create_memory(
            client, content="NS-A entry", namespace="project:repo-a"
        )
        _create_memory(
            client, content="NS-B entry", namespace="project:repo-b"
        )
        resp = client.post(
            "/memories/search",
            json={"query": "", "namespace": "project:repo-a"},
        )
        results = resp.json()
        assert len(results) == 1
        assert results[0]["namespace"] == "project:repo-a"

    def test_search_by_tags(self, client: TestClient) -> None:
        """Search can filter by tags."""
        _create_memory(
            client, content="Tagged entry", tags=["python", "testing"]
        )
        _create_memory(client, content="Untagged entry")
        resp = client.post(
            "/memories/search",
            json={"query": "", "tags": ["python"]},
        )
        results = resp.json()
        assert len(results) >= 1
        assert all("python" in r["tags"] for r in results)

    def test_search_by_min_importance(self, client: TestClient) -> None:
        """Search can filter by minimum importance."""
        _create_memory(client, content="Low importance", importance=0.2)
        _create_memory(client, content="High importance", importance=0.9)
        resp = client.post(
            "/memories/search",
            json={"query": "", "min_importance": 0.5},
        )
        results = resp.json()
        assert len(results) >= 1
        assert all(r["importance"] >= 0.5 for r in results)

    def test_search_with_limit(self, client: TestClient) -> None:
        """Search respects the limit parameter."""
        for i in range(5):
            _create_memory(client, content=f"Entry {i}")
        resp = client.post(
            "/memories/search", json={"query": "", "limit": 2}
        )
        results = resp.json()
        assert len(results) <= 2

    def test_search_by_status(self, client: TestClient) -> None:
        """Search can filter by status."""
        _create_memory(client, content="Active entry")
        resolved = _create_memory(client, content="Resolved entry")
        client.patch(
            f"/memories/{resolved['id']}", json={"status": "resolved"}
        )
        resp = client.post(
            "/memories/search",
            json={"query": "", "status": "resolved"},
        )
        results = resp.json()
        assert len(results) >= 1
        assert all(r["status"] == "resolved" for r in results)

    def test_search_invalid_status_returns_422(
        self, client: TestClient
    ) -> None:
        """Invalid status value returns 422."""
        resp = client.post(
            "/memories/search",
            json={"query": "", "status": "invalid_status"},
        )
        assert resp.status_code == 422

    def test_namespace_scoping_works(self, client: TestClient) -> None:
        """Entries in different namespaces are isolated in search results."""
        _create_memory(
            client,
            content="same keyword project-a",
            namespace="project:alpha",
        )
        _create_memory(
            client,
            content="same keyword project-b",
            namespace="project:beta",
        )
        resp_a = client.post(
            "/memories/search",
            json={"query": "keyword", "namespace": "project:alpha"},
        )
        resp_b = client.post(
            "/memories/search",
            json={"query": "keyword", "namespace": "project:beta"},
        )
        assert len(resp_a.json()) == 1
        assert len(resp_b.json()) == 1
        assert resp_a.json()[0]["namespace"] == "project:alpha"
        assert resp_b.json()[0]["namespace"] == "project:beta"
