"""Tests for background job management endpoints."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from trw_memory.api.app import create_app
from trw_memory.api.deps import get_backend
from trw_memory.api.router_jobs import reset_jobs
from trw_memory.storage.sqlite_backend import SQLiteBackend


@pytest.fixture()
def _tmp_backend(tmp_path: Path) -> SQLiteBackend:
    """Create a thread-safe temporary SQLiteBackend for testing."""
    db_path = tmp_path / "test.db"
    backend = SQLiteBackend(db_path)
    backend._conn.close()
    backend._conn = sqlite3.connect(str(db_path), check_same_thread=False)
    return backend


@pytest.fixture(autouse=True)
def _clear_jobs() -> None:
    """Reset the in-memory job store between tests."""
    reset_jobs()


@pytest.fixture()
def client(_tmp_backend: SQLiteBackend) -> TestClient:
    """Create a TestClient with a temporary backend."""
    app = create_app()
    app.dependency_overrides[get_backend] = lambda: _tmp_backend
    return TestClient(app, root_path="/v1")


class TestCreateJob:
    """POST /jobs submits a background job."""

    def test_create_consolidation_job(self, client: TestClient) -> None:
        """Creating a consolidation job returns 201."""
        resp = client.post("/jobs", json={"job_type": "consolidation"})
        assert resp.status_code == 201
        data = resp.json()
        assert data["id"].startswith("J-")
        assert data["job_type"] == "consolidation"

    def test_create_tier_sweep_job(self, client: TestClient) -> None:
        """Creating a tier_sweep job returns 201."""
        resp = client.post("/jobs", json={"job_type": "tier_sweep"})
        assert resp.status_code == 201
        assert resp.json()["job_type"] == "tier_sweep"

    def test_create_invalid_job_type_returns_422(
        self, client: TestClient
    ) -> None:
        """Invalid job type returns 422."""
        resp = client.post("/jobs", json={"job_type": "invalid_type"})
        assert resp.status_code == 422


class TestGetJob:
    """GET /jobs/{id} polls job status."""

    def test_get_existing_job(self, client: TestClient) -> None:
        """Retrieving an existing job returns its status."""
        created = client.post(
            "/jobs", json={"job_type": "consolidation"}
        ).json()
        resp = client.get(f"/jobs/{created['id']}")
        assert resp.status_code == 200
        assert resp.json()["id"] == created["id"]

    def test_get_unknown_job_returns_404(self, client: TestClient) -> None:
        """Requesting a non-existent job returns 404."""
        resp = client.get("/jobs/J-nonexistent99")
        assert resp.status_code == 404


class TestJobCompletion:
    """Jobs eventually reach completed or failed status."""

    def test_consolidation_completes(self, client: TestClient) -> None:
        """Consolidation job completes successfully."""
        created = client.post(
            "/jobs", json={"job_type": "consolidation"}
        ).json()
        # The job thread is joined in the create endpoint (timeout=2s),
        # so by the time we get the response it should be completed.
        resp = client.get(f"/jobs/{created['id']}")
        data = resp.json()
        assert data["status"] == "completed"
        assert "entries_scanned" in data["result"]

    def test_tier_sweep_completes(self, client: TestClient) -> None:
        """Tier sweep job completes successfully."""
        created = client.post(
            "/jobs", json={"job_type": "tier_sweep"}
        ).json()
        resp = client.get(f"/jobs/{created['id']}")
        data = resp.json()
        assert data["status"] == "completed"
        assert data["result"]["action"] == "tier_sweep_noop"

    def test_completed_job_has_timestamp(self, client: TestClient) -> None:
        """Completed jobs have a completed_at timestamp."""
        created = client.post(
            "/jobs", json={"job_type": "consolidation"}
        ).json()
        resp = client.get(f"/jobs/{created['id']}")
        assert resp.json()["completed_at"] is not None


class TestListJobs:
    """GET /jobs lists all recent jobs."""

    def test_list_empty(self, client: TestClient) -> None:
        """Empty job store returns empty list."""
        resp = client.get("/jobs")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_returns_all_jobs(self, client: TestClient) -> None:
        """List includes all created jobs."""
        client.post("/jobs", json={"job_type": "consolidation"})
        client.post("/jobs", json={"job_type": "tier_sweep"})
        resp = client.get("/jobs")
        assert len(resp.json()) == 2
