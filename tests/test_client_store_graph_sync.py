from __future__ import annotations

import asyncio
import sqlite3
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, patch

import pytest

from trw_memory.client import MemoryClient
from trw_memory.graph import update_entry_graph, wait_for_graph_updates
from trw_memory.integrations._backend import create_backend_from_config
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.storage.sqlite_backend import SQLiteBackend


class TestDbapiGraphCompatibility:
    def test_graph_update_accepts_dbapi_compatible_connection_proxy(self, tmp_path: Path) -> None:
        """Graph enrichment supports SQLCipher-like non-stdlib connection types."""

        class ConnectionProxy:
            def __init__(self, delegate: sqlite3.Connection) -> None:
                self._delegate = delegate

            def execute(self, *args: object, **kwargs: object) -> sqlite3.Cursor:
                return self._delegate.execute(*args, **kwargs)

            def commit(self) -> None:
                self._delegate.commit()

        backend = SQLiteBackend(tmp_path / "graph.db")
        candidate = MemoryEntry(
            id="M-proxy-candidate",
            content="candidate",
            namespace="default",
            tags=["python", "graph"],
        )
        target = MemoryEntry(
            id="M-proxy-target",
            content="target",
            namespace="default",
            tags=["python", "graph"],
        )
        real_connection = backend._conn
        try:
            backend.store(candidate)
            backend.store(target)
            backend._conn = ConnectionProxy(real_connection)
            result = update_entry_graph(target, backend)
            backend._conn = real_connection

            # PRD-CORE-245 FR07: tag co-occurrence is derived, not materialised,
            # so the graph pass reports no tag edge count at all.
            assert "tag_edges" not in result
            rows = real_connection.execute(
                "SELECT source_id, target_id FROM memory_graph_edges WHERE edge_type = 'tag_cooccurrence' "
                "ORDER BY source_id, target_id"
            ).fetchall()
            assert [tuple(row) for row in rows] == []
            # The relation still exists — it is an inverted-index lookup now, and
            # the proxy connection is what proves the write path reached SQLite.
            postings = real_connection.execute(
                "SELECT COUNT(*) FROM memory_tags WHERE entry_id IN (?, ?)",
                ("M-proxy-candidate", "M-proxy-target"),
            ).fetchone()[0]
            assert postings > 0
        finally:
            backend._conn = real_connection
            backend.close()


class TestRbacEnforcement:
    async def test_store_populates_similarity_and_tag_edges(self, client: MemoryClient) -> None:
        backend = cast("SQLiteBackend", client._get_backend())
        vector = [1.0, *([0.0] * (backend._dim - 1))]
        backend.store(
            MemoryEntry(
                id="M-existing",
                content="existing memory",
                namespace="default",
                tags=["python", "async", "sqlite"],
            )
        )
        backend.upsert_vector("M-existing", vector, namespace="default")

        fake_embedder = MagicMock()
        fake_embedder.embed.return_value = vector

        with patch.object(client, "_get_embedder", return_value=fake_embedder):
            stored = await client.store("new memory", tags=["python", "async", "graph"])
            await asyncio.to_thread(wait_for_graph_updates)

        edge_rows = backend._conn.execute(
            "SELECT edge_type, COUNT(*) FROM memory_graph_edges WHERE source_id IN (?, ?) OR target_id IN (?, ?) "
            "GROUP BY edge_type ORDER BY edge_type",
            (stored["memory_id"], "M-existing", stored["memory_id"], "M-existing"),
        ).fetchall()
        # PRD-CORE-245 FR07: no tag_cooccurrence row is ever written now.
        expected = [("similarity", 2)] if backend._vec_available else []
        assert [tuple(row) for row in edge_rows] == expected

    async def test_store_without_embedder_still_populates_tag_edges(self, client: MemoryClient) -> None:
        backend = cast("SQLiteBackend", client._get_backend())
        backend.store(
            MemoryEntry(
                id="M-existing-tags",
                content="existing memory",
                namespace="default",
                tags=["python", "async", "sqlite"],
            )
        )

        with patch.object(client, "_get_embedder", return_value=None):
            stored = await client.store("new memory", tags=["python", "async", "graph"])
            await asyncio.to_thread(wait_for_graph_updates)

        edge_rows = backend._conn.execute(
            "SELECT edge_type, COUNT(*) FROM memory_graph_edges WHERE source_id IN (?, ?) OR target_id IN (?, ?) "
            "GROUP BY edge_type ORDER BY edge_type",
            (stored["memory_id"], "M-existing-tags", stored["memory_id"], "M-existing-tags"),
        ).fetchall()
        # PRD-CORE-245 FR07: the tag relation lives in memory_tags now, so the
        # edge table stays empty and the inverted index carries the postings.
        assert [tuple(row) for row in edge_rows] == []
        tag_rows = backend._conn.execute(
            "SELECT COUNT(*) FROM memory_tags WHERE entry_id IN (?, ?)",
            (stored["memory_id"], "M-existing-tags"),
        ).fetchone()[0]
        assert tag_rows == 6

    async def test_store_registers_team_namespace_lifecycle_row(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "storage"))
        monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "sqlite")
        client = MemoryClient(namespace="team:sprint-24", mode="local")

        stored = await client.store("team finding", tags=["team"])
        backend = cast("SQLiteBackend", client._get_backend())
        row = backend._conn.execute(
            "SELECT team_id, expires_at, status FROM memory_namespaces WHERE namespace_id = ?",
            ("team:sprint-24",),
        ).fetchone()

        assert stored["status"] == "stored"
        assert tuple(row) == ("sprint-24", None, "active")
        await client.close()

    async def test_store_returns_before_graph_update_finishes(
        self,
        client: MemoryClient,
    ) -> None:
        release_update = threading.Event()

        def slow_graph_update(*_args: object, **_kwargs: object) -> dict[str, int]:
            release_update.wait(timeout=1.0)
            return {"similarity_edges": 0, "tag_edges": 0, "consolidation_edges": 0, "cross_validated_projects": 0}

        with (
            patch.object(client, "_get_embedder", return_value=None),
            patch("trw_memory.graph.update_entry_graph", side_effect=slow_graph_update),
        ):
            started = time.perf_counter()
            stored = await client.store("background graph update")
            elapsed = time.perf_counter() - started
            release_update.set()
            await asyncio.to_thread(wait_for_graph_updates)

            assert stored["status"] == "stored"
            assert elapsed < 0.1

    async def test_store_cross_validates_matching_project_entries(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "storage"))
        monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "sqlite")
        client = MemoryClient(namespace="project:default", mode="local")
        cfg = MemoryConfig(storage_backend="sqlite", storage_path=str(tmp_path / "storage"))

        with create_backend_from_config(cfg, "project:other") as storage:
            remote_backend = cast("SQLiteBackend", storage)
            remote_backend.store(
                MemoryEntry(
                    id="M-remote",
                    content="shared operational lesson",
                    namespace="project:other",
                    importance=0.6,
                )
            )

            embedding = [1.0] + ([0.0] * 383)
            fake_embedder = MagicMock()
            fake_embedder.embed.return_value = embedding

            @contextmanager
            def fake_discover(*_args: object, **_kwargs: object) -> Iterator[object]:
                yield [(["project:other"], remote_backend)]

            with (
                patch.object(
                    remote_backend,
                    "get_stored_embeddings",
                    return_value={"M-remote": embedding},
                ),
                patch.object(client, "_get_embedder", return_value=fake_embedder),
                patch("trw_memory.integrations._backend.discover_namespace_backends", fake_discover),
            ):
                stored = await client.store("shared operational lesson", importance=0.6)
                await asyncio.to_thread(wait_for_graph_updates)

            backend = cast("SQLiteBackend", client._get_backend())
            current_entry = backend.get(stored["memory_id"], namespace="project:default")
            remote_entry = remote_backend.get("M-remote", namespace="project:other")
            assert current_entry is not None
            assert remote_entry is not None
            assert current_entry.cross_validated is True
            assert remote_entry.cross_validated is True
            assert current_entry.importance == 0.65
            assert remote_entry.importance == 0.65
            assert any("cross_validated:project_id=other" in item for item in current_entry.outcome_history)
            assert any("cross_validated:project_id=default" in item for item in remote_entry.outcome_history)

        await client.close()

    async def test_store_sync_publish_marks_entry_as_published(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "storage"))
        monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "sqlite")
        monkeypatch.setenv("MEMORY_SYNC_ENABLED", "true")
        monkeypatch.setenv("MEMORY_LOCAL_ONLY", "false")
        monkeypatch.setenv("MEMORY_PLATFORM_URL", "https://api.test.com")
        monkeypatch.setenv("MEMORY_PLATFORM_API_KEY", "test-key")
        client = MemoryClient(namespace="default", mode="local")

        with patch(
            "trw_memory.client.publish_memory_result",
            return_value={"success": True, "remote_id": "42", "retryable": False},
        ):
            stored = await client.store("publish this entry", importance=0.9)
            await client.close()

        reopened = MemoryClient(namespace="default", mode="local")
        entry = reopened._get_backend().get(stored["memory_id"], namespace="default")
        assert entry is not None
        assert entry.published_to_platform is True
        assert entry.remote_id == "42"
        assert entry.vector_clock
        await reopened.close()

    async def test_store_sync_failure_enqueues_retry_payload(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "storage"))
        monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "sqlite")
        monkeypatch.setenv("MEMORY_SYNC_ENABLED", "true")
        monkeypatch.setenv("MEMORY_LOCAL_ONLY", "false")
        monkeypatch.setenv("MEMORY_PLATFORM_URL", "https://api.test.com")
        monkeypatch.setenv("MEMORY_PLATFORM_API_KEY", "test-key")
        client = MemoryClient(namespace="default", mode="local")

        with (
            patch(
                "trw_memory.client.publish_memory_result",
                return_value={"success": False, "remote_id": None, "retryable": True},
            ),
            patch("trw_memory.client._anonymize_entry", return_value={"summary": "queued"}),
        ):
            stored = await client.store("queue this entry", importance=0.9)
            await client.close()

        queue = client._retry_queue
        assert queue.depth() == 1
        lines = (Path(tmp_path) / "storage" / "sync_queue.jsonl").read_text(encoding="utf-8").splitlines()
        assert stored["memory_id"] in lines[0]

    async def test_store_does_not_wait_for_remote_publish_completion(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "storage"))
        monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "sqlite")
        monkeypatch.setenv("MEMORY_SYNC_ENABLED", "true")
        monkeypatch.setenv("MEMORY_PLATFORM_URL", "https://api.test.com")
        client = MemoryClient(namespace="default", mode="local")

        def slow_publish_result(*_args: object, **_kwargs: object) -> dict[str, object]:
            time.sleep(0.2)
            return {"success": True, "remote_id": "42", "retryable": False}

        started = asyncio.get_running_loop().time()
        with (
            patch.object(client, "_get_embedder", return_value=None),
            patch("trw_memory.client.publish_memory_result", side_effect=slow_publish_result),
        ):
            stored = await client.store("async publish", importance=0.9)
        elapsed = asyncio.get_running_loop().time() - started

        assert stored["status"] == "stored"
        assert elapsed < 0.1
        await client.close()

    async def test_store_invalid_platform_url_skips_publish_without_marking_synced(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "storage"))
        monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "sqlite")
        monkeypatch.setenv("MEMORY_SYNC_ENABLED", "true")
        monkeypatch.setenv("MEMORY_PLATFORM_URL", "file:///etc/passwd")
        client = MemoryClient(namespace="default", mode="local")

        stored = await client.store("skip invalid remote", importance=0.9)
        await client.close()

        reopened = MemoryClient(namespace="default", mode="local")
        entry = reopened._get_backend().get(stored["memory_id"], namespace="default")
        assert entry is not None
        assert entry.published_to_platform is False
        assert entry.remote_id is None
        assert reopened._retry_queue.depth() == 0
        await reopened.close()
