"""Focused helper tests for lifecycle/consolidation.py."""

from __future__ import annotations

from pathlib import Path
from typing import cast
from unittest.mock import MagicMock

import pytest

import trw_memory.lifecycle.consolidation as consolidation_module
from trw_memory.exceptions import StorageError
from trw_memory.graph import wait_for_graph_updates
from trw_memory.integrations._backend import create_backend_from_config
from trw_memory.lifecycle.consolidation import (
    _archive_originals,
    _create_consolidated_entry,
    _redact_paths,
    _summarize_cluster_fallback,
)
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.storage.sqlite_backend import SQLiteBackend

from ._test_consolidation_support import _V1, _InMemoryBackend, _make_embedder, _make_entry


class TestRedactPaths:
    def test_redacts_unix_home(self) -> None:
        result = _redact_paths("file at /home/user/project/file.py")
        assert "[REDACTED_PATH]" in result
        assert "/home/user" not in result

    def test_redacts_mnt_path(self) -> None:
        result = _redact_paths("stored at /mnt/c/Users/Tyler/project")
        assert "[REDACTED_PATH]" in result

    def test_redacts_windows_path(self) -> None:
        result = _redact_paths(r"path C:\Users\Tyler\project")
        assert "[REDACTED_PATH]" in result

    def test_no_paths_unchanged(self) -> None:
        text = "no filesystem paths here"
        assert _redact_paths(text) == text

    def test_redacts_tmp_path(self) -> None:
        result = _redact_paths("temp file /tmp/abc123")
        assert "[REDACTED_PATH]" in result


class TestSummarizeClusterFallback:
    def test_selects_longest_content(self) -> None:
        cluster = [
            _make_entry("e1", content="short", detail="x"),
            _make_entry("e2", content="much longer content here", detail="longer detail too"),
            _make_entry("e3", content="medium content", detail="ok"),
        ]
        result = _summarize_cluster_fallback(cluster)
        assert result["summary"] == "much longer content here"
        assert result["detail"] == "longer detail too"

    def test_returns_both_fields(self) -> None:
        cluster = [_make_entry("e1", content="c", detail="d")]
        result = _summarize_cluster_fallback(cluster)
        assert "summary" in result
        assert "detail" in result

    def test_single_entry_cluster(self) -> None:
        cluster = [_make_entry("e1", content="only entry", detail="some detail")]
        result = _summarize_cluster_fallback(cluster)
        assert result["summary"] == "only entry"
        assert result["detail"] == "some detail"

    def test_logs_fallback_event(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_logger = MagicMock()
        monkeypatch.setattr(consolidation_module, "logger", mock_logger)
        cluster = [
            _make_entry("e1", content="short", detail="x"),
            _make_entry("e2", content="much longer content here", detail="longer detail too"),
        ]

        result = _summarize_cluster_fallback(cluster)

        assert result["summary"] == "much longer content here"
        mock_logger.info.assert_called_once_with(
            "consolidation_llm_fallback",
            cluster_size=2,
            selected_id="e2",
        )


class TestCreateConsolidatedEntry:
    def test_creates_entry_with_correct_fields(self) -> None:
        storage = _InMemoryBackend()
        cluster = [
            _make_entry("e1", importance=0.8, tags=["a", "b"], evidence=["ev1"], recurrence=2, q_value=0.7),
            _make_entry("e2", importance=0.6, tags=["b", "c"], evidence=["ev2"], recurrence=3, q_value=0.5),
        ]
        result = _create_consolidated_entry(cluster, "summary text", "detail text", storage)
        assert result.id.startswith("M-")
        assert result.content == "summary text"
        assert result.detail == "detail text"
        assert result.source == "consolidated"
        assert result.importance == 0.8
        assert result.tags == ["a", "b", "c"]
        assert result.evidence == ["ev1", "ev2"]
        assert result.recurrence == 5
        assert result.q_value == 0.7
        assert set(result.consolidated_from) == {"e1", "e2"}

    def test_entry_persisted_in_storage(self) -> None:
        storage = _InMemoryBackend()
        cluster = [_make_entry("e1"), _make_entry("e2"), _make_entry("e3")]
        result = _create_consolidated_entry(cluster, "c", "d", storage)
        assert storage.get(result.id) is not None

    def test_entry_status_is_active(self) -> None:
        storage = _InMemoryBackend()
        cluster = [_make_entry("e1"), _make_entry("e2")]
        result = _create_consolidated_entry(cluster, "c", "d", storage)
        assert str(result.status) == "active"

    def test_evidence_deduplication(self) -> None:
        storage = _InMemoryBackend()
        cluster = [
            _make_entry("e1", evidence=["ev1", "ev2"]),
            _make_entry("e2", evidence=["ev2", "ev3"]),
        ]
        result = _create_consolidated_entry(cluster, "c", "d", storage)
        assert result.evidence == ["ev1", "ev2", "ev3"]

    def test_namespace_passed_correctly(self) -> None:
        storage = _InMemoryBackend()
        cluster = [_make_entry("e1"), _make_entry("e2")]
        result = _create_consolidated_entry(cluster, "c", "d", storage, namespace="team")
        assert result.namespace == "team"

    def test_persists_vector_when_embedder_available(self) -> None:
        storage = _InMemoryBackend()
        cluster = [_make_entry("e1"), _make_entry("e2")]
        embedder = _make_embedder(vectors=[_V1])

        result = _create_consolidated_entry(cluster, "summary", "detail", storage, embedder=embedder)

        assert storage.get_stored_embeddings([result.id])[result.id] == _V1

    def test_persists_consolidation_edges_for_sqlite_backend(self, tmp_path: Path) -> None:
        cfg = MemoryConfig(storage_backend="sqlite", storage_path=str(tmp_path), embedding_dim=4)

        with create_backend_from_config(cfg, "default") as sqlite_storage:
            storage = cast("SQLiteBackend", sqlite_storage)
            cluster = [_make_entry("e1"), _make_entry("e2")]
            for entry in cluster:
                storage.store(entry)

            result = _create_consolidated_entry(
                cluster,
                "summary",
                "detail",
                storage,
                embedder=_make_embedder(vectors=[_V1]),
            )
            wait_for_graph_updates()

            edge_rows = storage._conn.execute(
                "SELECT source_id, target_id, edge_type FROM memory_graph_edges WHERE source_id = ? ORDER BY target_id",
                (result.id,),
            ).fetchall()
            assert [tuple(row) for row in edge_rows] == [
                (result.id, "e1", "consolidation"),
                (result.id, "e2", "consolidation"),
            ]

    def test_rolls_back_entry_when_vector_write_fails(self) -> None:
        storage = _InMemoryBackend()
        cluster = [_make_entry("e1"), _make_entry("e2")]
        embedder = _make_embedder(vectors=[_V1])

        def _fail_vector(_entry_id: str, _embedding: list[float]) -> None:
            raise RuntimeError("vector write failed")

        storage.upsert_vector_override = _fail_vector

        with pytest.raises(StorageError, match="entry write was rolled back"):
            _create_consolidated_entry(cluster, "summary", "detail", storage, embedder=embedder)

        assert len(storage.list_entries()) == 0


class TestArchiveOriginals:
    def test_sets_consolidated_into_and_archived(self) -> None:
        storage = _InMemoryBackend()
        e1 = _make_entry("e1")
        e2 = _make_entry("e2")
        storage.store(e1)
        storage.store(e2)

        _archive_originals([e1, e2], "M-consolidated", storage)

        updated_e1 = storage.get("e1")
        updated_e2 = storage.get("e2")
        assert updated_e1 is not None
        assert updated_e2 is not None
        assert updated_e1.consolidated_into == "M-consolidated"
        assert updated_e2.consolidated_into == "M-consolidated"
        assert str(updated_e1.status) == "archived"
        assert str(updated_e2.status) == "archived"

    def test_raises_on_storage_error(self) -> None:
        storage = _InMemoryBackend()
        e1 = _make_entry("e1")

        def _fail_update(entry_id: str, fields: dict[str, object]) -> MemoryEntry | None:
            del entry_id, fields
            raise RuntimeError("storage error")

        storage.update_override = _fail_update
        with pytest.raises(RuntimeError):
            _archive_originals([e1], "M-cons", storage)

    def test_archival_is_atomic_partial_failure_rolls_back_all(self, tmp_path: Path) -> None:
        """S4: a crash mid-loop rolls back EVERY archival update (all-or-nothing).

        Two originals are archived against a real on-disk SQLiteBackend. We force
        a failure on the SECOND entry's UPDATE, after the FIRST has been staged
        inside the shared ``transaction()``. A fresh observer connection (reading
        committed-only state) must show BOTH originals still ACTIVE with no
        ``consolidated_into`` — proving the first entry's archival did not leak
        out on its own per-call commit (the pre-fix behaviour).
        """
        import sqlite3

        db_path = tmp_path / "archive_atomic.db"
        backend = SQLiteBackend(db_path)
        try:
            e1 = _make_entry("M-orig1")
            e2 = _make_entry("M-orig2")
            backend.store(e1)
            backend.store(e2)

            real_conn = backend._conn

            class _FailSecondUpdateConn:
                def execute(self, sql: str, params: object = (), /) -> object:
                    # Fail the UPDATE that targets the second original only, so
                    # the first original's UPDATE has already been staged inside
                    # the open transaction when the crash hits. The entry id is
                    # the trailing bound parameter of ``UPDATE ... WHERE id = ?``.
                    if sql.strip().startswith("UPDATE memories") and isinstance(params, (list, tuple)) and params and params[-1] == "M-orig2":
                        raise RuntimeError("crash-on-second-archival")
                    return real_conn.execute(sql, params)

                def __getattr__(self, name: str) -> object:
                    return getattr(real_conn, name)

            backend._conn = _FailSecondUpdateConn()
            try:
                with pytest.raises(RuntimeError, match="crash-on-second-archival"):
                    _archive_originals([e1, e2], "M-cons", backend)
            finally:
                backend._conn = real_conn

            observer = sqlite3.connect(str(db_path))
            try:
                rows = observer.execute(
                    "SELECT id, status, consolidated_into FROM memories "
                    "WHERE id IN ('M-orig1', 'M-orig2') ORDER BY id"
                ).fetchall()
            finally:
                observer.close()
            # Both originals survived UN-archived — the first entry's staged
            # UPDATE was rolled back together with the failed second one.
            assert rows == [
                ("M-orig1", "active", None),
                ("M-orig2", "active", None),
            ], f"partial archival leaked despite rollback: {rows}"
        finally:
            backend.close()
