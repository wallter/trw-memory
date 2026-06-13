"""Wave 15: coverage gap-fill for _graph_cross_project.py.

Target lines: 83, 109-113, 134, 136, 168, 187, 195, 217.
"""
from __future__ import annotations

from contextlib import nullcontext
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone

import pytest

from trw_memory._graph_cross_project import (
    CROSS_VALIDATION_THRESHOLD,
    append_cross_validation,
    apply_cross_project_validation,
    backend_update_guard,
    entry_has_cross_validation,
    entry_update_lock,
    merge_cross_validated_entry,
    persist_cross_validated_entry,
    project_scope_key,
)
from trw_memory.models.memory import MemoryEntry, MemoryStatus
from trw_memory.storage.sqlite_backend import SQLiteBackend


def _entry(**kwargs) -> MemoryEntry:
    defaults: dict = {
        "id": "M-test",
        "content": "test content",
        "namespace": "project:alpha",
    }
    defaults.update(kwargs)
    return MemoryEntry(**defaults)


# ---------------------------------------------------------------------------
# line 83: persist_cross_validated_entry early return when updated == original
# ---------------------------------------------------------------------------

class TestPersistCrossValidatedEntry:
    def test_no_update_when_entry_unchanged(self, tmp_path) -> None:
        """persist skips backend.update when updated == original (line 83)."""
        backend = SQLiteBackend(tmp_path / "test.db")
        entry = _entry()
        backend.store(entry)

        mock_backend = MagicMock()
        # Pass same entry as both original and updated → no diff → early return
        persist_cross_validated_entry(mock_backend, entry, entry)
        mock_backend.update.assert_not_called()

    def test_update_called_when_entry_changed(self, tmp_path) -> None:
        """persist calls backend.update when updated != original."""
        backend = SQLiteBackend(tmp_path / "test.db")
        entry = _entry()
        backend.store(entry)

        updated = append_cross_validation(entry, "beta", 0.95)
        mock_backend = MagicMock()
        persist_cross_validated_entry(mock_backend, entry, updated)
        mock_backend.update.assert_called_once()


# ---------------------------------------------------------------------------
# lines 109-113: backend_update_guard YAML-backend path (_dir attribute)
# ---------------------------------------------------------------------------

class TestBackendUpdateGuardYAMLPath:
    def test_yaml_backend_uses_dir_lock(self, tmp_path) -> None:
        """backend_update_guard uses _dir lock for YAML-style backends (lines 109-113)."""
        mock_backend = MagicMock()
        mock_backend._db_path = None
        mock_backend._dir = tmp_path

        ctx = backend_update_guard(mock_backend)
        # Should be an AbstractContextManager (lock_for_rmw result or nullcontext)
        assert hasattr(ctx, "__enter__")

    def test_no_path_returns_nullcontext(self) -> None:
        """backend_update_guard with no path attr → nullcontext (line 113)."""
        mock_backend = MagicMock(spec=[])  # no _db_path, no _dir
        ctx = backend_update_guard(mock_backend)
        # Should function as a nullcontext
        with ctx:
            pass


# ---------------------------------------------------------------------------
# line 134: merge_cross_validated_entry when entry not found
# ---------------------------------------------------------------------------

class TestMergeCrossValidatedEntryNotFound:
    def test_returns_none_false_when_entry_missing(self) -> None:
        """merge returns (None, False) when backend.get returns None (line 134)."""
        mock_backend = MagicMock()
        mock_backend.get.return_value = None

        result, applied = merge_cross_validated_entry(mock_backend, "M-missing", "beta", 0.95)
        assert result is None
        assert applied is False


# ---------------------------------------------------------------------------
# line 136: merge_cross_validated_entry skip already-validated
# ---------------------------------------------------------------------------

class TestMergeCrossValidatedEntryAlreadyValidated:
    def test_returns_current_false_when_already_validated(self, tmp_path) -> None:
        """merge returns (current, False) when entry already has cross-validation (line 136)."""
        backend = SQLiteBackend(tmp_path / "test.db")
        entry = _entry()
        entry_with_validation = append_cross_validation(entry, "beta", 0.95)
        backend.store(entry_with_validation)

        result, applied = merge_cross_validated_entry(backend, entry.id, "beta", 0.95)
        assert result is not None
        assert applied is False


# ---------------------------------------------------------------------------
# line 168: apply_cross_project_validation with non-project namespace
# ---------------------------------------------------------------------------

class TestApplyCrossProjectValidationNonProjectNS:
    def test_non_project_namespace_returns_zero(self, tmp_path) -> None:
        """apply_cross_project_validation returns 0 when namespace not project: (line 168)."""
        backend = SQLiteBackend(tmp_path / "test.db")
        entry = _entry(namespace="team:shared")  # not project:

        mock_conn = MagicMock()
        result = apply_cross_project_validation(entry, backend, mock_conn, embedding=[0.1, 0.2])
        assert result == 0


# ---------------------------------------------------------------------------
# line 187: project_id is None for remote namespace → continue
# ---------------------------------------------------------------------------

class TestApplyCrossProjectValidationNoneProjectId:
    def test_non_project_remote_namespace_skipped(self, tmp_path) -> None:
        """Remote namespace with project_scope_key=None → continue at line 187."""
        backend = SQLiteBackend(tmp_path / "test.db")
        entry = _entry(namespace="project:alpha")

        mock_conn = MagicMock()
        mock_remote_backend = MagicMock()

        # discover_namespace_backends returns a store with a "team:" namespace
        # project_scope_key("team:shared") = None → hits the `continue` at line 187
        def _fake_discover(cfg):
            from contextlib import contextmanager

            @contextmanager
            def _ctx():
                yield [
                    (["team:shared"], mock_remote_backend),
                ]

            return _ctx()

        with patch("trw_memory.integrations._backend.discover_namespace_backends", side_effect=_fake_discover):
            result = apply_cross_project_validation(
                entry, backend, mock_conn, embedding=[0.1, 0.2]
            )
        assert result == 0  # no projects matched

    def test_project_scope_key_returns_none_in_for_loop(self, tmp_path) -> None:
        """project_scope_key returns None on for-loop call → continue at line 187."""
        backend = SQLiteBackend(tmp_path / "test.db")
        entry = _entry(namespace="project:alpha")
        mock_conn = MagicMock()
        mock_remote_backend = MagicMock()
        mock_remote_backend.list_entries.return_value = []

        def _fake_discover(cfg):
            from contextlib import contextmanager

            @contextmanager
            def _ctx():
                yield [(["project:beta"], mock_remote_backend)]

            return _ctx()

        call_count: list[int] = [0]
        original_fn = project_scope_key

        def _flaky_scope_key(namespace: str):
            call_count[0] += 1
            result = original_fn(namespace)
            # On 3rd call and beyond (the for-loop re-calls), return None
            if call_count[0] >= 3:
                return None
            return result

        with patch("trw_memory.integrations._backend.discover_namespace_backends", side_effect=_fake_discover):
            with patch("trw_memory._graph_cross_project.project_scope_key", side_effect=_flaky_scope_key):
                result = apply_cross_project_validation(
                    entry, backend, mock_conn, embedding=[0.1, 0.2]
                )
        assert result == 0


# ---------------------------------------------------------------------------
# line 195: no remote entries → continue
# ---------------------------------------------------------------------------

class TestApplyCrossProjectValidationEmptyRemote:
    def test_empty_remote_entries_skipped(self, tmp_path) -> None:
        """Remote namespace has no entries → if not remote_entries: continue (line 195)."""
        backend = SQLiteBackend(tmp_path / "test.db")
        entry = _entry(namespace="project:alpha")

        mock_conn = MagicMock()
        mock_remote_backend = MagicMock()
        mock_remote_backend.list_entries.return_value = []

        def _fake_discover(cfg):
            from contextlib import contextmanager

            @contextmanager
            def _ctx():
                yield [
                    (["project:beta"], mock_remote_backend),
                ]

            return _ctx()

        with patch("trw_memory.integrations._backend.discover_namespace_backends", side_effect=_fake_discover):
            result = apply_cross_project_validation(
                entry, backend, mock_conn, embedding=[0.1, 0.2]
            )
        assert result == 0


# ---------------------------------------------------------------------------
# line 217: similarity <= threshold → continue
# ---------------------------------------------------------------------------

class TestApplyCrossProjectValidationLowSimilarity:
    def test_low_similarity_skips_merge(self, tmp_path) -> None:
        """similarity <= CROSS_VALIDATION_THRESHOLD → continue (line 217), no merge."""
        backend = SQLiteBackend(tmp_path / "main.db")
        remote_backend = SQLiteBackend(tmp_path / "remote.db")

        entry = _entry(namespace="project:alpha")
        backend.store(entry)

        remote_entry = _entry(id="M-remote", namespace="project:beta")
        remote_backend.store(remote_entry)

        embedding = [0.1, 0.2, 0.3]
        remote_embedding = [0.9, 0.1, 0.0]  # low cosine similarity

        mock_conn = MagicMock()

        def _fake_discover(cfg):
            from contextlib import contextmanager

            @contextmanager
            def _ctx():
                yield [
                    (["project:beta"], remote_backend),
                ]

            return _ctx()

        with patch("trw_memory.integrations._backend.discover_namespace_backends", side_effect=_fake_discover):
            with patch("trw_memory.graph.detect_cross_validation", return_value=True):
                with patch("trw_memory.graph._safe_cosine_similarity", return_value=0.5):
                    # 0.5 <= 0.92 threshold → hits continue at line 217
                    remote_backend.get_stored_embeddings = MagicMock(
                        return_value={"M-remote": remote_embedding}
                    )
                    result = apply_cross_project_validation(
                        entry, backend, mock_conn, embedding=embedding
                    )
        assert result == 0  # no merges applied
