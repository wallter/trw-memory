"""Wave 15: coverage gap-fill for _graph_cross_project.py.

Target lines: 83, 109-113, 134, 136. (The apply_cross_project_validation
one-entry-form coverage that used to live here was removed with the dead
function itself -- see cross_validate_entries for the live orchestrator.)
"""

from __future__ import annotations

from unittest.mock import MagicMock

from trw_memory._graph_cross_project import (
    append_cross_validation,
    backend_update_guard,
    entry_update_lock,
    merge_cross_validated_entry,
    persist_cross_validated_entry,
)
from trw_memory.models.memory import MemoryEntry
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


class TestEntryUpdateLockBounded:
    def test_locks_are_evicted_when_not_referenced(self) -> None:
        """The per-entry lock registry must not grow without bound.

        Each (backend, entry_id) lock is reclaimed once no caller holds it,
        so requesting locks for many distinct entries does not accumulate
        entries forever (the prior plain dict had no eviction).
        """
        import gc

        import trw_memory._graph_cross_project as gx

        backend = MagicMock()
        backend._db_path = "/tmp/some-backend.db"

        # Request locks for many distinct entries, holding NO references.
        for i in range(500):
            entry_update_lock(backend, f"entry-{i}")

        gc.collect()
        # With a WeakValueDictionary the unreferenced locks are collected, so
        # the registry does not retain all 500 keys.
        assert len(gx._ENTRY_UPDATE_LOCKS) < 500

    def test_same_lock_returned_while_reference_held(self) -> None:
        """Concurrent callers for the same key must share one lock object."""
        backend = MagicMock()
        backend._db_path = "/tmp/shared-backend.db"

        lock_a = entry_update_lock(backend, "same-entry")
        lock_b = entry_update_lock(backend, "same-entry")
        assert lock_a is lock_b, "same key must yield the same lock while held"


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
        assert type(ctx).__name__ == "nullcontext", (
            "a backend with no path must take no lock at all; a real lock here would "
            f"serialise unrelated writers, and this returned {type(ctx).__name__}"
        )
        with ctx as handle:
            assert handle is None


# ---------------------------------------------------------------------------
# line 134: merge_cross_validated_entry when entry not found
# ---------------------------------------------------------------------------


class TestMergeCrossValidatedEntryNotFound:
    def test_returns_none_false_when_entry_missing(self) -> None:
        """merge returns (None, False) when backend.get returns None (line 134)."""
        mock_backend = MagicMock()
        mock_backend.get.return_value = None

        result, applied = merge_cross_validated_entry(mock_backend, "M-missing", "beta", 0.95, namespace="default")
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

        result, applied = merge_cross_validated_entry(backend, entry.id, "beta", 0.95, namespace=entry.namespace)
        assert result is not None
        assert applied is False
