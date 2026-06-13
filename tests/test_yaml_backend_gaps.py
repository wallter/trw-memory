"""Wave 15: coverage gap-fill for storage/yaml_backend.py.

Target lines: 87-88, 119-120, 128-132, 206, 211, 224-230,
              273-276, 301-304, 313-314, 336-337, 361-362,
              476, 495-498, 516-536.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from trw_memory.exceptions import StorageError
from trw_memory.models.memory import MemoryEntry
from trw_memory.storage.yaml_backend import YAMLBackend, _dict_to_entry


def _base_data(**overrides: object) -> dict[str, object]:
    data: dict[str, object] = {
        "id": "M-001",
        "content": "test content",
        "namespace": "project:default",
    }
    data.update(overrides)
    return data


class TestDictToEntryHelpers:
    def test_int_helper_falls_back_on_bad_value(self) -> None:
        """_int() TypeError/ValueError → default (lines 87-88)."""
        entry = _dict_to_entry(_base_data(access_count=None))
        assert entry.access_count == 0  # None cannot be parsed by int(), falls back to 0

    def test_invalid_status_falls_back_to_active(self) -> None:
        """Invalid MemoryStatus string → MemoryStatus.ACTIVE (lines 119-120)."""
        entry = _dict_to_entry(_base_data(status="not_a_valid_status"))
        from trw_memory.models.memory import MemoryStatus
        assert entry.status == MemoryStatus.ACTIVE

    def test_bad_anchors_list_is_skipped(self) -> None:
        """Anchor.model_validate error → debug logged, anchors=[] (lines 128-132)."""
        entry = _dict_to_entry(_base_data(anchors=[{"bad_key": "value"}]))
        assert entry.anchors == []


class TestYAMLBackendRepr:
    def test_repr_includes_dir(self, tmp_path: Path) -> None:
        """__repr__ returns YAMLBackend(entries_dir=...) (line 206)."""
        backend = YAMLBackend(tmp_path / "entries")
        r = repr(backend)
        assert "YAMLBackend" in r
        assert "entries" in r


class TestYAMLBackendPathTraversal:
    def test_path_traversal_raises_storage_error(self, tmp_path: Path) -> None:
        """entry_id with path traversal → StorageError (line 211)."""
        backend = YAMLBackend(tmp_path / "entries")
        with pytest.raises(StorageError, match="path traversal"):
            backend._path("../evil")


class TestYAMLBackendLoadAllCorrupt:
    def test_corrupt_yaml_file_is_skipped(self, tmp_path: Path) -> None:
        """Corrupt YAML file during _load_all → skipped (lines 224-230)."""
        entries_dir = tmp_path / "entries"
        entries_dir.mkdir()
        corrupt = entries_dir / "bad-entry.yaml"
        corrupt.write_text("not: valid: yaml: content:\n  - [unclosed")

        backend = YAMLBackend(entries_dir)
        result = backend._load_all()
        assert result == []


class TestYAMLBackendGetDeserializeError:
    def test_get_value_error_raises_storage_error(self, tmp_path: Path) -> None:
        """get() ValueError during _dict_to_entry → StorageError (lines 273-276)."""
        backend = YAMLBackend(tmp_path / "entries")
        entry = MemoryEntry(id="M-get", content="hello", namespace="project:default")
        backend.store(entry)
        with patch("trw_memory.storage.yaml_backend._dict_to_entry", side_effect=ValueError("bad")):
            with pytest.raises(StorageError):
                backend.get("M-get")

    def test_get_storage_error_is_reraised(self, tmp_path: Path) -> None:
        """get() StorageError from read_yaml → re-raised directly (line 274)."""
        backend = YAMLBackend(tmp_path / "entries")
        entry = MemoryEntry(id="M-se", content="hello", namespace="project:default")
        backend.store(entry)
        with patch("trw_memory.storage.yaml_backend.read_yaml", side_effect=StorageError("disk")):
            with pytest.raises(StorageError, match="disk"):
                backend.get("M-se")


class TestYAMLBackendUpdateErrors:
    def test_update_read_error_raises_storage_error(self, tmp_path: Path) -> None:
        """update() OSError during read_yaml → StorageError (lines 301-304)."""
        backend = YAMLBackend(tmp_path / "entries")
        entry = MemoryEntry(id="M-upd", content="hello", namespace="project:default")
        backend.store(entry)
        with patch("trw_memory.storage.yaml_backend.read_yaml", side_effect=OSError("disk err")):
            with pytest.raises(StorageError):
                backend.update("M-upd", content="new")

    def test_update_storage_error_from_read_yaml_is_reraised(self, tmp_path: Path) -> None:
        """update() StorageError from read_yaml inside lock → re-raised (line 302)."""
        backend = YAMLBackend(tmp_path / "entries")
        entry = MemoryEntry(id="M-upd2", content="hello", namespace="project:default")
        backend.store(entry)
        with patch("trw_memory.storage.yaml_backend.read_yaml", side_effect=StorageError("lock")):
            with pytest.raises(StorageError, match="lock"):
                backend.update("M-upd2", content="new")

    def test_update_invalid_field_raises_storage_error(self, tmp_path: Path) -> None:
        """update() invalid field → StorageError (lines 313-314)."""
        backend = YAMLBackend(tmp_path / "entries")
        entry = MemoryEntry(id="M-inv", content="hello", namespace="project:default")
        backend.store(entry)
        with pytest.raises(StorageError):
            backend.update("M-inv", _invalid_field="oops")

    def test_update_deserialize_failure_raises_storage_error(self, tmp_path: Path) -> None:
        """update() final _dict_to_entry error → StorageError (lines 336-337)."""
        backend = YAMLBackend(tmp_path / "entries")
        entry = MemoryEntry(id="M-de", content="hello", namespace="project:default")
        backend.store(entry)

        call_count = 0

        def _mock_dict_to_entry(data: dict[str, object]) -> MemoryEntry:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return entry  # first call (inside lock) succeeds
            raise ValueError("corrupt on final deserialize")

        with patch("trw_memory.storage.yaml_backend._dict_to_entry", side_effect=_mock_dict_to_entry):
            with pytest.raises(StorageError):
                backend.update("M-de", content="updated")


class TestYAMLBackendDeleteOsError:
    def test_delete_oserror_raises_storage_error(self, tmp_path: Path) -> None:
        """delete() OSError → StorageError (lines 361-362)."""
        backend = YAMLBackend(tmp_path / "entries")
        entry = MemoryEntry(id="M-del", content="hello", namespace="project:default")
        backend.store(entry)
        with patch("trw_memory.storage.yaml_backend.Path.unlink", side_effect=OSError("busy")):
            with pytest.raises(StorageError):
                backend.delete("M-del")


class TestYAMLBackendListEntries:
    def test_min_importance_filter_skips_low_importance_entries(self, tmp_path: Path) -> None:
        """entry.importance < min_importance → continue (line 476)."""
        backend = YAMLBackend(tmp_path / "entries")
        backend.store(MemoryEntry(id="M-hi", content="important", namespace="project:default", importance=0.9))
        backend.store(MemoryEntry(id="M-lo", content="trivial", namespace="project:default", importance=0.1))
        results = backend.list_entries(namespace="project:default", min_importance=0.5)
        ids = [e.id for e in results]
        assert "M-hi" in ids
        assert "M-lo" not in ids


class TestYAMLBackendNamespaceOps:
    def test_list_namespaces_returns_distinct_sorted(self, tmp_path: Path) -> None:
        """list_namespaces() returns sorted unique namespaces (lines 495-498)."""
        backend = YAMLBackend(tmp_path / "entries")
        backend.store(MemoryEntry(id="M-a", content="x", namespace="project:alpha"))
        backend.store(MemoryEntry(id="M-b", content="y", namespace="project:beta"))
        backend.store(MemoryEntry(id="M-c", content="z", namespace="project:alpha"))
        ns = backend.list_namespaces()
        assert ns == ["project:alpha", "project:beta"]

    def test_delete_by_namespace_removes_matching_entries(self, tmp_path: Path) -> None:
        """delete_by_namespace() deletes all matching entries (lines 516-536)."""
        backend = YAMLBackend(tmp_path / "entries")
        backend.store(MemoryEntry(id="M-keep", content="keep", namespace="project:keep"))
        backend.store(MemoryEntry(id="M-del1", content="del1", namespace="project:delete"))
        backend.store(MemoryEntry(id="M-del2", content="del2", namespace="project:delete"))
        deleted = backend.delete_by_namespace("project:delete")
        assert deleted == 2
        assert backend.get("M-keep") is not None
        assert backend.get("M-del1") is None
        assert backend.get("M-del2") is None

    def test_delete_by_namespace_oserror_raises_storage_error(self, tmp_path: Path) -> None:
        """delete_by_namespace() OSError on unlink → StorageError (lines 525-529)."""
        backend = YAMLBackend(tmp_path / "entries")
        backend.store(MemoryEntry(id="M-osx", content="x", namespace="project:gone"))

        def _selective_unlink(self: Path) -> None:
            if self.suffix == ".yaml":
                raise OSError("disk full")

        with patch("trw_memory.storage.yaml_backend.Path.unlink", _selective_unlink):
            with pytest.raises(StorageError):
                backend.delete_by_namespace("project:gone")

    def test_delete_by_namespace_file_not_found_is_race_safe(self, tmp_path: Path) -> None:
        """delete_by_namespace() FileNotFoundError → pass (race-safe, line 524)."""
        backend = YAMLBackend(tmp_path / "entries")
        backend.store(MemoryEntry(id="M-race", content="x", namespace="project:race"))

        with patch("trw_memory.storage.yaml_backend.Path.unlink", side_effect=FileNotFoundError("gone")):
            deleted = backend.delete_by_namespace("project:race")
        assert deleted == 0  # FileNotFoundError swallowed, not counted
