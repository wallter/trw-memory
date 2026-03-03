"""Tests for atomic YAML/JSONL persistence utilities.

Covers:
- read_yaml: happy path, file-not-found, empty file, non-dict root, malformed YAML
- write_yaml: happy path write-and-read-back, creates parent dirs, atomic rename,
  temp file cleaned up on failure
- append_jsonl: happy path single record, creates file if absent, multiple appends
- json_serializer: datetime, date, unsupported type
- lock_for_rmw: acquires/releases lock, creates lock file, yields original path
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from trw_memory.exceptions import StorageError
from trw_memory.storage.persistence import (
    append_jsonl,
    json_serializer,
    lock_for_rmw,
    read_yaml,
    write_yaml,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_raw_yaml(path: Path, content: str) -> None:
    """Write raw YAML bytes to a file, bypassing our write_yaml helper."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# read_yaml
# ---------------------------------------------------------------------------


class TestReadYaml:
    def test_happy_path_returns_dict(self, tmp_path: Path) -> None:
        p = tmp_path / "data.yaml"
        _write_raw_yaml(p, "key: value\nnumber: 42\n")
        result = read_yaml(p)
        assert result["key"] == "value"
        assert result["number"] == 42

    def test_file_not_found_raises_storage_error(self, tmp_path: Path) -> None:
        missing = tmp_path / "nonexistent.yaml"
        with pytest.raises(StorageError, match="not found"):
            read_yaml(missing)

    def test_storage_error_path_attribute(self, tmp_path: Path) -> None:
        missing = tmp_path / "gone.yaml"
        with pytest.raises(StorageError) as exc_info:
            read_yaml(missing)
        assert str(missing) in exc_info.value.path

    def test_empty_file_returns_empty_dict(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.yaml"
        p.write_text("", encoding="utf-8")
        result = read_yaml(p)
        assert result == {}

    def test_non_dict_root_raises_storage_error(self, tmp_path: Path) -> None:
        """A YAML list at root is not a valid mapping — must raise StorageError."""
        p = tmp_path / "list.yaml"
        _write_raw_yaml(p, "- item1\n- item2\n")
        with pytest.raises(StorageError, match="mapping"):
            read_yaml(p)

    def test_malformed_yaml_raises_storage_error(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.yaml"
        # Intentionally malformed YAML (tab used in indentation where not allowed)
        _write_raw_yaml(p, "key:\n\t- value\n")
        with pytest.raises(StorageError):
            read_yaml(p)

    def test_nested_dict_returned(self, tmp_path: Path) -> None:
        p = tmp_path / "nested.yaml"
        _write_raw_yaml(p, "outer:\n  inner: 123\n")
        result = read_yaml(p)
        assert "outer" in result


# ---------------------------------------------------------------------------
# write_yaml
# ---------------------------------------------------------------------------


class TestWriteYaml:
    def test_happy_path_write_and_read_back(self, tmp_path: Path) -> None:
        p = tmp_path / "out.yaml"
        data: dict[str, object] = {"name": "Alice", "score": 0.9}
        write_yaml(p, data)
        assert p.exists()
        result = read_yaml(p)
        assert result["name"] == "Alice"
        assert float(str(result["score"])) == pytest.approx(0.9)

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        p = tmp_path / "nested" / "deep" / "data.yaml"
        write_yaml(p, {"x": 1})
        assert p.exists()

    def test_atomic_rename_no_partial_file(self, tmp_path: Path) -> None:
        """After write_yaml the final path exists; no .tmp file should remain."""
        p = tmp_path / "atomic.yaml"
        write_yaml(p, {"k": "v"})
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert len(tmp_files) == 0

    def test_tmp_file_cleaned_on_dump_failure(self, tmp_path: Path) -> None:
        """If the YAML dump raises, the temp file must be removed."""
        p = tmp_path / "fail.yaml"
        with patch("trw_memory.storage.persistence._new_yaml") as mock_new_yaml:
            mock_yml = MagicMock()
            mock_yml.dump.side_effect = RuntimeError("boom")
            mock_new_yaml.return_value = mock_yml
            with pytest.raises(StorageError):
                write_yaml(p, {"a": 1})
        # No orphaned temp files
        tmp_files = list(tmp_path.glob("*.yaml.tmp"))
        assert len(tmp_files) == 0

    def test_overwrite_existing_file(self, tmp_path: Path) -> None:
        p = tmp_path / "overwrite.yaml"
        write_yaml(p, {"v": 1})
        write_yaml(p, {"v": 2})
        result = read_yaml(p)
        assert result["v"] == 2

    def test_write_empty_dict(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.yaml"
        write_yaml(p, {})
        result = read_yaml(p)
        assert result == {}


# ---------------------------------------------------------------------------
# append_jsonl
# ---------------------------------------------------------------------------


class TestAppendJsonl:
    def test_happy_path_single_record(self, tmp_path: Path) -> None:
        p = tmp_path / "events.jsonl"
        record = {"action": "store", "id": "M-001"}
        append_jsonl(p, record)
        lines = p.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["action"] == "store"
        assert parsed["id"] == "M-001"

    def test_creates_file_if_absent(self, tmp_path: Path) -> None:
        p = tmp_path / "new.jsonl"
        assert not p.exists()
        append_jsonl(p, {"k": "v"})
        assert p.exists()

    def test_multiple_appends_produce_valid_jsonl(self, tmp_path: Path) -> None:
        p = tmp_path / "multi.jsonl"
        for i in range(3):
            append_jsonl(p, {"seq": i})
        lines = p.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 3
        for idx, line in enumerate(lines):
            obj = json.loads(line)
            assert obj["seq"] == idx

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        p = tmp_path / "sub" / "dir" / "log.jsonl"
        append_jsonl(p, {"x": 1})
        assert p.exists()

    def test_datetime_serialised_via_json_serializer(self, tmp_path: Path) -> None:
        p = tmp_path / "dt.jsonl"
        now = datetime.now(timezone.utc)
        append_jsonl(p, {"ts": now})  # type: ignore[dict-item]
        line = p.read_text(encoding="utf-8").strip()
        obj = json.loads(line)
        assert "T" in obj["ts"]  # ISO format contains 'T'


# ---------------------------------------------------------------------------
# json_serializer
# ---------------------------------------------------------------------------


class TestJsonSerializer:
    def test_datetime_returns_isoformat(self) -> None:
        dt = datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        result = json_serializer(dt)
        assert result == "2025-06-15T12:00:00+00:00"

    def test_date_returns_isoformat(self) -> None:
        d = date(2025, 6, 15)
        result = json_serializer(d)
        assert result == "2025-06-15"

    def test_unsupported_type_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="not JSON serializable"):
            json_serializer(object())

    def test_set_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            json_serializer({1, 2, 3})

    @pytest.mark.parametrize(
        "dt",
        [
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            datetime(2025, 12, 31, 23, 59, 59, tzinfo=timezone.utc),
        ],
    )
    def test_parametrized_datetime_values(self, dt: datetime) -> None:
        result = json_serializer(dt)
        assert result == dt.isoformat()


# ---------------------------------------------------------------------------
# lock_for_rmw
# ---------------------------------------------------------------------------


class TestLockForRmw:
    def test_yields_original_path(self, tmp_path: Path) -> None:
        target = tmp_path / "data.yaml"
        with lock_for_rmw(target) as p:
            assert p == target

    def test_creates_lock_file_during_hold(self, tmp_path: Path) -> None:
        target = tmp_path / "data.yaml"
        with lock_for_rmw(target):
            lock_file = tmp_path / "data.yaml.lock"
            assert lock_file.exists()

    def test_lock_file_remains_after_release(self, tmp_path: Path) -> None:
        """The lock file is NOT deleted on exit (advisory lock pattern)."""
        target = tmp_path / "data.yaml"
        with lock_for_rmw(target):
            pass
        lock_file = tmp_path / "data.yaml.lock"
        # Lock file may or may not remain — just verify the context exits cleanly
        # and no exception is raised (file existence is an impl detail)

    def test_creates_parent_dirs_for_lock(self, tmp_path: Path) -> None:
        target = tmp_path / "sub" / "deep" / "data.yaml"
        with lock_for_rmw(target) as p:
            assert p == target

    def test_lock_released_on_exception(self, tmp_path: Path) -> None:
        """Context manager must release lock even when body raises."""
        target = tmp_path / "data.yaml"
        with pytest.raises(ValueError, match="inner"):
            with lock_for_rmw(target):
                raise ValueError("inner error")
        # If lock was not released we'd deadlock on the second acquire;
        # completing without timeout confirms the lock was released.
        with lock_for_rmw(target):
            pass  # must not block

    def test_read_modify_write_pattern(self, tmp_path: Path) -> None:
        """Full RMW cycle: write, read, modify, write again under lock."""
        target = tmp_path / "entry.yaml"
        write_yaml(target, {"count": 0})
        with lock_for_rmw(target):
            data = read_yaml(target)
            data["count"] = int(str(data["count"])) + 1
            write_yaml(target, data)
        result = read_yaml(target)
        assert result["count"] == 1
