"""Wave 15: coverage gap-fill for storage/persistence.py (lines 92-96, 105-110, 193, 229-230)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from trw_memory.exceptions import StorageError
from trw_memory.storage.persistence import _close_handle, _handle_fileno, append_jsonl, write_yaml


class TestHandleFilenoWrappedPath:
    def test_wrapped_handle_with_fh_fileno(self) -> None:
        """_handle_fileno falls back to handle._fh.fileno() (lines 92-94)."""
        inner = MagicMock()
        inner.fileno.return_value = 5
        wrapper = MagicMock(spec=[])  # no direct fileno attribute
        wrapper._fh = inner
        type(wrapper).fileno = property(lambda self: None)  # type: ignore[misc]

        # Build a simple wrapper object that has ._fh but no direct fileno callable
        class WrappedHandle:
            def __init__(self, fh: object) -> None:
                self._fh = fh

        obj = WrappedHandle(inner)
        result = _handle_fileno(obj)
        assert result == 5

    def test_raises_when_no_fileno_at_all(self) -> None:
        """_handle_fileno raises TypeError when neither fileno nor _fh.fileno exist (line 96)."""

        class NoFileno:
            pass

        with pytest.raises(TypeError, match="fileno"):
            _handle_fileno(NoFileno())


class TestCloseHandleWrappedPath:
    def test_wrapped_handle_with_fh_close(self) -> None:
        """_close_handle falls back to handle._fh.close() (lines 105-108)."""
        inner = MagicMock()

        class WrappedHandle:
            def __init__(self, fh: object) -> None:
                self._fh = fh

        obj = WrappedHandle(inner)
        _close_handle(obj)
        inner.close.assert_called_once()
        assert obj._fh is inner

    def test_raises_when_no_close_at_all(self) -> None:
        """_close_handle raises TypeError when no close method exists (lines 109-110)."""

        class NoClose:
            pass

        with pytest.raises(TypeError, match="close"):
            _close_handle(NoClose())


class TestWriteYamlStorageError:
    def test_oserror_in_write_raises_storage_error(self, tmp_path: Path) -> None:
        """OSError from yaml.dump → outer except wraps as StorageError (line 193)."""
        target = tmp_path / "data.yaml"
        mock_yaml = MagicMock()
        mock_yaml.dump.side_effect = OSError("disk full")
        with patch("trw_memory.storage.persistence._new_yaml", return_value=mock_yaml):
            with pytest.raises(StorageError, match="Failed to write YAML"):
                write_yaml(target, {"key": "value"})


class TestAppendJsonlStorageError:
    def test_oserror_in_append_raises_storage_error(self, tmp_path: Path) -> None:
        """OSError from Path.open → StorageError (lines 229-230)."""
        target = tmp_path / "log.jsonl"
        with patch.object(Path, "open", side_effect=OSError("no space")):
            with pytest.raises(StorageError, match="Failed to append JSONL"):
                append_jsonl(target, {"event": "test"})
