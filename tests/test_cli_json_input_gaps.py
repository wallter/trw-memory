"""Wave 15: coverage gap-fill for cli_json_input.py (lines 44-45, 82, 84, 88, 91-93)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from trw_memory.cli_json_input import JsonInputError, json_type_name, read_source_text


class TestReadSourceTextOsError:
    def test_oserror_raises_json_input_error(self, tmp_path: Path) -> None:
        """OSError (not FileNotFoundError/IsADirectoryError) → JsonInputError (lines 44-45)."""
        target = tmp_path / "file.json"
        target.write_bytes(b"")
        with patch.object(Path, "read_bytes", side_effect=PermissionError("denied")):
            with pytest.raises(JsonInputError, match="cannot read"):
                read_source_text(target, source="file.json")


class TestJsonTypeName:
    def test_none_returns_null(self) -> None:
        """None → 'null' (line 82)."""
        assert json_type_name(None) == "null"

    def test_bool_returns_boolean(self) -> None:
        """bool → 'boolean' (line 84)."""
        assert json_type_name(True) == "boolean"
        assert json_type_name(False) == "boolean"

    def test_int_returns_number(self) -> None:
        """int → 'number' (line 88)."""
        assert json_type_name(42) == "number"

    def test_float_returns_number(self) -> None:
        """float → 'number' (line 88)."""
        assert json_type_name(3.14) == "number"

    def test_list_returns_array(self) -> None:
        """list → 'array' (line 91-92)."""
        assert json_type_name([1, 2, 3]) == "array"

    def test_unknown_type_returns_class_name(self) -> None:
        """Unrecognised type → type.__name__ (line 93)."""

        class MyObj:
            pass

        assert json_type_name(MyObj()) == "MyObj"
