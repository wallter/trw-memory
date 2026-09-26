"""Wave 15: coverage gap-fill for cli_json_input.py (lines 44-45)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from trw_memory.cli_json_input import JsonInputError, read_source_text


class TestReadSourceTextOsError:
    def test_oserror_raises_json_input_error(self, tmp_path: Path) -> None:
        """OSError (not FileNotFoundError/IsADirectoryError) → JsonInputError (lines 44-45)."""
        target = tmp_path / "file.json"
        target.write_bytes(b"")
        with patch.object(Path, "read_bytes", side_effect=PermissionError("denied")):
            with pytest.raises(JsonInputError, match="cannot read"):
                read_source_text(target, source="file.json")
