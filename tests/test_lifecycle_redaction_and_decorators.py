"""Wave 12: coverage for lifecycle/_redaction.py and decorators.py."""

from __future__ import annotations

import pytest


class TestRedactPaths:
    def test_consolidation_alias_uses_canonical_redactor(self) -> None:
        from trw_memory.lifecycle._redaction import redact_paths
        from trw_memory.lifecycle.consolidation import _redact_paths

        assert _redact_paths is redact_paths

    def test_linux_home_path_redacted(self) -> None:
        from trw_memory.lifecycle._redaction import redact_paths

        result = redact_paths("stored at /home/alice/projects/foo/bar.db")
        assert "[REDACTED_PATH]" in result
        assert "/home/alice" not in result

    def test_windows_drive_path_redacted(self) -> None:
        from trw_memory.lifecycle._redaction import redact_paths

        result = redact_paths("stored at C:\\Users\\bob\\docs")
        assert "[REDACTED_PATH]" in result

    def test_tmp_path_redacted(self) -> None:
        from trw_memory.lifecycle._redaction import redact_paths

        result = redact_paths("file: /tmp/trw-1234/memory.db")
        assert "[REDACTED_PATH]" in result

    def test_non_path_text_unchanged(self) -> None:
        from trw_memory.lifecycle._redaction import redact_paths

        result = redact_paths("no paths here, just text")
        assert result == "no paths here, just text"

    @pytest.mark.parametrize(
        "path",
        ["~/projects/secrets/key.pem", "./config.yaml", "../certs/ca.pem"],
    )
    def test_relative_path_redacted(self, path: str) -> None:
        from trw_memory.lifecycle._redaction import redact_paths

        result = redact_paths(f"stored at {path} here")
        assert "[REDACTED_PATH]" in result
        assert path not in result

    def test_plain_prose_with_punctuation_not_redacted(self) -> None:
        from trw_memory.lifecycle._redaction import redact_paths

        text = "Consider the options, e.g. retry. Done."
        assert redact_paths(text) == text


class TestDecoratorsReExport:
    def test_memory_client_importable_from_decorators(self) -> None:
        from trw_memory.decorators import MemoryClient

        assert MemoryClient is not None
