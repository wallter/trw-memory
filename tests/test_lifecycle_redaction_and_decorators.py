"""Wave 12: coverage for lifecycle/_redaction.py and decorators.py."""
from __future__ import annotations


class TestRedactPaths:
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


class TestDecoratorsReExport:
    def test_memory_client_importable_from_decorators(self) -> None:
        from trw_memory.decorators import MemoryClient

        assert MemoryClient is not None
