"""Tests for trw_memory.cli_formatters."""

from __future__ import annotations

import json
from typing import Any

from trw_memory.cli_formatters import (
    format_export_summary,
    format_import_summary,
    format_results,
    format_status,
    format_store_result,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


_SENTINEL_TAGS: list[str] = ["py", "test"]


def _make_result(
    memory_id: str = "M-abc12345",
    score: float = 0.85,
    importance: float = 0.9,
    tags: list[str] | None = None,
    content: str = "Pydantic v2 requires strict=True",
) -> dict[str, Any]:
    return {
        "memory_id": memory_id,
        "score": score,
        "importance": importance,
        "tags": _SENTINEL_TAGS if tags is None else tags,
        "content": content,
        "detail": "",
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
        "namespace": "default",
    }


# ---------------------------------------------------------------------------
# format_results — table
# ---------------------------------------------------------------------------


class TestFormatResultsTable:
    def test_empty_results(self) -> None:
        out = format_results([], fmt="table")
        assert out == "No results found."

    def test_single_result_has_header_and_row(self) -> None:
        results = [_make_result()]
        out = format_results(results, fmt="table")
        lines = out.split("\n")
        assert len(lines) == 3  # header + separator + 1 row
        assert "ID" in lines[0]
        assert "Score" in lines[0]
        assert "Importance" in lines[0]
        assert "M-abc12345" in lines[2]

    def test_multiple_results(self) -> None:
        results = [
            _make_result(memory_id="M-aaa"),
            _make_result(memory_id="M-bbb", score=0.5),
        ]
        out = format_results(results, fmt="table")
        lines = out.split("\n")
        assert len(lines) == 4  # header + sep + 2 rows

    def test_long_content_truncated(self) -> None:
        long_content = "A" * 200
        results = [_make_result(content=long_content)]
        out = format_results(results, fmt="table")
        assert "..." in out
        # Full 200-char content should NOT appear
        assert long_content not in out

    def test_empty_tags(self) -> None:
        results = [_make_result(tags=[])]
        out = format_results(results, fmt="table")
        assert "[]" in out

    def test_default_format_is_table(self) -> None:
        results = [_make_result()]
        out = format_results(results)
        assert "ID" in out  # Table header present


# ---------------------------------------------------------------------------
# format_results — json
# ---------------------------------------------------------------------------


class TestFormatResultsJson:
    def test_empty_results_json(self) -> None:
        out = format_results([], fmt="json")
        parsed = json.loads(out)
        assert parsed == []

    def test_single_result_json(self) -> None:
        results = [_make_result()]
        out = format_results(results, fmt="json")
        parsed = json.loads(out)
        assert len(parsed) == 1
        assert parsed[0]["memory_id"] == "M-abc12345"

    def test_json_preserves_all_fields(self) -> None:
        r = _make_result()
        out = format_results([r], fmt="json")
        parsed = json.loads(out)
        assert parsed[0]["score"] == 0.85
        assert parsed[0]["tags"] == ["py", "test"]


# ---------------------------------------------------------------------------
# format_results — compact
# ---------------------------------------------------------------------------


class TestFormatResultsCompact:
    def test_empty_results_compact(self) -> None:
        out = format_results([], fmt="compact")
        assert out == "No results found."

    def test_single_result_compact(self) -> None:
        results = [_make_result()]
        out = format_results(results, fmt="compact")
        assert "M-abc12345" in out
        assert "score=0.85" in out
        assert "Pydantic" in out

    def test_multiple_results_compact_one_per_line(self) -> None:
        results = [_make_result(memory_id=f"M-{i}") for i in range(3)]
        out = format_results(results, fmt="compact")
        lines = out.strip().split("\n")
        assert len(lines) == 3


# ---------------------------------------------------------------------------
# format_status
# ---------------------------------------------------------------------------


class TestFormatStatus:
    def test_table_format(self) -> None:
        status = {"namespace": "default", "entry_count": 42, "backend": "sqlite"}
        out = format_status(status, fmt="table")
        assert "Memory System Status" in out
        assert "Namespace" in out
        assert "42" in out

    def test_json_format(self) -> None:
        status = {"namespace": "default", "entry_count": 42}
        out = format_status(status, fmt="json")
        parsed = json.loads(out)
        assert parsed["entry_count"] == 42

    def test_empty_status(self) -> None:
        out = format_status({}, fmt="table")
        assert "Memory System Status" in out


# ---------------------------------------------------------------------------
# format_store_result
# ---------------------------------------------------------------------------


class TestFormatStoreResult:
    def test_basic_output(self) -> None:
        result = {
            "memory_id": "M-abc123",
            "namespace": "default",
            "status": "stored",
            "timestamp": "2026-01-01T00:00:00+00:00",
        }
        out = format_store_result(result)
        assert "Stored:" in out
        assert "M-abc123" in out
        assert "default" in out

    def test_missing_fields(self) -> None:
        out = format_store_result({})
        assert "Stored:" in out
        assert "unknown" in out


# ---------------------------------------------------------------------------
# format_export_summary / format_import_summary
# ---------------------------------------------------------------------------


class TestExportImportSummaries:
    def test_export_to_file(self) -> None:
        out = format_export_summary(10, "/tmp/export.json")
        assert "10 entries" in out
        assert "/tmp/export.json" in out

    def test_export_to_stdout(self) -> None:
        out = format_export_summary(5, None)
        assert "5 entries" in out
        assert "stdout" in out

    def test_import_summary(self) -> None:
        out = format_import_summary(8, 2)
        assert "8" in out
        assert "2" in out
        assert "Imported" in out
        assert "skipped" in out

    def test_import_zero_skipped(self) -> None:
        out = format_import_summary(3, 0)
        assert "3" in out
        assert "0" in out
