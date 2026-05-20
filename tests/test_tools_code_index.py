"""Tests for MCP-compatible explicit code-index tools."""

from __future__ import annotations

from pathlib import Path

from trw_memory.tools.code_index import memory_code_index_impl, memory_code_search_impl, memory_code_symbol_impl


def test_memory_code_index_and_search_are_explicit_and_bounded(tmp_path: Path) -> None:
    source = tmp_path / "src" / "service.py"
    source.parent.mkdir()
    source.write_text("def calculate_total(items: list[int]) -> int:\n    return sum(items)\n", encoding="utf-8")

    indexed = memory_code_index_impl(str(tmp_path), namespace="project")
    searched = memory_code_search_impl(str(tmp_path), "calculate total", namespace="project", limit=5)

    assert indexed["status"] == "ok"
    assert indexed["stats"]["indexed_files"] == 1
    assert searched["status"] == "ok"
    assert searched["results"][0]["file"] == "src/service.py"
    assert len(searched["results"][0]["snippet"]) <= 400


def test_memory_code_symbol_returns_disambiguated_matches(tmp_path: Path) -> None:
    source = tmp_path / "src" / "service.py"
    source.parent.mkdir()
    source.write_text("def calculate_total(items: list[int]) -> int:\n    return sum(items)\n", encoding="utf-8")

    memory_code_index_impl(str(tmp_path), namespace="project")
    result = memory_code_symbol_impl(str(tmp_path), "calculate_total", namespace="project", kind="function")

    assert result["status"] == "ok"
    assert result["results"][0]["name"] == "calculate_total"
    assert result["results"][0]["disambiguation"].startswith("python:function:src/service.py")


def test_memory_code_index_returns_structured_failure_for_invalid_root(tmp_path: Path) -> None:
    not_dir = tmp_path / "file.txt"
    not_dir.write_text("not a directory", encoding="utf-8")

    result = memory_code_index_impl(str(not_dir))

    assert result["status"] == "failed"
    assert result["error_code"] == "invalid_root"
