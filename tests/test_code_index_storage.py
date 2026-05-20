"""Tests for pure in-memory/file-backed code indexing behavior."""

from __future__ import annotations

from pathlib import Path

from trw_memory.code_index.indexer import CodeIndexer, InMemoryCodeIndex


def test_indexer_skips_unchanged_files_by_hash_and_reindexes_changed_files(tmp_path: Path) -> None:
    source = tmp_path / "pkg" / "service.py"
    source.parent.mkdir()
    source.write_text("def alpha() -> int:\n    return 1\n", encoding="utf-8")
    store = InMemoryCodeIndex()
    indexer = CodeIndexer(root=tmp_path, store=store, namespace="project")

    first = indexer.index()
    second = indexer.index()
    source.write_text("def alpha() -> int:\n    return 2\n", encoding="utf-8")
    third = indexer.index()

    assert first.indexed_files == 1
    assert first.skipped_unchanged == 0
    assert second.indexed_files == 0
    assert second.skipped_unchanged == 1
    assert third.indexed_files == 1
    assert third.skipped_unchanged == 0
    assert len(store.list_chunks(namespace="project")) == 1


def test_indexer_removes_chunks_and_symbols_for_deleted_files(tmp_path: Path) -> None:
    source = tmp_path / "pkg" / "service.py"
    source.parent.mkdir()
    source.write_text("def alpha() -> int:\n    return 1\n", encoding="utf-8")
    store = InMemoryCodeIndex()
    indexer = CodeIndexer(root=tmp_path, store=store, namespace="project")

    indexer.index()
    source.unlink()
    result = indexer.index()

    assert result.deleted_files == 1
    assert store.list_files(namespace="project") == []
    assert store.list_chunks(namespace="project") == []
    assert store.list_symbols(namespace="project") == []


def test_indexer_default_excludes_skip_vendor_build_binary_large_and_secret_like_files(tmp_path: Path) -> None:
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "dep.py").write_text("def dep():\n    return 1\n", encoding="utf-8")
    (tmp_path / "build").mkdir()
    (tmp_path / "build" / "generated.py").write_text("def generated():\n    return 1\n", encoding="utf-8")
    (tmp_path / "secrets.env").write_text("TOKEN=abc", encoding="utf-8")
    (tmp_path / "image.png").write_bytes(b"\x89PNG\x00")
    (tmp_path / "app.py").write_text("def ok() -> int:\n    return 1\n", encoding="utf-8")
    store = InMemoryCodeIndex()

    result = CodeIndexer(root=tmp_path, store=store, namespace="project").index()

    assert result.indexed_files == 1
    assert result.skipped_excluded == 4
    assert [code_file.path for code_file in store.list_files(namespace="project")] == ["app.py"]
