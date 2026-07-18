"""Tests for pure in-memory/file-backed code indexing behavior."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

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
    (tmp_path / "vendor").mkdir()
    (tmp_path / "vendor" / "vendored.py").write_text("def vendored():\n    return 1\n", encoding="utf-8")
    (tmp_path / "build").mkdir()
    (tmp_path / "build" / "generated.py").write_text("def generated():\n    return 1\n", encoding="utf-8")
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "passwords.py").write_text("PASSWORD = 'example'\n", encoding="utf-8")
    (tmp_path / "secrets.env").write_text("TOKEN=abc", encoding="utf-8")
    (tmp_path / "image.png").write_bytes(b"\x89PNG\x00")
    (tmp_path / "app.py").write_text("def ok() -> int:\n    return 1\n", encoding="utf-8")
    store = InMemoryCodeIndex()

    result = CodeIndexer(root=tmp_path, store=store, namespace="project").index()

    assert result.indexed_files == 1
    assert result.skipped_excluded == 6
    assert [code_file.path for code_file in store.list_files(namespace="project")] == ["app.py"]


def test_indexer_rejects_external_source_symlink(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("API_KEY = 'outside-secret'\n", encoding="utf-8")
    link = root / "innocent.py"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks unavailable")
    store = InMemoryCodeIndex()

    result = CodeIndexer(root=root, store=store, namespace="project").index()

    assert result.indexed_files == 0
    assert result.skipped_excluded == 1
    assert store.list_files(namespace="project") == []
    assert "outside-secret" not in repr(store.list_chunks(namespace="project"))


def test_indexer_rejects_internal_source_symlink_alias(tmp_path: Path) -> None:
    source = tmp_path / "real.py"
    source.write_text("def real() -> int:\n    return 1\n", encoding="utf-8")
    alias = tmp_path / "alias.py"
    try:
        alias.symlink_to(source)
    except OSError:
        pytest.skip("symlinks unavailable")
    store = InMemoryCodeIndex()

    result = CodeIndexer(root=tmp_path, store=store, namespace="project").index()

    assert result.indexed_files == 1
    assert result.skipped_excluded == 1
    assert [code_file.path for code_file in store.list_files(namespace="project")] == ["real.py"]


def test_indexer_contains_parent_directory_swap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    if os.open not in os.supports_dir_fd or not hasattr(os, "O_NOFOLLOW"):
        pytest.skip("secure directory-descriptor traversal unavailable")
    root = tmp_path / "repo"
    source_dir = root / "pkg"
    source_dir.mkdir(parents=True)
    (source_dir / "service.py").write_text("SAFE = True\n", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "service.py").write_text("API_KEY = 'outside-secret'\n", encoding="utf-8")
    original_open = os.open
    opened_leaf_fds: list[int] = []
    swapped = False

    def _swap_parent(path: str | bytes | Path, flags: int, mode: int = 0o777, *, dir_fd: int | None = None) -> int:
        nonlocal swapped
        if path == "service.py" and dir_fd is not None and not swapped:
            swapped = True
            source_dir.rename(root / "pkg-original")
            source_dir.symlink_to(outside, target_is_directory=True)
        fd = original_open(path, flags, mode, dir_fd=dir_fd)
        if path == "service.py":
            opened_leaf_fds.append(fd)
        return fd

    monkeypatch.setattr("trw_memory.code_index.indexer.os.open", _swap_parent)
    store = InMemoryCodeIndex()

    result = CodeIndexer(root=root, store=store, namespace="project").index()

    assert result.indexed_files == 0
    assert result.skipped_excluded == 1
    assert "outside-secret" not in repr(store.list_chunks(namespace="project"))
    assert opened_leaf_fds
    with pytest.raises(OSError):
        os.fstat(opened_leaf_fds[0])
