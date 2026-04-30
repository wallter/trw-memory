from __future__ import annotations

from pathlib import Path

import pytest

from trw_memory.client import MemoryClient


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> MemoryClient:
    """Isolated MemoryClient backed by SQLite in tmp_path."""
    monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "e2e_storage"))
    monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "sqlite")
    return MemoryClient(namespace="default", mode="local")


@pytest.fixture()
def client_ns_a(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> MemoryClient:
    """MemoryClient in namespace 'project:ns-a'."""
    monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "e2e_storage"))
    monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "sqlite")
    return MemoryClient(namespace="project:ns-a", mode="local")


@pytest.fixture()
def client_ns_b(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> MemoryClient:
    """MemoryClient in namespace 'project:ns-b'."""
    monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "e2e_storage"))
    monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "sqlite")
    return MemoryClient(namespace="project:ns-b", mode="local")
