"""Shared helpers for split SQLite storage tests."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

import pytest

from trw_memory.models.memory import MemoryEntry, MemoryStatus
from trw_memory.storage.sqlite_backend import SQLiteBackend


def make_entry(
    entry_id: str,
    content: str = "test content",
    *,
    detail: str = "",
    tags: list[str] | None = None,
    importance: float = 0.5,
    status: MemoryStatus = MemoryStatus.ACTIVE,
    namespace: str = "default",
    source: str = "agent",
) -> MemoryEntry:
    now = datetime.now(timezone.utc)
    return MemoryEntry(
        id=entry_id,
        content=content,
        detail=detail,
        tags=tags or [],
        importance=importance,
        status=status,
        namespace=namespace,
        source=source,
        created_at=now,
        updated_at=now,
    )


@pytest.fixture()
def backend(tmp_path: Path) -> Iterator[SQLiteBackend]:
    db = SQLiteBackend(tmp_path / "test.db")
    yield db
    db.close()
