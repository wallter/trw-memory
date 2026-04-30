"""Shared helpers for the split ``test_cold_rebuild*`` family."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
import structlog

from trw_memory.models.memory import MemoryEntry, MemoryStatus
from trw_memory.storage._schema import ensure_schema
from trw_memory.storage.persistence import write_yaml
from trw_memory.storage.sqlite_backend import SQLiteBackend


@pytest.fixture(autouse=True)
def _configure_structlog() -> Iterator[None]:
    """Ensure structlog routes through the testing capture."""
    structlog.reset_defaults()
    yield
    structlog.reset_defaults()


def _make_yaml(base_dir: Path, entry_id: str, **overrides: Any) -> Path:
    """Write a cold-tier YAML under ``base_dir/memory/cold/2026/04``."""
    cold_dir = base_dir / "memory" / "cold" / "2026" / "04"
    cold_dir.mkdir(parents=True, exist_ok=True)
    data: dict[str, object] = {
        "id": entry_id,
        "summary": f"summary for {entry_id}",
        "detail": f"detail for {entry_id}",
        "impact": 0.7,
        "status": "active",
        "recurrence": 1,
        "namespace": "default",
        "created": "2026-04-12T10:00:00+00:00",
        "updated": "2026-04-12T10:00:00+00:00",
        "tags": ["alpha", "beta"],
        "evidence": [],
        "source_type": "agent",
        "metadata": {},
        "vector_clock": {},
    }
    data.update(overrides)
    path = cold_dir / f"{entry_id}.yaml"
    write_yaml(path, data)
    return path


def _open_fresh_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    ensure_schema(conn)
    return conn


def _corrupt_sqlite_master(db_path: Path) -> None:
    """Destroy ``sqlite_master`` the same way the 2026-04-12 incident did."""
    data = db_path.read_bytes()
    db_path.write_bytes(b"\x00\xff\xfe\xfd" * 512 + data[2048:])


def _populate_real_db(db_path: Path, *, entries: int = 3) -> None:
    """Create a non-empty, structurally-valid SQLite DB."""
    backend = SQLiteBackend(db_path)
    now = datetime.now(timezone.utc)
    for idx in range(entries):
        backend.store(
            MemoryEntry(
                id=f"L-HOT{idx}",
                content=f"hot {idx}",
                importance=0.5,
                status=MemoryStatus.ACTIVE,
                namespace="default",
                source="agent",
                created_at=now,
                updated_at=now,
            )
        )
    backend.close()
