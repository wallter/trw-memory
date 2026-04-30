"""Shared test fixtures for the trw-memory test suite.

Test Tiering Philosophy
-----------------------
Tests are classified by their resource usage:

- **unit**: Pure logic — in-memory backends or no I/O at all, no ``tmp_path``.
  Target: <30s for the full unit tier.
- **integration**: Tests that write files, use SQLite on disk, or exercise
  the full ``MemoryClient`` stack with real storage.
- **slow**: Tests loading sentence-transformer models or running full
  consolidation cycles (individual runtime >5s).

To classify a test file:
  1. Uses ``tmp_path`` or real disk backends → integration (default).
  2. Only patches/mocks or uses ``:memory:`` SQLite → unit.
  3. Loads sentence-transformers or runs 100+ dedup cycles → slow.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from trw_memory.client import MemoryClient
from trw_memory.graph import wait_for_graph_updates
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry, MemoryStatus
from trw_memory.security.keys import clear_key_cache
from trw_memory.storage.sqlite_backend import SQLiteBackend


@pytest.fixture(autouse=True)
def drain_background_graph_updates() -> Iterator[None]:
    """Finish graph worker threads before pytest closes per-test capture streams."""
    yield
    try:
        wait_for_graph_updates(timeout=1.0)
    except TimeoutError:
        # Graph enrichment is best-effort; tests should not hang if a worker is
        # already blocked on an intentionally fault-injected backend.
        pass


@pytest.fixture(autouse=True)
def clear_master_key_cache_fixture() -> Iterator[None]:
    clear_key_cache()
    yield
    clear_key_cache()


# ---------------------------------------------------------------------------
# SQLiteBackend fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def sqlite_backend(tmp_path: Path) -> Iterator[SQLiteBackend]:
    """Return an initialized SQLiteBackend using a temp-dir database.

    Use this in integration tests that need a real disk-backed store.
    For unit tests that need a backend, use ``sqlite_memory_backend`` instead.
    """
    db = SQLiteBackend(tmp_path / "test.db")
    yield db
    db.close()


@pytest.fixture()
def sqlite_memory_backend() -> Iterator[SQLiteBackend]:
    """Return an initialized in-memory SQLiteBackend.

    Use this in unit tests — no filesystem I/O, safe for the unit tier.
    """
    db = SQLiteBackend(Path(":memory:"))
    yield db
    db.close()


# ---------------------------------------------------------------------------
# MemoryClient fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def memory_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> MemoryClient:
    """Return a MemoryClient backed by a SQLite store in ``tmp_path``.

    Sets ``MEMORY_STORAGE_PATH`` and ``MEMORY_STORAGE_BACKEND`` env vars so
    the client does not accidentally write to the real user store.
    """
    monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "mem_storage"))
    monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "sqlite")
    return MemoryClient(namespace="default", mode="local")


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> MemoryClient:
    monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "storage"))
    monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "sqlite")
    return MemoryClient(namespace="default", mode="local")


@pytest.fixture()
def yaml_memory_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> MemoryClient:
    """Return a MemoryClient backed by a YAML store in ``tmp_path``."""
    monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "yaml_storage"))
    monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "yaml")
    return MemoryClient(namespace="default", mode="local")


@pytest.fixture()
def yaml_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> MemoryClient:
    monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "yaml_storage"))
    monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "yaml")
    return MemoryClient(namespace="default", mode="local")


# ---------------------------------------------------------------------------
# MemoryEntry factory helpers
# ---------------------------------------------------------------------------


def make_entry(
    *,
    entry_id: str = "M-001",
    content: str = "test content",
    detail: str = "",
    tags: list[str] | None = None,
    importance: float = 0.5,
    q_value: float = 0.5,
    q_observations: int = 0,
    recurrence: int = 1,
    access_count: int = 0,
    source: str = "agent",
    status: MemoryStatus = MemoryStatus.ACTIVE,
    created_at: datetime | None = None,
    last_accessed_at: datetime | None = None,
    namespace: str = "default",
    metadata: dict[str, str] | None = None,
) -> MemoryEntry:
    """Create a ``MemoryEntry`` with sensible defaults.

    This is the canonical factory for unit tests — it avoids repetitive
    per-file ``_make_entry`` / ``_entry`` helpers that were duplicated
    across 10+ test files.

    Example::

        entry = make_entry(content="use absolute paths", tags=["gotcha"])
        assert entry.importance == 0.5
    """
    now = datetime.now(timezone.utc)
    return MemoryEntry(
        id=entry_id,
        content=content,
        detail=detail,
        tags=tags or [],
        importance=importance,
        q_value=q_value,
        q_observations=q_observations,
        recurrence=recurrence,
        access_count=access_count,
        source=source,  # type: ignore[arg-type]  # validator coerces str to Literal
        status=status,
        created_at=created_at or now,
        last_accessed_at=last_accessed_at or now,
        namespace=namespace,
        metadata=metadata or {},
    )


def make_entry_dict(
    *,
    entry_id: str = "M-001",
    content: str = "test content",
    detail: str = "",
    tags: list[str] | None = None,
    importance: float = 0.5,
    q_value: float = 0.5,
    q_observations: int = 0,
    recurrence: int = 1,
    access_count: int = 0,
    source: str = "agent",
    status: str = "active",
    created_at: datetime | None = None,
    last_accessed_at: datetime | None = None,
) -> dict[str, Any]:
    """Create a minimal entry dict matching the ``MemoryEntry`` serialised shape.

    Use this when the code under test expects a plain dict rather than a
    ``MemoryEntry`` model (e.g., scoring functions, lifecycle utilities).
    """
    now = datetime.now(timezone.utc)
    return {
        "id": entry_id,
        "content": content,
        "detail": detail,
        "tags": tags or [],
        "importance": importance,
        "q_value": q_value,
        "q_observations": q_observations,
        "recurrence": recurrence,
        "access_count": access_count,
        "source": source,
        "status": status,
        "created_at": (created_at or now).isoformat(),
        "last_accessed_at": (last_accessed_at or now).isoformat(),
    }


# ---------------------------------------------------------------------------
# MemoryConfig fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def memory_config(tmp_path: Path) -> MemoryConfig:
    """Return a MemoryConfig pointing to ``tmp_path`` for storage.

    Avoids hardcoding paths in tests that need a config object but don't
    care about the storage backend specifics.
    """
    return MemoryConfig(storage_path=str(tmp_path / "mem"))
