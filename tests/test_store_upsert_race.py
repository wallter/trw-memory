"""C12 rc8: ``memory_store(entry_id=...)`` reads the row before embedding, off the serialized lane.

A forget, update or other store that lands in that window must not be overwritten by the stale
revision (it restored an ACTIVE status over a retirement); a recall's counter bump must not
refuse the store.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryStatus
from trw_memory.storage.sqlite_backend import SQLiteBackend
from trw_memory.tools import store as store_module
from trw_memory.tools.store import memory_store_impl

_NS = "project:default"


@pytest.fixture
def backend(tmp_path: Path) -> Iterator[SQLiteBackend]:
    store = SQLiteBackend(tmp_path / "memory.db")
    try:
        yield store
    finally:
        store.close()


def _store_with(
    backend: SQLiteBackend, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, meanwhile: Callable[[], object]
) -> dict[str, object]:
    """Store M-1's second revision, running *meanwhile* after the row is read, before the write."""
    cfg = MemoryConfig(storage_path=str(tmp_path / "mem"))
    assert memory_store_impl("first", _NS, backend=backend, config=cfg, entry_id="M-1")["status"] == "stored"
    prepare = store_module.prepare_entry_for_store

    def racing(*args: object, **kwargs: object) -> object:
        meanwhile()
        return prepare(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(store_module, "prepare_entry_for_store", racing)
    return memory_store_impl("second", _NS, backend=backend, config=cfg, entry_id="M-1")


def test_a_retirement_landing_mid_store_is_not_undone(
    backend: SQLiteBackend, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _store_with(
        backend, tmp_path, monkeypatch, lambda: backend.update("M-1", namespace=_NS, status=MemoryStatus.OBSOLETE)
    )

    assert result["status"] == "conflict"
    row = backend.get("M-1", namespace=_NS)
    assert row is not None
    assert (row.status, row.content) == (MemoryStatus.OBSOLETE, "first")


def test_a_forget_landing_mid_store_is_not_resurrected(
    backend: SQLiteBackend, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _store_with(backend, tmp_path, monkeypatch, lambda: backend.delete("M-1", namespace=_NS))

    assert result["status"] == "conflict"
    assert backend.get("M-1", namespace=_NS) is None


def test_a_recall_bump_landing_mid_store_does_not_refuse_it(
    backend: SQLiteBackend, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _store_with(
        backend, tmp_path, monkeypatch, lambda: backend.increment_recall_access(["M-1"], namespace=_NS)
    )

    assert result["status"] in ("stored", "updated")
    row = backend.get("M-1", namespace=_NS)
    assert row is not None
    assert row.content == "second"


def test_conflicting_stores_charge_no_rate_limit_slot(
    backend: SQLiteBackend, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B71-81: a store refused as ``conflict`` wrote nothing, so it must not spend a write slot."""
    cfg = MemoryConfig(
        storage_path=str(tmp_path / "mem"),
        rate_limit_state_path=str(tmp_path / "rl.yaml"),
        max_memory_writes_per_minute=2,
    )

    def store(content: str) -> dict[str, object]:
        return memory_store_impl(content, _NS, backend=backend, config=cfg, entry_id="M-1", session_id="s")

    assert store("first")["status"] == "stored"  # slot 1 of 2
    prepare = store_module.prepare_entry_for_store
    edits = iter(range(10))

    def racing(*args: object, **kwargs: object) -> object:
        backend.update("M-1", namespace=_NS, content=f"edit {next(edits)}")
        return prepare(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(store_module, "prepare_entry_for_store", racing)
    assert [store("stale")["status"] for _ in range(3)] == ["conflict"] * 3
    monkeypatch.setattr(store_module, "prepare_entry_for_store", prepare)

    assert store("second")["status"] == "updated"  # slot 2 of 2: the conflicts spent none
    assert store("third")["status"] == "rate_limited"  # the limiter itself still binds
