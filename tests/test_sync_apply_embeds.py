"""SHARED-RECALL-LOCAL item 1: a pulled team learning is embedded as it is stored.

A pulled row used to land with no vector: 599 rows pulled in one day dropped a
store's embedding coverage to 0.82 until an operator ran ``memory reembed``, and
until then dense recall could not find them. ``memory_sync_apply`` now encodes
the row the way ``memory_store`` does, in the same transaction as the row.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.storage._shared import revision_of
from trw_memory.storage.sqlite_backend import SQLiteBackend
from trw_memory.tools import sync as sync_tools
from trw_memory.tools.sync import memory_sync_apply_impl

_NS = "project:alpha-11111111"


class _Embedder:
    """Deterministic 384-dim encoder that records every text it encoded."""

    def __init__(self, *, fail: bool = False) -> None:
        self.texts: list[str] = []
        self._fail = fail

    def embed(self, text: str) -> list[float] | None:
        self.texts.append(text)
        if self._fail:
            raise RuntimeError("encoder crashed")
        return [float(len(text) % 7 + 1)] + [0.0] * 383

    def embed_batch(self, texts: list[str]) -> list[list[float] | None]:
        return [self.embed(text) for text in texts]

    def available(self) -> bool:
        return True

    def dim(self) -> int:
        return 384


@pytest.fixture
def backend(tmp_path: Path) -> Iterator[SQLiteBackend]:
    store = SQLiteBackend(tmp_path / "memory.db")
    if not store.supports_vectors():
        pytest.skip("sqlite-vec is not installed")
    yield store
    store.close()


@pytest.fixture
def config(tmp_path: Path) -> MemoryConfig:
    return MemoryConfig(storage_path=str(tmp_path))


def _pulled(entry_id: str, content: str, detail: str = "") -> dict[str, object]:
    row = MemoryEntry(id=entry_id, content=content, detail=detail, namespace=_NS, remote_id=f"R-{entry_id}")
    return row.model_copy(update={"source": "team_sync"}).model_dump(mode="json")


def test_a_pulled_row_is_stored_with_its_vector(
    backend: SQLiteBackend, config: MemoryConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    embedder = _Embedder()
    monkeypatch.setattr(sync_tools, "resolve_embedder", lambda _cfg, *, surface: embedder)

    answer = memory_sync_apply_impl(
        _NS, _pulled("T-1", "pin the wheel cache", "before a release"), backend=backend, config=config, if_revision=None
    )

    assert answer == {"status": "stored", "reason": ""}
    assert backend.vector_exists("T-1", namespace=_NS)
    assert embedder.texts == ["pin the wheel cache before a release"]


def test_a_pulled_row_still_lands_when_no_embedder_can_load(
    backend: SQLiteBackend, config: MemoryConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keyword-only hosts keep pulling; the row waits for ``memory reembed`` and coverage shows it."""
    monkeypatch.setattr(
        sync_tools, "resolve_embedder", lambda _cfg, *, surface: {"status": "unavailable", "reason": "model_not_cached"}
    )

    answer = memory_sync_apply_impl(
        _NS, _pulled("T-2", "no model here"), backend=backend, config=config, if_revision=None
    )

    assert answer["status"] == "stored"
    assert backend.get("T-2", namespace=_NS) is not None
    assert not backend.vector_exists("T-2", namespace=_NS)


def test_an_encoder_failure_stores_the_row_without_a_vector(
    backend: SQLiteBackend, config: MemoryConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One bad encode must not hold the pull cursor on that item (PRD-FIX-138 shape)."""
    monkeypatch.setattr(sync_tools, "resolve_embedder", lambda _cfg, *, surface: _Embedder(fail=True))

    answer = memory_sync_apply_impl(_NS, _pulled("T-3", "crashy"), backend=backend, config=config, if_revision=None)

    assert answer["status"] == "stored"
    assert not backend.vector_exists("T-3", namespace=_NS)


def test_a_merge_that_changes_the_text_replaces_the_vector(
    backend: SQLiteBackend, config: MemoryConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    embedder = _Embedder()
    monkeypatch.setattr(sync_tools, "resolve_embedder", lambda _cfg, *, surface: embedder)
    memory_sync_apply_impl(_NS, _pulled("T-4", "first"), backend=backend, config=config, if_revision=None)
    before = backend.get_vector_records(["T-4"], namespace=_NS)["T-4"]
    current = backend.get("T-4", namespace=_NS)

    revised = {**_pulled("T-4", "first, now with a much longer revision"), "sync_seq": current.sync_seq}  # type: ignore[union-attr]
    answer = memory_sync_apply_impl(_NS, revised, backend=backend, config=config, if_revision=revision_of(current))

    after = backend.get_vector_records(["T-4"], namespace=_NS)["T-4"]
    assert answer["status"] == "stored"
    assert embedder.texts[-1] == "first, now with a much longer revision "
    assert list(after.embedding) != list(before.embedding)


def test_a_text_change_with_no_encoder_drops_the_stale_vector(
    backend: SQLiteBackend, config: MemoryConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A vector is never left beside text it was not computed from (PRD-CORE-302 C5)."""
    monkeypatch.setattr(sync_tools, "resolve_embedder", lambda _cfg, *, surface: _Embedder())
    memory_sync_apply_impl(_NS, _pulled("T-5", "old text"), backend=backend, config=config, if_revision=None)
    current = backend.get("T-5", namespace=_NS)
    monkeypatch.setattr(
        sync_tools, "resolve_embedder", lambda _cfg, *, surface: {"status": "unavailable", "reason": "embedder_error"}
    )

    answer = memory_sync_apply_impl(
        _NS, _pulled("T-5", "new text"), backend=backend, config=config, if_revision=revision_of(current)
    )

    assert answer["status"] == "stored"
    assert not backend.vector_exists("T-5", namespace=_NS)
