"""Offline with no cached model, the daemon's tools fall back to keyword search (L-0P5T).

``get_local_embedder`` refuses an uncached model under ``TRW_OFFLINE`` by raising
``LocalOnlyViolationError``. Its keyword-only degradation used to live in
trw-mcp's embedder wrapper; since recall runs in the daemon nothing caught it, so
every recall, store and consolidate failed on an offline machine without the model.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from trw_memory.embeddings import reset_provider_cache
from trw_memory.exceptions import RemoteCodeNotPermittedError
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.storage.sqlite_backend import SQLiteBackend
from trw_memory.tools.consolidate import memory_consolidate_impl
from trw_memory.tools.recall import memory_recall_impl
from trw_memory.tools.store import memory_store_impl

pytest.importorskip("sentence_transformers", reason="the refusal comes from the real model loader")

_NS = "project:default"


@pytest.fixture
def offline_without_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """TRW_OFFLINE with every model cache pointed at an empty directory."""
    monkeypatch.setenv("TRW_OFFLINE", "1")
    for var in ("HF_HOME", "SENTENCE_TRANSFORMERS_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE"):
        monkeypatch.setenv(var, str(tmp_path / "empty-model-cache"))
    reset_provider_cache()
    yield
    reset_provider_cache()


@pytest.fixture
def backend(tmp_path: Path) -> Iterator[SQLiteBackend]:
    store = SQLiteBackend(tmp_path / "memory.db")
    store.store(MemoryEntry(id="L-async", content="python async event loop", namespace=_NS))
    store.store(MemoryEntry(id="L-other", content="terraform state locking", namespace=_NS))
    yield store
    store.close()


@pytest.mark.usefixtures("offline_without_model")
def test_recall_answers_by_keyword_and_says_the_dense_arm_is_off(backend: SQLiteBackend) -> None:
    result = memory_recall_impl("python async", _NS, backend=backend, config=MemoryConfig())

    ids = [row["id"] for row in result["memories"]]  # type: ignore[union-attr]
    assert ids[:1] == ["L-async"]
    assert str(result["dense"]).startswith("unavailable: ")
    assert "BAAI/bge-small-en-v1.5" in str(result["dense"])


@pytest.mark.usefixtures("offline_without_model")
def test_store_writes_the_row_without_a_vector(backend: SQLiteBackend) -> None:
    result = memory_store_impl("offline learning about queues", _NS, backend=backend, config=MemoryConfig())

    assert "error" not in result, result
    stored = backend.get(str(result["memory_id"]), namespace=_NS)
    assert stored is not None and stored.content == "offline learning about queues"


@pytest.mark.usefixtures("offline_without_model")
def test_consolidate_runs_without_dense_similarity(backend: SQLiteBackend) -> None:
    result = memory_consolidate_impl(_NS, backend=backend, dry_run=True, config=MemoryConfig())

    assert "error" not in result


def test_a_remote_code_refusal_still_raises(backend: SQLiteBackend, monkeypatch: pytest.MonkeyPatch) -> None:
    """Only the offline refusal degrades: a model that needs remote code is a configuration to fix, not a mode."""

    def refuse(**_kwargs: object) -> None:
        raise RemoteCodeNotPermittedError("model requires trust_remote_code")

    monkeypatch.setattr("trw_memory.tools.recall.get_local_embedder", refuse)
    reset_provider_cache()

    with pytest.raises(RemoteCodeNotPermittedError):
        memory_recall_impl("python async", _NS, backend=backend, config=MemoryConfig())
