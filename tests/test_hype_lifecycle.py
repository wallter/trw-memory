"""PRD-CORE-195 FR05 — HyPE sibling lifecycle (forget purge, re-store overwrite)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("sqlite_vec")

from trw_memory.client import MemoryClient
from tests.test_hype_store import _FakeEmbedder, _ListGenerator


def _client(tmp_path: Path, generator: object) -> MemoryClient:
    client = MemoryClient(
        "default",
        mode="local",
        db_path=tmp_path / "lifecycle.db",
        question_generator=generator,  # type: ignore[arg-type]
    )
    client._config.hype_enabled = True
    client._get_embedder = lambda: _FakeEmbedder()  # type: ignore[method-assign]
    return client


async def test_forget_purges_siblings(tmp_path: Path) -> None:
    client = _client(tmp_path, _ListGenerator(["a good question about strict mode"]))
    try:
        await client.store("content", entry_id="P1")
        assert client._get_backend().hype_sibling_ids("P1")
        await client.forget("P1")
        assert client._get_backend().hype_sibling_ids("P1") == []
    finally:
        await client.close()


async def test_restore_overwrites_without_stale_accumulation(tmp_path: Path) -> None:
    client = _client(tmp_path, _ListGenerator(["question one padded out", "question two padded out"]))
    try:
        await client.store("content v1", entry_id="P2")
        first = set(client._get_backend().hype_sibling_ids("P2"))
        assert len(first) == 2
        # Re-store with a generator yielding a SINGLE question → siblings must
        # be overwritten (purge-then-regenerate), not appended.
        client._question_generator = _ListGenerator(["only one question now padded"])
        await client.store("content v2", entry_id="P2")
        second = client._get_backend().hype_sibling_ids("P2")
        assert len(second) == 1  # no stale accumulation
    finally:
        await client.close()


async def test_delete_hype_siblings_idempotent(tmp_path: Path) -> None:
    client = _client(tmp_path, _ListGenerator(["a good question about strict mode"]))
    try:
        await client.store("content", entry_id="P3")
        backend = client._get_backend()
        removed_first = backend.delete_hype_siblings("P3")
        removed_second = backend.delete_hype_siblings("P3")
        assert removed_first >= 1
        assert removed_second == 0  # idempotent double-delete no-ops
    finally:
        await client.close()


def test_delete_hype_siblings_noop_without_vectors(tmp_path: Path) -> None:
    # NFR03: a backend without vec support no-ops, never raises.
    from trw_memory.storage.yaml_backend import YAMLBackend

    backend = YAMLBackend(tmp_path / "yaml")
    assert backend.delete_hype_siblings("anything") == 0
    assert backend.hype_sibling_ids("anything") == []
