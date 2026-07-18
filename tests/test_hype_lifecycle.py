"""PRD-CORE-195 FR05 — HyPE sibling lifecycle (forget purge, re-store overwrite)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("sqlite_vec")

from tests.conftest import make_entry
from tests.test_hype_store import _FakeEmbedder, _ListGenerator
from trw_memory.client import MemoryClient


@pytest.fixture(autouse=True)
def _isolate_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "tier-storage"))


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


@pytest.mark.parametrize("parent_id", ["P%", "P_", "P\\"])
async def test_delete_hype_siblings_treats_parent_like_wildcards_literally(tmp_path: Path, parent_id: str) -> None:
    client = _client(tmp_path, _ListGenerator(["a good question about strict mode"]))
    try:
        backend = client._get_backend()
        backend.upsert_vector(f"{parent_id}#hype0", [0.0] * 384)
        backend.upsert_vector("P1#hype0", [0.0] * 384)

        assert backend.hype_sibling_ids(parent_id) == [f"{parent_id}#hype0"]
        assert backend.delete_hype_siblings(parent_id) == 1
        assert backend.hype_sibling_ids(parent_id) == []
        assert backend.hype_sibling_ids("P1") == ["P1#hype0"]
    finally:
        await client.close()


def test_delete_hype_siblings_noop_without_vectors(tmp_path: Path) -> None:
    # NFR03: a backend without vec support no-ops, never raises.
    from trw_memory.storage.yaml_backend import YAMLBackend

    backend = YAMLBackend(tmp_path / "yaml")
    assert backend.delete_hype_siblings("anything") == 0
    assert backend.hype_sibling_ids("anything") == []


async def test_delete_hype_siblings_preserves_canonical_and_nested_foreign_ids(tmp_path: Path) -> None:
    client = _client(tmp_path, _ListGenerator(["a good question about strict mode"]))
    try:
        backend = client._get_backend()
        embedding = [0.0] * 384
        backend.store(make_entry(entry_id="foo#hypevictim"))
        backend.store(make_entry(entry_id="foo#hype0"))
        backend.upsert_vector("foo#hypevictim", embedding)
        backend.upsert_vector("foo#hypevictim#hype0", embedding)
        backend.upsert_vector("foo#hype0", embedding)

        assert backend.hype_sibling_ids("foo") == []
        assert backend.delete_hype_siblings("foo") == 0
        assert backend.vector_exists("foo#hypevictim")
        assert backend.vector_exists("foo#hypevictim#hype0")
        assert backend.vector_exists("foo#hype0")
    finally:
        await client.close()


async def test_hype_expansion_does_not_overwrite_canonical_numeric_collision(tmp_path: Path) -> None:
    client = _client(tmp_path, _ListGenerator(["a good question about strict mode"]))
    try:
        client._config.hype_enabled = False
        await client.store("canonical collision content", entry_id="foo#hype0")
        backend = client._get_backend()
        original = backend.get_stored_embeddings(["foo#hype0"])["foo#hype0"]

        client._config.hype_enabled = True
        await client.store("parent content", entry_id="foo")

        assert backend.get_stored_embeddings(["foo#hype0"])["foo#hype0"] == original
        assert backend.hype_sibling_ids("foo") == []
    finally:
        await client.close()
