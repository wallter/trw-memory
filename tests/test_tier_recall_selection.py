"""Model-free tier discovery: policy before caps, restoration only after selection."""

import json
import sqlite3
from datetime import datetime, timezone

import pytest

from trw_memory.lifecycle.tiers import _runtime
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry, MemoryStatus
from trw_memory.retrieval.recall_selection import RecallInvocation
from trw_memory.retrieval.source_policy import SourcePolicy
from trw_memory.retrieval.temporal_selection import TemporalSelection
from trw_memory.security.namespace_scope import NamespaceScopeError
from trw_memory.storage.persistence import write_yaml
from trw_memory.storage.sqlite_backend import SQLiteBackend


def entry(id, kind="episodic", **kwargs):
    return MemoryEntry(
        id=id,
        content="needle",
        created_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
        metadata={"source_kind": kind},
        **kwargs,
    )


def policy(**kwargs):
    return RecallInvocation(SourcePolicy.resolve(**kwargs), TemporalSelection(), "default")


@pytest.fixture
def tiers(tmp_path):
    config = MemoryConfig(storage_path=str(tmp_path), hot_max_entries=100)
    manager = _runtime.get_tier_manager(config, "default")
    with SQLiteBackend(tmp_path / "canonical.db") as backend:
        yield config, manager, backend
    manager.close()


def discover(manager, backend, invocation=None, **kwargs):
    return manager.search(
        ["needle"],
        invocation=invocation or policy(),
        resolve_entry=lambda id: backend.get(id, namespace="default"),
        top_k=1,
        **kwargs,
    )


def archive(manager, value):
    path = manager._cold_dir() / f"{value.id}.yaml"
    write_yaml(path, value.model_dump(mode="json"))
    return path


def snapshot(path):
    return {
        str(p.relative_to(path)): (p.read_bytes(), p.stat().st_mtime_ns)
        for p in path.rglob("*")
        if p.is_file() and not p.name.endswith(("-wal", "-shm"))
    }


@pytest.mark.parametrize("tier", ["hot", "warm", "cold", "vector"])
def test_source_policy_before_competitive_cap(tiers, tier):
    config, manager, backend = tiers
    if tier == "vector":
        pytest.importorskip("sqlite_vec")
    for i in range(30):
        value = entry(f"e{i:02}", importance=0.9)
        if tier == "hot":
            manager.hot_put(value.id, value)
        elif tier in {"warm", "vector"}:
            manager.warm_add(value.id, value.model_dump(mode="json"), [1.0, 0.0] if tier == "vector" else None)
        else:
            archive(manager, value)
    durable = entry("zzdurable", "semantic_memory", importance=0.1)
    if tier == "hot":
        manager.hot_put(durable.id, durable)
    elif tier in {"warm", "vector"}:
        manager.warm_add(durable.id, durable.model_dump(mode="json"), [0.0, 1.0] if tier == "vector" else None)
    else:
        archive(manager, durable)
    manager._warm_store.close()
    before = snapshot(manager._base_dir)
    hot_before = [(key, item.model_dump()) for key, item in manager._hot.items()]
    for invocation in (policy(), policy(exclude_source_kinds=["episodic"])):
        rows = discover(manager, backend, invocation, query_embedding=[1.0, 0.0] if tier == "vector" else None)
        assert [r.entry.id for r in rows] == [durable.id]
        if tier == "vector":
            assert rows[0].relevance_hint == pytest.approx(0.0)
    assert snapshot(manager._base_dir) == before
    assert [(key, item.model_dump()) for key, item in manager._hot.items()] == hot_before


@pytest.mark.parametrize("tier", ["hot", "warm", "cold"])
def test_canonical_invalidation_overrides_cached_validity(tiers, tier):
    _, manager, backend = tiers
    value = entry("stale")
    if tier == "hot":
        manager.hot_put(value.id, value)
    elif tier == "warm":
        manager.warm_add(value.id, value.model_dump(mode="json"), None)
    else:
        archive(manager, value)
    value.invalid_from = datetime(2021, 1, 1, tzinfo=timezone.utc)
    value.invalidated_by = "replacement"
    backend.store(value)
    assert discover(manager, backend) == []


def test_missing_sidecar_does_not_create_directory(tiers):
    _, manager, backend = tiers
    assert not manager._base_dir.exists()
    assert discover(manager, backend, query_embedding=[1.0, 0.0]) == []
    assert not manager._base_dir.exists()


def test_partial_legacy_sidecar_requires_canonical_authority(tiers):
    _, manager, backend = tiers
    sidecar = manager._warm_store._warm_sidecar_path()
    sidecar.write_text(json.dumps({"id": "partial", "summary": "needle", "tags": []}) + "\n")
    assert discover(manager, backend) == []
    backend.store(entry("partial"))
    assert discover(manager, backend)[0].entry.id == "partial"


def test_cold_discovery_restores_only_selected(tiers):
    config, manager, backend = tiers
    winner = archive(manager, entry("winner", "semantic_memory"))
    loser = archive(manager, entry("loser"))
    before = snapshot(manager._base_dir)
    selected = discover(manager, backend)
    assert snapshot(manager._base_dir) == before
    assert backend.get("winner", namespace="default") is None
    assert _runtime.restore_selected_cold(config, "default", backend, selected) == set()
    assert backend.get("winner", namespace="default") is not None
    assert backend.get("loser", namespace="default") is None
    assert not winner.exists() and loser.exists()


def test_selected_restore_failure_rolls_back(tiers, monkeypatch):
    config, manager, backend = tiers
    path = archive(manager, entry("winner"))
    selected = discover(manager, backend)
    before = path.read_bytes()

    def fail(*args, **kwargs):
        raise OSError("warm write failed")

    monkeypatch.setattr(manager._warm_store, "warm_add", fail)
    assert _runtime.restore_selected_cold(config, "default", backend, selected) == {("default", "winner")}
    assert backend.get("winner", namespace="default") is None
    assert path.read_bytes() == before


def test_namespace_assertion_precedes_source_exclusion(tiers):
    _, manager, backend = tiers
    manager.hot_put("foreign", entry("foreign", namespace="project:other"))
    with pytest.raises(NamespaceScopeError):
        discover(manager, backend, policy(exclude_source_kinds=["episodic"]))


def test_legacy_search_still_returns_dicts_and_promotes(tiers):
    _, manager, _ = tiers
    path = archive(manager, entry("legacy"))
    rows = manager.search(["needle"], top_k=1)
    assert rows[0]["id"] == "legacy"
    assert not path.exists()


def test_runtime_discovery_skips_warmup_and_empty_selection_does_not_restore(tiers, monkeypatch):
    config, manager, backend = tiers
    path = archive(manager, entry("candidate"))

    def forbidden(*args, **kwargs):
        raise AssertionError("discovery must not warm up or promote")

    monkeypatch.setattr(_runtime, "warmup_tier_manager", forbidden)
    monkeypatch.setattr(manager, "cold_promote", forbidden)
    before = snapshot(manager._base_dir)
    rows = _runtime.tier_candidates(config, "default", backend, query="needle", tags=None, limit=1, invocation=policy())
    assert rows[0].cold
    _runtime.restore_selected_cold(config, "default", backend, [])
    assert path.exists() and snapshot(manager._base_dir) == before


def test_complete_legacy_sidecar_uses_created_at_without_synthesizing(tiers):
    _, manager, backend = tiers
    value = entry("legacy-complete")
    payload = value.model_dump(mode="json")
    payload.pop("valid_from")
    manager.warm_add(value.id, payload, None)
    rows = discover(manager, backend)
    assert rows[0].entry.created_at == value.created_at
    assert rows[0].entry.valid_from == value.created_at


def test_canonical_warm_metadata_wins_and_is_not_marked_cold(tiers):
    _, manager, backend = tiers
    stale = entry("same", "episodic")
    archive(manager, stale)
    current = entry("same", "semantic_memory")
    backend.store(current)
    rows = discover(manager, backend, policy(exclude_source_kinds=["episodic"]))
    assert rows[0].entry.metadata == current.metadata
    assert not rows[0].cold


def test_vector_discovery_connection_denies_writes(tiers, monkeypatch):
    pytest.importorskip("sqlite_vec")
    _, manager, backend = tiers
    value = entry("readonly")
    manager.warm_add(value.id, value.model_dump(mode="json"), [1.0, 0.0])
    manager._warm_store.close()
    original_connect = sqlite3.connect
    connections = []

    def connect(*args, **kwargs):
        conn = original_connect(*args, **kwargs)
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            conn.execute("CREATE TABLE forbidden_write (value TEXT)")
        connections.append(conn)
        return conn

    monkeypatch.setattr(sqlite3, "connect", connect)
    hot_before = [(key, item.model_dump()) for key, item in manager._hot.items()]
    before = snapshot(manager._base_dir)
    assert discover(manager, backend, query_embedding=[1.0, 0.0])[0].relevance_hint == 1.0
    assert snapshot(manager._base_dir) == before
    assert [(key, item.model_dump()) for key, item in manager._hot.items()] == hot_before
    assert connections
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        connections[0].execute("SELECT 1")


def test_legacy_missing_namespace_can_resolve_authorized_canonical(tiers):
    _, manager, backend = tiers
    sidecar = manager._warm_store._warm_sidecar_path()
    sidecar.write_text(json.dumps({"id": "partial", "summary": "needle"}) + "\n")
    value = entry("partial", namespace="project:one")
    backend.store(value)
    invocation = RecallInvocation(SourcePolicy.resolve(), TemporalSelection(), "project:one")
    rows = manager.search(
        ["needle"], invocation=invocation, top_k=1, resolve_entry=lambda id: backend.get(id, namespace="project:one")
    )
    assert rows[0].entry.namespace == "project:one"


def test_canonical_obsolete_status_overrides_active_snapshot(tiers):
    _, manager, backend = tiers
    cached = entry("old")
    manager.warm_add(cached.id, cached.model_dump(mode="json"), None)
    backend.store(cached.model_copy(update={"status": MemoryStatus.OBSOLETE}))
    assert discover(manager, backend) == []


def test_temporal_eligibility_precedes_source_bucket(tiers):
    _, manager, backend = tiers
    stale = entry(
        "closed-durable",
        "semantic_memory",
        importance=1.0,
        invalid_from=datetime(2021, 1, 1, tzinfo=timezone.utc),
        invalidated_by="replacement",
    )
    live = entry("live", importance=0.1)
    manager.hot_put(stale.id, stale)
    manager.hot_put(live.id, live)
    invocation = RecallInvocation(SourcePolicy.resolve(), TemporalSelection(include_superseded=True), "default")
    assert discover(manager, backend, invocation)[0].entry.id == live.id


def test_nonlexical_hot_snapshot_does_not_hide_matching_warm_vector(tiers):
    pytest.importorskip("sqlite_vec")
    _, manager, backend = tiers
    value = entry("vector-only").model_copy(update={"content": "unrelated text"})
    manager.hot_put(value.id, value)
    manager.warm_add(value.id, value.model_dump(mode="json"), [1.0, 0.0])
    rows = discover(manager, backend, query_embedding=[1.0, 0.0])
    assert [row.entry.id for row in rows] == [value.id]
    assert rows[0].relevance_hint == 1.0


def test_cold_iterator_is_lazy_and_close_preserves_early_stop_cache_cleanup(tiers, monkeypatch):
    _, manager, _ = tiers
    for name in ("a", "b", "c"):
        archive(manager, entry(name))
    store = manager._cold_store
    assert len(store.cold_search(["needle"])) == 3
    reads = []
    original = store._cached_search_entry

    def read(path):
        reads.append(path.name)
        return original(path)

    monkeypatch.setattr(store, "_cached_search_entry", read)
    rows = store.iter_search(["needle"])
    assert reads == []
    assert next(rows)["id"] == "a"
    assert reads == ["a.yaml"]
    rows.close()
    assert [path.rsplit("/", 1)[-1] for path in store._search_cache] == ["a.yaml"]
    reads.clear()
    assert [row["id"] for row in store.cold_search(["needle"], top_k=1)] == ["a"]
    assert reads == ["a.yaml"]


def test_discovery_passes_captured_reference_to_composite_scoring(tiers, monkeypatch):
    from trw_memory.lifecycle.tiers import _manager_search

    _, manager, backend = tiers
    manager.hot_put("clock", entry("clock"))
    invocation = policy()
    original = _manager_search.compute_importance_score
    references = []

    def score(*args, **kwargs):
        references.append(kwargs.get("reference_time"))
        return original(*args, **kwargs)

    monkeypatch.setattr(_manager_search, "compute_importance_score", score)
    assert discover(manager, backend, invocation)
    assert references == [invocation.temporal.reference_time]


@pytest.fixture
def public_client(tmp_path, monkeypatch):
    from trw_memory.client import MemoryClient

    for key in ("HOME", "TRW_HOME", "XDG_CONFIG_HOME"):
        monkeypatch.setenv(key, str(tmp_path))
    monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "storage"))
    monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "sqlite")
    client = MemoryClient(namespace="default", mode="local")
    monkeypatch.setattr(client, "_get_embedder", lambda: None)
    yield client
    _runtime.get_tier_manager(client._config, "default").close()
    client._get_backend().close()


async def test_public_recall_restores_only_returned_cold_entry(public_client):
    client = public_client
    manager = _runtime.get_tier_manager(client._config, "default")
    selected = archive(manager, entry("durable", "semantic_memory"))
    unselected = archive(manager, entry("episodic"))
    before = (unselected.read_bytes(), unselected.stat().st_mtime_ns)
    rows = await client.recall("needle", limit=1, include_shared=False, include_org_memories=False)
    assert [row["memory_id"] for row in rows] == ["durable"]
    restored = client._get_backend().get("durable", namespace="default")
    assert restored is not None and restored.access_count == 1
    assert restored.last_accessed_at is not None
    assert not selected.exists()
    assert client._get_backend().get("episodic", namespace="default") is None
    assert (unselected.read_bytes(), unselected.stat().st_mtime_ns) == before


async def test_public_restore_failure_omits_failed_hit_but_returns_successful_peer(public_client, monkeypatch):
    from unittest.mock import AsyncMock

    client = public_client
    manager = _runtime.get_tier_manager(client._config, "default")
    path = archive(manager, entry("failed"))
    peer = archive(manager, entry("successful"))
    before = path.read_bytes()
    warm_add = manager._warm_store.warm_add

    def fail_one(entry_id, *args, **kwargs):
        if entry_id == "failed":
            raise OSError("warm write failed")
        return warm_add(entry_id, *args, **kwargs)

    monkeypatch.setattr(manager._warm_store, "warm_add", fail_one)
    access = AsyncMock(wraps=client._record_recall_access)
    monkeypatch.setattr(client, "_record_recall_access", access)
    rows = await client.recall("needle", limit=2, include_shared=False, include_org_memories=False)
    assert [row["memory_id"] for row in rows] == ["successful"]
    access.assert_awaited_once()
    assert [row["memory_id"] for row in access.call_args.args[0]] == ["successful"]
    restored = client._get_backend().get("successful", namespace="default")
    assert restored is not None and restored.access_count == 1
    assert not peer.exists()
    assert client._get_backend().get("failed", namespace="default") is None
    assert path.read_bytes() == before
