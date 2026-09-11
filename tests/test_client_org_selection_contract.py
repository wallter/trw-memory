"""Actual authorized org producer applies policy before candidate cuts; no models."""

from datetime import datetime, timedelta, timezone

import pytest

from trw_memory.client import MemoryClient
from trw_memory.graph import list_org_shared_entries
from trw_memory.integrations._backend import create_backend_from_config
from trw_memory.models.memory import MemoryEntry
from trw_memory.retrieval.recall_selection import RecallInvocation
from trw_memory.retrieval.source_policy import SourcePolicy
from trw_memory.retrieval.temporal_selection import TemporalSelection


@pytest.fixture(params=["sqlite", "yaml"])
def client(request, tmp_path, monkeypatch):
    for key in ("HOME", "TRW_HOME", "XDG_CONFIG_HOME"):
        monkeypatch.setenv(key, str(tmp_path))
    monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "storage"))
    monkeypatch.setenv("MEMORY_STORAGE_BACKEND", request.param)
    value = MemoryClient(namespace="project:caller", mode="local")
    monkeypatch.setattr(value, "_get_embedder", lambda: None)
    yield value
    value._get_backend().close()


def item(id, *, namespace="project:sibling", kind="episodic", closed=False, content=None, offset=0):
    return MemoryEntry(
        id=id,
        namespace=namespace,
        content=content or f"needle {id}",
        importance=0.9,
        cross_validated=True,
        metadata={"source_kind": kind},
        created_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
        updated_at=datetime(2020, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=offset),
        invalid_from=datetime(2021, 1, 1, tzinfo=timezone.utc) if closed else None,
        invalidated_by="replacement" if closed else None,
    )


def store(client, entries):
    with create_backend_from_config(client._config, entries[0].namespace) as backend:
        for entry in entries:
            backend.store(entry)


@pytest.mark.parametrize("exclusion", ["source", "temporal", "query"])
async def test_public_org_policy_before_old_25_cut(client, exclusion):
    competitors = [
        item(
            f"excluded{i}",
            closed=exclusion == "temporal",
            content="unrelated" if exclusion == "query" else None,
            offset=i + 1,
        )
        for i in range(40)
    ]
    store(client, [item("eligible", kind="semantic_memory"), *competitors])
    rows = await client.recall(
        "needle",
        limit=1,
        include_shared=False,
        include_org_memories=True,
        exclude_source_kinds=["episodic"] if exclusion == "source" else None,
    )
    assert [row["memory_id"] for row in rows] == ["eligible"]
    assert rows[0]["source"] == "org"


def test_actual_authorization_precedes_entry_filter(client):
    store(client, [item("allowed", kind="semantic_memory")])
    store(client, [item("secret", namespace="project:secret", kind="semantic_memory")])
    client._config.rbac_enabled = True
    client._config.namespace_roles = {"project:secret": "none"}
    seen = []

    def observe(entry):
        seen.append(entry.namespace)
        return True

    invocation = RecallInvocation(SourcePolicy.resolve(), TemporalSelection(), "project:caller")
    rows = list_org_shared_entries(client._config, client._namespace, invocation=invocation, entry_filter=observe)
    assert [entry.id for entry in rows] == ["allowed"]
    assert seen and set(seen) == {"project:sibling"}


def test_pages_do_not_cap_late_better_source_and_legacy_order_retained(client):
    store(client, [item("durable", kind="semantic_memory"), *(item(f"episodic{i}", offset=i + 1) for i in range(270))])
    invocation = RecallInvocation(SourcePolicy.resolve(), TemporalSelection(), "project:caller")
    native = list_org_shared_entries(client._config, client._namespace, limit=1, invocation=invocation)
    assert [entry.id for entry in native] == ["durable"]
    legacy = list_org_shared_entries(client._config, client._namespace, limit=1)
    assert [entry.id for entry in legacy] == ["episodic269"]


def test_explicit_source_weights_use_org_provenance(client):
    store(client, [item("durable", kind="semantic_memory"), item("episodic", offset=1)])
    invocation = RecallInvocation(
        SourcePolicy.resolve(source_weights={"semantic_memory": 0.1, "episodic": 1.0}),
        TemporalSelection(),
        "project:caller",
    )
    rows = list_org_shared_entries(client._config, client._namespace, limit=1, invocation=invocation)
    assert [entry.id for entry in rows] == ["episodic"]


@pytest.mark.parametrize("duplicate", ["content", "bare_id"])
async def test_public_org_local_duplicates_do_not_consume_candidate_slots(client, duplicate):
    local = item("local", namespace="project:caller", kind="semantic_memory")
    client._get_backend().store(local)
    for index in range(2):
        copy = item(
            "local" if duplicate == "bare_id" else f"copy{index}",
            namespace=f"project:sibling{index}",
            kind="semantic_memory",
            content=local.content if duplicate == "content" else f"needle copy{index}",
            offset=index + 1,
        )
        store(client, [copy])
    store(client, [item("distinct", kind="semantic_memory")])
    rows = await client.recall("needle", limit=2, include_shared=False, include_org_memories=True)
    assert {row["memory_id"] for row in rows} == {"local", "distinct"}
