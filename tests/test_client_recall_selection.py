"""Invocation policy is frozen before asynchronous acquisition."""

from datetime import datetime, timezone

from trw_memory.models.memory import MemoryEntry


def test_invocation_keeps_source_and_temporal_decisions_together():
    from trw_memory.retrieval.recall_selection import RecallInvocation
    from trw_memory.retrieval.source_policy import SourcePolicy
    from trw_memory.retrieval.temporal_selection import TemporalSelection

    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    weights = {"episodic": 2.0}
    invocation = RecallInvocation(
        source=SourcePolicy.resolve(source_weights=weights, reference_time=now),
        temporal=TemporalSelection(reference_time=now),
        namespace="default",
    )
    weights["episodic"] = 0.0
    entry = MemoryEntry(id="e", content="entry", metadata={"source_kind": "episodic"})
    assert invocation.allows_entry(entry)
    assert invocation.rank_key(entry, 0.5) == (0, 0, -1.0)
    assert invocation.source.reference_time is invocation.temporal.reference_time


def test_invocation_asserts_namespace_before_source_exclusion():
    import pytest

    from trw_memory.retrieval.recall_selection import RecallInvocation
    from trw_memory.retrieval.source_policy import SourcePolicy
    from trw_memory.retrieval.temporal_selection import TemporalSelection
    from trw_memory.security.namespace_scope import NamespaceScopeError

    invocation = RecallInvocation(
        source=SourcePolicy.resolve(exclude_source_kinds=["episodic"]),
        temporal=TemporalSelection(),
        namespace="default",
    )
    foreign = MemoryEntry(id="e", content="entry", namespace="foreign", metadata={"source_kind": "episodic"})
    with pytest.raises(NamespaceScopeError):
        invocation.allows_entry(foreign)


async def test_public_call_snapshots_options_before_first_await(tmp_path, monkeypatch):
    from unittest.mock import AsyncMock

    from trw_memory import _client_lifecycle, _client_recall
    from trw_memory.client import MemoryClient
    from trw_memory.lifecycle.tiers import _runtime

    for key in ("HOME", "TRW_HOME", "XDG_CONFIG_HOME"):
        monkeypatch.setenv(key, str(tmp_path))
    monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "memory"))
    for module in (_client_recall, _client_lifecycle, _runtime):
        monkeypatch.setattr(module, "tier_runtime_enabled", lambda _: False)
    client = MemoryClient(namespace="default", mode="local")
    monkeypatch.setattr(client, "_get_embedder", lambda: None)
    monkeypatch.setattr(client, "_try_hybrid_recall", AsyncMock(return_value=None))
    clocks = []

    class Clock:
        @staticmethod
        def now(tz):
            clocks.append(tz)
            return datetime(2026, 1, 1, tzinfo=timezone.utc)

    monkeypatch.setattr(_client_recall, "datetime", Clock)
    weights = {"episodic": 2.0}
    tags = ["original"]
    backend = client._get_backend()
    backend.store(MemoryEntry(id="e", content="snapshot probe", tags=tags, metadata={"source_kind": "episodic"}))

    async def mutate_after_capture():
        weights["episodic"] = 0.0
        tags[:] = ["different"]
        client._config.recall_confidence_filter = 1.0

    monkeypatch.setattr(client, "_apply_pending_remote_retirements", mutate_after_capture)
    try:
        result = await client.recall("snapshot", tags=tags, source_weights=weights, include_org_memories=False)
        assert [r["memory_id"] for r in result] == ["e"]
        assert result[0]["score"] > 1.0
        assert clocks == [timezone.utc]
    finally:
        backend.close()


def test_remote_supplied_windows_and_unknown_are_distinct():
    from trw_memory.retrieval.recall_selection import RemoteCandidate
    from trw_memory.retrieval.temporal_selection import TemporalSelection

    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for as_of in (None, now):
        temporal = TemporalSelection(as_of=as_of, reference_time=now)
        assert RemoteCandidate({}).temporal_eligibility(temporal) is None
        assert RemoteCandidate({"valid_from": "2020-01-01Z"}).temporal_eligibility(temporal) is None
        assert (
            RemoteCandidate({"valid_from": "2020-01-01T00:00:00Z", "invalid_from": None}).temporal_eligibility(temporal)
            is True
        )
        assert RemoteCandidate({"invalid_from": "2024-01-01T00:00:00Z"}).temporal_eligibility(temporal) is False
        assert RemoteCandidate({"expires": "2025-12-31"}).temporal_eligibility(temporal) is False
    assert RemoteCandidate({"valid_from": "2027-01-01T00:00:00Z"}).temporal_eligibility(temporal) is False


async def test_public_shared_window_finishing_preserves_unknown_and_defers_closed(tmp_path, monkeypatch):
    from unittest.mock import AsyncMock

    from trw_memory import _client_lifecycle, _client_recall
    from trw_memory.client import MemoryClient
    from trw_memory.lifecycle.tiers import _runtime

    for key in ("HOME", "TRW_HOME", "XDG_CONFIG_HOME"):
        monkeypatch.setenv(key, str(tmp_path))
    monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "memory"))
    for module in (_client_recall, _client_lifecycle, _runtime):
        monkeypatch.setattr(module, "tier_runtime_enabled", lambda _: False)
    client = MemoryClient(namespace="default", mode="local")
    monkeypatch.setattr(client, "_get_embedder", lambda: None)
    monkeypatch.setattr(client, "_try_hybrid_recall", AsyncMock(return_value=None))

    def row(identifier, score, **window):
        return dict(
            memory_id=identifier,
            content=identifier,
            detail="",
            tags=[],
            importance=0.5,
            score=score,
            namespace="remote",
            source="shared",
            created_at="",
            updated_at="",
            **window,
        )

    monkeypatch.setattr(
        client,
        "_merge_shared_results",
        AsyncMock(
            return_value=[
                row("closed", 1.0, valid_from="2020-01-01T00:00:00Z", invalid_from="2024-01-01T00:00:00Z"),
                row("unknown", 0.7),
                row("open", 0.6, valid_from="2020-01-01T00:00:00Z", invalid_from=None),
            ]
        ),
    )
    try:
        for as_of in (None, datetime(2026, 1, 1, tzinfo=timezone.utc)):
            result = await client.recall("", include_shared=True, include_org_memories=False, as_of=as_of)
            assert [r["memory_id"] for r in result] == ["unknown", "open"]
            assert all("valid_from" not in r and "invalid_from" not in r for r in result)
            deferred = await client.recall(
                "", include_shared=True, include_org_memories=False, as_of=as_of, include_superseded=True
            )
            assert [r["memory_id"] for r in deferred] == ["unknown", "open", "closed"]
    finally:
        client._get_backend().close()


async def test_public_hybrid_does_not_turn_foreign_excluded_candidate_into_fallback(tmp_path, monkeypatch):
    from unittest.mock import AsyncMock

    import pytest

    from trw_memory import _client_lifecycle, _client_recall
    from trw_memory.client import MemoryClient
    from trw_memory.lifecycle.tiers import _runtime
    from trw_memory.security.namespace_scope import NamespaceScopeError

    for key in ("HOME", "TRW_HOME", "XDG_CONFIG_HOME"):
        monkeypatch.setenv(key, str(tmp_path))
    monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "memory"))
    for module in (_client_recall, _client_lifecycle, _runtime):
        monkeypatch.setattr(module, "tier_runtime_enabled", lambda _: False)
    client = MemoryClient(namespace="default", mode="local")
    monkeypatch.setattr(client, "_get_embedder", lambda: None)
    fallback = AsyncMock(side_effect=AssertionError("namespace violation must not degrade to fallback"))
    monkeypatch.setattr(client, "_fallback_recall", fallback)
    backend = client._get_backend()
    foreign = MemoryEntry(id="foreign", namespace="other", content="forbidden", metadata={"source_kind": "episodic"})
    monkeypatch.setattr(backend, "list_entries", lambda **kwargs: [foreign])
    monkeypatch.setattr(backend, "_fts_available", False)
    try:
        with pytest.raises(NamespaceScopeError):
            await client.recall("forbidden", exclude_source_kinds=["episodic"], include_org_memories=False)
        fallback.assert_not_called()
    finally:
        backend.close()


def test_refresh_keeps_authoritative_window_but_only_returned_masked_content(monkeypatch):
    from unittest.mock import Mock

    from trw_memory._client_distilled_tiering import entry_to_result
    from trw_memory._client_recall_helpers import remember_selected_candidates
    from trw_memory.retrieval.recall_selection import LocalCandidate

    valid_from = datetime(2020, 1, 1, tzinfo=timezone.utc)
    invalid_from = datetime(2024, 1, 1, tzinfo=timezone.utc)
    entry = MemoryEntry(
        id="e",
        content="private",
        detail="private detail",
        valid_from=valid_from,
        invalid_from=invalid_from,
        invalidated_by="replacement",
        metadata={"source_kind": "semantic_memory"},
    )
    row = entry_to_result(entry, 0.5)
    row.update(content="masked", detail="masked detail")
    captured = []
    monkeypatch.setattr(
        "trw_memory._client_recall_helpers.remember_entry_data_in_tiers",
        lambda config, payload: captured.append(payload),
    )
    remember_selected_candidates(Mock(), [LocalCandidate(entry, 0.5)], [row])
    assert len(captured) == 1
    restored = MemoryEntry.model_validate(captured[0])
    assert (restored.content, restored.detail) == ("masked", "masked detail")
    assert (restored.valid_from, restored.invalid_from, restored.invalidated_by) == (
        valid_from,
        invalid_from,
        "replacement",
    )
    assert restored.metadata == entry.metadata
    assert entry.content == "private"


def test_partition_acquisition_preserves_total_cap_and_disjoint_candidates():
    from trw_memory.retrieval.recall_selection import RecallInvocation
    from trw_memory.retrieval.source_policy import SourcePolicy
    from trw_memory.retrieval.temporal_selection import TemporalSelection

    invocation = RecallInvocation(SourcePolicy.resolve(), TemporalSelection(include_superseded=True), "default")
    entries = [
        MemoryEntry(id=f"transient-{i}", content="transient", metadata={"source_kind": "episodic"}) for i in range(8)
    ]
    entries.append(MemoryEntry(id="durable", content="durable"))
    requested = []

    def fetch(predicate, remaining):
        requested.append(remaining)
        return [entry for entry in entries if predicate(entry)][:remaining]

    result = invocation.acquire(fetch, limit=3)
    assert [entry.id for entry in result] == ["durable", "transient-0", "transient-1"]
    assert requested == [3, 2]
    assert len({entry.id for entry in result}) == len(result) == 3
