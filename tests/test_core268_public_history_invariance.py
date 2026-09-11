"""Paired real SQLite retrieval, not merely shared-scorer algebra (CORE268 FR01)."""

from datetime import datetime, timezone
from pathlib import Path

import pytest

from trw_memory.client import MemoryClient
from trw_memory.lifecycle.tiers._runtime import get_tier_manager
from trw_memory.models.memory import MemoryEntry
from trw_memory.tools.recall import memory_recall_impl


@pytest.mark.parametrize("consumer", ["standalone", "sdk"])
@pytest.mark.parametrize("query", ["", "invariant"])
async def test_public_order_ignores_only_historical_q(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, consumer: str, query: str
) -> None:
    # Independent stores avoid first-recall access writes contaminating the pair.
    # Acquisition, SQLite, tier merge and scoring are real; only inference is off.
    monkeypatch.setattr("trw_memory.tools.recall.get_local_embedder", lambda **kwargs: None)
    stamp = datetime(2026, 9, 1, tzinfo=timezone.utc)
    observed = []
    cohorts = []
    for reversed_history in (False, True):
        monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / str(reversed_history)))
        monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "sqlite")
        async with MemoryClient(namespace="default", mode="local") as client:
            monkeypatch.setattr(client, "_get_embedder", lambda: None)
            backend = client._get_backend()
            manager = get_tier_manager(client._config, "default")
            cohort = []
            for index, name in enumerate(("amber", "birch", "cedar")):
                q = float(bool(index % 2) != reversed_history)
                entry = MemoryEntry(
                    id=f"history-{name}",
                    namespace="default",
                    content=f"invariant policy {name}",
                    importance=0.6,
                    created_at=stamp,
                    updated_at=stamp,
                    last_accessed_at=stamp,
                    q_value=q,
                    q_observations=100 if reversed_history else 10,
                    outcome_history=[f"2026-09-01:{q}:delivered"],
                )
                backend.store(entry)
                manager.warm_add(entry.id, entry.model_dump(mode="json"), None)
                persisted = backend.get(entry.id, namespace="default")
                assert persisted is not None
                cohort.append(persisted.model_dump(exclude={"q_value", "q_observations", "outcome_history"}))
            cohorts.append(cohort)
            if consumer == "standalone":
                rows = memory_recall_impl(
                    query,
                    "default",
                    backend=backend,
                    config=client._config,
                    limit=3,
                    include_org_memories=False,
                )["memories"]
                observed.append([(row["id"], row["score"]) for row in rows])
            else:
                rows = await client.recall(query, limit=3, include_shared=False, include_org_memories=False)
                observed.append([(row["memory_id"], row["score"]) for row in rows])
    assert cohorts[0] == cohorts[1]  # all nonhistorical persisted fields really match
    assert len(observed[0]) == 3  # no vacuous empty/equally filtered success
    assert {row[0] for row in observed[0]} == {"history-amber", "history-birch", "history-cedar"}
    assert observed[0] == observed[1]  # final order AND score, including equal-impact ties
