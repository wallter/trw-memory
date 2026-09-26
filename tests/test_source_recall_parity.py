"""Parity tests for source-aware policy across both recall branches."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from trw_memory.client import MemoryClient
from trw_memory.models.memory import MemoryEntry
from trw_memory.retrieval.recall_selection import LocalCandidate


def _candidate(
    *,
    memory_id: str,
    score: float,
    metadata: dict[str, str] | None = None,
    expires: str = "",
) -> LocalCandidate:
    """The invocation seam carries authoritative entries, never projected results."""
    return LocalCandidate(
        MemoryEntry(
            id=memory_id,
            content=memory_id,
            importance=0.5,
            namespace="default",
            created_at=datetime.fromisoformat("2026-04-23T12:00:00+00:00"),
            updated_at=datetime.fromisoformat("2026-04-23T12:00:00+00:00"),
            metadata=metadata or {},
            expires=expires,
        ),
        score,
    )


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> MemoryClient:
    monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "storage"))
    monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "sqlite")
    return MemoryClient(namespace="default", mode="local")


def _prepare_common_recall_mocks(client: MemoryClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(client, "_get_embedder", lambda: None)
    monkeypatch.setattr("trw_memory._client_recall.remember_results_in_tiers", lambda _client, _results: None)
    monkeypatch.setattr(client, "_record_recall_access", AsyncMock(return_value=None))
    monkeypatch.setattr(client, "_apply_pending_remote_retirements", AsyncMock(return_value=None))


@pytest.mark.asyncio
async def test_recall_fallback_applies_source_policy(client: MemoryClient, monkeypatch: pytest.MonkeyPatch) -> None:
    _prepare_common_recall_mocks(client, monkeypatch)
    monkeypatch.setattr(client, "_try_hybrid_recall", AsyncMock(return_value=None))
    monkeypatch.setattr(
        client,
        "_fallback_recall",
        AsyncMock(
            return_value=[
                _candidate(memory_id="instruction", score=0.8, metadata={"source_kind": "instruction_rule"}),
                _candidate(
                    memory_id="expired-lifecycle",
                    score=0.95,
                    metadata={"source_kind": "lifecycle"},
                    expires="2020-01-01T00:00:00+00:00",
                ),
            ]
        ),
    )

    out = await client.recall("source policy", include_source_kinds=["instruction_rule", "lifecycle"])

    assert [result["memory_id"] for result in out] == ["instruction"]


@pytest.mark.asyncio
async def test_recall_hybrid_applies_same_source_policy(client: MemoryClient, monkeypatch: pytest.MonkeyPatch) -> None:
    _prepare_common_recall_mocks(client, monkeypatch)
    monkeypatch.setattr(
        client,
        "_try_hybrid_recall",
        AsyncMock(
            return_value=[
                _candidate(memory_id="semantic", score=0.9, metadata={"source_kind": "semantic_memory"}),
                _candidate(memory_id="git", score=0.88, metadata={"source": "distilled:git:aaa..bbb"}),
            ]
        ),
    )

    out = await client.recall(
        "source policy",
        include_source_kinds=["semantic_memory", "git_distilled"],
        source_weights={"semantic_memory": 0.5},
    )

    assert [result["memory_id"] for result in out] == ["git", "semantic"]
    assert out[1]["score"] == pytest.approx(0.45)
