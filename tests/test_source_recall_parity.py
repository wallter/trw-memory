"""Parity tests for source-aware policy across both recall branches."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

from trw_memory.client import MemoryClient, MemoryResultDict


def _result(
    *,
    memory_id: str,
    score: float,
    metadata: dict[str, str] | None = None,
    expires: str = "",
) -> MemoryResultDict:
    raw: dict[str, Any] = {
        "memory_id": memory_id,
        "content": memory_id,
        "detail": "",
        "tags": [],
        "importance": 0.5,
        "score": score,
        "created_at": "2026-04-23T12:00:00+00:00",
        "updated_at": "2026-04-23T12:00:00+00:00",
        "namespace": "default",
        "source": "local",
    }
    if metadata is not None:
        raw["metadata"] = metadata
    if expires:
        raw["expires"] = expires
    return cast(MemoryResultDict, raw)


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> MemoryClient:
    monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "storage"))
    monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "sqlite")
    return MemoryClient(namespace="default", mode="local")


def _prepare_common_recall_mocks(client: MemoryClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(client, "_get_embedder", lambda: None)
    monkeypatch.setattr(client, "_merge_tier_results", lambda results, *_args: results)
    monkeypatch.setattr(client, "_remember_results_in_tiers", lambda _results: None)
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
                _result(memory_id="instruction", score=0.8, metadata={"source_kind": "instruction_rule"}),
                _result(
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
                _result(memory_id="semantic", score=0.9, metadata={"source_kind": "semantic_memory"}),
                _result(memory_id="git", score=0.88, metadata={"source": "distilled:git:aaa..bbb"}),
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
