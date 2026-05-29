"""Tests for source-aware recall policy and expiry round-trips."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from trw_memory.client import MemoryClient, MemoryResultDict
from trw_memory.retrieval.source_policy import apply_source_policy, classify_source_family


def _result(
    *,
    memory_id: str,
    score: float,
    tags: list[str] | None = None,
    metadata: dict[str, str] | None = None,
    expires: str = "",
) -> MemoryResultDict:
    raw: dict[str, Any] = {
        "memory_id": memory_id,
        "content": memory_id,
        "detail": "",
        "tags": list(tags or []),
        "importance": 0.5,
        "score": score,
        "created_at": "2026-04-23T12:00:00+00:00",
        "updated_at": "2026-04-23T12:00:00+00:00",
        "namespace": "test",
        "source": "local",
    }
    if metadata is not None:
        raw["metadata"] = metadata
    if expires:
        raw["expires"] = expires
    return cast("MemoryResultDict", raw)


def test_classify_source_family_recognizes_multi_source_metadata() -> None:
    assert classify_source_family(_result(memory_id="g", score=1.0, metadata={"source": "distilled:git:aaa..bbb"})) == (
        "git_distilled"
    )
    assert classify_source_family(_result(memory_id="i", score=1.0, metadata={"source_kind": "instruction_rule"})) == (
        "instruction_rule"
    )
    assert classify_source_family(_result(memory_id="s", score=1.0, metadata={"source_kind": "semantic_memory"})) == (
        "semantic_memory"
    )
    assert classify_source_family(_result(memory_id="l", score=1.0, metadata={"source_kind": "lifecycle"})) == (
        "lifecycle"
    )
    assert classify_source_family(_result(memory_id="e", score=1.0, metadata={"source_kind": "episodic"})) == (
        "episodic"
    )


def test_apply_source_policy_filters_expired_transient_records() -> None:
    results = [
        _result(memory_id="curated", score=0.8),
        _result(
            memory_id="expired-bulletin",
            score=0.95,
            metadata={"source_kind": "lifecycle"},
            expires="2020-01-01T00:00:00+00:00",
        ),
        _result(
            memory_id="fresh-episode",
            score=0.9,
            metadata={"source_kind": "episodic"},
            expires="2999-01-01T00:00:00+00:00",
        ),
    ]

    out = apply_source_policy(cast("list[dict[str, Any]]", results))

    assert [result["memory_id"] for result in out] == ["curated", "fresh-episode"]


def test_apply_source_policy_respects_family_filters_and_weights() -> None:
    results = [
        _result(memory_id="git", score=0.9, metadata={"source": "distilled:git:aaa..bbb"}),
        _result(memory_id="instruction", score=0.8, metadata={"source_kind": "instruction_rule"}),
        _result(memory_id="semantic", score=0.85, metadata={"source_kind": "semantic_memory"}),
    ]

    out = apply_source_policy(
        cast("list[dict[str, Any]]", results),
        include_source_kinds=["instruction_rule", "semantic_memory"],
        source_weights={"semantic_memory": 0.5},
    )

    assert [result["memory_id"] for result in out] == ["instruction", "semantic"]
    assert out[1]["score"] == pytest.approx(0.425)


def test_apply_source_policy_prioritizes_durable_context_over_transient_by_default() -> None:
    results = [
        _result(memory_id="durable", score=0.6, metadata={"source_kind": "instruction_rule"}),
        _result(
            memory_id="transient",
            score=0.99,
            metadata={"source_kind": "lifecycle"},
            expires="2099-01-01T00:00:00+00:00",
        ),
    ]

    out = apply_source_policy(cast("list[dict[str, Any]]", results))

    assert [result["memory_id"] for result in out] == ["durable", "transient"]


def test_apply_source_policy_allows_explicit_transient_weight_override() -> None:
    results = [
        _result(memory_id="durable", score=0.6, metadata={"source_kind": "instruction_rule"}),
        _result(
            memory_id="transient",
            score=0.99,
            metadata={"source_kind": "lifecycle"},
            expires="2099-01-01T00:00:00+00:00",
        ),
    ]

    out = apply_source_policy(
        cast("list[dict[str, Any]]", results),
        source_weights={"lifecycle": 2.0},
    )

    assert [result["memory_id"] for result in out] == ["transient", "durable"]
    assert out[0]["score"] == pytest.approx(1.98)


@pytest.mark.asyncio
async def test_memory_client_store_preserves_expires_on_update(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "storage"))
    monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "sqlite")
    client = MemoryClient(namespace="default", mode="local")

    await client.store(
        "Lifecycle bulletin for sprint 99",
        metadata={"source_kind": "lifecycle"},
        entry_id="M-fixed-expiry",
        expires="2099-01-01T00:00:00+00:00",
    )
    await client.store(
        "Lifecycle bulletin for sprint 99",
        metadata={"source_kind": "lifecycle"},
        entry_id="M-fixed-expiry",
    )

    results = await client.recall("bulletin", include_source_kinds=["lifecycle"])

    assert results[0]["memory_id"] == "M-fixed-expiry"
    assert results[0]["expires"] == "2099-01-01T00:00:00+00:00"
