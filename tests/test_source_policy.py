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


def test_classify_source_family_git_source_kind_metadata() -> None:
    r = _result(memory_id="g", score=1.0, metadata={"source_kind": "git"})
    assert classify_source_family(r) == "git_distilled"


def test_classify_source_family_distilled_bulletin_prefix() -> None:
    r = _result(memory_id="b", score=1.0, metadata={"source": "distilled:bulletin:sprint-99"})
    assert classify_source_family(r) == "lifecycle"


def test_classify_source_family_tag_source_kind_git() -> None:
    r = _result(memory_id="t", score=1.0, tags=["source_kind:git"])
    assert classify_source_family(r) == "git_distilled"


def test_classify_source_family_tag_source_kind_semantic() -> None:
    r = _result(memory_id="t", score=1.0, tags=["source_kind:semantic_memory"])
    assert classify_source_family(r) == "semantic_memory"


def test_classify_source_family_tag_distill_prefix() -> None:
    r = _result(memory_id="t", score=1.0, tags=["distill:abc123"])
    assert classify_source_family(r) == "git_distilled"


def test_classify_source_family_tag_change_bulletin() -> None:
    r = _result(memory_id="t", score=1.0, tags=["change_bulletin"])
    assert classify_source_family(r) == "lifecycle"


def test_classify_source_family_unknown_when_no_match() -> None:
    r = _result(memory_id="u", score=1.0)
    assert classify_source_family(r) == "unknown"


def test_resolve_expiry_uses_metadata_fallback() -> None:
    from trw_memory.retrieval.source_policy import resolve_expiry

    r = _result(memory_id="m", score=1.0, metadata={"expires": "2099-06-01T00:00:00+00:00"})
    assert resolve_expiry(cast("dict[str, Any]", r)) == "2099-06-01T00:00:00+00:00"


def test_is_expired_result_invalid_date_returns_false() -> None:
    from trw_memory.retrieval.source_policy import is_expired_result

    r = _result(memory_id="x", score=1.0, expires="not-a-date")
    assert is_expired_result(cast("dict[str, Any]", r)) is False


def test_apply_source_policy_exclude_source_kinds() -> None:
    results = [
        _result(memory_id="ins", score=0.9, metadata={"source_kind": "instruction_rule"}),
        _result(memory_id="sem", score=0.8, metadata={"source_kind": "semantic_memory"}),
    ]
    out = apply_source_policy(
        cast("list[dict[str, Any]]", results),
        exclude_source_kinds=["semantic_memory"],
    )
    assert [r["memory_id"] for r in out] == ["ins"]


def test_apply_source_policy_distilled_weight_override() -> None:
    results = [
        _result(memory_id="git", score=1.0, metadata={"source": "distilled:git:aaa..bbb"}),
        _result(memory_id="ins", score=1.0, metadata={"source_kind": "instruction_rule"}),
    ]
    out = apply_source_policy(
        cast("list[dict[str, Any]]", results),
        distilled_weight=2.0,
    )
    assert out[0]["memory_id"] == "git"
    assert out[0]["score"] == pytest.approx(2.0)


def test_classify_source_family_non_string_tag_skipped() -> None:
    from trw_memory.retrieval.source_policy import classify_source_family

    # Non-string in tags list must not crash — should fall through to "unknown"
    r: dict[str, Any] = {
        "memory_id": "t",
        "score": 1.0,
        "tags": [42, None, "change_bulletin"],  # non-strings before a valid tag
        "metadata": None,
    }
    assert classify_source_family(r) == "lifecycle"


def test_resolve_expiry_returns_empty_when_no_expires_anywhere() -> None:
    from trw_memory.retrieval.source_policy import resolve_expiry

    r = _result(memory_id="n", score=1.0, metadata={"other_key": "value"})
    assert resolve_expiry(cast("dict[str, Any]", r)) == ""


def test_is_expired_no_expiry_returns_false_early() -> None:
    from trw_memory.retrieval.source_policy import is_expired_result

    # A lifecycle result with no expires field → is_expired returns False via "not raw" path
    r = _result(memory_id="lc", score=1.0, metadata={"source_kind": "lifecycle"})
    assert is_expired_result(cast("dict[str, Any]", r)) is False


def test_apply_source_policy_include_distilled_false_excludes_git() -> None:
    results = [
        _result(memory_id="git", score=0.9, metadata={"source": "distilled:git:aaa..bbb"}),
        _result(memory_id="ins", score=0.7, metadata={"source_kind": "instruction_rule"}),
    ]
    out = apply_source_policy(
        cast("list[dict[str, Any]]", results),
        include_distilled=False,
    )
    assert [r["memory_id"] for r in out] == ["ins"]


def test_apply_source_policy_org_source_containment_bucket() -> None:
    # "org" source → containment_bucket=1, ranked after bucket=0 (durable) at same score
    results_raw: list[dict[str, Any]] = [
        {
            "memory_id": "org_entry",
            "content": "org content",
            "detail": "",
            "tags": [],
            "importance": 0.5,
            "score": 1.0,
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:00:00+00:00",
            "namespace": "test",
            "source": "org",
            "metadata": {"source_kind": "semantic_memory"},
        },
        {
            "memory_id": "local_entry",
            "content": "local content",
            "detail": "",
            "tags": [],
            "importance": 0.5,
            "score": 1.0,
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:00:00+00:00",
            "namespace": "test",
            "source": "local",
            "metadata": {"source_kind": "semantic_memory"},
        },
    ]
    out = apply_source_policy(cast("list[dict[str, Any]]", results_raw))
    # local_entry has bucket=0, org_entry has bucket=1 → local first at equal score
    assert out[0]["memory_id"] == "local_entry"
    assert out[1]["memory_id"] == "org_entry"


def test_apply_source_policy_zero_weight_excludes_family() -> None:
    results = [
        _result(memory_id="git", score=0.9, metadata={"source": "distilled:git:aaa..bbb"}),
        _result(memory_id="ins", score=0.7, metadata={"source_kind": "instruction_rule"}),
    ]
    out = apply_source_policy(
        cast("list[dict[str, Any]]", results),
        source_weights={"git_distilled": 0.0},
    )
    assert [r["memory_id"] for r in out] == ["ins"]


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
