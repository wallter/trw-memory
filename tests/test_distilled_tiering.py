"""PRD-DIST-005 FR-6: recall-side tiering for distilled records.

Unit tests for the pure helpers in ``trw_memory.client``:
  - _is_distilled_result — detection via tags or metadata.source
  - _get_distilled_recall_weight — env override + validation
  - apply_distilled_tiering — dampen / exclude / no-op paths

The full MemoryClient.recall integration is covered by the broader
client test suite; this file pins the FR-6-specific invariants.
"""

from __future__ import annotations

from typing import Any, cast

import pytest

from trw_memory.client import (
    DEFAULT_DISTILLED_RECALL_WEIGHT,
    MemoryResultDict,
    _get_distilled_recall_weight,
    _is_distilled_result,
    apply_distilled_tiering,
)


def _make_result(
    *,
    memory_id: str,
    score: float,
    tags: list[str] | None = None,
    metadata: dict[str, str] | None = None,
) -> MemoryResultDict:
    """Build a MemoryResultDict with just the fields FR-6 touches."""
    r: dict[str, Any] = {
        "memory_id": memory_id,
        "content": f"content-{memory_id}",
        "detail": "",
        "tags": list(tags or []),
        "importance": 0.5,
        "score": score,
        "created_at": "2026-04-18T00:00:00+00:00",
        "updated_at": "2026-04-18T00:00:00+00:00",
        "namespace": "test",
        "source": "local",
    }
    if metadata is not None:
        r["metadata"] = metadata
    return cast("MemoryResultDict", r)


# --- _is_distilled_result ---


def test_is_distilled_detects_distill_colon_tag() -> None:
    r = _make_result(memory_id="a", score=1.0, tags=["distill:decision", "adr"])
    assert _is_distilled_result(r) is True


def test_is_distilled_detects_distilled_colon_tag() -> None:
    r = _make_result(memory_id="a", score=1.0, tags=["distilled:git:abc..def"])
    assert _is_distilled_result(r) is True


def test_is_distilled_detects_metadata_source_distilled_prefix() -> None:
    r = _make_result(
        memory_id="a",
        score=1.0,
        metadata={"source": "distilled:git:abc..def"},
    )
    assert _is_distilled_result(r) is True


def test_is_distilled_false_on_curated_record() -> None:
    r = _make_result(memory_id="a", score=1.0, tags=["curated", "ops"])
    assert _is_distilled_result(r) is False


def test_is_distilled_false_on_empty_tags_and_metadata() -> None:
    r = _make_result(memory_id="a", score=1.0)
    assert _is_distilled_result(r) is False


# --- _get_distilled_recall_weight ---


def test_weight_defaults_to_0_75() -> None:
    assert DEFAULT_DISTILLED_RECALL_WEIGHT == 0.75
    assert _get_distilled_recall_weight() == pytest.approx(0.75)


def test_weight_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRW_MEMORY_DISTILLED_RECALL_WEIGHT", "0.5")
    assert _get_distilled_recall_weight() == pytest.approx(0.5)


def test_weight_env_invalid_falls_back_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRW_MEMORY_DISTILLED_RECALL_WEIGHT", "not-a-number")
    assert _get_distilled_recall_weight() == pytest.approx(0.75)


def test_weight_env_out_of_range_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRW_MEMORY_DISTILLED_RECALL_WEIGHT", "1.5")
    assert _get_distilled_recall_weight() == pytest.approx(0.75)

    monkeypatch.setenv("TRW_MEMORY_DISTILLED_RECALL_WEIGHT", "-0.1")
    assert _get_distilled_recall_weight() == pytest.approx(0.75)


# --- apply_distilled_tiering: include_distilled=False ---


def test_tiering_excludes_distilled_when_flag_off() -> None:
    curated = [_make_result(memory_id=f"c{i}", score=0.8) for i in range(3)]
    distilled = [_make_result(memory_id=f"d{i}", score=0.8, tags=["distill:decision"]) for i in range(3)]
    results = curated + distilled
    out = apply_distilled_tiering(results, include_distilled=False)
    assert len(out) == 3
    assert all(r["memory_id"].startswith("c") for r in out)


def test_tiering_include_false_keeps_all_curated() -> None:
    # No distilled records — include_distilled=False is a no-op.
    curated = [_make_result(memory_id=f"c{i}", score=0.8) for i in range(3)]
    assert apply_distilled_tiering(curated, include_distilled=False) == curated


# --- apply_distilled_tiering: dampening ---


def test_tiering_dampens_distilled_scores() -> None:
    """FR-6 core: 5 curated + 5 distilled at equal score → distilled rank lower."""
    curated = [_make_result(memory_id=f"c{i}", score=0.80) for i in range(5)]
    distilled = [_make_result(memory_id=f"d{i}", score=0.80, tags=["distill:decision"]) for i in range(5)]
    results = curated + distilled
    out = apply_distilled_tiering(results, weight=0.75)
    # First 5 should be curated (untouched score 0.80)
    assert [r["memory_id"] for r in out[:5]] == [f"c{i}" for i in range(5)]
    # Distilled scores become 0.80 * 0.75 = 0.60
    for r in out[5:]:
        assert r["score"] == pytest.approx(0.60)


def test_tiering_weight_1_0_is_noop() -> None:
    curated = [_make_result(memory_id="c1", score=0.80)]
    distilled = [_make_result(memory_id="d1", score=0.80, tags=["distill:decision"])]
    out = apply_distilled_tiering(curated + distilled, weight=1.0)
    # Order may be curated-first (stable) since no dampening changed ordering.
    assert {r["memory_id"] for r in out} == {"c1", "d1"}
    # Scores unchanged
    scores = {r["memory_id"]: r["score"] for r in out}
    assert scores["c1"] == pytest.approx(0.80)
    assert scores["d1"] == pytest.approx(0.80)


def test_tiering_does_not_mutate_input() -> None:
    distilled = [_make_result(memory_id="d1", score=0.90, tags=["distill:decision"])]
    original_score = distilled[0]["score"]
    _ = apply_distilled_tiering(distilled, weight=0.5)
    assert distilled[0]["score"] == original_score


def test_tiering_preserves_metadata_on_dampened_records() -> None:
    distilled = [
        _make_result(
            memory_id="d1",
            score=0.90,
            tags=["distill:decision"],
            metadata={"source": "distilled:git:abc..def", "distill_type": "decision"},
        )
    ]
    out = apply_distilled_tiering(distilled, weight=0.5)
    assert out[0]["metadata"]["source"] == "distilled:git:abc..def"
    assert out[0]["metadata"]["distill_type"] == "decision"


def test_tiering_resorts_after_dampening() -> None:
    # Setup: distilled at 0.95, curated at 0.80.
    # After dampening weight=0.5 → distilled=0.475, curated stays 0.80.
    # → curated should rank first.
    distilled = [_make_result(memory_id=f"d{i}", score=0.95, tags=["distill:decision"]) for i in range(3)]
    curated = [_make_result(memory_id=f"c{i}", score=0.80) for i in range(3)]
    out = apply_distilled_tiering(distilled + curated, weight=0.5)
    assert [r["memory_id"] for r in out[:3]] == [f"c{i}" for i in range(3)]
