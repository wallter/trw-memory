"""PRD-CORE-336 FR01/FR03: the one scoring point for the distilled weight.

``apply_distilled_tiering`` (PRD-DIST-005 FR-6) had no production caller and is
deleted; its cases now run against ``source_policy.weight_distilled``, which
``hybrid_search_scored`` calls for both recall routes. The
``TRW_MEMORY_DISTILLED_RECALL_WEIGHT`` override is read by
``SourcePolicy.resolve`` and nothing else.
"""

from __future__ import annotations

import importlib
from datetime import datetime, timezone
from typing import Any

import pytest

from trw_memory.models.memory import MemoryEntry
from trw_memory.retrieval.pipeline import hybrid_search_scored
from trw_memory.retrieval.source_policy import DEFAULT_SOURCE_WEIGHTS, SourcePolicy, weight_distilled

from ._test_scope_support import DEFAULT_SCOPE

_ENV = "TRW_MEMORY_DISTILLED_RECALL_WEIGHT"
_STAMP = datetime(2026, 1, 5, tzinfo=timezone.utc)


def _entry(entry_id: str, content: str = "", *, distilled: bool = False, tags: list[str] | None = None) -> MemoryEntry:
    return MemoryEntry(
        id=entry_id,
        content=content or entry_id,
        namespace="default",
        created_at=_STAMP,
        updated_at=_STAMP,
        valid_from=_STAMP,
        tags=tags or [],
        metadata={"source": "distilled:git:aaa..bbb"} if distilled else {},
    )


# --- the scoring point ---


@pytest.mark.parametrize(
    ("weight", "expected_order", "expected_scores"),
    [
        # Equal pre-weight scores: distilled rows drop below every curated row.
        (0.75, ["c0", "c1", "d0", "d1"], [0.8, 0.8, 0.6000000000000001, 0.6000000000000001]),
        # A crossing: 0.95 * 0.5 = 0.475 falls below the curated 0.8.
        (0.5, ["c0", "c1", "d0", "d1"], [0.8, 0.8, 0.4, 0.4]),
        # Weight 1.0 multiplies by one: order and scores unchanged.
        (1.0, ["d0", "c0", "d1", "c1"], [0.8, 0.8, 0.8, 0.8]),
    ],
)
def test_weight_distilled_multiplies_only_distilled_rows_and_resorts_stably(
    weight: float, expected_order: list[str], expected_scores: list[float]
) -> None:
    entries = [_entry("d0", distilled=True), _entry("c0"), _entry("d1", distilled=True), _entry("c1")]
    weighted = weight_distilled(entries, [0.8, 0.8, 0.8, 0.8], weight)
    assert [entry.id for entry, _, _ in weighted] == expected_order
    assert [score for _, score, _ in weighted] == expected_scores
    assert {entry.id: flag for entry, _, flag in weighted} == {"d0": True, "d1": True, "c0": False, "c1": False}


def test_weight_distilled_recognizes_every_git_distilled_marker_and_nothing_else() -> None:
    entries = [
        _entry("tag", tags=["distill:decision"]),
        _entry("kind", tags=["source_kind:git"]),
        _entry("bulletin", tags=["change_bulletin"]),
        _entry("plain"),
    ]
    weighted = weight_distilled(entries, [1.0, 1.0, 1.0, 1.0], 0.5)
    assert {entry.id: score for entry, score, _ in weighted} == {"tag": 0.5, "kind": 0.5, "bulletin": 1.0, "plain": 1.0}


def test_weight_distilled_does_not_mutate_its_inputs() -> None:
    entries = [_entry("d0", distilled=True), _entry("c0")]
    scores = [0.9, 0.8]
    weight_distilled(entries, scores, 0.5)
    assert scores == [0.9, 0.8]
    assert [entry.id for entry in entries] == ["d0", "c0"]


# --- the pipeline calls it, once, after rerank and before the top_k cut ---


def _search(monkeypatch: pytest.MonkeyPatch, *, rerank: bool, top_k: int = 10, **kwargs: Any) -> list[Any]:
    entries = [
        _entry("M-git", "retry backoff queue retry backoff", distilled=True),
        _entry("M-a", "retry backoff queue"),
        _entry("M-b", "retry backoff"),
        _entry("M-c", "retry"),
    ]

    def cross_encode_scores(_query: str, rows: list[MemoryEntry], **_: Any) -> list[tuple[MemoryEntry, float]]:
        return sorted(((row, float(ord(row.id[-1]))) for row in rows), key=lambda pair: pair[1], reverse=True)

    monkeypatch.setattr("trw_memory.retrieval.reranker.cross_encode_scores", cross_encode_scores)
    return hybrid_search_scored(
        "retry backoff queue",
        entries,
        scope=DEFAULT_SCOPE,
        rerank=rerank,
        top_k=top_k,
        **kwargs,
    )


@pytest.mark.parametrize("rerank", [False, True])
@pytest.mark.parametrize("neutral", [{}, {"distilled_weight": None}, {"distilled_weight": 1.0}])
def test_hybrid_search_without_a_distilled_weight_keeps_todays_output(
    monkeypatch: pytest.MonkeyPatch, rerank: bool, neutral: dict[str, Any]
) -> None:
    baseline = _search(monkeypatch, rerank=rerank)
    assert _search(monkeypatch, rerank=rerank, **neutral) == baseline
    assert not any(candidate.distilled_weighted for candidate in baseline)


@pytest.mark.parametrize(("rerank", "basis"), [(False, "fused"), (True, "position")])
def test_hybrid_search_weights_distilled_rows_on_the_calls_basis(
    monkeypatch: pytest.MonkeyPatch, rerank: bool, basis: str
) -> None:
    before = {c.entry.id: c for c in _search(monkeypatch, rerank=rerank)}
    after = _search(monkeypatch, rerank=rerank, distilled_weight=0.25)
    assert {c.basis for c in after} == {basis}
    for candidate in after:
        prior = before[candidate.entry.id]
        expected = prior.score * 0.25 if candidate.entry.id == "M-git" else prior.score
        assert candidate.score == expected
        assert candidate.distilled_weighted is (candidate.entry.id == "M-git")
    assert [c.score for c in after] == sorted((c.score for c in after), reverse=True)


def test_hybrid_search_weights_before_the_top_k_cut(monkeypatch: pytest.MonkeyPatch) -> None:
    """A distilled row that the weight moves below the cut leaves the set; the next row enters."""
    unweighted = [c.entry.id for c in _search(monkeypatch, rerank=False)]
    assert unweighted[0] == "M-git"
    cut = [c.entry.id for c in _search(monkeypatch, rerank=False, top_k=2, distilled_weight=0.01)]
    assert cut == [row for row in unweighted if row != "M-git"][:2]


# --- FR03: one override reader, no second weighting path ---


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, DEFAULT_SOURCE_WEIGHTS["git_distilled"]),
        ("0.5", 0.5),
        ("0", 0.0),
        ("1", 1.0),
        ("not-a-number", DEFAULT_SOURCE_WEIGHTS["git_distilled"]),
        ("1.5", DEFAULT_SOURCE_WEIGHTS["git_distilled"]),
        ("-0.1", DEFAULT_SOURCE_WEIGHTS["git_distilled"]),
        ("nan", DEFAULT_SOURCE_WEIGHTS["git_distilled"]),
    ],
)
def test_source_policy_reads_the_env_override_with_range_validation(
    monkeypatch: pytest.MonkeyPatch, raw: str | None, expected: float
) -> None:
    if raw is None:
        monkeypatch.delenv(_ENV, raising=False)
    else:
        monkeypatch.setenv(_ENV, raw)
    policy = SourcePolicy.resolve()
    assert policy.weights["git_distilled"] == expected
    assert policy.explicit_distilled_weight is False


@pytest.mark.parametrize(
    ("options", "expected"),
    [
        ({"distilled_weight": 0.9}, 0.9),
        ({"source_weights": {"git_distilled": 0.3}}, 0.3),
        ({"source_weights": {"semantic_memory": 0.3}}, 0.5),
    ],
)
def test_a_caller_weight_outranks_the_env_override(
    monkeypatch: pytest.MonkeyPatch, options: dict[str, Any], expected: float
) -> None:
    monkeypatch.setenv(_ENV, "0.5")
    assert SourcePolicy.resolve(**options).weights["git_distilled"] == expected


def test_the_uncalled_second_weighting_path_is_gone() -> None:
    tiering = importlib.import_module("trw_memory._client_distilled_tiering")
    client = importlib.import_module("trw_memory.client")
    for name in ("apply_distilled_tiering", "get_distilled_recall_weight", "is_distilled_result"):
        assert not hasattr(tiering, name), name
    for name in ("apply_distilled_tiering", "DEFAULT_DISTILLED_RECALL_WEIGHT"):
        assert not hasattr(client, name), name
