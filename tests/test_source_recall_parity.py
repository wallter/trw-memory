"""Parity tests for source-aware policy across both recall branches."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from trw_memory.client import MemoryClient
from trw_memory.models.memory import MemoryEntry
from trw_memory.retrieval.recall_selection import LocalCandidate
from trw_memory.tools import recall as recall_tool


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


# --- PRD-CORE-336 FR01: one scoring point for the distilled weight, on both routes ---


@pytest.mark.asyncio
@pytest.mark.parametrize("rerank", ["off", "reordering"])
async def test_weighted_order_and_score_parity_library_vs_daemon(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rerank: str
) -> None:
    """Soundness scope: git_distilled + ``unknown`` rows, default bucket, no tags.

    Proves the two routes return one id order and apply the weight once. It does
    not claim cross-route numeric equality: the daemon reports the pipeline's
    basis score, the library its pre-weight positional score (PRD-CORE-278).

    Both routes fuse with ``importance_alpha`` 1.0 here. By default the library
    fuses with the configured 0.7 and the daemon with 1.0 (CORE116 RA2), so their
    ``fused``-basis scores sit on different scales and one weight can move a row
    across a neighbour on one route and not the other: that divergence predates
    PRD-CORE-336 and is outside its soundness scope.
    """
    from tests import _source_weighting_support as support

    monkeypatch.setenv("MEMORY_RRF_IMPORTANCE_ALPHA", "1.0")
    support.stub_cross_encoder(monkeypatch, None if rerank == "off" else support.fewest_retries_rerank)
    store = support.make_client(tmp_path, monkeypatch)
    support.seed(store, support.PARITY_ROWS + support.PARITY_PADDING)
    monkeypatch.setenv("TRW_MEMORY_DISTILLED_RECALL_WEIGHT", "1.0")
    basis = {support.row_id(row): row["score"] for row in support.daemon_recall(store)["memories"]}
    unweighted_library = await support.library_recall(store)
    monkeypatch.setenv("TRW_MEMORY_DISTILLED_RECALL_WEIGHT", "0.5")

    daemon = support.daemon_recall(store)["memories"]
    library = await support.library_recall(store)

    assert [support.row_id(row) for row in library] == [support.row_id(row) for row in daemon]
    assert len(daemon) == len(support.PARITY_ROWS)
    daemon_scores = [float(row["score"]) for row in daemon]
    assert daemon_scores == sorted(daemon_scores, reverse=True)  # reported score IS the sort score
    for row in daemon:
        factor = 0.5 if support.is_distilled(row) else 1.0
        assert row["score"] == basis[support.row_id(row)] * factor
    # NFR01 (CORE-336-KI1): the library keeps each non-distilled row's unweighted
    # positional score, and its scores still never rise along the shared order.
    assert support.non_distilled_view(library) == support.non_distilled_view(unweighted_library)
    library_scores = [float(row["score"]) for row in library]
    assert library_scores == sorted(library_scores, reverse=True)
    # The weight moved at least one distilled row, so the parity above is not vacuous.
    assert [support.row_id(row) for row in daemon] != sorted(basis, key=lambda key: -basis[key])


@pytest.mark.asyncio
async def test_include_distilled_false_excludes_distilled_rows_on_both_routes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests import _source_weighting_support as support

    support.stub_cross_encoder(monkeypatch, None)
    store = support.make_client(tmp_path, monkeypatch)
    support.seed(store, support.PARITY_ROWS)
    library = await store.recall(support.QUERY, limit=20, include_org_memories=False, include_distilled=False)
    daemon = support.daemon_recall(store, include_distilled=False)["memories"]
    for rows in (library, daemon):
        ids = {support.row_id(dict(row)) for row in rows}
        assert ids and not any(row_id.startswith("M-git") for row_id in ids)


# --- weighted at most once: rows that never pass through the pipeline keep one weighting ---


@pytest.mark.asyncio
async def test_library_weights_pipeline_rows_once_and_other_rows_once(
    client: MemoryClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pipeline row already carries the weight in its order; tier, graph and fallback rows get it here."""
    from dataclasses import replace

    _prepare_common_recall_mocks(client, monkeypatch)
    git = {"source": "distilled:git:aaa..bbb"}
    pipeline_row = replace(_candidate(memory_id="pipeline-git", score=0.5, metadata=git), distilled_weighted=True)
    tier_row = _candidate(memory_id="tier-git", score=0.5, metadata=git)
    monkeypatch.setattr(client, "_try_hybrid_recall", AsyncMock(return_value=[pipeline_row, tier_row]))

    out = await client.recall("source policy", distilled_weight=0.5)

    assert {row["memory_id"]: row["score"] for row in out} == {"pipeline-git": 0.5, "tier-git": 0.25}


def test_remote_rows_take_the_policy_weight_once() -> None:
    from trw_memory.retrieval.source_policy import SourcePolicy

    policy = SourcePolicy.resolve(distilled_weight=0.5)
    remote = {"score": 0.8, "metadata": {"source": "distilled:git:aaa..bbb"}, "source": "shared"}
    assert policy.rank_key(remote) == (1, -0.4)
    assert policy.rank_key(remote, pipeline_weighted=True) == (1, -0.8)


def test_daemon_wildcard_and_supplementary_rows_are_not_weighted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Wildcard rows carry utility and supplements a rescaled rank: neither passes the weighting step."""
    from tests import _source_weighting_support as support

    support.stub_cross_encoder(monkeypatch, None)
    store = support.make_client(tmp_path, monkeypatch)
    support.seed(store, support.PARITY_ROWS)
    supplement = support.DISTILLED_ROWS[0]
    monkeypatch.setattr(recall_tool, "supports_tier_runtime", lambda _backend: True)
    monkeypatch.setattr(
        recall_tool,
        "tier_candidates",
        lambda *_a, **_k: [
            {
                "id": "M-tier-git",
                "namespace": "default",
                "content": supplement[1],
                "score": 0.9,
                "metadata": supplement[2],
            }
        ],
    )

    def run(weight: str, query: str) -> list[tuple[str, object]]:
        monkeypatch.setenv("TRW_MEMORY_DISTILLED_RECALL_WEIGHT", weight)
        return [(support.row_id(row), row["score"]) for row in support.daemon_recall(store, query)["memories"]]

    assert run("0.5", "") == run("1.0", "")
    tier_score = {weight: dict(run(weight, "consumer lag"))["M-tier-git"] for weight in ("0.5", "1.0")}
    assert tier_score["0.5"] == tier_score["1.0"]


@pytest.mark.asyncio
async def test_library_distilled_move_keeps_cross_family_non_distilled_order_and_scores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CORE-336-KI1 (NFR01): a distilled row dropping past two non-distilled rows of
    different families must not flip them on the library route.

    Pre-weight pipeline order is semantic (weight 0.45), distilled, unknown (1.0):
    positions 1, 1/2, 1/3, so semantic leads unknown 0.45 to 0.333. Re-scoring
    positions over the WEIGHTED order gives unknown 1/2, ahead of semantic.
    """
    from tests import _source_weighting_support as support

    support.stub_cross_encoder(monkeypatch, None)
    store = support.make_client(tmp_path, monkeypatch)
    rows = (
        support._graded("M-sem", 8, {"source_kind": "semantic_memory"}),
        support._graded("M-git", 7, {"source": "distilled:git:aaa..bbb"}),
        support._graded("M-unk", 6, {}),
    )
    support.seed(store, rows + support.PARITY_PADDING)

    async def run(weight: float) -> list[dict[str, object]]:
        out = await store.recall(
            support.QUERY,
            limit=20,
            include_org_memories=False,
            distilled_weight=weight,
            source_weights={"semantic_memory": 0.45},
        )
        return [dict(row) for row in out if not str(support.row_id(dict(row))).startswith("M-pad")]

    before, after = await run(1.0), await run(0.5)

    assert [row_id for row_id, _ in support.non_distilled_view(before)] == ["M-sem", "M-unk"]
    # The weight moved the distilled row past both non-distilled rows (not vacuous).
    assert support.row_id(after[-1]) == "M-git"
    assert support.non_distilled_view(after) == support.non_distilled_view(before)
    scores = [float(str(row["score"])) for row in after]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.parametrize(
    ("rows", "expected"),
    [
        # No scores observed (pipeline patched out): positions over the final order.
        ([None, None, None], [1.0, 0.5, 0.3333]),
        # Weighting ran but moved nothing: identical to positions.
        ([("a", 0, False), ("b", 1, False)], [1.0, 0.5]),
        # A distilled row (pre 0) dropped past two others: they keep pre-weight
        # positions 1/2 and 1/3; the distilled row ties the row it now follows.
        ([("b", 1, False), ("c", 2, False), ("d", 0, True)], [0.5, 0.3333, 0.3333]),
        # Two distilled rows swapped order between themselves: never rising.
        ([("d2", 1, True), ("d1", 0, True), ("n", 2, False)], [0.5, 0.5, 0.3333]),
        # A distilled row lifted above a non-distilled one (weight > 1): it is
        # floored at that row's score, which keeps its own pre-weight position.
        ([("d", 1, True), ("n", 0, False)], [1.0, 1.0]),
    ],
)
def test_library_positional_scores_keep_preweight_positions(
    rows: list[tuple[str, int, bool] | None], expected: list[float]
) -> None:
    from trw_memory._client_recall_hybrid import positional_scores
    from trw_memory.retrieval.pipeline import ScoredCandidate

    candidates = [
        None
        if row is None
        else ScoredCandidate(_candidate(memory_id=row[0], score=0.0).entry, 0.0, "position", row[2], row[1])
        for row in rows
    ]
    assert positional_scores(candidates) == expected


@pytest.mark.asyncio
async def test_library_never_promotes_a_distilled_row_above_a_weighted_family_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """core336-ki1 r1 reviewer case: the pipeline ranks semantic ahead of distilled.

    Clamping the distilled row to the semantic row's RAW position (0.5) put it
    above that row's effective key (0.5 x 0.9 = 0.45) and undid the demotion.
    """
    from tests import _source_weighting_support as support

    support.stub_cross_encoder(monkeypatch, None)
    store = support.make_client(tmp_path, monkeypatch)
    rows = (
        support._graded("M-git", 8, {"source": "distilled:git:aaa..bbb"}),
        support._graded("M-sem", 7, {"source_kind": "semantic_memory"}),
    )
    support.seed(store, rows + support.PARITY_PADDING)

    async def run(weight: float) -> list[dict[str, object]]:
        out = await store.recall(support.QUERY, limit=20, include_org_memories=False, distilled_weight=weight)
        return [dict(row) for row in out if not str(support.row_id(dict(row))).startswith("M-pad")]

    before, after = await run(1.0), await run(0.5)
    assert [support.row_id(row) for row in before] == ["M-git", "M-sem"]
    assert [support.row_id(row) for row in after] == ["M-sem", "M-git"]
    assert support.non_distilled_view(after) == support.non_distilled_view(before)


_FAMILIES: tuple[dict[str, str], ...] = (
    {},
    {"source_kind": "semantic_memory"},
    {"source_kind": "instruction_rule"},
    {"source_kind": "lifecycle"},
    {"source_kind": "episodic"},
    {"source": "distilled:git:aaa..bbb"},
    {"source": "distilled:git:aaa..bbb"},
)


@pytest.mark.parametrize(
    "overrides",
    [None, {"semantic_memory": 0.3, "instruction_rule": 1.4, "lifecycle": 0.8, "unknown": 0.6}],
    ids=["default-weights", "override-weights"],
)
def test_library_order_property_random_families_and_distilled_positions(overrides: dict[str, float] | None) -> None:
    """Random families, distilled positions and weights, through the real
    ``weight_distilled`` and ``SourcePolicy.rank_key`` (sorted as ``finish_candidates`` does):

    - NFR01: non-distilled rows keep the weight-1.0 run's scores and relative order;
    - a distilled row never sorts above a same-bucket row the pipeline ranked ahead of it;
    - whenever the non-distilled keys already agree with the pipeline order (so no
      pre-existing cross-family reorder is in play), each bucket's final order IS the
      pipeline order. Unconditional equality is unsatisfiable together with NFR01:
      with S ahead of D ahead of U and U's family weight lifting it above S, the
      pipeline asks for S < D < U while NFR01 fixes U before S.
    """
    import random

    from trw_memory._client_recall_hybrid import positional_scores
    from trw_memory.retrieval.pipeline import ScoredCandidate
    from trw_memory.retrieval.recall_selection import entry_policy_fields
    from trw_memory.retrieval.source_policy import SourcePolicy, classify_source_family, weight_distilled

    policy = SourcePolicy.resolve(source_weights=overrides)

    def key(entry: MemoryEntry, score: float) -> tuple[int, float]:
        return policy.rank_key(entry_policy_fields(entry, score=score), pipeline_weighted=True)

    def library(weight: float, entries: list[MemoryEntry]) -> tuple[list[str], dict[str, float], list[str]]:
        pre = [1.0 / (1 + rank) for rank in range(len(entries))]
        weighted = (
            weight_distilled(entries, pre, weight) if weight != 1.0 else [(e, s, False) for e, s in zip(entries, pre)]
        )
        rank_of = {e.id: rank for rank, e in enumerate(entries)}  # entries ARE the pre-weight order
        pipeline = [ScoredCandidate(e, s, "position", w, rank_of[e.id]) for e, s, w in weighted]
        scores = positional_scores(list(pipeline), key)
        final = sorted(zip(pipeline, scores), key=lambda pair: key(pair[0].entry, pair[1]))
        return [c.entry.id for c in pipeline], {c.entry.id: s for c, s in final}, [c.entry.id for c, _ in final]

    rng = random.Random(336)
    agreeing_moves = 0
    for _ in range(400):
        entries = [
            _candidate(memory_id=f"r{i}", score=0.0, metadata=rng.choice(_FAMILIES)).entry
            for i in range(rng.randint(2, 12))
        ]
        is_git = {e.id: classify_source_family({"metadata": e.metadata}) == "git_distilled" for e in entries}
        pipeline, scores, final = library(rng.choice([0.2, 0.5, 0.75]), entries)
        _, base_scores, base_final = library(1.0, entries)
        by_id = {e.id: e for e in entries}
        nd = [i for i in final if not is_git[i]]
        assert nd == [i for i in base_final if not is_git[i]]
        assert {i: scores[i] for i in nd} == {i: base_scores[i] for i in nd}
        bucket = {i: key(by_id[i], scores[i])[0] for i in pipeline}
        for d in (i for i in pipeline if is_git[i]):
            ahead = pipeline[: pipeline.index(d)]
            assert all(final.index(a) < final.index(d) for a in ahead if bucket[a] == bucket[d]), (pipeline, final)
        for b in set(bucket.values()):
            in_b = [i for i in pipeline if bucket[i] == b]
            nd_keys = [key(by_id[i], scores[i])[1] for i in in_b if not is_git[i]]
            if nd_keys == sorted(nd_keys):
                assert [i for i in final if bucket[i] == b] == in_b, (pipeline, final)
                agreeing_moves += pipeline != base_final
    assert agreeing_moves > 20  # the equality arm is exercised on lists the weight reordered
