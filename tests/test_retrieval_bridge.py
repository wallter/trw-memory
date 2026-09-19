"""Entity-bridge second hop: term selection, tail promotion, pipeline + recall wiring."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch

import pytest

pytest.importorskip("rank_bm25")

from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.retrieval.bridge import bridge_terms, extend_with_bridge
from trw_memory.retrieval.pipeline import hybrid_search, hybrid_search_scored
from trw_memory.security.namespace_scope import authorize_namespaces
from trw_memory.security.rbac import Permission

NS = "project:bridge"


def _e(eid: str, content: str, **kw: object) -> MemoryEntry:
    return MemoryEntry(id=eid, content=content, namespace=NS, tags=[], **kw)


def _corpus() -> list[MemoryEntry]:
    """``seed`` answers the question; ``bridge`` shares only the rare word "pottery" with it."""
    rows = [_e("seed", "Melanie: I signed up for a pottery class yesterday, it is like therapy")]
    # Filler so "pottery" is rare (low df) while "Melanie" is everywhere; it also
    # shares "class" with the query so fusion ranks every filler above "bridge".
    rows += [_e(f"f{i}", f"Melanie: the class talked about the weather and lunch number {i}") for i in range(60)]
    rows.append(_e("bridge", "Melanie: finished my first pottery bowl this weekend"))
    return rows


def _scope():
    return authorize_namespaces(MemoryConfig(rbac_enabled=False), [NS], Permission.READ, "test")


def test_bridge_terms_pick_rare_seed_terms_not_in_the_query() -> None:
    entries = _corpus()
    terms = bridge_terms("What activities does Melanie do?", [entries[0]], entries)
    assert "pottery" in terms
    # present in every row -> no idf -> never a bridge; and query words are excluded
    assert "melanie" not in terms
    assert "activiti" not in terms and "activity" not in terms


def test_bridge_terms_empty_inputs() -> None:
    entries = _corpus()
    assert bridge_terms("q", [], entries) == []
    assert bridge_terms("q", [entries[0]], []) == []
    assert bridge_terms("q", [entries[0]], entries, max_terms=0) == []


def test_extend_promotes_tail_rows_the_bridge_query_finds() -> None:
    entries = _corpus()
    seed, bridge = entries[0], entries[-1]
    tail = [e for e in entries if e.id != "seed"]
    calls: list[list[str]] = []

    def score(fresh: list[MemoryEntry]) -> list[tuple[MemoryEntry, float]]:
        calls.append([e.id for e in fresh])
        return [(e, 2.0 if e.id == "bridge" else -9.0) for e in fresh]

    scored, new_tail, changed = extend_with_bridge(
        "What activities does Melanie do?", [(seed, 5.0)], tail, entries, score=score
    )
    assert changed is True
    # only rows holding a bridge term reach the cross-encoder; fillers match query words alone
    assert calls == [["bridge"]]
    assert [e.id for e, _ in scored][:2] == ["seed", "bridge"]
    assert bridge.id not in {e.id for e in new_tail}
    # every promoted row left the tail; nothing is duplicated or lost
    assert len(scored) + len(new_tail) == 1 + len(tail)


def test_extend_is_a_no_op_when_the_scorer_is_unavailable_or_tail_empty() -> None:
    entries = _corpus()
    scored = [(entries[0], 5.0)]
    tail = entries[1:]
    assert extend_with_bridge("q Melanie", scored, tail, entries, score=lambda fresh: None) == (scored, tail, False)
    called: list[int] = []
    out = extend_with_bridge("q", scored, [], entries, score=lambda fresh: called.append(1) or [])
    assert out == (scored, [], False) and called == []


def _table_scores(query, entries, *, model_name, local_only=False):
    table = {"seed": 5.0, "bridge": 3.0}
    return sorted(((e, table.get(e.id, -9.0)) for e in entries), key=lambda x: x[1], reverse=True)


def _search(entries: list[MemoryEntry], *, bridge_hop: bool):
    # A flat dense signal puts EVERY row in the fused list, as production's
    # namespace-sized vector pool does; the tail is then all non-reranked rows.
    vectors = {e.id: [1.0, 0.0] for e in entries}
    with patch("trw_memory.retrieval.reranker.cross_encode_scores", side_effect=_table_scores):
        return hybrid_search_scored(
            "What activities does Melanie do in therapy class?",
            entries,
            scope=_scope(),
            rerank=True,
            rerank_candidates=3,
            rerank_min_score=None,
            top_k=10,
            bridge_hop=bridge_hop,
            query_embedding=[1.0, 0.0],
            stored_embeddings=vectors,
            vector_candidates=len(entries),
        )


def test_pipeline_bridge_lifts_a_tail_row_into_the_head() -> None:
    entries = _corpus()
    off = [c.entry.id for c in _search(entries, bridge_hop=False)]
    assert off[0] == "seed" and "bridge" not in off[:3]
    on = _search(entries, bridge_hop=True)
    assert [c.entry.id for c in on][:2] == ["seed", "bridge"]
    # the order no longer follows fusion scores, so scores are positional
    assert {c.basis for c in on} == {"position"}


def test_pipeline_bridge_never_resurrects_an_ineligible_row() -> None:
    entries = _corpus()
    entries[-1] = _e(
        "bridge",
        entries[-1].content,
        valid_from=datetime(2019, 1, 1, tzinfo=timezone.utc),
        invalid_from=datetime(2020, 1, 1, tzinfo=timezone.utc),
        invalidated_by="seed",
    )
    ids = [c.entry.id for c in _search(entries, bridge_hop=True)]
    assert ids[0] == "seed" and "bridge" not in ids


def test_pipeline_bridge_defaults_off() -> None:
    entries = _corpus()
    with patch("trw_memory.retrieval.bridge.extend_with_bridge") as ext:
        with patch("trw_memory.retrieval.reranker.cross_encode_scores", side_effect=_table_scores):
            hybrid_search("pottery", entries, scope=_scope(), rerank=True, rerank_candidates=3)
    ext.assert_not_called()


@pytest.mark.parametrize(("env", "expected"), [(None, True), ("false", False)])
async def test_recall_turns_the_bridge_on_with_rerank(tmp_path, monkeypatch, env, expected) -> None:
    from trw_memory.client import MemoryClient
    from trw_memory.retrieval import pipeline as pipeline_mod

    monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "mem"))
    monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "sqlite")
    if env is not None:
        monkeypatch.setenv("MEMORY_RECALL_BRIDGE_HOP", env)
    client = MemoryClient(namespace="default", mode="local")
    await client.store("bridge wiring entry")
    captured: dict[str, object] = {}
    original = pipeline_mod.hybrid_search

    def spy(*args: object, **kwargs: object):
        captured.update(kwargs)
        return original(*args, **kwargs)

    with (
        patch.object(pipeline_mod, "hybrid_search", side_effect=spy),
        patch(
            "trw_memory.retrieval.reranker.cross_encode_scores",
            side_effect=lambda query, entries, **kwargs: [(e, 1.0) for e in entries],
        ),
    ):
        await client.recall("bridge wiring")
    assert captured.get("bridge_hop") is expected
