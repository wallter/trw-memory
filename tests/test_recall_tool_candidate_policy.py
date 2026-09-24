"""PRD-CORE-298 FR05 step 2 -- the daemon recall tool acquires candidates like the other surfaces.

``MemoryClient.recall`` and trw-mcp's ``trw_recall`` take their candidates from
``retrieval/recall_policy.acquire_candidates``: the recency pool plus the rows
full-text search finds past it. The daemon's ``memory_recall`` read the recency
pool alone, so a relevant row older than the pool window could never reach its
ranker, and the three surfaces answered one query three ways. These tests watch
what reaches the ranker (``build_scored_candidates``), which isolates
acquisition from the tier supplement that is merged after ranking.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from mcp.server.auth.middleware.auth_context import auth_context_var
from mcp.server.auth.middleware.bearer_auth import AuthenticatedUser
from mcp.server.auth.provider import AccessToken

from tests.conftest import make_entry
from trw_memory.integrations._backend import create_backend_from_config
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryStatus
from trw_memory.retrieval.recall_policy import RECALL_PREFETCH_MULTIPLIER
from trw_memory.storage.interface import StorageBackend
from trw_memory.tools import recall as recall_tool

_NS = "project:policy-11111111"
_OTHER = "project:other-22222222"
_POOL = 10
#: The recency pool the tool reads for ``limit=2``: trw_recall's ranking depth
#: (2 x RECALL_PREFETCH_MULTIPLIER) times five, or the configured pool if larger.
_WINDOW = max(2 * RECALL_PREFETCH_MULTIPLIER * 5, _POOL)
_OLD = "2020-01-01T00:00:00+00:00"


@pytest.fixture
def config(tmp_path: Path) -> MemoryConfig:
    return MemoryConfig(
        storage_path=str(tmp_path),
        memory_single_store_path=str(tmp_path / "memory.db"),
        hybrid_search_candidate_pool_size=_POOL,
    )


@pytest.fixture
def ranked_ids(monkeypatch: pytest.MonkeyPatch) -> list[set[str]]:
    """The ids of every candidate set handed to the ranker."""
    seen: list[set[str]] = []
    real = recall_tool.build_scored_candidates

    def record(query: str, entries: list[Any], **kwargs: Any) -> Any:
        seen.append({entry.id for entry in entries})
        return real(query, entries, **kwargs)

    monkeypatch.setattr(recall_tool, "build_scored_candidates", record)
    return seen


def _seed(backend: StorageBackend, old: list[tuple[str, str, MemoryStatus]]) -> None:
    """*old* rows first, aged past the pool window, then a full pool of newer distractors."""
    for entry_id, content, status in old:
        backend.store(make_entry(entry_id=entry_id, content=content, namespace=_NS, status=status))
    backend._conn.execute("UPDATE memories SET created_at = ?, updated_at = ?", (_OLD, _OLD))  # type: ignore[attr-defined]
    backend._conn.commit()  # type: ignore[attr-defined]
    for index in range(_WINDOW + 5):
        backend.store(make_entry(entry_id=f"M-new-{index:02d}", content=f"routine note {index}", namespace=_NS))


def _recall(
    backend: StorageBackend, config: MemoryConfig, query: str, *, limit_override: int = 2, **kwargs: Any
) -> dict[str, object]:
    return recall_tool.memory_recall_impl(
        query, _NS, backend=backend, limit=limit_override, config=config, include_org_memories=False, **kwargs
    )


def test_a_row_past_the_recency_window_reaches_the_ranker(config: MemoryConfig, ranked_ids: list[set[str]]) -> None:
    with create_backend_from_config(config, _NS) as backend:
        _seed(backend, [("M-old", "zephyrine cache eviction rule", MemoryStatus.ACTIVE)])
        _recall(backend, config, "zephyrine")

    assert ranked_ids and "M-old" in ranked_ids[0]
    assert len(ranked_ids[0]) == _WINDOW + 1, "the full recency pool plus the one full-text addition"


def test_the_full_text_leg_admits_only_active_rows(config: MemoryConfig, ranked_ids: list[set[str]]) -> None:
    with create_backend_from_config(config, _NS) as backend:
        _seed(backend, [("M-retired", "zephyrine retired convention", MemoryStatus.OBSOLETE)])
        _recall(backend, config, "zephyrine")

    assert ranked_ids and "M-retired" not in ranked_ids[0]


@pytest.fixture
def granted_only_ns() -> Iterator[None]:
    reset = auth_context_var.set(AuthenticatedUser(AccessToken(token="t", client_id="c", scopes=[f"ns:{_NS}"])))
    yield
    auth_context_var.reset(reset)


def test_an_ungranted_extra_namespace_is_never_acquired_from(
    config: MemoryConfig, monkeypatch: pytest.MonkeyPatch, granted_only_ns: None
) -> None:
    acquired: list[str | None] = []
    real = recall_tool.acquire_candidates

    def record(backend: StorageBackend, query: str, **kwargs: Any) -> Any:
        acquired.append(kwargs["namespace"])
        return real(backend, query, **kwargs)

    monkeypatch.setattr(recall_tool, "acquire_candidates", record)
    with create_backend_from_config(config, _NS) as backend:
        backend.store(make_entry(entry_id="M-foreign", content="zephyrine foreign", namespace=_OTHER))
        result = _recall(
            backend,
            config,
            "zephyrine",
            include_namespaces=[_OTHER],
            namespace_backend_factory=lambda ns: create_backend_from_config(config, ns),
        )

    assert acquired == [_NS]
    assert "M-foreign" not in str(result)


def test_a_wrong_tag_full_text_match_never_reaches_the_ranker(config: MemoryConfig, ranked_ids: list[set[str]]) -> None:
    """The full-text leg honours ``tags`` as the recency leg does (Codex P1 on b5fe0557a)."""
    with create_backend_from_config(config, _NS) as backend:
        backend.store(make_entry(entry_id="M-wrong", content="zephyrine wrong tag", namespace=_NS, tags=["other"]))
        backend.store(make_entry(entry_id="M-right", content="zephyrine right tag", namespace=_NS, tags=["wanted"]))
        backend._conn.execute("UPDATE memories SET created_at = ?, updated_at = ?", (_OLD, _OLD))  # type: ignore[attr-defined]
        backend._conn.commit()  # type: ignore[attr-defined]
        for index in range(_POOL + 5):
            backend.store(
                make_entry(entry_id=f"M-new-{index:02d}", content=f"note {index}", namespace=_NS, tags=["wanted"])
            )
        _recall(backend, config, "zephyrine", tags=["wanted"])

    assert ranked_ids and "M-right" in ranked_ids[0]
    assert "M-wrong" not in ranked_ids[0]


def test_full_text_filters_tags_before_its_limit(config: MemoryConfig) -> None:
    """A better-matching wrong-tag row must not take the only slot."""
    with create_backend_from_config(config, _NS) as backend:
        backend.store(
            make_entry(entry_id="M-wrong", content="zephyrine zephyrine zephyrine", namespace=_NS, tags=["other"])
        )
        backend.store(
            make_entry(entry_id="M-right", content="zephyrine and more words here", namespace=_NS, tags=["wanted"])
        )
        found = backend.search_fts("zephyrine", top_k=1, namespace=_NS, tags=["wanted"])

    assert [entry.id for entry in found] == ["M-right"]


def test_the_tool_ranks_with_the_shared_ranking_arguments(
    config: MemoryConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same pool, same arguments as trw_recall: rerank, fusion, the adaptive floor and the bridge hop."""
    from trw_memory.retrieval.recall_policy import ranking_arguments, resolve_query

    seen: list[dict[str, Any]] = []
    real = recall_tool.hybrid_search_scored

    def record(**kwargs: Any) -> Any:
        seen.append(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(recall_tool, "hybrid_search_scored", record)
    with create_backend_from_config(config, _NS) as backend:
        _seed(backend, [("M-old", "zephyrine cache eviction rule", MemoryStatus.ACTIVE)])
        _recall(backend, config, "zephyrine")

    expected = ranking_arguments(
        config,
        limit=2 * RECALL_PREFETCH_MULTIPLIER,  # trw_recall's depth; the tool caps to 2 last
        pool_size=_WINDOW + 1,
        recency_weight=resolve_query("zephyrine", config).recency_weight,
    )
    assert seen and {key: seen[0][key] for key in expected} == expected


def test_a_query_keeps_the_retrieval_order(config: MemoryConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    """Tied retrieval scores are not reordered by utility, as trw_recall keeps the pipeline's order."""
    from trw_memory.retrieval.pipeline import ScoredCandidate

    def tied(**kwargs: Any) -> list[ScoredCandidate]:
        by_id = {entry.id: entry for entry in kwargs["entries"]}
        return [ScoredCandidate(entry=by_id[i], score=0.5, basis="fused") for i in ("M-low", "M-high")]

    monkeypatch.setattr(recall_tool, "hybrid_search_scored", tied)
    monkeypatch.setattr(recall_tool, "supports_tier_runtime", lambda backend: False)
    with create_backend_from_config(config, _NS) as backend:
        backend.store(make_entry(entry_id="M-low", content="zephyrine low", namespace=_NS, importance=0.1))
        backend.store(make_entry(entry_id="M-high", content="zephyrine high", namespace=_NS, importance=0.9))
        result = _recall(backend, config, "zephyrine")

    assert [row["id"] for row in result["memories"]] == ["M-low", "M-high"]  # type: ignore[index, union-attr]


def test_org_discovery_reuses_the_open_store_instead_of_reopening_it(
    config: MemoryConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Under the single store, org-memory discovery used to open the recall's own file a second time."""
    from trw_memory.integrations import _backend

    opened: list[object] = []
    real = _backend.open_namespace_store

    def record(cfg: MemoryConfig, location: Any) -> Any:
        opened.append(location.db_path)
        return real(cfg, location)

    monkeypatch.setattr(_backend, "open_namespace_store", record)
    with create_backend_from_config(config, _NS) as backend:
        backend.store(make_entry(entry_id="M-own", content="zephyrine own", namespace=_NS))
        sibling = make_entry(entry_id="M-sib", content="zephyrine sibling", namespace=_OTHER, importance=0.9)
        sibling.cross_validated = True
        backend.store(sibling)
        result = recall_tool.memory_recall_impl(
            "zephyrine", _NS, backend=backend, limit=5, config=config, include_org_memories=True
        )

    assert opened == []
    assert "M-sib" in str(result), "the org leg still reads sibling rows through the reused store"


def _fixed_order(monkeypatch: pytest.MonkeyPatch, order: list[str]) -> None:
    """The pipeline returns *order*, with falling scores, cut to the ``top_k`` it is asked for."""
    from trw_memory.retrieval.pipeline import ScoredCandidate

    def ranked(**kwargs: Any) -> list[ScoredCandidate]:
        by_id = {entry.id: entry for entry in kwargs["entries"]}
        return [
            ScoredCandidate(entry=by_id[i], score=1.0 / (1 + n), basis="position")
            for n, i in enumerate(order[: kwargs["top_k"]])
        ]

    monkeypatch.setattr(recall_tool, "hybrid_search_scored", ranked)
    monkeypatch.setattr(recall_tool, "supports_tier_runtime", lambda backend: False)


def test_mixed_sources_keep_the_pipeline_order(config: MemoryConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    """An episodic top hit is not demoted below durable rows, as trw_recall does not demote it (Codex P1)."""
    _fixed_order(monkeypatch, ["M-episode", "M-durable", "M-git"])
    with create_backend_from_config(config, _NS) as backend:
        backend.store(
            make_entry(entry_id="M-episode", content="zephyrine episode", namespace=_NS, tags=["source_kind:episodic"])
        )
        backend.store(make_entry(entry_id="M-durable", content="zephyrine durable", namespace=_NS))
        backend.store(make_entry(entry_id="M-git", content="zephyrine git", namespace=_NS, tags=["source_kind:git"]))
        result = _recall(backend, config, "zephyrine", limit_override=3)

    assert [row["id"] for row in result["memories"]] == ["M-episode", "M-durable", "M-git"]  # type: ignore[index, union-attr]


def test_a_top_hit_admission_drops_is_refilled_from_the_ranked_tail(
    config: MemoryConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Admission runs after ranking, so the tool ranks past ``limit`` and caps last (Codex P1)."""
    _fixed_order(monkeypatch, ["M-git", "M-durable", "M-other"])
    with create_backend_from_config(config, _NS) as backend:
        backend.store(make_entry(entry_id="M-git", content="zephyrine git", namespace=_NS, tags=["source_kind:git"]))
        backend.store(make_entry(entry_id="M-durable", content="zephyrine durable", namespace=_NS))
        backend.store(make_entry(entry_id="M-other", content="zephyrine other", namespace=_NS))
        result = _recall(backend, config, "zephyrine", limit_override=1, include_distilled=False)

    assert [row["id"] for row in result["memories"]] == ["M-durable"]  # type: ignore[index, union-attr]
