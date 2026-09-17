"""PRD-CORE-278 FR01-FR04, FR07 — the scored candidate contract.

What these tests pin is a CHAIN, not a function: the retrieval pipeline computes
a ranking, and every stage after it has to be unable to quietly replace that
ranking with a weaker one. Each class below owns one link.
"""

from __future__ import annotations

import math
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock, patch

from trw_memory.integrations._backend import create_backend_from_config
from trw_memory.lifecycle._recall import FUSED_SCORE_KEY, rank_by_utility
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.retrieval import hybrid_search, hybrid_search_scored
from trw_memory.retrieval.bm25 import _normalize_text, _split_identifiers, _tokenize_entry
from trw_memory.retrieval.lexical import lexical_relevance, tokenize_query
from trw_memory.security.namespace_scope import authorize_namespaces
from trw_memory.security.rbac import Permission
from trw_memory.tools.recall import memory_recall_impl

NAMESPACE = "project:default"


def _scope(cfg: MemoryConfig | None = None) -> Any:
    return authorize_namespaces(cfg or MemoryConfig(), [NAMESPACE], Permission.READ, "recall")


def _entries(*contents: str) -> list[MemoryEntry]:
    return [MemoryEntry(id=f"e{i}", content=text, namespace=NAMESPACE) for i, text in enumerate(contents)]


class TestPipelineReturnsScoredCandidates:
    def test_scores_are_finite_and_non_increasing(self) -> None:
        entries = _entries(
            "pydantic validation error on a nested model",
            "pydantic settings loader",
            "unrelated filler entry",
        )
        candidates = hybrid_search_scored("pydantic validation", entries, scope=_scope())
        assert candidates
        scores = [candidate.score for candidate in candidates]
        assert all(math.isfinite(score) for score in scores)
        assert scores == sorted(scores, reverse=True)

    def test_entry_view_matches_scored_view(self) -> None:
        entries = _entries("pydantic validation error", "pydantic settings loader", "filler")
        scored = hybrid_search_scored("pydantic", entries, scope=_scope())
        plain = hybrid_search("pydantic", entries, scope=_scope())
        assert [candidate.entry.id for candidate in scored] == [entry.id for entry in plain]

    def test_basis_is_fused_when_nothing_reordered(self) -> None:
        entries = _entries("pydantic validation error", "filler entry")
        candidates = hybrid_search_scored("pydantic", entries, scope=_scope())
        assert {candidate.basis for candidate in candidates} == {"fused"}

    def test_reranker_reversal_switches_basis_and_rescores(self) -> None:
        """A cross-encoder that reverses fusion order must not be explained by
        the fusion numbers it just overruled."""
        entries = _entries("pydantic validation error", "pydantic settings loader", "pydantic model config")
        with patch(
            "trw_memory.retrieval.reranker.cross_encode_scores",
            side_effect=lambda _q, items, **_kw: [(e, float(i)) for i, e in enumerate(items)][::-1],
        ):
            candidates = hybrid_search_scored("pydantic", entries, scope=_scope(), rerank=True)
        assert len(candidates) > 1
        assert {candidate.basis for candidate in candidates} == {"position"}
        scores = [candidate.score for candidate in candidates]
        assert scores == sorted(scores, reverse=True)

    def test_superseded_record_never_outranks_an_open_one(self) -> None:
        now = datetime.now(timezone.utc)
        open_entry = MemoryEntry(id="open", content="pydantic validation error", namespace=NAMESPACE, valid_from=now)
        superseded = MemoryEntry(
            id="old",
            content="pydantic validation error older wording",
            namespace=NAMESPACE,
            valid_from=now - timedelta(days=2),
            invalid_from=now,
            invalidated_by="open",
        )
        candidates = hybrid_search_scored(
            "pydantic validation", [superseded, open_entry], scope=_scope(), include_superseded=True
        )
        ids = [candidate.entry.id for candidate in candidates]
        assert ids.index("open") < ids.index("old")
        scores = [candidate.score for candidate in candidates]
        assert scores == sorted(scores, reverse=True)


class TestRankerConsumesTheFusedScore:
    def test_retrieval_score_outranks_utility(self) -> None:
        strong = {"id": "strong", "content": "anything", FUSED_SCORE_KEY: 0.9, "importance": 0.1}
        popular = {
            "id": "popular",
            "content": "anything",
            FUSED_SCORE_KEY: 0.2,
            "importance": 1.0,
            "access_count": 50,
            "q_value": 0.99,
            "q_observations": 20,
        }
        ranked = rank_by_utility([popular, strong], tokenize_query("anything"))
        assert [row["id"] for row in ranked] == ["strong", "popular"]

    def test_utility_only_breaks_ties(self) -> None:
        low = {"id": "low", "content": "same", FUSED_SCORE_KEY: 0.5, "importance": 0.1}
        high = {"id": "high", "content": "same", FUSED_SCORE_KEY: 0.5, "importance": 0.9}
        ranked = rank_by_utility([low, high], tokenize_query("same"))
        assert [row["id"] for row in ranked] == ["high", "low"]

    def test_no_lambda_weight_parameter_exists(self) -> None:
        """NFR01: the blend factor is gone and no knob replaced it."""
        import inspect

        assert "lambda_weight" not in inspect.signature(rank_by_utility).parameters

    def test_whole_word_matching_replaces_substring_matching(self) -> None:
        tokens = tokenize_query("lang")
        assert lexical_relevance({"content": "favourite_language: Go"}, tokens) == 0.0
        assert lexical_relevance({"content": "favourite_language: Go"}, tokenize_query("language")) > 0.0

    def test_stopwords_are_removed_from_the_query(self) -> None:
        assert tokenize_query("who is my boss") == ["boss"]

    def test_stopword_only_query_keeps_its_tokens(self) -> None:
        assert tokenize_query("is it the") == ["is", "it", "the"]

    def test_all_zero_scores_do_not_divide_by_zero(self) -> None:
        rows = [
            {"id": "a", "content": "x", FUSED_SCORE_KEY: 0.0, "importance": 0.2},
            {"id": "b", "content": "x", FUSED_SCORE_KEY: 0.0, "importance": 0.9},
        ]
        assert [row["id"] for row in rank_by_utility(rows, tokenize_query("x"))] == ["b", "a"]

    def test_a_single_non_finite_score_is_rejected_individually(self) -> None:
        """A broken neighbour must not collapse everyone else's relevance."""
        rows = [
            {"id": "broken", "content": "pydantic", FUSED_SCORE_KEY: float("nan")},
            {"id": "good", "content": "pydantic", FUSED_SCORE_KEY: 0.9},
            {"id": "unrelated", "content": "nothing here", FUSED_SCORE_KEY: 0.9},
        ]
        ranked = [row["id"] for row in rank_by_utility(rows, tokenize_query("pydantic"))]
        # "broken" falls back to lexical relevance (1.0) rather than to zero.
        assert ranked.index("broken") < ranked.index("unrelated")

    def test_wildcard_query_orders_by_utility(self) -> None:
        rows = [
            {"id": "dull", "content": "a", "importance": 0.1},
            {"id": "bright", "content": "b", "importance": 0.9},
        ]
        assert [row["id"] for row in rank_by_utility(rows, [])] == ["bright", "dull"]


class TestToolRecallPreservesScoreOrder:
    def _store(self, tmp: Path, cfg: MemoryConfig, rows: list[tuple[str, str]]) -> Any:
        backend = create_backend_from_config(cfg, NAMESPACE)
        for entry_id, content in rows:
            backend.store(MemoryEntry(id=entry_id, content=content, namespace=NAMESPACE))
        return backend

    def test_returned_order_and_scores_are_the_retrieval_ranking(self) -> None:
        """The candidates the pipeline produced ARE what the caller receives.

        Captured at the pipeline boundary rather than recomputed, so the test
        cannot accidentally grade a different retrieval policy than the tool ran.
        """
        captured: list[Any] = []
        with tempfile.TemporaryDirectory() as td:
            cfg = MemoryConfig(storage_backend="sqlite", storage_path=td)
            with self._store(
                Path(td),
                cfg,
                [
                    ("target", "favourite_language: Go"),
                    ("weak", "note_1: filler language"),
                    ("absent", "nothing relevant at all"),
                ],
            ) as backend:
                real = hybrid_search_scored

                def spy(**kwargs: Any) -> Any:
                    produced = real(**kwargs)
                    captured.extend(produced)
                    return produced

                with (
                    patch("trw_memory.tools.recall.get_local_embedder", return_value=None),
                    patch("trw_memory.tools.recall.hybrid_search_scored", spy),
                ):
                    result = memory_recall_impl(
                        "favourite language",
                        NAMESPACE,
                        backend=backend,
                        config=cfg,
                        limit=5,
                        include_org_memories=False,
                    )
        memories = cast("list[dict[str, object]]", result["memories"])
        assert captured
        assert [row["id"] for row in memories] == [candidate.entry.id for candidate in captured]
        by_id = {candidate.entry.id: candidate.score for candidate in captured}
        for row in memories:
            assert round(float(cast("float", row["score"])), 4) == round(by_id[str(row["id"])], 4)

    def test_scores_are_not_a_constant(self) -> None:
        """The defect this PRD measures: every row came back with one number."""
        with tempfile.TemporaryDirectory() as td:
            cfg = MemoryConfig(storage_backend="sqlite", storage_path=td)
            with self._store(
                Path(td),
                cfg,
                [("a", "deploy runbook for the api"), ("b", "deploy notes"), ("c", "api reference")],
            ) as backend:
                with patch("trw_memory.tools.recall.get_local_embedder", return_value=None):
                    result = memory_recall_impl(
                        "deploy api", NAMESPACE, backend=backend, config=cfg, limit=5, include_org_memories=False
                    )
        scores = [float(cast("float", row["score"])) for row in cast("list[dict[str, object]]", result["memories"])]
        assert len(scores) > 1
        assert len(set(scores)) > 1
        assert scores == sorted(scores, reverse=True)

    def test_a_tier_only_candidate_cannot_overtake_a_retrieval_hit(self) -> None:
        """Through the WHOLE path, not just the merge helper.

        A merge-level assertion is not enough: ``apply_source_policy`` sorts the
        merged list by score afterwards, so a tier row carrying an absolute
        utility score (~0.8) would climb back over a realistic fused score
        (~0.16) after the merge had correctly placed it below.
        """
        from trw_memory.lifecycle.tiers._runtime import get_tier_manager

        with tempfile.TemporaryDirectory() as td:
            cfg = MemoryConfig(storage_backend="sqlite", storage_path=td)
            with self._store(Path(td), cfg, [("retrieved", "deploy runbook for the api")]) as backend:
                manager = get_tier_manager(cfg, NAMESPACE)
                manager.warm_add(
                    "tier-only",
                    MemoryEntry(
                        id="tier-only",
                        content="unrelated warm entry",
                        namespace=NAMESPACE,
                        importance=0.99,
                        q_value=0.99,
                        q_observations=20,
                    ).model_dump(mode="json"),
                    [1.0, 0.0],
                )
                with patch("trw_memory.tools.recall.get_local_embedder", return_value=None):
                    result = memory_recall_impl(
                        "deploy runbook",
                        NAMESPACE,
                        backend=backend,
                        config=cfg,
                        limit=5,
                        include_org_memories=False,
                    )
        memories = cast("list[dict[str, object]]", result["memories"])
        ids = [str(row["id"]) for row in memories]
        assert ids[0] == "retrieved"
        if "tier-only" in ids:
            assert ids.index("retrieved") < ids.index("tier-only")
        scores = [float(cast("float", row["score"])) for row in memories]
        assert scores == sorted(scores, reverse=True)

    def test_min_score_filters_the_number_the_response_reports(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = MemoryConfig(storage_backend="sqlite", storage_path=td)
            with self._store(Path(td), cfg, [("a", "deploy runbook for the api"), ("b", "deploy notes")]) as backend:
                with patch("trw_memory.tools.recall.get_local_embedder", return_value=None):
                    unfiltered = memory_recall_impl(
                        "deploy api", NAMESPACE, backend=backend, config=cfg, include_org_memories=False
                    )
                    rows = cast("list[dict[str, object]]", unfiltered["memories"])
                    floor = float(cast("float", rows[-1]["score"]))
                    filtered = memory_recall_impl(
                        "deploy api",
                        NAMESPACE,
                        backend=backend,
                        config=cfg,
                        include_org_memories=False,
                        min_score=floor,
                    )
        kept = [float(cast("float", row["score"])) for row in cast("list[dict[str, object]]", filtered["memories"])]
        assert kept
        assert all(score >= floor for score in kept)

    def test_a_wildcard_recall_reports_a_score_and_honours_min_score(self) -> None:
        """The empty-query path has no retrieval score; it must still report one."""
        with tempfile.TemporaryDirectory() as td:
            cfg = MemoryConfig(storage_backend="sqlite", storage_path=td)
            with self._store(Path(td), cfg, [("a", "first"), ("b", "second")]) as backend:
                with patch("trw_memory.tools.recall.get_local_embedder", return_value=None):
                    result = memory_recall_impl(
                        "", NAMESPACE, backend=backend, config=cfg, include_org_memories=False, min_score=0.0001
                    )
        memories = cast("list[dict[str, object]]", result["memories"])
        assert memories, "a wildcard recall with a tiny floor returned nothing"
        assert all(float(cast("float", row["score"])) > 0.0 for row in memories)

    def test_response_shape_is_unchanged(self) -> None:
        """NFR04: only the VALUE of score changes meaning."""
        with tempfile.TemporaryDirectory() as td:
            cfg = MemoryConfig(storage_backend="sqlite", storage_path=td)
            with self._store(Path(td), cfg, [("a", "deploy runbook")]) as backend:
                with patch("trw_memory.tools.recall.get_local_embedder", return_value=None):
                    result = memory_recall_impl(
                        "deploy", NAMESPACE, backend=backend, config=cfg, include_org_memories=False
                    )
        assert set(result) >= {
            "memories",
            "total_matches",
            "query",
            "tokens_used",
            "tokens_budget",
            "tokens_truncated",
        }


class TestDegradedLexicalFallback:
    def test_identifier_words_are_indexed_separately(self) -> None:
        tokens = _tokenize_entry(MemoryEntry(id="x", content="favourite_language: Go", namespace=NAMESPACE))
        assert {"favourite", "language", "favourite_language"} <= set(tokens)

    def test_identifier_split_keeps_the_composite_token(self) -> None:
        assert _split_identifiers(["manager_name"]) == ["manager_name", "manager", "name"]
        assert _normalize_text("HybridSearch") == "hybrid search"

    def test_no_retrieval_source_still_answers_a_lexical_query(self) -> None:
        entries = _entries("favourite_language: Go", "note_1: unrelated filler entry 1")
        with patch("trw_memory.retrieval.bm25._BM25_AVAILABLE", False):
            candidates = hybrid_search_scored("favourite language", entries, scope=_scope())
        assert [candidate.entry.content for candidate in candidates] == ["favourite_language: Go"]

    def test_no_lexical_match_returns_nothing_rather_than_filler(self) -> None:
        entries = _entries("note_1: unrelated filler entry 1", "note_2: unrelated filler entry 2")
        with patch("trw_memory.retrieval.bm25._BM25_AVAILABLE", False):
            assert hybrid_search_scored("kubernetes cluster", entries, scope=_scope()) == []

    def test_fallback_still_honours_the_validity_prior(self) -> None:
        """The fallback lives INSIDE the pipeline so exclusions still apply."""
        now = datetime.now(timezone.utc)
        open_entry = MemoryEntry(id="open", content="pydantic validation", namespace=NAMESPACE, valid_from=now)
        superseded = MemoryEntry(
            id="old",
            content="pydantic validation",
            namespace=NAMESPACE,
            valid_from=now - timedelta(days=2),
            invalid_from=now,
            invalidated_by="open",
        )
        with patch("trw_memory.retrieval.bm25._BM25_AVAILABLE", False):
            candidates = hybrid_search_scored("pydantic validation", [superseded, open_entry], scope=_scope())
        assert [candidate.entry.id for candidate in candidates] == ["open"]


class TestEmptyNamespaceWarmsTheEmbedder:
    def test_query_in_an_empty_namespace_resolves_the_embedder(self) -> None:
        factory = MagicMock(return_value=None)
        with tempfile.TemporaryDirectory() as td:
            cfg = MemoryConfig(storage_backend="sqlite", storage_path=td)
            with create_backend_from_config(cfg, NAMESPACE) as backend:
                with patch("trw_memory.tools.recall.get_local_embedder", factory):
                    memory_recall_impl(
                        "readiness probe", NAMESPACE, backend=backend, config=cfg, include_org_memories=False
                    )
        assert factory.call_count == 1

    def test_empty_query_resolves_nothing(self) -> None:
        factory = MagicMock(return_value=None)
        with tempfile.TemporaryDirectory() as td:
            cfg = MemoryConfig(storage_backend="sqlite", storage_path=td)
            with create_backend_from_config(cfg, NAMESPACE) as backend:
                with patch("trw_memory.tools.recall.get_local_embedder", factory):
                    memory_recall_impl("", NAMESPACE, backend=backend, config=cfg, include_org_memories=False)
        assert factory.call_count == 0
