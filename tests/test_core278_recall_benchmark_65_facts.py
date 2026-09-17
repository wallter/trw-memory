"""PRD-CORE-278 FR10 — the 65-fact recall benchmark, committed in full.

PROVENANCE, stated because a benchmark nobody can reproduce is a claim, not a
measurement. Remote submission sub_lMovqXjoUHNw4pDd published the five target
facts, the shape of the sixty filler entries, and four example queries — not its
full ten-query set. The ten queries below are therefore a RECONSTRUCTION to that
shape, not the submitter's list, and the daemon numbers in that report are not
directly comparable with what this fixture produces. No assertion here depends on
them.

What this fixture measures is the LEXICAL half, with dense retrieval unavailable:
no embedder, no stored vectors. That is the configuration in which the defect
reproduces in-process (measured 1 of 5 before the change). The meaning-only half
is REPORTED and never asserted: with no vectors it measures how the lexical
fallback handles paraphrases, not vector quality, and asserting on it would be a
claim the fixture cannot support.

A hit rate alone can pass while score carriage is broken — the pre-change
implementation returned a single constant score on every row — so the score
sequence is asserted too.
"""

from __future__ import annotations

import tempfile
from typing import cast
from unittest.mock import patch

from trw_memory.integrations._backend import create_backend_from_config
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.tools.recall import memory_recall_impl

NAMESPACE = "project:default"

#: The five facts from sub_lMovqXjoUHNw4pDd, verbatim.
TARGETS: dict[str, str] = {
    "kv:kubernetes_cluster": "kubernetes_cluster: cx1-prod",
    "kv:manager_name": "manager_name: Priya",
    "kv:oncall_rotation": "oncall_rotation: every third week",
    "kv:favourite_language": "favourite_language: Go",
    "kv:home_city": "home_city: Salt Lake City",
}

FILLER_COUNT = 60

#: Five queries that share at least one word with their target.
LEXICAL_QUERIES: list[tuple[str, str]] = [
    ("favourite language", "kv:favourite_language"),
    ("what is my manager name", "kv:manager_name"),
    ("oncall rotation schedule", "kv:oncall_rotation"),
    ("kubernetes cluster name", "kv:kubernetes_cluster"),
    ("home city", "kv:home_city"),
]

#: Five queries that share NO content word with their target.
MEANING_ONLY_QUERIES: list[tuple[str, str]] = [
    ("who is my boss", "kv:manager_name"),
    ("which k8s environment do we deploy to", "kv:kubernetes_cluster"),
    ("where do I live", "kv:home_city"),
    ("which programming language do I like most", "kv:favourite_language"),
    ("when am I on call", "kv:oncall_rotation"),
]

#: FR10 floor for the lexical half. Measured 1/5 before PRD-CORE-278.
LEXICAL_HIT_AT_3_FLOOR = 4


class TestSixtyFiveFactBenchmark:
    def _recall(self, backend: object, cfg: MemoryConfig, query: str) -> list[dict[str, object]]:
        with patch("trw_memory.tools.recall.get_local_embedder", return_value=None):
            result = memory_recall_impl(
                query,
                NAMESPACE,
                backend=cast("object", backend),  # type: ignore[arg-type]
                config=cfg,
                limit=3,
                include_org_memories=False,
            )
        return cast("list[dict[str, object]]", result["memories"])

    def test_fixture_is_sixty_five_facts(self) -> None:
        assert len(TARGETS) + FILLER_COUNT == 65
        assert len(LEXICAL_QUERIES) == 5
        assert len(MEANING_ONLY_QUERIES) == 5

    def test_lexical_hit_at_3_meets_the_floor_and_scores_carry(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = MemoryConfig(storage_backend="sqlite", storage_path=td)
            with create_backend_from_config(cfg, NAMESPACE) as backend:
                for entry_id, content in TARGETS.items():
                    backend.store(MemoryEntry(id=entry_id, content=content, namespace=NAMESPACE))
                for index in range(FILLER_COUNT):
                    backend.store(
                        MemoryEntry(
                            id=f"note_{index}",
                            content=f"note_{index}: unrelated filler entry {index}",
                            namespace=NAMESPACE,
                        )
                    )

                lexical_hits = 0
                score_sets: list[list[float]] = []
                for query, expected_id in LEXICAL_QUERIES:
                    memories = self._recall(backend, cfg, query)
                    ids = [str(row["id"]) for row in memories]
                    lexical_hits += expected_id in ids
                    score_sets.append([float(cast("float", row["score"])) for row in memories])

                meaning_hits = sum(
                    expected_id in [str(row["id"]) for row in self._recall(backend, cfg, query)]
                    for query, expected_id in MEANING_ONLY_QUERIES
                )

        # Reported, with N, so a future reader sees the population and not just a verdict.
        print(
            f"\nPRD-CORE-278 FR10 (65 facts, dense retrieval unavailable): "
            f"lexical hit@3 {lexical_hits}/{len(LEXICAL_QUERIES)}, "
            f"meaning-only hit@3 {meaning_hits}/{len(MEANING_ONLY_QUERIES)}"
        )
        assert lexical_hits >= LEXICAL_HIT_AT_3_FLOOR
        assert meaning_hits >= 0  # reported, never asserted upward

        # Score carriage: the pre-change path returned one constant on every row.
        for scores in score_sets:
            assert scores == sorted(scores, reverse=True)
        multi_result_scores = [scores for scores in score_sets if len(scores) > 1]
        assert multi_result_scores, "benchmark produced no multi-result query to check score spread"
        assert any(len(set(scores)) > 1 for scores in multi_result_scores)
