"""Corpus, fixture, and retrieval helper tests for the benchmark suite."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from benchmarks._retrieval import rank_entries
from benchmarks.corpus import (
    create_dedup_set,
    create_golden_set,
    generate_corpus,
    generate_query_set,
)
from trw_memory.models.memory import MemoryEntry


class TestCorpusGeneration:
    """Tests for the synthetic corpus generator."""

    def test_generate_corpus_exact_size(self) -> None:
        """generate_corpus(100) produces exactly 100 entries."""
        corpus = generate_corpus(100)
        assert len(corpus) == 100

    def test_generate_corpus_deterministic(self) -> None:
        """Same seed produces identical corpus on repeated calls."""
        c1 = generate_corpus(50, seed=42)
        c2 = generate_corpus(50, seed=42)
        assert len(c1) == len(c2)
        for a, b in zip(c1, c2):
            assert a.id == b.id
            assert a.content == b.content
            assert a.tags == b.tags
            assert a.importance == b.importance

    def test_generate_corpus_different_seeds(self) -> None:
        """Different seeds produce different corpora."""
        c1 = generate_corpus(50, seed=1)
        c2 = generate_corpus(50, seed=2)
        ids_1 = {e.id for e in c1}
        ids_2 = {e.id for e in c2}
        assert ids_1 != ids_2

    def test_generate_corpus_valid_fields(self) -> None:
        """Generated entries have valid MemoryEntry fields."""
        corpus = generate_corpus(20)
        for entry in corpus:
            assert entry.id.startswith("M-")
            assert len(entry.id) == 10
            assert len(entry.content) > 10
            assert len(entry.detail) > 0
            assert len(entry.tags) >= 2
            assert 0.1 <= entry.importance <= 1.0
            assert entry.namespace == "benchmark"
            assert entry.source == "agent"
            assert entry.recurrence >= 1

    def test_generate_corpus_large(self) -> None:
        """Corpus generation works at 1000 entries."""
        corpus = generate_corpus(1000)
        assert len(corpus) == 1000
        ids = [e.id for e in corpus]
        assert len(set(ids)) == 1000

    def test_generate_query_set_count(self) -> None:
        """generate_query_set produces the requested number of queries."""
        corpus = generate_corpus(100)
        queries = generate_query_set(corpus, num_queries=30)
        assert len(queries) == 30

    def test_generate_query_set_structure(self) -> None:
        """Each query has the expected keys."""
        corpus = generate_corpus(100)
        queries = generate_query_set(corpus, num_queries=10)
        for q in queries:
            assert "query" in q
            assert "expected_ids" in q
            assert "expected_tags" in q
            assert isinstance(q["query"], str)
            assert isinstance(q["expected_ids"], list)
            assert isinstance(q["expected_tags"], list)

    def test_generate_query_set_deterministic(self) -> None:
        """Same corpus and seed produce identical query sets."""
        corpus = generate_corpus(100, seed=7)
        q1 = generate_query_set(corpus, num_queries=10, seed=7)
        q2 = generate_query_set(corpus, num_queries=10, seed=7)
        assert len(q1) == len(q2)
        for a, b in zip(q1, q2):
            assert a["query"] == b["query"]


class TestFixtures:
    """Tests for golden set and dedup set fixture generation and loading."""

    def test_golden_set_generation(self, tmp_path: Path) -> None:
        """create_golden_set writes 50 entries to JSON."""
        out = tmp_path / "golden.json"
        create_golden_set(out)
        assert out.exists()
        data = json.loads(out.read_text())
        assert "entries" in data
        assert len(data["entries"]) == 50

    def test_golden_set_entry_structure(self, tmp_path: Path) -> None:
        """Each golden entry has required fields."""
        out = tmp_path / "golden.json"
        create_golden_set(out)
        data = json.loads(out.read_text())
        for entry in data["entries"]:
            assert "id" in entry
            assert "content" in entry
            assert "tags" in entry
            assert "importance" in entry
            assert "queries" in entry
            assert entry["id"].startswith("golden-")
            assert len(entry["queries"]) >= 2
            for q in entry["queries"]:
                assert "query" in q
                assert "relevant" in q

    def test_golden_set_unique_ids(self, tmp_path: Path) -> None:
        """All golden entry IDs are unique."""
        out = tmp_path / "golden.json"
        create_golden_set(out)
        data = json.loads(out.read_text())
        ids = [e["id"] for e in data["entries"]]
        assert len(set(ids)) == 50

    def test_dedup_set_generation(self, tmp_path: Path) -> None:
        """create_dedup_set writes 30 pairs to JSON."""
        out = tmp_path / "dedup.json"
        create_dedup_set(out)
        assert out.exists()
        data = json.loads(out.read_text())
        assert "pairs" in data
        assert len(data["pairs"]) == 30

    def test_dedup_set_pair_structure(self, tmp_path: Path) -> None:
        """Each dedup pair has required fields."""
        out = tmp_path / "dedup.json"
        create_dedup_set(out)
        data = json.loads(out.read_text())
        for pair in data["pairs"]:
            assert "id" in pair
            assert "entry_a" in pair
            assert "entry_b" in pair
            assert "expected_duplicate" in pair
            assert "content" in pair["entry_a"]
            assert "tags" in pair["entry_a"]
            assert "content" in pair["entry_b"]
            assert "tags" in pair["entry_b"]
            assert isinstance(pair["expected_duplicate"], bool)

    def test_dedup_set_has_both_labels(self, tmp_path: Path) -> None:
        """Dedup set contains both true and false duplicate pairs."""
        out = tmp_path / "dedup.json"
        create_dedup_set(out)
        data = json.loads(out.read_text())
        true_count = sum(1 for p in data["pairs"] if p["expected_duplicate"])
        false_count = sum(1 for p in data["pairs"] if not p["expected_duplicate"])
        assert true_count > 0, "Should have true duplicate pairs"
        assert false_count > 0, "Should have false duplicate pairs"
        assert true_count + false_count == 30


class TestBenchmarkRetrieval:
    """Tests for benchmark ranking helpers."""

    def test_rank_entries_matches_expected_golden_entry(self, tmp_path: Path) -> None:
        """Fallback benchmark ranking surfaces the intended golden entry."""
        out = tmp_path / "golden.json"
        create_golden_set(out)
        data = json.loads(out.read_text())
        now = datetime.now(timezone.utc)

        entries = [
            MemoryEntry(
                id=str(entry["id"]),
                content=str(entry["content"]),
                detail="",
                tags=[str(tag) for tag in entry["tags"]],
                importance=float(entry["importance"]),
                namespace="golden",
                created_at=now,
                updated_at=now,
                source="agent",
            )
            for entry in data["entries"]
        ]

        results = rank_entries("pydantic strict mode", entries, top_k=3)
        assert results
        assert "golden-001" in {entry.id for entry in results}
