"""Batched graph enrichment: one pass per write batch, same edges as per-entry enrichment.

Regression cover for the ingest slowdown where every ``bulk_store`` row ran its
own enrichment -- a thread, a backend, a ``CANDIDATE_LIMIT`` candidate decode,
and a walk of every sibling project store with one pure-Python cosine per pair
-- so ingest time grew with the number of projects in the store.
"""

from __future__ import annotations

import math
import random
import sqlite3
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from trw_memory import graph
from trw_memory._graph_primitives import CandidateVectors, _safe_cosine_similarity
from trw_memory.embeddings.provenance import EmbeddingSpace, VectorProvenance
from trw_memory.integrations import _backend as backend_module
from trw_memory.integrations._backend import create_backend_from_config
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.storage.interface import StorageBackend

DIM = 8
SPACE = EmbeddingSpace("a" * 64, "test-encoder:graph-batch", DIM)
OTHER_SPACE = EmbeddingSpace("b" * 64, "test-encoder:other", DIM)


def _cluster_vectors(count: int, seed: int) -> list[list[float]]:
    """Vectors in a few tight clusters, so some pairs clear 0.75 and most do not."""
    rng = random.Random(seed)
    centres = [[rng.gauss(0, 1) for _ in range(DIM)] for _ in range(3)]
    return [[c + rng.gauss(0, 0.35) for c in centres[i % 3]] for i in range(count)]


def _config(tmp_path: Path) -> MemoryConfig:
    return MemoryConfig(storage_backend="sqlite", storage_path=str(tmp_path / "store"), embedding_dim=DIM)


def _put(
    backend: StorageBackend,
    entry_id: str,
    namespace: str,
    vector: list[float],
    space: EmbeddingSpace | None = SPACE,
) -> MemoryEntry:
    entry = MemoryEntry(id=entry_id, content=f"content {entry_id}", namespace=namespace)
    backend.store(entry)
    proof = {"provenance": VectorProvenance.for_vector(space, entry.content, vector)} if space is not None else {}
    backend.upsert_vector(entry_id, vector, namespace=namespace, **proof)
    return entry


def _edges(backend: StorageBackend) -> dict[tuple[str, str], float]:
    conn: sqlite3.Connection = backend._conn  # type: ignore[attr-defined]
    rows = conn.execute(
        "SELECT source_id, target_id, weight FROM memory_graph_edges WHERE edge_type = 'similarity'"
    ).fetchall()
    return {(str(src), str(dst)): float(weight) for src, dst, weight in rows}


def _cos(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True)) / (math.hypot(*a) * math.hypot(*b))


class TestCandidateVectors:
    @pytest.mark.parametrize("use_numpy", [True, False])
    def test_scores_match_the_pairwise_cosine(self, use_numpy: bool, monkeypatch: pytest.MonkeyPatch) -> None:
        if use_numpy:
            pytest.importorskip("numpy")
        else:
            monkeypatch.setattr("trw_memory._graph_primitives._numpy", lambda: None)
        vectors = _cluster_vectors(30, seed=3)
        candidates = CandidateVectors((f"c{i}", vec) for i, vec in enumerate(vectors))
        query = vectors[0]

        scored = dict(candidates.above(query, -1.0))

        assert len(scored) == len(vectors)
        for i, vec in enumerate(vectors):
            assert scored[f"c{i}"] == pytest.approx(_safe_cosine_similarity(query, vec), abs=1e-9)

    @pytest.mark.parametrize("use_numpy", [True, False])
    def test_threshold_zero_vectors_and_other_dimensions(
        self, use_numpy: bool, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        if use_numpy:
            pytest.importorskip("numpy")
        else:
            monkeypatch.setattr("trw_memory._graph_primitives._numpy", lambda: None)
        candidates = CandidateVectors(
            [("same", [1.0, 0.0, 0.0]), ("orthogonal", [0.0, 1.0, 0.0]), ("zero", [0.0, 0.0, 0.0]), ("2d", [1.0, 0.0])]
        )

        assert candidates.above([2.0, 0.0, 0.0], 0.5) == [("same", pytest.approx(1.0))]
        assert candidates.above([0.0, 0.0, 0.0], -1.0) == []  # a zero query scores nothing
        assert candidates.above([1.0, 0.0, 0.0, 0.0], -1.0) == []  # no candidate of that dimension
        assert len(candidates) == 3  # the zero vector is dropped


class TestUpdateEntriesGraph:
    def test_batch_writes_exactly_the_edges_brute_force_predicts(self, tmp_path: Path) -> None:
        ns = "project:alpha"
        vectors = _cluster_vectors(24, seed=11)
        with create_backend_from_config(_config(tmp_path), ns) as backend:
            entries = [_put(backend, f"M-{i:02d}", ns, vec) for i, vec in enumerate(vectors)]
            batch = list(range(12, 24))  # the "just written" half

            counts = graph.update_entries_graph([(entries[i], vectors[i]) for i in batch], backend)

            expected = {}
            for i in batch:
                for j in range(len(vectors)):
                    if i != j and (sim := _cos(vectors[i], vectors[j])) > graph.SIMILARITY_THRESHOLD:
                        expected[(f"M-{i:02d}", f"M-{j:02d}")] = sim
                        expected[(f"M-{j:02d}", f"M-{i:02d}")] = sim
            edges = _edges(backend)

        assert expected, "fixture must produce at least one edge"
        assert set(edges) == set(expected)
        for key, sim in expected.items():
            assert edges[key] == pytest.approx(round(sim, 4), abs=1e-4)
        assert counts["similarity_edges"] >= len(expected)

    def test_candidates_outside_the_entry_space_get_no_edge(self, tmp_path: Path) -> None:
        ns = "project:alpha"
        vec = [1.0] + [0.0] * (DIM - 1)
        with create_backend_from_config(_config(tmp_path), ns) as backend:
            new = _put(backend, "M-new", ns, vec)
            _put(backend, "M-same", ns, vec)
            _put(backend, "M-other-space", ns, vec, space=OTHER_SPACE)
            _put(backend, "M-no-proof", ns, vec, space=None)

            graph.update_entries_graph([(new, vec)], backend)
            edges = _edges(backend)

        assert set(edges) == {("M-new", "M-same"), ("M-same", "M-new")}

    def test_single_entry_form_matches_the_batch(self, tmp_path: Path) -> None:
        ns = "project:alpha"
        vectors = _cluster_vectors(10, seed=5)
        cfg_a = _config(tmp_path / "a")
        cfg_b = _config(tmp_path / "b")
        with create_backend_from_config(cfg_a, ns) as one, create_backend_from_config(cfg_b, ns) as many:
            entries = [_put(one, f"M-{i}", ns, v) for i, v in enumerate(vectors)]
            for i, v in enumerate(vectors):
                _put(many, f"M-{i}", ns, v)
            for entry, vec in zip(entries, vectors, strict=True):
                graph.update_entry_graph(entry, one, embedding=vec)
            graph.update_entries_graph(list(zip(entries, vectors, strict=True)), many)

            assert _edges(one) == _edges(many)


class TestCrossValidateEntries:
    def test_one_sibling_walk_per_batch_and_matches_still_validate(self, tmp_path: Path) -> None:
        cfg = _config(tmp_path)
        match = [0.0] * (DIM - 1) + [1.0]
        vectors = [[1.0] + [0.0] * (DIM - 1), [0.0, 1.0] + [0.0] * (DIM - 2), match]
        real_locations = backend_module.namespace_store_locations
        walks: list[int] = []

        def counting_locations(config: MemoryConfig) -> list[backend_module.NamespaceStoreLocation]:
            walks.append(1)
            return real_locations(config)

        with create_backend_from_config(cfg, "project:beta") as remote:
            _put(remote, "M-remote", "project:beta", match)
            _put(remote, "M-remote-far", "project:beta", [0.0, 0.0, 1.0] + [0.0] * (DIM - 3))
        with create_backend_from_config(cfg, "project:alpha") as backend:
            entries = [_put(backend, f"M-{i}", "project:alpha", v) for i, v in enumerate(vectors)]
            with patch.object(backend_module, "namespace_store_locations", counting_locations):
                counts = graph.update_entries_graph(list(zip(entries, vectors, strict=True)), backend, config=cfg)
            local = backend.get("M-2", namespace="project:alpha")
            untouched = backend.get("M-0", namespace="project:alpha")
        with create_backend_from_config(cfg, "project:beta") as remote:
            remote_entry = remote.get("M-remote", namespace="project:beta")
            far = remote.get("M-remote-far", namespace="project:beta")

        assert walks == [1]
        assert counts["cross_validated_projects"] == 1
        assert local is not None and local.cross_validated is True
        assert untouched is not None and untouched.cross_validated is False
        assert remote_entry is not None and remote_entry.cross_validated is True
        assert any("cross_validated:project_id=alpha" in item for item in remote_entry.outcome_history)
        assert far is not None and far.cross_validated is False


class TestBulkStoreSchedulesOnePass:
    async def test_bulk_store_dispatches_one_enrichment_for_the_whole_batch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from trw_memory._client_bulk_store import BulkStoreRequest
        from trw_memory.client import MemoryClient

        monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "sqlite")
        monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "store"))
        monkeypatch.setattr("trw_memory.client.MemoryClient._get_embedder", lambda self: None)
        calls: list[list[str]] = []

        def spy(items: list[tuple[MemoryEntry, Any]], backend: StorageBackend, **_kw: object) -> dict[str, int]:
            calls.append([entry.id for entry, _vec in items])
            return {}

        client = MemoryClient(namespace="project:bulk", mode="local")
        with patch.object(graph, "update_entries_graph", spy):
            summary = await client.bulk_store([BulkStoreRequest(content=f"row {i}") for i in range(7)])
            graph.wait_for_graph_updates(timeout=30)
        await client.close()

        assert summary.stored == 7
        assert len(calls) == 1
        assert sorted(calls[0]) == sorted(item.memory_id for item in summary.items)


class TestRecentVectorRecords:
    def test_sqlite_selects_exactly_the_list_entries_candidate_set(self, tmp_path: Path) -> None:
        from datetime import datetime, timedelta, timezone

        from trw_memory.models.memory import MemoryStatus

        ns = "project:alpha"
        vectors = _cluster_vectors(12, seed=2)
        t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
        with create_backend_from_config(_config(tmp_path), ns) as backend:
            for i, vec in enumerate(vectors):
                _put(backend, f"M-{i:02d}", ns, vec)
                backend.update(f"M-{i:02d}", namespace=ns, updated_at=t0 + timedelta(minutes=(i * 7) % 12))
            backend.update("M-03", namespace=ns, status=MemoryStatus.OBSOLETE)
            _put(backend, "M-other", "project:beta", vectors[0])

            fast = backend.recent_vector_records(namespace=ns, limit=5)
            default = StorageBackend.recent_vector_records(backend, namespace=ns, limit=5)

        assert list(fast) == list(default)
        assert len(fast) == 5
        assert "M-03" not in fast
        assert "M-other" not in fast
        assert all(record.provenance is not None for record in fast.values())
