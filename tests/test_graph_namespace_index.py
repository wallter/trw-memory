"""Graph enrichment scores each entry against its WHOLE namespace through a per-process index.

Regression cover for two defects of the ``CANDIDATE_LIMIT`` most-recent window:
per-row cost grew with the store (every single-row write re-read and
re-normalised up to 500 vectors: 4 -> ~30 ms per row), and a row older than the
newest 500 was never compared, so related older learnings were never linked.
The index must not trade that for staleness -- rows written, deleted, retired,
or re-embedded by another process must all be honoured -- and its memory stays
bounded.
"""

from __future__ import annotations

import math
import random
import sqlite3
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

import trw_memory
from trw_memory import _graph_namespace_index as nsindex
from trw_memory import graph
from trw_memory.embeddings.provenance import EmbeddingSpace, VectorProvenance
from trw_memory.integrations._backend import create_backend_from_config, resolve_backend_db_path
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry, MemoryStatus
from trw_memory.storage.interface import StorageBackend

pytest.importorskip("sqlite_vec")
pytest.importorskip("numpy")

DIM = 8
NS = "project:alpha"
SPACE = EmbeddingSpace("a" * 64, "test-encoder:namespace-index", DIM)
OTHER_SPACE = EmbeddingSpace("b" * 64, "test-encoder:namespace-index-other", DIM)
E0 = [1.0] + [0.0] * (DIM - 1)
E7 = [0.0] * (DIM - 1) + [1.0]


@pytest.fixture(autouse=True)
def _fresh_index() -> Iterator[None]:
    nsindex.NAMESPACE_INDEX.clear()
    yield
    nsindex.NAMESPACE_INDEX.clear()


def _config(tmp_path: Path) -> MemoryConfig:
    return MemoryConfig(storage_backend="sqlite", storage_path=str(tmp_path / "store"), embedding_dim=DIM)


def _clusters(count: int, seed: int) -> list[list[float]]:
    """Vectors in a few tight clusters, so some pairs clear 0.75 and most do not."""
    rng = random.Random(seed)
    centres = [[rng.gauss(0, 1) for _ in range(DIM)] for _ in range(4)]
    return [[c + rng.gauss(0, 0.4) for c in centres[i % 4]] for i in range(count)]


def _put(backend: StorageBackend, entry_id: str, vector: list[float], space: EmbeddingSpace = SPACE) -> MemoryEntry:
    entry = MemoryEntry(id=entry_id, content=f"content {entry_id}", namespace=NS)
    backend.store(entry)
    backend.upsert_vector(
        entry_id, vector, namespace=NS, provenance=VectorProvenance.for_vector(space, entry.content, vector)
    )
    return entry


def _write(backend: StorageBackend, entry_id: str, vector: list[float]) -> int:
    """Store *entry_id* and enrich it as a single-row writer does; returns its similarity edges."""
    entry = _put(backend, entry_id, vector)
    return graph.update_entry_graph(entry, backend, embedding=vector)["similarity_edges"]


def _edges(backend: StorageBackend) -> set[tuple[str, str]]:
    conn: sqlite3.Connection = backend._conn  # type: ignore[attr-defined]
    rows = conn.execute("SELECT source_id, target_id FROM memory_graph_edges WHERE edge_type = 'similarity'")
    return {(str(src), str(dst)) for src, dst in rows}


def _cos(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True)) / (math.hypot(*a) * math.hypot(*b))


def _brute_force(vectors: list[list[float]]) -> set[tuple[str, str]]:
    pairs = set()
    for i, a in enumerate(vectors):
        for j, b in enumerate(vectors):
            if i != j and _cos(a, b) > graph.SIMILARITY_THRESHOLD:
                pairs.add((f"M-{i:03d}", f"M-{j:03d}"))
    return pairs


def _other_process(cfg: MemoryConfig, body: str) -> None:
    """Run *body* in a fresh interpreter with ``backend`` open on *cfg*'s ``project:alpha`` store."""
    script = (
        "from trw_memory.embeddings.provenance import EmbeddingSpace, VectorProvenance\n"
        "from trw_memory.integrations._backend import create_backend_from_config\n"
        "from trw_memory.models.config import MemoryConfig\n"
        "from trw_memory.models.memory import MemoryEntry\n"
        f"SPACE = EmbeddingSpace('a' * 64, {SPACE.encoding!r}, {DIM})\n"
        f"cfg = MemoryConfig(storage_backend='sqlite', storage_path={cfg.storage_path!r}, embedding_dim={DIM})\n"
        f"with create_backend_from_config(cfg, {NS!r}) as backend:\n"
        + "".join(f"    {line}\n" for line in body.strip().splitlines())
    )
    src = str(Path(trw_memory.__file__).resolve().parents[1])
    subprocess.run([sys.executable, "-c", script], check=True, env={"PYTHONPATH": src, "PATH": ""}, timeout=120)


class TestWholeNamespaceEdges:
    def test_single_row_writes_link_exactly_the_brute_force_pairs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(graph, "CANDIDATE_LIMIT", 10)  # the old window would miss most older pairs
        vectors = _clusters(120, seed=7)
        with create_backend_from_config(_config(tmp_path), NS) as backend:
            for i, vector in enumerate(vectors):
                _write(backend, f"M-{i:03d}", vector)
            edges = _edges(backend)

        expected = _brute_force(vectors)
        assert len(expected) > 200, "fixture must produce pairs far apart in write order"
        assert edges == expected

    def test_the_window_fallback_misses_the_older_pairs(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Non-vacuity: without the index the same writes link only pairs inside the recent window."""
        monkeypatch.setattr(graph, "CANDIDATE_LIMIT", 10)
        monkeypatch.setattr("trw_memory._graph_batch.namespace_candidates", lambda *_a, **_k: None)
        vectors = _clusters(120, seed=7)
        with create_backend_from_config(_config(tmp_path), NS) as backend:
            for i, vector in enumerate(vectors):
                _write(backend, f"M-{i:03d}", vector)
            edges = _edges(backend)

        assert edges < _brute_force(vectors)

    def test_other_spaces_and_retired_rows_are_never_candidates(self, tmp_path: Path) -> None:
        with create_backend_from_config(_config(tmp_path), NS) as backend:
            _write(backend, "M-same", E0)
            _put(backend, "M-other-space", E0, space=OTHER_SPACE)
            _write(backend, "M-retired", E0)
            backend.update("M-retired", namespace=NS, status=MemoryStatus.OBSOLETE)
            conn: sqlite3.Connection = backend._conn  # type: ignore[attr-defined]
            conn.execute("DELETE FROM memory_graph_edges")
            conn.commit()

            assert _write(backend, "M-new", E0) == 2
            assert _edges(backend) == {("M-new", "M-same"), ("M-same", "M-new")}


class TestSpaceRows:
    def test_random_puts_replacements_and_removals_score_like_brute_force(self) -> None:
        import numpy as np

        rng = random.Random(1)
        rows = nsindex._SpaceRows(np, DIM)
        live: dict[str, list[float]] = {}
        vectors = _clusters(300, seed=9)
        for step in range(1500):  # enough removals to compact several times
            entry_id = f"M-{rng.randrange(150)}"
            if rng.random() < 0.45:
                rows.remove(entry_id)
                live.pop(entry_id, None)
            else:
                live[entry_id] = vectors[rng.randrange(300)] if step % 97 else [0.0] * DIM
                rows.put(entry_id, live[entry_id])
                if not any(live[entry_id]):
                    live.pop(entry_id)  # a zero vector is never a candidate
            if step % 50 == 0:
                query = vectors[rng.randrange(300)]
                got = dict(rows.above(query, 0.6))
                want = {k: _cos(query, v) for k, v in live.items() if _cos(query, v) > 0.6}
                assert set(got) == set(want)
                assert all(got[k] == pytest.approx(want[k], abs=1e-5) for k in want)
        assert len(rows.ids) < 2 * max(64, len(rows.pos)) + 1, "removed rows must be compacted away"
        assert rows.above([1.0] * (DIM + 1), -1.0) == []  # another dimension is never a candidate


class TestIncrementalReads:
    def test_a_warm_single_row_write_decodes_only_the_changed_rows(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        vectors = _clusters(60, seed=3)
        with create_backend_from_config(_config(tmp_path), NS) as backend:
            for i, vector in enumerate(vectors[:50]):
                _put(backend, f"M-{i:03d}", vector)
            _write(backend, "M-050", vectors[50])  # the first use loads the whole namespace
            decoded: list[int] = []
            real = type(backend).get_vector_records

            def counting(self: StorageBackend, entry_ids: list[str], *, namespace: str) -> object:
                decoded.append(len(entry_ids))
                return real(self, entry_ids, namespace=namespace)

            monkeypatch.setattr(type(backend), "get_vector_records", counting)
            for i in range(51, 60):
                _write(backend, f"M-{i:03d}", vectors[i])

        assert decoded and max(decoded) <= 2, f"a warm write must not re-read the namespace: {decoded}"


class TestStaleness:
    def test_a_vector_committed_after_its_row_is_indexed_by_its_own_enrichment(self, tmp_path: Path) -> None:
        """trw-mcp commits the row and the vector separately; another enrichment can run in between."""
        with create_backend_from_config(_config(tmp_path), NS) as backend:
            late = MemoryEntry(id="M-late", content="content M-late", namespace=NS)
            backend.store(late)  # the row lands, its vector not yet
            _write(backend, "M-between", E0)  # refreshes the index past M-late's row
            backend.upsert_vector(
                "M-late", E7, namespace=NS, provenance=VectorProvenance.for_vector(SPACE, late.content, E7)
            )
            graph.update_entry_graph(late, backend, embedding=E7)

            assert _write(backend, "M-new", E7) == 2
            assert ("M-new", "M-late") in _edges(backend)

    def test_rows_written_by_another_process_are_candidates(self, tmp_path: Path) -> None:
        cfg = _config(tmp_path)
        with create_backend_from_config(cfg, NS) as backend:
            _write(backend, "M-far", E0)  # warms the index
            _other_process(
                cfg,
                """
entry = MemoryEntry(id="M-remote", content="remote", namespace="project:alpha")
backend.store(entry)
vec = [0.0] * 7 + [1.0]
backend.upsert_vector("M-remote", vec, namespace="project:alpha",
                      provenance=VectorProvenance.for_vector(SPACE, entry.content, vec))
""",
            )
            assert _write(backend, "M-new", E7) == 2
            assert ("M-new", "M-remote") in _edges(backend)

    def test_a_row_deleted_by_another_process_gets_no_edge(self, tmp_path: Path) -> None:
        cfg = _config(tmp_path)
        with create_backend_from_config(cfg, NS) as backend:
            _write(backend, "M-doomed", E7)
            _write(backend, "M-other", E0)  # the newest row: deleting M-doomed lowers no maximum
            _other_process(cfg, 'backend.delete("M-doomed", namespace="project:alpha")')

            assert _write(backend, "M-new", E7) == 0
            assert not any("M-doomed" in pair for pair in _edges(backend))

    def test_a_vector_re_embedded_by_another_process_is_seen_by_the_reconcile(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cfg = _config(tmp_path)
        with create_backend_from_config(cfg, NS) as backend:
            _write(backend, "M-moved", E0)
            _write(backend, "M-anchor", [0.0, 1.0] + [0.0] * (DIM - 2))
            _other_process(  # replaces the vector only: no row write, so the change token cannot see it
                cfg,
                """
vec = [0.0] * 7 + [1.0]
backend.upsert_vector("M-moved", vec, namespace="project:alpha",
                      provenance=VectorProvenance.for_vector(SPACE, "content M-moved", vec))
""",
            )
            clock = [nsindex.monotonic()]
            monkeypatch.setattr(nsindex, "monotonic", lambda: clock[0])
            clock[0] += nsindex.RECONCILE_SECONDS + 1

            assert _write(backend, "M-new", E7) == 2
            assert ("M-new", "M-moved") in _edges(backend)

    def test_a_database_file_replaced_under_the_same_path_starts_over(self, tmp_path: Path) -> None:
        cfg = _config(tmp_path)
        with create_backend_from_config(cfg, NS) as backend:
            _write(backend, "M-old", E0)
            db_path = resolve_backend_db_path(cfg, NS)
        for suffix in ("", "-wal", "-shm"):
            db_path.with_name(db_path.name + suffix).unlink(missing_ok=True)
        with create_backend_from_config(cfg, NS) as backend:
            assert _write(backend, "M-new", E0) == 0
            assert _edges(backend) == set()

    def test_a_retired_row_stops_being_a_candidate(self, tmp_path: Path) -> None:
        with create_backend_from_config(_config(tmp_path), NS) as backend:
            _write(backend, "M-retired", E0)
            _write(backend, "M-first", E0)
            backend.update("M-retired", namespace=NS, status=MemoryStatus.OBSOLETE)
            before = _edges(backend)

            assert _write(backend, "M-new", E0) == 2
            assert _edges(backend) - before == {("M-new", "M-first"), ("M-first", "M-new")}


class TestBoundedMemory:
    def test_the_cache_stays_within_its_byte_budget(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        budget = 12_000
        monkeypatch.setattr(nsindex, "NAMESPACE_INDEX", nsindex.NamespaceIndexCache(max_bytes=budget))
        cfg = _config(tmp_path)
        vectors = _clusters(40, seed=5)
        for n in range(4):
            namespace = f"project:ns{n}"
            with create_backend_from_config(cfg, namespace) as backend:
                for i, vector in enumerate(vectors):
                    entry = MemoryEntry(id=f"M-{i}", content=f"c {i}", namespace=namespace)
                    backend.store(entry)
                    proof = VectorProvenance.for_vector(SPACE, entry.content, vector)
                    backend.upsert_vector(entry.id, vector, namespace=namespace, provenance=proof)
                graph.update_entry_graph(entry, backend, embedding=vector)
            assert nsindex.NAMESPACE_INDEX.nbytes <= budget
        assert 0 < len(nsindex.NAMESPACE_INDEX) < 4

    def test_a_namespace_over_the_budget_keeps_the_recent_window(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(nsindex, "NAMESPACE_INDEX", nsindex.NamespaceIndexCache(max_bytes=64))
        monkeypatch.setattr(graph, "CANDIDATE_LIMIT", 3)
        with create_backend_from_config(_config(tmp_path), NS) as backend:
            _write(backend, "M-oldest", E0)
            for i in range(3):
                _write(backend, f"M-{i}", E7)

            assert nsindex.namespace_candidates(backend, NS) is None
            assert nsindex.NAMESPACE_INDEX.nbytes == 0
            assert _write(backend, "M-new", E0) == 0  # M-oldest is outside the 3-row window

    def test_an_index_that_grows_past_the_budget_is_released(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(nsindex, "NAMESPACE_INDEX", nsindex.NamespaceIndexCache(max_bytes=3000))
        with create_backend_from_config(_config(tmp_path), NS) as backend:
            assert _write(backend, "M-first", E0) == 0
            assert nsindex.NAMESPACE_INDEX.nbytes > 0  # small enough to index at first
            for i in range(30):
                _write(backend, f"M-{i:02d}", E7)

            assert nsindex.namespace_candidates(backend, NS) is None
            assert nsindex.NAMESPACE_INDEX.nbytes == 0

    def test_a_namespace_over_the_row_cap_keeps_the_recent_window(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(nsindex, "MAX_INDEX_ROWS", 2)
        with create_backend_from_config(_config(tmp_path), NS) as backend:
            for i in range(3):
                _put(backend, f"M-{i}", E0)

            assert nsindex.namespace_candidates(backend, NS) is None
            assert _write(backend, "M-new", E0) == 6  # the window still links the three recent rows


class TestBackendsWithoutAnIndex:
    def test_in_memory_store_is_indexed_and_forgotten_with_its_backend(self) -> None:
        import gc

        from trw_memory.storage.sqlite_backend import SQLiteBackend

        backend = SQLiteBackend(Path(":memory:"), dim=DIM)
        if not backend.supports_vectors():
            pytest.skip("sqlite-vec did not load for the in-memory store")
        _write(backend, "M-a", E0)
        assert _write(backend, "M-b", E0) == 2
        assert len(nsindex.NAMESPACE_INDEX) == 1
        backend.close()
        del backend
        gc.collect()
        assert len(nsindex.NAMESPACE_INDEX) == 0

    def test_without_numpy_the_window_is_used(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("trw_memory._graph_primitives._numpy", lambda: None)
        with create_backend_from_config(_config(tmp_path), NS) as backend:
            _write(backend, "M-a", E0)
            assert nsindex.namespace_candidates(backend, NS) is None
            assert _write(backend, "M-b", E0) == 2
        assert len(nsindex.NAMESPACE_INDEX) == 0


def test_empty_namespace_states_have_a_count_bound_and_lru_eviction() -> None:
    cache = nsindex.NamespaceIndexCache()
    first = cache.state(("store", "0"), None)
    for i in range(1, 128):
        cache.state(("store", str(i)), None)
    assert cache.state(("store", "0"), None) is first  # refresh the oldest
    cache.state(("store", "new"), None)
    cache.evict()
    assert len(cache) == 128
    assert ("store", "0") in cache._states
    assert ("store", "1") not in cache._states
    assert cache.nbytes == 0
