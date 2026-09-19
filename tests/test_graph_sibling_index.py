"""Cross-project validation reads sibling stores through a per-process cache.

Regression cover for per-row store cost growing with the number of project
namespaces: every single-row write re-opened every sibling store and re-read
and re-normalised its ``CANDIDATE_LIMIT`` vectors (42 ms/row with 1 sibling,
438 ms/row with 20). The cache must not trade that for staleness: a sibling's
vector replaced in place, rows written by another process, and a deleted and
recreated store must all be seen by the next write, and memory stays bounded.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from time import monotonic

import pytest

import trw_memory
from trw_memory import _graph_sibling_index as index
from trw_memory import graph
from trw_memory._graph_primitives import CandidateVectors
from trw_memory.embeddings._space_gate import select_space_vectors
from trw_memory.embeddings.provenance import EmbeddingSpace, StoredVector, VectorProvenance
from trw_memory.integrations import _backend as backend_module
from trw_memory.integrations._backend import create_backend_from_config, resolve_backend_db_path
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.storage.interface import StorageBackend

pytest.importorskip("sqlite_vec")

DIM = 8
SPACE = EmbeddingSpace("a" * 64, "test-encoder:sibling-index", DIM)
OTHER_SPACE = EmbeddingSpace("b" * 64, "test-encoder:other", DIM)
E0 = [1.0] + [0.0] * (DIM - 1)
E1 = [0.0, 1.0] + [0.0] * (DIM - 2)
E7 = [0.0] * (DIM - 1) + [1.0]


@pytest.fixture(autouse=True)
def _fresh_cache() -> Iterator[None]:
    index.SIBLING_CACHE.clear()
    yield
    index.SIBLING_CACHE.clear()


def _config(tmp_path: Path) -> MemoryConfig:
    return MemoryConfig(storage_backend="sqlite", storage_path=str(tmp_path / "store"), embedding_dim=DIM)


def _put(backend: StorageBackend, entry_id: str, namespace: str, vector: list[float]) -> MemoryEntry:
    entry = MemoryEntry(id=entry_id, content=f"content {entry_id}", namespace=namespace)
    backend.store(entry)
    backend.upsert_vector(
        entry_id, vector, namespace=namespace, provenance=VectorProvenance.for_vector(SPACE, entry.content, vector)
    )
    return entry


def _validate(cfg: MemoryConfig, entry_id: str, vector: list[float]) -> int:
    """Write *entry_id* into ``project:alpha`` and cross-validate it (one single-row batch)."""
    with create_backend_from_config(cfg, "project:alpha") as backend:
        entry = _put(backend, entry_id, "project:alpha", vector)
        return graph.cross_validate_entries([(entry, vector, SPACE)], backend, config=cfg)[entry_id]


def _remote(cfg: MemoryConfig, entry_id: str) -> MemoryEntry:
    with create_backend_from_config(cfg, "project:beta") as remote:
        entry = remote.get(entry_id, namespace="project:beta")
    assert entry is not None
    return entry


@pytest.fixture
def opens(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Every sibling store cross-validation opens (the writer's own store is reused, never opened)."""
    real_open = backend_module.open_namespace_store
    opened: list[Path] = []

    def counting_open(config: MemoryConfig, location: backend_module.NamespaceStoreLocation) -> StorageBackend:
        opened.append(location.db_path)
        return real_open(config, location)

    monkeypatch.setattr(backend_module, "open_namespace_store", counting_open)
    return opened


class TestCrossValidationCache:
    def test_idle_sibling_is_opened_once_across_single_row_writes(self, tmp_path: Path, opens: list[Path]) -> None:
        cfg = _config(tmp_path)
        with create_backend_from_config(cfg, "project:beta") as remote:
            _put(remote, "M-far", "project:beta", E0)

        assert _validate(cfg, "M-1", E1) == 0
        assert len(opens) == 1
        for i in range(2, 6):
            assert _validate(cfg, f"M-{i}", E1) == 0
        assert len(opens) == 1, "an unchanged sibling must be served from the cache, not reopened per write"

    def test_a_match_is_written_back_to_the_sibling(self, tmp_path: Path, opens: list[Path]) -> None:
        cfg = _config(tmp_path)
        with create_backend_from_config(cfg, "project:beta") as remote:
            _put(remote, "M-remote", "project:beta", E7)
        assert _validate(cfg, "M-warm", E1) == 0  # warms the cache

        assert _validate(cfg, "M-hit", E7) == 1

        assert "cross_validated:project_id=alpha" in " ".join(_remote(cfg, "M-remote").outcome_history)
        assert len(opens) == 2  # the warm-up read, then the write-back of the match

    def test_vector_replaced_in_place_is_seen(self, tmp_path: Path) -> None:
        cfg = _config(tmp_path)
        with create_backend_from_config(cfg, "project:beta") as remote:
            _put(remote, "M-remote", "project:beta", E0)
            assert _validate(cfg, "M-1", E7) == 0  # E7 misses E0; beta is now cached
            # Same row, same store, same row count: only the vector changes.
            remote.upsert_vector(
                "M-remote",
                E7,
                namespace="project:beta",
                provenance=VectorProvenance.for_vector(SPACE, "content M-remote", E7),
            )

        assert _validate(cfg, "M-2", E7) == 1

    def test_rows_written_by_another_process_are_seen(self, tmp_path: Path) -> None:
        cfg = _config(tmp_path)
        with create_backend_from_config(cfg, "project:beta") as remote:
            _put(remote, "M-far", "project:beta", E0)
        assert _validate(cfg, "M-1", E7) == 0

        script = (
            "import sys\n"
            "from trw_memory.embeddings.provenance import EmbeddingSpace, VectorProvenance\n"
            "from trw_memory.integrations._backend import create_backend_from_config\n"
            "from trw_memory.models.config import MemoryConfig\n"
            "from trw_memory.models.memory import MemoryEntry\n"
            f"dim = {DIM}\n"
            "vec = [0.0] * (dim - 1) + [1.0]\n"
            f"space = EmbeddingSpace({'a' * 64!r}, {SPACE.encoding!r}, dim)\n"
            "cfg = MemoryConfig(storage_backend='sqlite', storage_path=sys.argv[1], embedding_dim=dim)\n"
            "with create_backend_from_config(cfg, 'project:beta') as b:\n"
            "    b.store(MemoryEntry(id='M-new', content='content M-new', namespace='project:beta'))\n"
            "    b.upsert_vector('M-new', vec, namespace='project:beta',\n"
            "                    provenance=VectorProvenance.for_vector(space, 'content M-new', vec))\n"
        )
        env = {**os.environ, "PYTHONPATH": str(Path(trw_memory.__file__).resolve().parents[1])}
        subprocess.run([sys.executable, "-c", script, cfg.storage_path], check=True, env=env, timeout=120)

        assert _validate(cfg, "M-2", E7) == 1
        assert _remote(cfg, "M-new").cross_validated is True

    def test_deleted_and_recreated_sibling_is_seen(self, tmp_path: Path) -> None:
        cfg = _config(tmp_path)
        with create_backend_from_config(cfg, "project:beta") as remote:
            _put(remote, "M-old", "project:beta", E7)
        assert _validate(cfg, "M-1", E0) == 0
        shutil.rmtree(resolve_backend_db_path(cfg, "project:beta").parent)
        with create_backend_from_config(cfg, "project:beta") as remote:
            _put(remote, "M-new", "project:beta", E0)

        assert _validate(cfg, "M-2", E0) == 1
        assert _remote(cfg, "M-new").cross_validated is True

    def test_entry_older_than_max_age_is_reread(self, tmp_path: Path, opens: list[Path]) -> None:
        cfg = _config(tmp_path)
        with create_backend_from_config(cfg, "project:beta") as remote:
            _put(remote, "M-far", "project:beta", E0)
        _validate(cfg, "M-1", E7)
        index.SIBLING_CACHE.max_age = -1.0  # every entry is now too old
        try:
            _validate(cfg, "M-2", E7)
        finally:
            index.SIBLING_CACHE.max_age = index.MAX_AGE_SECONDS
        assert len(opens) == 2


class TestFileToken:
    def test_changes_on_an_in_place_vector_replacement(self, tmp_path: Path) -> None:
        cfg = _config(tmp_path)
        with create_backend_from_config(cfg, "project:beta") as remote:
            _put(remote, "M-remote", "project:beta", E0)
            db_path = Path(remote._db_path)  # type: ignore[attr-defined]
            before = index.file_token(db_path)
            remote.upsert_vector(
                "M-remote", E7, namespace="project:beta", provenance=VectorProvenance.for_vector(SPACE, "x", E7)
            )
            after = index.file_token(db_path)
        assert before is not None and after is not None
        assert before != after

    def test_missing_database_is_unfingerprintable(self, tmp_path: Path) -> None:
        assert index.file_token(tmp_path / "absent.db") is None


class TestBoundedMemory:
    def _state(self, rows: int) -> tuple[index._StoreState, dict[EmbeddingSpace, CandidateVectors]]:
        vectors = [(f"c{i}", [float(i + 1)] + [1.0] * (DIM - 1)) for i in range(rows)]
        state = index._StoreState(((0, 0, 0, 0), None), monotonic(), ("project:x",))
        return state, {SPACE: CandidateVectors(vectors, compact=True)}

    def test_byte_budget_evicts_least_recently_used(self) -> None:
        cache = index.SiblingCandidateCache(max_bytes=10_000)
        for key in ("a", "b", "c"):
            state, candidates = self._state(100)  # 100 x 8 x 4 bytes (float32) = 3200
            cache.install(key, state)
            cache.add_namespace(key, state, "project:x", candidates)
        cache.lookup("a", state.token)  # "a" is now the most recently used
        state, candidates = self._state(100)
        cache.install("d", state)
        cache.add_namespace("d", state, "project:x", candidates)

        assert cache.nbytes <= cache.max_bytes
        assert cache.lookup("b", state.token) is None  # least recently used went first
        assert cache.lookup("a", state.token) is not None
        assert cache.lookup("d", state.token) is not None

    def test_a_store_larger_than_the_budget_is_not_retained(self) -> None:
        cache = index.SiblingCandidateCache(max_bytes=1_000)
        state, candidates = self._state(100)
        cache.install("big", state)
        cache.add_namespace("big", state, "project:x", candidates)

        assert len(cache) == 0
        assert cache.nbytes == 0
        assert state.by_namespace["project:x"] is candidates  # still usable by the batch that read it

    def test_cached_matrices_are_float32(self) -> None:
        pytest.importorskip("numpy")
        exact = CandidateVectors([("c", E0)])
        compact = CandidateVectors([("c", E0)], compact=True)
        assert compact.nbytes * 2 == exact.nbytes


class TestGroupBySpace:
    def test_equals_select_space_vectors_per_space(self) -> None:
        def record(vector: list[float], space: EmbeddingSpace | None) -> StoredVector:
            proof = VectorProvenance.for_vector(space, "t", vector) if space is not None else None
            return StoredVector(tuple(vector), proof)

        records = {
            "a": record(E0, SPACE),
            "b": record(E7, OTHER_SPACE),
            "c": record(E1, SPACE),
            "d": record(E0, None),
        }
        grouped = index.group_by_space(records)

        assert set(grouped) == {SPACE, OTHER_SPACE}
        for space in (SPACE, OTHER_SPACE):
            expected = CandidateVectors(select_space_vectors(records, space).vectors.items())
            got, want = grouped[space].above(E0, -1.0), expected.above(E0, -1.0)
            assert [candidate_id for candidate_id, _score in got] == [candidate_id for candidate_id, _score in want]
            assert [score for _id, score in got] == pytest.approx([score for _id, score in want], abs=1e-6)


def test_repeated_namespace_install_accounts_only_the_current_payload() -> None:
    candidates = {SPACE: CandidateVectors([("one", E0)], compact=True)}
    larger = {SPACE: CandidateVectors([("one", E0), ("two", E1)], compact=True)}
    size = candidates[SPACE].nbytes
    cache = index.SiblingCandidateCache(max_bytes=3 * size)
    namespace = "project:alpha"
    state = index._StoreState(((1, 2, 3, 4), None), monotonic(), (namespace,))
    cache.install("store", state)
    for payload in (candidates, candidates, larger, candidates):
        cache.add_namespace("store", state, namespace, payload)
        expected = sum(v.nbytes for v in payload.values())
        assert state.nbytes == expected
        assert cache.nbytes == expected
        assert len(cache) == 1
    cache.add_namespace("store", state, namespace, {})
    assert state.nbytes == cache.nbytes == 0
