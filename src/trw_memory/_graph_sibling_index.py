"""Per-process cache of sibling project stores' cross-validation candidates.

Belongs to the cross-project validation cluster (``_graph_cross_project``).

Why a cache: cross-project validation compares every written entry with the
``CANDIDATE_LIMIT`` most recent vectors of every sibling project namespace.
Opening each sibling store, decoding its vectors and their provenance, and
normalising them cost ~20 ms per sibling on EVERY write batch -- and a
single-row writer (``MemoryClient.store``, trw-mcp's ``trw_learn``, the LOCOMO
REST shim) is a batch of one. Per-row cost therefore grew linearly with the
number of project namespaces in the store: 42 ms with 1 sibling, 438 ms with
20 (2026-09-18). Siblings are almost always idle while one namespace is being
written, so their candidate matrices are cached here and a store is opened only
when its file changed or a match must be written back to it.

Staleness: an entry is served only while its store's file token -- ``(dev,
inode, size, mtime_ns)`` of the database AND its ``-wal`` file -- is unchanged,
so a row or vector written by any process (an in-place vector replacement
appends WAL frames), a checkpoint, or a delete-and-recreate invalidates it. The
token is read BEFORE the data, so a write racing a load only makes the entry
look stale. Filesystems with coarse mtimes could hide two same-size writes in
one tick (a WAL restart after a checkpoint), so an entry is also re-read once it
is ``MAX_AGE_SECONDS`` old.

Memory: bounded by ``MAX_CACHED_BYTES`` over the normalised float32 matrices,
evicting least-recently-used stores. A store that alone exceeds the budget is
used for the batch that read it and not retained.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager, ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic

from trw_memory._graph_primitives import CandidateVectors
from trw_memory.embeddings.provenance import EmbeddingSpace, StoredVector
from trw_memory.storage.interface import StorageBackend

MAX_CACHED_BYTES = 64 * 1024 * 1024
MAX_AGE_SECONDS = 30.0

_StatKey = tuple[int, int, int, int]
#: ``(database, -wal)`` stat keys; ``None`` for an absent ``-wal``.
FileToken = tuple[_StatKey, _StatKey | None]


def _stat_key(path: Path) -> _StatKey:
    st = path.stat()
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)


def file_token(db_path: Path) -> FileToken | None:
    """Fingerprint of *db_path*'s on-disk state, or ``None`` when it cannot be read."""
    wal = db_path.with_name(db_path.name + "-wal")
    try:
        db_key = _stat_key(db_path)
    except OSError:  # trw-fail-silent-allow: None means "unfingerprintable" and is never served from cache; the store is read fresh, and opening it reports any real error
        return None
    try:
        wal_key: _StatKey | None = _stat_key(wal)
    except FileNotFoundError:  # trw-fail-silent-allow: no -wal is a real state (checkpointed store)
        wal_key = None
    except OSError:  # trw-fail-silent-allow: None means "unfingerprintable" and is never served from cache
        return None
    return (db_key, wal_key)


def group_by_space(records: Mapping[str, StoredVector]) -> dict[EmbeddingSpace, CandidateVectors]:
    """Candidates per recorded space; vectors without provenance match no space and are dropped.

    Equal, per space, to ``select_space_vectors(records, space)`` in insertion order.
    """
    grouped: dict[EmbeddingSpace, list[tuple[str, tuple[float, ...]]]] = {}
    for entry_id, record in records.items():
        if record.provenance is not None:
            grouped.setdefault(record.provenance.space, []).append((entry_id, record.embedding))
    return {space: CandidateVectors(vectors, compact=True) for space, vectors in grouped.items()}


@dataclass
class _StoreState:
    token: FileToken
    loaded_at: float
    namespaces: tuple[str, ...]
    by_namespace: dict[str, dict[EmbeddingSpace, CandidateVectors]] = field(default_factory=dict)
    nbytes: int = 0


class SiblingCandidateCache:
    """LRU of sibling stores' namespaces and candidates, bounded in bytes and age."""

    def __init__(self, max_bytes: int = MAX_CACHED_BYTES, max_age: float = MAX_AGE_SECONDS) -> None:
        self.max_bytes = max_bytes
        self.max_age = max_age
        self._stores: OrderedDict[str, _StoreState] = OrderedDict()
        self._nbytes = 0
        self._guard = threading.Lock()

    @property
    def nbytes(self) -> int:
        return self._nbytes

    def __len__(self) -> int:
        return len(self._stores)

    def clear(self) -> None:
        with self._guard:
            self._stores.clear()
            self._nbytes = 0

    def lookup(self, key: str, token: FileToken) -> _StoreState | None:
        with self._guard:
            state = self._stores.get(key)
            if state is None:
                return None
            if state.token != token or monotonic() - state.loaded_at > self.max_age:
                self._drop(key)
                return None
            self._stores.move_to_end(key)
            return state

    def install(self, key: str, state: _StoreState) -> None:
        with self._guard:
            if key in self._stores:
                self._drop(key)
            self._stores[key] = state
            self._nbytes += state.nbytes
            self._evict()

    def add_namespace(
        self, key: str, state: _StoreState, namespace: str, candidates: dict[EmbeddingSpace, CandidateVectors]
    ) -> None:
        size = sum(vectors.nbytes for vectors in candidates.values())
        with self._guard:
            previous = state.by_namespace.get(namespace, {})
            delta = size - sum(vectors.nbytes for vectors in previous.values())
            state.by_namespace[namespace] = candidates
            state.nbytes += delta
            if self._stores.get(key) is state:
                self._nbytes += delta
                self._evict()

    def _drop(self, key: str) -> None:
        self._nbytes -= self._stores.pop(key).nbytes

    def _evict(self) -> None:
        while self._nbytes > self.max_bytes and self._stores:
            self._drop(next(iter(self._stores)))


#: The process-wide cache every cross-validation pass shares.
SIBLING_CACHE = SiblingCandidateCache()


class SiblingStoreView:
    """One write batch's view of one sibling store: cached candidates, the store opened only on demand."""

    def __init__(
        self,
        db_path: Path,
        opener: Callable[[], AbstractContextManager[StorageBackend]],
        *,
        cache: SiblingCandidateCache | None = None,
        candidate_limit: int,
    ) -> None:
        self._cache = cache if cache is not None else SIBLING_CACHE
        self._key = str(db_path)
        self._opener = opener
        self._limit = candidate_limit
        self._stack = ExitStack()
        self._backend: StorageBackend | None = None
        self._token = file_token(db_path)  # read BEFORE the data it vouches for

    def __enter__(self) -> SiblingStoreView:
        token = self._token
        state = self._cache.lookup(self._key, token) if token is not None else None
        if state is None:
            try:
                namespaces = tuple(self.backend().list_namespaces())
            except BaseException:
                self._stack.close()
                raise
            state = _StoreState(token or ((0, 0, 0, 0), None), monotonic(), namespaces)
            if token is not None:
                self._cache.install(self._key, state)
        self._state = state
        return self

    def __exit__(self, *_exc: object) -> None:
        self._stack.close()

    @property
    def namespaces(self) -> tuple[str, ...]:
        return self._state.namespaces

    def backend(self) -> StorageBackend:
        """The opened store (opened on first use, closed with the view)."""
        if self._backend is None:
            self._backend = self._stack.enter_context(self._opener())
        return self._backend

    def candidates(self, namespace: str, space: EmbeddingSpace) -> CandidateVectors | None:
        """*namespace*'s recent candidates recorded in *space* (``None`` when it has none)."""
        by_space = self._state.by_namespace.get(namespace)
        if by_space is None:
            by_space = group_by_space(self.backend().recent_vector_records(namespace=namespace, limit=self._limit))
            self._cache.add_namespace(self._key, self._state, namespace, by_space)
        return by_space.get(space)
