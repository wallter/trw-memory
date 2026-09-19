"""Per-process similarity index over the WRITTEN namespace's own vectors.

Belongs to graph enrichment (``_graph_batch``); the sibling-store counterpart
for cross-project validation is ``_graph_sibling_index``.

Why an index: every write batch compared its entries with the namespace's
``CANDIDATE_LIMIT`` (500) most recently updated ACTIVE rows, decoding and
normalising those vectors from scratch each time. A single-row writer
(``MemoryClient.store``, trw-mcp's ``trw_learn``, which enriches
synchronously) is a batch of one, so enrichment rose from 4 to ~30 ms per row
over the first 500 rows (2026-09-18) -- and rows older than the newest 500
were never compared at all, so at TRW scale an older related learning was
never linked. Here each namespace's ACTIVE vectors are held once per process
as float32 unit rows per embedding space, updated incrementally, and every
entry is scored against the WHOLE namespace with one matrix product.

Freshness, in order of cost:

- The namespace change token (``StorageBackend.namespace_change_token``, two
  index seeks) is read BEFORE the data; unchanged means nothing to do. A
  changed token is caught up through the change feed: only the rows inserted
  or stamped since are re-read, whichever process wrote them.
- A full reconcile compares every ACTIVE row's stored vector proof
  (``vec_index.provenance_json``, which names the vector's sha256 and is
  rewritten by every vector upsert) with the proof the index was built from,
  and re-reads only the rows that differ. It runs on the first use, when the
  feed overflows or this process deleted rows, when the database file was
  replaced, and at least every ``RECONCILE_SECONDS``. That bounds what the
  token cannot see: a vector replaced without a row write (re-embedding) and a
  delete by another process.
- Every candidate above the threshold is re-checked against the live row
  before an edge is written, so a delete the index has not seen yet never
  produces an edge.

Memory: bounded by ``MAX_INDEX_BYTES`` over all cached namespaces, least
recently used evicted first. A namespace with more than ``MAX_INDEX_ROWS``
vectors, or whose index alone exceeds the budget, is not indexed: it keeps
the ``CANDIDATE_LIMIT`` most-recent window. So is any backend without a
change token (non-SQLite) or without numpy.
"""

from __future__ import annotations

import threading
import weakref
from collections import OrderedDict
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic
from typing import Any

from trw_memory import _graph_primitives
from trw_memory.embeddings.provenance import EmbeddingSpace
from trw_memory.models.memory import MemoryStatus
from trw_memory.storage._sql_utils import iter_bind_chunks
from trw_memory.storage.interface import NamespaceChangeToken, StorageBackend

MAX_INDEX_BYTES = 128 * 1024 * 1024
MAX_INDEX_ROWS = 100_000
MAX_CACHED_NAMESPACES = 128
RECONCILE_SECONDS = 30.0
FEED_LIMIT = 256

_ACTIVE = MemoryStatus.ACTIVE.value
_PROOFS_SQL = (
    "SELECT vi.entry_id, vi.provenance_json FROM vec_index vi "
    "JOIN memories m ON m.namespace = vi.namespace AND m.id = vi.entry_id "
    "WHERE vi.namespace = ? AND m.status = ? AND vi.provenance_json IS NOT NULL"
)
_LIVE_SQL = "SELECT id FROM memories WHERE namespace = ? AND status = ? AND id IN "


class _SpaceRows:
    """Unit float32 rows of one embedding space; removed rows are zeroed and compacted lazily."""

    def __init__(self, np: Any, dim: int) -> None:
        self._np = np
        self.dim = dim
        self.ids: list[str | None] = []
        self.pos: dict[str, int] = {}
        self.rows: Any = np.zeros((16, dim), dtype=np.float32)

    @property
    def nbytes(self) -> int:
        return int(self.rows.nbytes) + 120 * len(self.ids)  # the matrix plus the id bookkeeping

    def put(self, entry_id: str, vector: Sequence[float]) -> None:
        np = self._np
        row = np.asarray(vector, dtype=np.float64)
        norm = float(np.sqrt(row @ row))
        if norm == 0.0:  # a zero vector scores 0.0 against everything; never a candidate
            self.remove(entry_id)
            return
        index = self.pos.get(entry_id)
        if index is None:
            index = len(self.ids)
            if index == len(self.rows):
                grown = np.zeros((2 * len(self.rows), self.dim), dtype=np.float32)
                grown[:index] = self.rows
                self.rows = grown
            self.ids.append(entry_id)
            self.pos[entry_id] = index
        self.rows[index] = row / norm

    def remove(self, entry_id: str) -> None:
        index = self.pos.pop(entry_id, None)
        if index is None:
            return
        self.ids[index] = None
        self.rows[index] = 0.0
        if len(self.ids) - len(self.pos) > max(64, len(self.pos)):
            live = [i for i, candidate in enumerate(self.ids) if candidate is not None]
            self.rows = self.rows[live] if live else self.rows[:0]
            self.rows = self._np.concatenate([self.rows, self._np.zeros((16, self.dim), dtype=self._np.float32)])
            self.ids = [self.ids[i] for i in live]
            self.pos = {candidate: i for i, candidate in enumerate(self.ids) if candidate is not None}

    def above(self, query: Sequence[float], threshold: float) -> list[tuple[str, float]]:
        np = self._np
        if len(query) != self.dim or not self.pos:
            return []
        unit = np.asarray(query, dtype=np.float64)
        norm = float(np.sqrt(unit @ unit))
        if norm == 0.0:
            return []
        scores = self.rows[: len(self.ids)] @ (unit / norm).astype(np.float32)
        hits = []
        for index in np.flatnonzero(scores > threshold).tolist():
            candidate = self.ids[index]
            if candidate is not None:
                hits.append((candidate, float(scores[index])))
        return hits


@dataclass
class _IndexState:
    file_id: tuple[int, int] | None
    token: NamespaceChangeToken | None = None
    reconciled_at: float = float("-inf")
    proofs: dict[str, int] = field(default_factory=dict)  # entry id -> hash of the proof it was indexed from
    spaces: dict[EmbeddingSpace, _SpaceRows] = field(default_factory=dict)
    where: dict[str, EmbeddingSpace] = field(default_factory=dict)
    oversize: bool = False
    nbytes: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def put(self, entry_id: str, vector: Sequence[float], space: EmbeddingSpace, np: Any) -> None:
        previous = self.where.get(entry_id)
        if previous is not None and previous != space:
            self.spaces[previous].remove(entry_id)
        rows = self.spaces.get(space)
        if rows is None:
            rows = self.spaces[space] = _SpaceRows(np, space.dimensions)
        if len(vector) != rows.dim:
            self.drop(entry_id)
            return
        rows.put(entry_id, vector)
        self.where[entry_id] = space

    def drop(self, entry_id: str) -> None:
        space = self.where.pop(entry_id, None)
        if space is not None:
            self.spaces[space].remove(entry_id)

    def clear(self) -> None:
        self.proofs.clear()
        self.spaces.clear()
        self.where.clear()


def _file_id(db_path: object) -> tuple[int, int] | None:
    try:
        st = Path(str(db_path)).stat()
    except (
        OSError,
        ValueError,
    ):  # trw-fail-silent-allow: in-memory and unstattable stores have no file identity; the change token and reconcile still guard them
        return None
    return (st.st_dev, st.st_ino)


class NamespaceIndexCache:
    """LRU of per-namespace indexes, bounded in bytes and entry count."""

    def __init__(self, max_bytes: int = MAX_INDEX_BYTES) -> None:
        self.max_bytes = max_bytes
        self._states: OrderedDict[tuple[str, str], _IndexState] = OrderedDict()
        self._guard = threading.Lock()

    @property
    def nbytes(self) -> int:
        with self._guard:
            return sum(state.nbytes for state in self._states.values())

    def __len__(self) -> int:
        return len(self._states)

    def clear(self) -> None:
        with self._guard:
            self._states.clear()

    def discard(self, key: tuple[str, str]) -> None:
        with self._guard:
            self._states.pop(key, None)

    def state(self, key: tuple[str, str], file_id: tuple[int, int] | None) -> _IndexState:
        with self._guard:
            state = self._states.get(key)
            if state is None or state.file_id != file_id:  # a replaced database file starts over
                state = self._states[key] = _IndexState(file_id)
            self._states.move_to_end(key)
            return state

    def evict(self) -> None:
        """Drop least-recently-used indexes until the cached total fits the budget."""
        with self._guard:
            total = sum(state.nbytes for state in self._states.values())
            while len(self._states) > 1 and (total > self.max_bytes or len(self._states) > MAX_CACHED_NAMESPACES):
                _key, removed = self._states.popitem(last=False)
                total -= removed.nbytes


#: The process-wide cache every graph enrichment pass shares.
NAMESPACE_INDEX = NamespaceIndexCache()


class NamespaceCandidates:
    """One batch's handle on a namespace's index: score against every indexed row of a space."""

    def __init__(self, backend: StorageBackend, namespace: str, state: _IndexState, np: Any) -> None:
        self._backend = backend
        self._namespace = namespace
        self._state = state
        self._np = np

    def add(self, entry_id: str, embedding: Sequence[float], space: EmbeddingSpace | None) -> None:
        """Index a row this batch wrote (its vector is proven to be in *space*)."""
        if space is not None:
            with self._state.lock:
                self._state.put(entry_id, embedding, space, self._np)

    def in_space(self, space: EmbeddingSpace | None) -> SpaceCandidates:
        return SpaceCandidates(self, space)

    def score(self, space: EmbeddingSpace | None, query: Sequence[float], threshold: float) -> list[tuple[str, float]]:
        if space is None:
            return []
        with self._state.lock:
            rows = self._state.spaces.get(space)
            hits = rows.above(query, threshold) if rows is not None else []
        return self._live(hits)

    def _live(self, hits: list[tuple[str, float]]) -> list[tuple[str, float]]:
        """*hits* whose row is still ACTIVE (a delete the index has not seen yet makes no edge)."""
        if not hits:
            return hits
        conn: Any = self._backend._conn  # type: ignore[attr-defined]
        live: set[str] = set()
        with self._backend._lock:  # type: ignore[attr-defined]
            for chunk in iter_bind_chunks([entry_id for entry_id, _score in hits], reserved_bindings=2):
                sql = _LIVE_SQL + "(" + ", ".join("?" for _ in chunk) + ")"
                live.update(str(row[0]) for row in conn.execute(sql, [self._namespace, _ACTIVE, *chunk]))
        return [hit for hit in hits if hit[0] in live]


class SpaceCandidates:
    """The ``CandidateVectors.above`` interface over one space of a :class:`NamespaceCandidates`."""

    def __init__(self, owner: NamespaceCandidates, space: EmbeddingSpace | None) -> None:
        self._owner = owner
        self._space = space

    def above(self, query: Sequence[float], threshold: float) -> list[tuple[str, float]]:
        return self._owner.score(self._space, query, threshold)


def namespace_candidates(
    backend: StorageBackend, namespace: str, *, cache: NamespaceIndexCache | None = None
) -> NamespaceCandidates | None:
    """The up-to-date index of *namespace*, or ``None`` when the recent window must be used instead."""
    np = _graph_primitives._numpy()  # looked up per call so tests can take numpy away
    probe = backend.namespace_change_token(namespace) if np is not None else None
    if probe is None or not backend.supports_vectors() or not _has_proof_column(backend):
        return None
    cache = cache if cache is not None else NAMESPACE_INDEX
    key = (probe.store, namespace)
    file_id = _file_id(getattr(backend, "_db_path", ""))
    state = cache.state(key, file_id)
    if file_id is None and state.token is None:  # an in-memory store: forget its index with the backend
        weakref.finalize(backend, cache.discard, key)
    with state.lock:
        token = backend.namespace_change_token(namespace)  # read BEFORE the data, under the index lock
        if token is None:
            return None
        _refresh(backend, namespace, state, token, np, cache.max_bytes)
        state.nbytes = sum(rows.nbytes for rows in state.spaces.values())
        if state.nbytes > cache.max_bytes:  # never retained; the namespace keeps the recent window
            state.oversize = True
            state.clear()
            state.nbytes = 0
        oversize = state.oversize
    cache.evict()
    return None if oversize else NamespaceCandidates(backend, namespace, state, np)


def _refresh(
    backend: StorageBackend, namespace: str, state: _IndexState, token: NamespaceChangeToken, np: Any, max_bytes: int
) -> None:
    previous = state.token
    changed: list[str] | None = None
    if previous is not None and monotonic() - state.reconciled_at <= RECONCILE_SECONDS:
        if token == previous or state.oversize:
            return
        if token.delete_epoch == previous.delete_epoch and token.insert_seq >= previous.insert_seq:
            feed = backend.entries_changed_since(namespace, previous, limit=FEED_LIMIT)
            changed = None if feed is None else [entry.id for entry in feed]
    if changed is None:  # first use, feed overflow, a delete, or the reconcile is due
        state.oversize = False
        _sync(backend, namespace, state, np, None, max_bytes)
        state.reconciled_at = monotonic()
    elif changed:
        _sync(backend, namespace, state, np, changed, max_bytes)
    state.token = token


def _sync(
    backend: StorageBackend, namespace: str, state: _IndexState, np: Any, ids: Collection[str] | None, max_bytes: int
) -> None:
    """Bring the index up to date for *ids* (``None``: the whole namespace), re-reading only changed vectors."""
    proofs = _read_proofs(backend, namespace, ids)
    row_bytes = 4 * int(getattr(backend, "_dim", 384)) + 120  # checked BEFORE decoding what would not be kept
    if ids is None and (len(proofs) > MAX_INDEX_ROWS or len(proofs) * row_bytes > max_bytes):
        state.oversize = True
        state.clear()
        return
    scope = state.proofs.keys() if ids is None else ids
    for entry_id in [entry_id for entry_id in scope if entry_id not in proofs]:
        state.proofs.pop(entry_id, None)
        state.drop(entry_id)
    stale = [entry_id for entry_id, proof in proofs.items() if state.proofs.get(entry_id) != proof]
    records = backend.get_vector_records(stale, namespace=namespace) if stale else {}
    for entry_id in stale:
        record = records.get(entry_id)
        if record is None or record.provenance is None:  # unreadable or unproven: not a candidate, retried next time
            state.proofs.pop(entry_id, None)
            state.drop(entry_id)
        else:
            state.proofs[entry_id] = proofs[entry_id]
            state.put(entry_id, record.embedding, record.provenance.space, np)


def _read_proofs(backend: StorageBackend, namespace: str, ids: Collection[str] | None) -> dict[str, int]:
    """``{entry id: hash(provenance_json)}`` for the ACTIVE rows of *namespace* (restricted to *ids*)."""
    conn: Any = backend._conn  # type: ignore[attr-defined]
    proofs: dict[str, int] = {}
    with backend._lock:  # type: ignore[attr-defined]
        if ids is None:
            chunks: list[Sequence[str] | None] = [None]
        else:
            chunks = list(iter_bind_chunks(list(ids), reserved_bindings=2))
        for chunk in chunks:
            sql, params = _PROOFS_SQL, [namespace, _ACTIVE]
            if chunk is not None:
                sql += " AND vi.entry_id IN (" + ", ".join("?" for _ in chunk) + ")"
                params.extend(chunk)
            proofs.update((str(entry_id), hash(proof)) for entry_id, proof in conn.execute(sql, params))
    return proofs


def _has_proof_column(backend: StorageBackend) -> bool:
    """Whether *backend* exposes the SQLite vector tables with recorded proofs (legacy layouts lack the column)."""
    conn: Any = getattr(backend, "_conn", None)
    if not callable(getattr(conn, "execute", None)) or getattr(backend, "_lock", None) is None:
        return False
    with backend._lock:  # type: ignore[attr-defined]
        columns = {row[1] for row in conn.execute("PRAGMA table_info(vec_index)").fetchall()}
    return "provenance_json" in columns
