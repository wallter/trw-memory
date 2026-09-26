"""The two halves of ``memory_import_checkout``'s merge: the untrusted reads, and the write (rc9).

:func:`plan_import` checks, opens and reads the checkout's copy whole, a page at a time, and compares
its collisions with the namespace. It runs on its caller's thread, never on the daemon's one-thread
write lane, so a crafted or large copy costs that caller's time alone. :func:`write_import` is the
lane's half: it re-checks the compared rows and merges. The copy is registered with
``untrusted_store`` in both, so every connection opened on it -- a reopen or a decode fallback too --
runs under that phase's deadline and caps every value it builds or reads.
"""

from __future__ import annotations

import contextlib
import hashlib
import sqlite3
import time
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

from trw_memory.exceptions import StorageError
from trw_memory.models.memory import MemoryEntry
from trw_memory.namespaces.curate import NamespaceStores, merge_namespace
from trw_memory.namespaces.validation import DEFAULT_NAMESPACE
from trw_memory.storage._connection import untrusted_store
from trw_memory.storage._stale_handle import ensure_connection_fresh
from trw_memory.storage._untrusted_store import verify_untrusted_store
from trw_memory.storage.interface import EntryCursor
from trw_memory.storage.sqlite_backend import SQLiteBackend

#: SQLite VM instructions between deadline checks: frequent enough to stop a long scan promptly.
_DEADLINE_CHECK_INTERVAL = 10_000

#: Assigned by the store on every write (``store()`` bumps the sequence, recomputes the hash
#: and clears the sync mark), so an identical copy differs in these alone.
_STORE_ASSIGNED: set[str] = {"namespace", "sync_seq", "sync_hash", "last_synced_at"}


class _Refused(Exception):
    """Raised inside the import's transactions so both roll back; carries the reply."""

    def __init__(self, reply: dict[str, object]) -> None:
        super().__init__(reply["error"])
        self.reply = reply


def _invalid(why: object) -> dict[str, object]:
    return {"error": f"not a project store trw-memory can import: {why}", "status": "invalid"}


def _busy(deadline_seconds: float) -> dict[str, object]:
    return {
        "error": f"the import did not finish within {deadline_seconds:g}s and was rolled back; "
        "the store is busy or the copy is large: retry",
        "status": "busy",
    }


@contextlib.contextmanager
def _lock_wait_until(store: SQLiteBackend, deadline: float) -> Iterator[None]:
    """Bound *store*'s waits for its locks by *deadline*: its in-process lock, and SQLite's busy wait,
    which runs no progress handler.

    The connection is refreshed first, under the store's lock, so the timeout is set on the
    connection the transaction then opens (its own freshness check is cached for a second).
    """
    if not store._lock.acquire(timeout=max(0.0, deadline - time.monotonic())):  # another thread holds the store
        raise sqlite3.OperationalError("database is locked: another thread holds the store past the deadline")
    try:
        ensure_connection_fresh(store)
        conn = store._conn
        previous = int(conn.execute("PRAGMA busy_timeout").fetchone()[0])
        conn.execute(f"PRAGMA busy_timeout = {max(1, int((deadline - time.monotonic()) * 1000))}")
        try:
            yield
        finally:
            conn.execute(f"PRAGMA busy_timeout = {previous}")
    finally:
        store._lock.release()


@contextlib.contextmanager
def _interrupted_after(store: SQLiteBackend, deadline: float) -> Iterator[None]:
    """Abort any statement *store* runs past *deadline* (``sqlite3.OperationalError: interrupted``)."""
    conn = store._conn
    conn.set_progress_handler(lambda: int(time.monotonic() > deadline), _DEADLINE_CHECK_INTERVAL)
    try:
        yield
    finally:
        conn.set_progress_handler(None, 0)


#: Rows per page of every read of the copy and of the collisions' re-check (the merge moves 2,000 at a time).
_READ_PAGE = 2_000

#: The most graph edges a copy may carry: the merge's edge check holds their keys in memory, and no
#: page bounds that read (rc9). trw-memory files a handful per learning.
IMPORT_MAX_EDGES = 500_000


class _Plan(NamedTuple):
    """What the reads of the copy decided, for the write: nothing in it is the copy's own text."""

    #: The copy's edge keys (source, target, type), which the namespace must hold after the merge.
    edges: set[tuple[str, str, str]]
    #: Ids whose vectors have another dimension: another embedding space, which the namespace's vector
    #: write skips. Their rows still move; the reply names them for ``memory_reembed``.
    other_space: list[str]
    #: Each colliding id, with a digest of what the namespace held for it when compared.
    collisions: dict[str, str]


def plan_import(
    path: Path, destination: SQLiteBackend, namespace: str, deadline_seconds: float
) -> _Plan | dict[str, object]:
    """Check, open and read the copy whole, and compare its collisions: the untrusted half, off the write lane.

    The backend's open admits a file whose tables can still hold another build's shape or rows this
    build cannot parse (C9): read here, a failure belongs to the copy and gets ``invalid``, or ``busy``
    past the deadline.
    """
    deadline = time.monotonic() + deadline_seconds
    try:
        with untrusted_store(path, deadline):
            verify_untrusted_store(path)
            source = SQLiteBackend(path, dim=destination._dim, check_integrity_once=True)
            try:
                with _lock_wait_until(destination, deadline):  # its reads wait for its locks no longer than that
                    return _read_and_compare(source, destination, namespace, deadline)
            finally:
                source.close()
    except (StorageError, RuntimeError, sqlite3.DatabaseError, ValueError) as exc:  # ValidationError is a ValueError
        return _busy(deadline_seconds) if time.monotonic() > deadline else _invalid(exc)


def _read_and_compare(
    source: SQLiteBackend, destination: SQLiteBackend, namespace: str, deadline: float
) -> _Plan | dict[str, object]:
    """Read every row, vector and edge the merge will read, once, in pages; compare each collision.

    The destination wins a collision, so one whose row or vector differs from the copy's in anything but
    what the store assigns is refused by id: the copy's row would not be imported. Never the whole copy
    in memory: a page at a time, the deadline checked between pages (rc9).
    """
    dim = destination._dim
    counted = source._conn.execute("SELECT count(*) FROM memory_graph_edges WHERE namespace = ?", (DEFAULT_NAMESPACE,))
    if (edge_count := int(counted.fetchone()[0])) > IMPORT_MAX_EDGES:
        return _invalid(f"it holds {edge_count} graph edges, more than the {IMPORT_MAX_EDGES} an import takes")
    edges = {edge[:3] for edge in source.graph_edges(DEFAULT_NAMESPACE)}
    other_space: list[str] = []
    collisions: dict[str, str] = {}
    conflicts: list[str] = []
    listed = 0
    cursor: EntryCursor | None = None
    while page := source.list_entries(namespace=DEFAULT_NAMESPACE, limit=_READ_PAGE, after=cursor):
        if time.monotonic() > deadline:
            raise sqlite3.OperationalError("interrupted")
        cursor = EntryCursor.from_entry(page[-1])
        listed += len(page)
        ids = [entry.id for entry in page]
        not_carried: set[str] = set()
        if source.supports_vectors():
            records = source.get_vector_records(ids, namespace=DEFAULT_NAMESPACE)
            stored = source.get_stored_embeddings(ids, namespace=DEFAULT_NAMESPACE)
            wrong = {i for i, r in records.items() if len(r.embedding) != dim}
            not_carried = wrong | {i for i, e in stored.items() if len(e) != dim}
            other_space += sorted(not_carried)
        held = {entry.id: kept for entry in page if (kept := destination.get(entry.id, namespace=namespace))}
        if not held:
            continue
        try:  # an unread vector is not an absent one
            theirs = source.vector_records_or_raise(list(held), namespace=DEFAULT_NAMESPACE)
            ours = destination.vector_records_or_raise(list(held), namespace=namespace)
        except StorageError as exc:
            return {"error": f"could not read the vectors to compare: {exc}", "status": "conflict"}
        for entry in page:
            if (kept := held.get(entry.id)) is None:
                continue
            # A vector of another dimension is one the merge does not carry, so it is not compared either:
            # a retry after an import that moved the row but not that vector converges (C12 pre-rc9).
            vector_differs = entry.id not in not_carried and theirs.get(entry.id) != ours.get(entry.id)
            if kept.model_dump(exclude=_STORE_ASSIGNED) != entry.model_dump(exclude=_STORE_ASSIGNED) or vector_differs:
                conflicts.append(entry.id)  # absent on one side, or a different embedding or provenance
            else:
                collisions[entry.id] = _digest(kept, ours.get(entry.id))
    if listed != source.count(namespace=DEFAULT_NAMESPACE):  # a row it cannot list back cannot be compared
        return {"error": "could not list every row of the project store to compare", "status": "conflict"}
    if conflicts:
        return {
            "error": f"{namespace} already holds different content for {sorted(conflicts)}",
            "status": "conflict",
            "conflicts": sorted(conflicts),
        }
    return _Plan(edges, other_space, collisions)


def _digest(held: MemoryEntry, vector: object) -> str:
    """What the namespace holds for one id, as compared: every field the store does not assign, and its vector."""
    return hashlib.sha256(repr((held.model_dump(exclude=_STORE_ASSIGNED), vector)).encode()).hexdigest()


def write_import(
    path: Path, namespace: str, plan: _Plan, deadline_seconds: float, destination: SQLiteBackend
) -> dict[str, object]:
    """The write lane's half: merge the checked copy into *namespace* if its collisions are as compared.

    One transaction on each side across the re-check, the merge and the skip check: no other writer
    changes a compared row before the merge, and a refusal rolls back whatever the merge had copied or
    removed. The deadline bounds the whole hold of the daemon's write lock, the wait for it included;
    the progress handlers are off again before either commit runs. The destination commits first.
    """
    deadline = time.monotonic() + deadline_seconds
    committed = False  # the destination's commit ran: from here nothing can claim a rollback
    try:
        with untrusted_store(path, deadline):
            source = SQLiteBackend(path, dim=destination._dim, check_integrity_once=True)  # checked by _plan
            try:
                with _lock_wait_until(source, deadline), _lock_wait_until(destination, deadline), source.transaction():
                    with (
                        destination.transaction(),
                        _interrupted_after(source, deadline),
                        _interrupted_after(destination, deadline),
                    ):
                        if _changed(destination, namespace, plan.collisions, deadline):
                            raise _Refused(
                                {"error": f"{namespace} changed while the import compared it: retry", "status": "busy"}
                            )
                        result = merge_namespace(
                            NamespaceStores(source=source, destination=destination), DEFAULT_NAMESPACE, namespace
                        )
                        if result.skipped != len(plan.collisions):  # it skipped rows it was not asked to: unchecked
                            raise _Refused(
                                {
                                    "error": f"the merge skipped {result.skipped} rows, not the "
                                    f"{len(plan.collisions)} compared",
                                    "status": "conflict",
                                }
                            )
                        # The merge files edges with INSERT OR IGNORE, which drops an edge that breaks a CHECK too.
                        kept = {edge[:3] for edge in destination.graph_edges(namespace)}
                        if dropped := sorted(key for key in plan.edges if key not in kept):
                            raise _Refused(
                                _invalid(f"{len(dropped)} of its edges break this store's constraints ({dropped[0]})")
                            )
                        if time.monotonic() > deadline:
                            raise _Refused(_busy(deadline_seconds))
                    committed = True
            finally:
                source.close()
    except _Refused as refused:
        return _busy(deadline_seconds) if time.monotonic() > deadline else refused.reply
    except Exception as exc:
        if committed:  # the namespace took the merge but the copy kept its rows
            return {
                "error": f"the namespace committed the import but the copy's commit failed ({exc}); "
                "rerun: the import is idempotent",
                "status": "uncertain",
            }
        if isinstance(exc, (sqlite3.OperationalError, StorageError)) and (
            time.monotonic() > deadline or "locked" in str(exc) or "busy" in str(exc)
        ):
            return _busy(deadline_seconds)
        raise
    return {"status": "ok", "moved": result.moved, "skipped": result.skipped}


def _changed(destination: SQLiteBackend, namespace: str, collisions: dict[str, str], deadline: float) -> bool:
    """Whether any collision's row or vector in *namespace* differs from what was compared off the lane.

    Content, as the compare judged it: a row replaced by an identical one is still identical, and the
    merge skips it either way.
    """
    ids = list(collisions)
    for start in range(0, len(ids), _READ_PAGE):
        if time.monotonic() > deadline:
            raise _Refused({"error": "the import's deadline passed", "status": "busy"})
        chunk = ids[start : start + _READ_PAGE]
        ours = destination.vector_records_or_raise(chunk, namespace=namespace)
        for entry_id in chunk:
            held = destination.get(entry_id, namespace=namespace)
            if held is None or _digest(held, ours.get(entry_id)) != collisions[entry_id]:
                return True
    return False


__all__ = ["plan_import", "write_import"]
