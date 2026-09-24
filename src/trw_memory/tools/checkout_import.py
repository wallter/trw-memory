"""MCP tool: fold a checkout's project store into its namespace (PRD-CORE-280 FR03).

``trw-mcp memory migrate --to user`` never opens the user store while the daemon
serves it: it hands the daemon a working copy of the checkout's project store (a
file inside the checkout its token was minted for), and ``memory_import_checkout``
merges that copy's ``default`` rows, vectors and graph edges into the granted
namespace. The destination wins an id collision; one whose row or vector differs from
the copy's in anything the store did not assign, or holds a vector only one side has,
is refused by id, since the copy's row would not be imported. The compare and the merge share one transaction on each store,
so a refused import leaves the namespace, and the copy, as they were. An import that
outlives ``IMPORT_DEADLINE_SECONDS``, waiting for the write lock included, is rolled back
and answers ``busy``. The destination commits first: if the copy's commit then fails the
answer is ``uncertain`` (rerun it), never a claimed rollback. A rerun after a lost reply
or a killed caller moves nothing twice. The reply counts what the namespace now
holds of the migrated ids, which the caller checks before it cuts over.
"""

from __future__ import annotations

import contextlib
import sqlite3
import time
from collections.abc import Iterator
from pathlib import Path

from trw_memory.exceptions import StorageError
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.namespaces.curate import NamespaceCurateResult, NamespaceStores, merge_namespace
from trw_memory.namespaces.validation import DEFAULT_NAMESPACE
from trw_memory.security.rbac import Permission
from trw_memory.storage._stale_handle import ensure_connection_fresh
from trw_memory.storage.interface import StorageBackend
from trw_memory.storage.sqlite_backend import SQLiteBackend
from trw_memory.tools._types import McpServer
from trw_memory.tools.entry import checkout_path, in_namespace, refused_namespace

#: How long the import may hold the daemon's write lock before it gives up and rolls back.
IMPORT_DEADLINE_SECONDS = 10.0

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


def memory_import_checkout_impl(
    namespace: str,
    source_path: str,
    ids: list[str],
    *,
    backend: StorageBackend,
    deadline_seconds: float = IMPORT_DEADLINE_SECONDS,
) -> dict[str, object]:
    """``{"status": "ok", "moved", "skipped", "held": {"rows", "vectors", "edges"}}`` -- held among *ids*."""
    if not Path(source_path).is_file():
        return {"error": f"no project store at {source_path}", "status": "invalid"}
    if not isinstance(backend, SQLiteBackend):
        return {"error": "memory_import_checkout needs a SQLite store to compare vectors", "status": "invalid"}
    source = SQLiteBackend(Path(source_path), dim=getattr(backend, "_dim", 384))
    deadline = time.monotonic() + deadline_seconds
    committed = False  # the destination's commit ran: from here nothing can claim a rollback
    try:
        # One transaction on each side across the compare, the merge and the skip
        # check: no other writer changes a compared row before the merge, and a
        # refusal rolls back whatever the merge had copied or removed. The deadline
        # bounds the whole hold of the daemon's write lock, the wait for it included;
        # the progress handlers are off again before either commit runs.
        with _lock_wait_until(source, deadline), _lock_wait_until(backend, deadline), source.transaction():
            with backend.transaction(), _interrupted_after(source, deadline), _interrupted_after(backend, deadline):
                result = _merge_if_identical(source, backend, namespace)
                if time.monotonic() > deadline:
                    raise _Refused(_busy(deadline_seconds))
            committed = True
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
    finally:
        source.close()
    wanted = set(ids)
    held = {
        "rows": sum(1 for entry_id in ids if backend.get(entry_id, namespace=namespace) is not None),
        "vectors": len(backend.existing_vector_ids(namespace=namespace) & wanted) if backend.supports_vectors() else 0,
        "edges": sum(1 for edge in backend.graph_edges(namespace) if edge.source_id in wanted),
    }
    return {"status": "ok", "moved": result.moved, "skipped": result.skipped, "held": held}


def _busy(deadline_seconds: float) -> dict[str, object]:
    return {
        "error": f"the import did not finish within {deadline_seconds:g}s and was rolled back; "
        "the store is busy or the copy is large: retry",
        "status": "busy",
    }


@contextlib.contextmanager
def _lock_wait_until(store: SQLiteBackend, deadline: float) -> Iterator[None]:
    """Bound *store*'s wait for a lock by *deadline*: SQLite's busy wait runs no progress handler.

    The connection is refreshed first, under the store's lock, so the timeout is set on the
    connection the transaction then opens (its own freshness check is cached for a second).
    """
    with store._lock:
        ensure_connection_fresh(store)
        conn = store._conn
        previous = int(conn.execute("PRAGMA busy_timeout").fetchone()[0])
        conn.execute(f"PRAGMA busy_timeout = {max(1, int((deadline - time.monotonic()) * 1000))}")
        try:
            yield
        finally:
            conn.execute(f"PRAGMA busy_timeout = {previous}")


@contextlib.contextmanager
def _interrupted_after(store: SQLiteBackend, deadline: float) -> Iterator[None]:
    """Abort any statement *store* runs past *deadline* (``sqlite3.OperationalError: interrupted``)."""
    conn = store._conn
    conn.set_progress_handler(lambda: int(time.monotonic() > deadline), _DEADLINE_CHECK_INTERVAL)
    try:
        yield
    finally:
        conn.set_progress_handler(None, 0)


def _merge_if_identical(source: SQLiteBackend, destination: SQLiteBackend, namespace: str) -> NamespaceCurateResult:
    """Merge the copy into *namespace* when every collision is identical; raises :class:`_Refused` otherwise."""
    collisions = _collisions(source, destination, namespace)
    if collisions is None:  # a row it cannot list back cannot be compared: fail closed
        raise _Refused({"error": "could not list every row of the project store to compare", "status": "conflict"})
    try:
        conflicts = _differing(source, destination, namespace, collisions)
    except StorageError as exc:  # an unread vector is not an absent one
        raise _Refused({"error": f"could not read the vectors to compare: {exc}", "status": "conflict"}) from exc
    if conflicts:
        raise _Refused(
            {
                "error": f"{namespace} already holds different content for {conflicts}",
                "status": "conflict",
                "conflicts": conflicts,
            }
        )
    result = merge_namespace(NamespaceStores(source=source, destination=destination), DEFAULT_NAMESPACE, namespace)
    if result.skipped != len(collisions):  # the merge skipped rows it was not asked to: they went unchecked
        raise _Refused(
            {
                "error": f"the merge skipped {result.skipped} rows, not the {len(collisions)} compared",
                "status": "conflict",
            }
        )
    return result


def _collisions(source: StorageBackend, destination: StorageBackend, namespace: str) -> list[MemoryEntry] | None:
    """The copy's rows whose id *namespace* already holds, or ``None`` when the copy did not list back whole."""
    total = source.count(namespace=DEFAULT_NAMESPACE)
    rows = source.list_entries(namespace=DEFAULT_NAMESPACE, limit=total + 1)
    if len(rows) != total:
        return None
    return [entry for entry in rows if destination.get(entry.id, namespace=namespace) is not None]


def _differing(
    source: SQLiteBackend, destination: SQLiteBackend, namespace: str, collisions: list[MemoryEntry]
) -> list[str]:
    """Ids whose held row or vector differs from the copy's in anything but what the store assigns."""
    ids = [entry.id for entry in collisions]
    theirs = source.vector_records_or_raise(ids, namespace=DEFAULT_NAMESPACE)
    ours = destination.vector_records_or_raise(ids, namespace=namespace)
    differing = []
    for entry in collisions:
        held = destination.get(entry.id, namespace=namespace)
        vector, kept = theirs.get(entry.id), ours.get(entry.id)
        if (
            held is None
            or held.model_dump(exclude=_STORE_ASSIGNED) != entry.model_dump(exclude=_STORE_ASSIGNED)
            or (vector is None) != (kept is None)
            or (vector is not None and kept is not None and kept.embedding != vector.embedding)
        ):
            differing.append(entry.id)
    return sorted(differing)


def register_checkout_import_tools(mcp: McpServer) -> None:
    """Register memory_import_checkout with a FastMCP server instance."""

    async def memory_import_checkout(namespace: str, source_path: str, ids: list[str]) -> dict[str, object]:
        """Merge the project store at *source_path* (inside the granted checkout) into *namespace*."""
        if refused := refused_namespace(namespace, Permission.WRITE, "import_checkout", MemoryConfig()):
            return refused
        source = checkout_path(source_path, "memory_import_checkout", within=True)
        if not isinstance(source, str):
            return source or {"error": "memory_import_checkout needs a source_path", "status": "invalid"}
        return in_namespace(
            namespace,
            Permission.WRITE,
            "import_checkout",
            lambda backend, _config: memory_import_checkout_impl(namespace, source, ids, backend=backend),
        )

    mcp.tool()(memory_import_checkout)


__all__ = ["memory_import_checkout_impl", "register_checkout_import_tools"]
