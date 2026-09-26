"""Delta tracking for sync pipeline -- PRD-INFRA-051."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from contextlib import nullcontext
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import structlog

from trw_memory.models.memory import MemoryEntry

if TYPE_CHECKING:
    from trw_memory.models.config import MemoryConfig
    from trw_memory.storage.interface import StorageBackend

logger = structlog.get_logger(__name__)

_FALLBACK_SCAN_ATTEMPTS = 3

# Fields included in sync hash (content-bearing fields)
_HASH_FIELDS = (
    "content",
    "detail",
    "tags",
    "evidence",
    "importance",
    "status",
    "type",
    "confidence",
    "domain",
    "phase_affinity",
    "metadata",
    # PRD-CORE-194 FR04: a supersession write closes the validity window
    # (sets invalid_from + invalidated_by) without touching content, so it must
    # mark the entry dirty for sync. ``valid_from`` is deliberately EXCLUDED: it
    # defaults to per-construction ``now()`` for an entry built without an
    # explicit created_at, so hashing it would make two otherwise-identical
    # entries diverge purely on construction instant (breaks the content-hash
    # contract). For a persisted row valid_from is stable; the supersession
    # signal we need to propagate is the close pair below.
    "invalid_from",
    "invalidated_by",
)


def _normalize_hash_value(value: object) -> object:
    """Normalize values into the PRD's canonical JSON-hash representation."""
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, float):
        return float(f"{value:.6f}")
    if isinstance(value, dict):
        return {
            str(key): _normalize_hash_value(val) for key, val in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_hash_value(item) for item in value]
    return value


class DeltaTracker:
    """Tracks which entries are dirty and need syncing."""

    @staticmethod
    def compute_sync_hash(entry: MemoryEntry) -> str:
        """SHA-256 of canonical serialization of content fields."""
        d = entry.to_dict()
        canonical = {k: _normalize_hash_value(d.get(k)) for k in _HASH_FIELDS}
        raw = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(raw.encode()).hexdigest()

    @staticmethod
    def get_dirty_entries(
        backend: StorageBackend, since_seq: int = 0, *, namespace: str | None = None, limit: int | None = None
    ) -> list[MemoryEntry]:
        """Get entries needing sync (sync_seq > since_seq and not yet synced), oldest first.

        *namespace* keeps one tenant's push from paging another's rows; *limit* bounds the page.
        """
        # Try SQLite direct query for efficiency
        conn = getattr(backend, "_conn", None)
        lock = getattr(backend, "_lock", None)
        if conn is not None:
            from trw_memory.storage._row_mapper import row_to_entry
            from trw_memory.storage._shared import ENTRY_COLUMNS

            cols = ", ".join("expires_at AS expires" if c == "expires_at" else c for c in ENTRY_COLUMNS)
            sql = (
                f"SELECT {cols} FROM memories "  # noqa: S608
                f"WHERE sync_seq > ? AND (last_synced_at IS NULL OR last_synced_at = '') "
                f"AND (? IS NULL OR namespace = ?) ORDER BY sync_seq ASC LIMIT ?"
            )
            params = (since_seq, namespace, namespace, -1 if limit is None else limit)
            # Acquire backend._lock to match the locking pattern used by every
            # other SQLite query in this backend (Bug: missing lock could race a
            # concurrent write on the same connection).
            if lock is not None:
                with lock:
                    rows = conn.execute(sql, params).fetchall()
            else:
                rows = conn.execute(sql, params).fetchall()
            return [row_to_entry(tuple(r)) for r in rows]
        # Fallback: hydrate a complete snapshot before filtering. Filtering
        # after a fixed limit can permanently hide an older dirty row behind
        # newer synced rows. Recount after each read so concurrent inserts that
        # could displace the tail trigger a larger retry.
        scan_limit = max(1, backend.count() + 1)
        all_entries: list[MemoryEntry] = []
        for _ in range(_FALLBACK_SCAN_ATTEMPTS):
            all_entries = backend.list_entries(limit=scan_limit)
            current_count = backend.count()
            if current_count <= scan_limit:
                break
            scan_limit = current_count + 1
        else:
            # A stable, complete snapshot is impossible while writes outpace
            # the scan. Return the newest bounded snapshot; the next sync pass
            # can collect rows inserted concurrently without livelocking this
            # one.
            logger.warning("delta_fallback_scan_unstable", attempts=_FALLBACK_SCAN_ATTEMPTS)
        dirty = [
            e
            for e in sorted(all_entries, key=lambda e: e.sync_seq)
            if e.sync_seq > since_seq and e.last_synced_at is None and namespace in (None, e.namespace)
        ]
        return dirty if limit is None else dirty[:limit]

    @staticmethod
    def mark_synced(
        entry_ids: list[str],
        backend: StorageBackend,
        *,
        namespace: str,
        expected_seq: Mapping[str, int] | None = None,
    ) -> int:
        """Set last_synced_at = now() on successfully pushed entries in *namespace*.

        With *expected_seq*, a row is marked only while its ``sync_seq`` still
        equals the one it was pushed at; the compare and the mark share one
        write transaction, so an edit landing after the push keeps the row dirty.
        A backend without a real transaction (only the interface's no-op) raises
        ``TypeError`` rather than performing a compare-and-mark that is not atomic.
        """
        from trw_memory.storage.interface import StorageBackend

        now = datetime.now(tz=timezone.utc)
        count = 0
        if expected_seq is not None and type(backend).transaction is StorageBackend.transaction:
            raise TypeError(
                f"a conditional sync ack needs a transactional backend; {type(backend).__name__} has no transaction"
            )
        with backend.transaction() if expected_seq is not None else nullcontext():
            for eid in entry_ids:
                try:
                    if expected_seq is not None:
                        current = backend.get(eid, namespace=namespace)
                        if current is None or current.sync_seq != expected_seq.get(eid):
                            continue
                    if backend.update(eid, namespace=namespace, last_synced_at=now) is not None:
                        count += 1
                except Exception:
                    logger.warning("delta_mark_synced_failed", entry_id=eid, exc_info=True)
        return count


def ack_publish(backend: StorageBackend, entry: MemoryEntry, **fields: object) -> bool:
    """Record a publish of *entry*'s snapshot: *fields* always, ``last_synced_at`` only while the row is
    still the published revision (``sync_seq`` and ``sync_hash``), so an edit made since stays dirty for the next
    push (C12 rc7)."""
    with backend.transaction():
        if (current := backend.get(entry.id, namespace=entry.namespace)) is None:
            return False
        # The hash too: store() numbers a revision from the writer's copy, so two contents can share a seq.
        clean = (current.sync_seq, current.sync_hash) == (entry.sync_seq, entry.sync_hash)
        backend.update(
            entry.id,
            namespace=entry.namespace,
            **fields,
            **({"last_synced_at": datetime.now(tz=timezone.utc)} if clean else {}),
        )
        return clean


def find_synced_entry(backend: StorageBackend, namespace: str, remote_id: str, ids: list[str]) -> MemoryEntry | None:
    """The row in *namespace* that a pulled learning maps to: its ``remote_id`` or one of *ids*.

    Every match is namespace-qualified, so two peers emitting the same remote id into two
    namespaces stay two rows (PRD-CORE-245 P1).
    """
    conn = getattr(backend, "_conn", None)
    if conn is not None:
        from trw_memory.storage._row_mapper import row_to_entry

        marks = ", ".join("?" for _ in ids) or "NULL"
        sql = f"SELECT * FROM memories WHERE namespace = ? AND (remote_id = ? OR id IN ({marks})) LIMIT 1"  # noqa: S608
        row = conn.execute(sql, (namespace, remote_id, *ids)).fetchone()
        return row_to_entry(tuple(row)) if row is not None else None
    for candidate in backend.list_entries(namespace=namespace, limit=max(backend.count(namespace=namespace), 1)):
        if candidate.remote_id == remote_id or candidate.id in ids:
            return candidate
    return None


def apply_synced_entry(
    backend: StorageBackend, config: MemoryConfig, entry: MemoryEntry, *, synced: bool = True
) -> tuple[str, str]:
    """Write a merged pulled row through the write gate and leave it synced, or dirty when *synced* is false.

    Returns ``("stored" | "quarantined" | "blocked", reason)``. A security refusal is a judged
    decision, not a store failure (PRD-FIX-138-FR01), so it is reported rather than raised.
    """
    from trw_memory.exceptions import PIIBlockError, PoisoningError
    from trw_memory.security.runtime import prepare_entry_for_store, store_quarantined_entry

    try:
        decision = prepare_entry_for_store(entry, backend=backend, config=config, session_id=None)
    except (PoisoningError, PIIBlockError) as exc:
        return "blocked", getattr(exc, "reason", "") or type(exc).__name__
    if decision.quarantined:
        store_quarantined_entry(config, decision.entry)
        return "quarantined", ""
    # One commit: an edit that lands between the write and its ack must not be marked clean (C12 rc7).
    with backend.transaction():
        backend.store(decision.entry)
        if synced:
            DeltaTracker.mark_synced([entry.id], backend, namespace=entry.namespace)
    return "stored", ""
