"""Delta tracking for sync pipeline -- PRD-INFRA-051."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import structlog

from trw_memory.models.memory import MemoryEntry

if TYPE_CHECKING:
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
    def mark_dirty(entry_id: str, backend: StorageBackend, *, namespace: str) -> None:
        """Mark the ``(namespace, entry_id)`` entry as dirty (needs re-sync)."""
        entry = backend.get(entry_id, namespace=namespace)
        if entry is None:
            return
        new_seq = entry.sync_seq + 1
        new_hash = DeltaTracker.compute_sync_hash(entry)
        backend.update(entry_id, namespace=namespace, sync_seq=new_seq, sync_hash=new_hash, last_synced_at=None)

    @staticmethod
    def get_dirty_entries(backend: StorageBackend, since_seq: int = 0) -> list[MemoryEntry]:
        """Get entries needing sync (sync_seq > since_seq and not yet synced)."""
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
                f"ORDER BY sync_seq ASC"
            )
            # Acquire backend._lock to match the locking pattern used by every
            # other SQLite query in this backend (Bug: missing lock could race a
            # concurrent write on the same connection).
            if lock is not None:
                with lock:
                    rows = conn.execute(sql, (since_seq,)).fetchall()
            else:
                rows = conn.execute(sql, (since_seq,)).fetchall()
            return [row_to_entry(tuple(r)) for r in rows]
        # Fallback: hydrate a complete snapshot before filtering. Filtering
        # after a fixed limit can permanently hide an older dirty row behind
        # newer synced rows. Recount after each read so concurrent inserts that
        # could displace the tail trigger a larger retry.
        limit = max(1, backend.count() + 1)
        all_entries: list[MemoryEntry] = []
        for _ in range(_FALLBACK_SCAN_ATTEMPTS):
            all_entries = backend.list_entries(limit=limit)
            current_count = backend.count()
            if current_count <= limit:
                break
            limit = current_count + 1
        else:
            # A stable, complete snapshot is impossible while writes outpace
            # the scan. Return the newest bounded snapshot; the next sync pass
            # can collect rows inserted concurrently without livelocking this
            # one.
            logger.warning("delta_fallback_scan_unstable", attempts=_FALLBACK_SCAN_ATTEMPTS)
        return [e for e in all_entries if e.sync_seq > since_seq and e.last_synced_at is None]

    @staticmethod
    def mark_synced(entry_ids: list[str], backend: StorageBackend, *, namespace: str) -> int:
        """Set last_synced_at = now() on successfully pushed entries in *namespace*."""
        now = datetime.now(tz=timezone.utc)
        count = 0
        for eid in entry_ids:
            try:
                result = backend.update(eid, namespace=namespace, last_synced_at=now)
                if result is not None:
                    count += 1
            except Exception:
                logger.warning("delta_mark_synced_failed", entry_id=eid, exc_info=True)
        return count
