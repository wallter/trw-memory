"""Delta tracking for sync pipeline -- PRD-INFRA-051."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from trw_memory.models.memory import MemoryEntry

if TYPE_CHECKING:
    from trw_memory.storage.interface import StorageBackend

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
)


class DeltaTracker:
    """Tracks which entries are dirty and need syncing."""

    @staticmethod
    def compute_sync_hash(entry: MemoryEntry) -> str:
        """SHA-256 of canonical serialization of content fields."""
        d = entry.to_dict()
        canonical = {k: d.get(k) for k in _HASH_FIELDS}
        # Stable serialization
        raw = json.dumps(canonical, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()

    @staticmethod
    def mark_dirty(entry_id: str, backend: StorageBackend) -> None:
        """Mark a single entry as dirty (needs re-sync)."""
        entry = backend.get(entry_id)
        if entry is None:
            return
        new_seq = entry.sync_seq + 1
        new_hash = DeltaTracker.compute_sync_hash(entry)
        backend.update(entry_id, sync_seq=new_seq, sync_hash=new_hash, last_synced_at=None)

    @staticmethod
    def get_dirty_entries(backend: StorageBackend, since_seq: int = 0) -> list[MemoryEntry]:
        """Get entries needing sync (sync_seq > since_seq and not yet synced)."""
        # Try SQLite direct query for efficiency
        conn = getattr(backend, "_conn", None)
        if conn is not None:
            from trw_memory.storage._row_mapper import row_to_entry
            from trw_memory.storage._shared import ENTRY_COLUMNS

            cols = ", ".join(
                "expires_at AS expires" if c == "expires_at" else c for c in ENTRY_COLUMNS
            )
            sql = (
                f"SELECT {cols} FROM memories "  # noqa: S608
                f"WHERE sync_seq > ? AND (last_synced_at IS NULL OR last_synced_at = '') "
                f"ORDER BY sync_seq ASC"
            )
            rows = conn.execute(sql, (since_seq,)).fetchall()
            return [row_to_entry(tuple(r)) for r in rows]
        # Fallback: list all and filter
        all_entries = backend.list_entries(limit=10000)
        return [e for e in all_entries if e.sync_seq > since_seq and e.last_synced_at is None]

    @staticmethod
    def mark_synced(entry_ids: list[str], backend: StorageBackend) -> int:
        """Set last_synced_at = now() on successfully pushed entries."""
        now = datetime.now(tz=timezone.utc)
        count = 0
        for eid in entry_ids:
            try:
                result = backend.update(eid, last_synced_at=now)
                if result is not None:
                    count += 1
            except Exception:
                pass
        return count
