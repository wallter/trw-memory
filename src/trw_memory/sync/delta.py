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
from trw_memory.security._evidence_invariant import served_view

if TYPE_CHECKING:
    from trw_memory.embeddings.interface import EmbeddingProvider
    from trw_memory.embeddings.provenance import VectorProvenance
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
        backend: StorageBackend,
        since_seq: int = 0,
        *,
        namespace: str | None = None,
        limit: int | None = None,
        after: tuple[int, str] | None = None,
    ) -> list[MemoryEntry]:
        """Get entries needing sync (sync_seq > since_seq and not yet synced), oldest first.

        *namespace* keeps one tenant's push from paging another's rows; *limit* bounds the page. *after* is a keyset
        position ``(sync_seq, id)``: only rows strictly behind it are returned, so a caller that will not send the rows
        at the front of the queue can page past them (SYNC-PUSH-HELD-STALL); it takes the place of *since_seq*.
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
                f"WHERE (sync_seq > ? OR (? IS NOT NULL AND sync_seq = ? AND id > ?)) "
                f"AND (last_synced_at IS NULL OR last_synced_at = '') "
                f"AND (? IS NULL OR namespace = ?) ORDER BY sync_seq ASC, id ASC LIMIT ?"
            )
            # PRD-CORE-333 (CORE-333-PUBLISHER-BYPASS): a row the quarantine ledger blocks never leaves the host. It
            # stays dirty and sorts first, so the page is filled from the rows behind it instead of coming back short.
            page: list[MemoryEntry] = []
            # Keyset paging on (sync_seq, id): sync_seq is a per-row revision count, not a unique cursor, and a
            # position (OFFSET) would skip or repeat rows that another call acknowledges between pages.
            while True:
                last_seq, last_id = after if after is not None else (since_seq, None)
                cap = -1 if limit is None else limit
                params = (last_seq, last_id, last_seq, last_id, namespace, namespace, cap)
                # Acquire backend._lock to match the locking pattern used by every
                # other SQLite query in this backend (Bug: missing lock could race a
                # concurrent write on the same connection).
                if lock is not None:
                    with lock:
                        rows = conn.execute(sql, params).fetchall()
                else:
                    rows = conn.execute(sql, params).fetchall()
                # PRD-CORE-312: dirty rows are served (pushed, returned by the daemon), so demote.
                entries = [served_view(row_to_entry(tuple(r))) for r in rows]
                page += backend.filter_quarantined(entries)
                if limit is None or len(rows) < limit or len(page) >= limit:
                    return page if limit is None else page[:limit]
                after = (entries[-1].sync_seq, entries[-1].id)
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
        dirty = backend.filter_quarantined(
            [
                e
                for e in sorted(all_entries, key=lambda e: (e.sync_seq, e.id))
                if (
                    (e.sync_seq, e.id) > after if after is not None else e.sync_seq > since_seq
                )  # a cursor replaces since_seq, as in the SQL path
                and e.last_synced_at is None
                and namespace in (None, e.namespace)
            ]
        )
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
        from trw_memory.storage.interface import is_transactional

        now = datetime.now(tz=timezone.utc)
        count = 0
        if expected_seq is not None and not is_transactional(backend):
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
    return ack_revision(backend, entry.id, entry.namespace, (entry.sync_seq, entry.sync_hash), **fields)


def ack_revision(
    backend: StorageBackend,
    entry_id: str,
    namespace: str,
    revision: tuple[int, str] | None,
    **fields: object,
) -> bool:
    """Write *fields* to the row, and stamp ``last_synced_at`` only while it is still *revision*.

    *revision* is the ``(sync_seq, sync_hash)`` the published snapshot was taken at; ``None`` (a retry record
    queued before revisions were recorded) never stamps, so the row stays dirty and is pushed again rather
    than marked synced over an edit it never carried (B71-77). The compare and the write share one
    transaction; a backend without a real one (the YAML store) never stamps either, as ``mark_synced``
    refuses a conditional ack there. Returns whether the row was stamped.
    """
    from trw_memory.storage.interface import is_transactional

    with backend.transaction():
        if (current := backend.get(entry_id, namespace=namespace)) is None:
            return False
        # The hash too: store() numbers a revision from the writer's copy, so two contents can share a seq.
        clean = revision is not None and is_transactional(backend) and (current.sync_seq, current.sync_hash) == revision
        backend.update(
            entry_id,
            namespace=namespace,
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
        return served_view(row_to_entry(tuple(row))) if row is not None else None
    for candidate in backend.list_entries(namespace=namespace, limit=max(backend.count(namespace=namespace), 1)):
        if candidate.remote_id == remote_id or candidate.id in ids:
            return candidate
    return None


#: Remote ids, then ids, per SQL statement: well inside SQLite's bound-variable limit.
_FIND_MANY_CHUNK = 400


def find_synced_entries(
    backend: StorageBackend, namespace: str, remote_ids: list[str], ids: list[str]
) -> list[MemoryEntry]:
    """Every row in *namespace* whose ``remote_id`` is in *remote_ids* or whose id is in *ids*.

    :func:`find_synced_entry` for a whole pulled page in one statement per chunk; the caller maps each row back to
    the learning it answers. Namespace-qualified like the single find (PRD-CORE-245 P1).
    """
    conn = getattr(backend, "_conn", None)
    if conn is None:
        wanted_remote, wanted_ids = set(remote_ids), set(ids)
        everything = backend.list_entries(namespace=namespace, limit=max(backend.count(namespace=namespace), 1))
        return [e for e in everything if e.remote_id in wanted_remote or e.id in wanted_ids]
    from trw_memory.storage._row_mapper import row_to_entry

    found: dict[str, MemoryEntry] = {}
    for start in range(0, max(len(remote_ids), len(ids), 1), _FIND_MANY_CHUNK):
        remote_chunk = remote_ids[start : start + _FIND_MANY_CHUNK]
        id_chunk = ids[start : start + _FIND_MANY_CHUNK]
        if not remote_chunk and not id_chunk:
            continue
        remote_marks = ", ".join("?" for _ in remote_chunk) or "NULL"
        id_marks = ", ".join("?" for _ in id_chunk) or "NULL"
        sql = f"SELECT * FROM memories WHERE namespace = ? AND (remote_id IN ({remote_marks}) OR id IN ({id_marks}))"  # noqa: S608
        for row in conn.execute(sql, (namespace, *remote_chunk, *id_chunk)).fetchall():
            entry = served_view(row_to_entry(tuple(row)))
            found[entry.id] = entry
    return list(found.values())


def _encode_pulled(
    embedder: EmbeddingProvider | None, text: str
) -> tuple[list[float] | None, dict[str, VectorProvenance]]:
    """*text*'s vector and provenance, encoded before the write transaction; ``(None, {})`` when it cannot be.

    An encode failure is logged and the row lands without a vector, which ``memory reembed``
    backfills and coverage reports: one bad encode must not hold the pull cursor on that item.
    """
    from trw_memory.embeddings.provenance import generation_provenance_kwargs

    if embedder is None:
        return None, {}
    try:
        vector = embedder.embed(text)
    except Exception:  # justified: fail-open per pulled item, the row still lands and coverage shows the gap
        logger.warning("sync_apply_embed_failed", event_type="sync_team_merge", outcome="error", exc_info=True)
        return None, {}
    return vector, generation_provenance_kwargs(embedder, text, vector) if vector is not None else {}


def apply_synced_entry(
    backend: StorageBackend,
    config: MemoryConfig,
    entry: MemoryEntry,
    *,
    if_revision: str | None,
    synced: bool = True,
    embedder: EmbeddingProvider | None = None,
) -> tuple[str, str]:
    """Write a merged pulled row through the write gate and leave it synced, or dirty when *synced* is false.

    The write is conditional (PRD-CORE-308, B71-90): *if_revision* is the ``revision_of`` the row
    the caller merged from, ``None`` when it found none. A row that moved since (a local edit, a
    row created meanwhile) answers ``conflict`` and nothing is written; the caller re-reads.
    Returns ``("stored" | "quarantined" | "blocked" | "conflict" | "invalid", reason)``. A
    security refusal is a judged decision, not a store failure (PRD-FIX-138-FR01), so it is
    reported rather than raised.

    With *embedder* the row is encoded as ``memory_store`` encodes a row, and its vector
    commits with it, so a pulled learning is semantically recallable without a manual
    reembed. A text change with no vector to replace the old one drops the old one.
    """
    from trw_memory.exceptions import PIIBlockError, PoisoningError
    from trw_memory.security.runtime import prepare_entry_for_store, store_quarantined_entry
    from trw_memory.storage._shared import revision_of
    from trw_memory.storage.interface import is_transactional

    if not is_transactional(backend):  # the compare and the write must share one lock
        return "invalid", f"a conditional sync apply needs a transactional backend, not {type(backend).__name__}"
    try:
        decision = prepare_entry_for_store(entry, backend=backend, config=config, session_id=None)
    except (PoisoningError, PIIBlockError) as exc:
        return "blocked", getattr(exc, "reason", "") or type(exc).__name__
    if decision.quarantined:
        store_quarantined_entry(config, decision.entry)
        return "quarantined", ""
    row = decision.entry
    text = f"{row.content} {row.detail}"
    vector, proof = _encode_pulled(embedder, text)
    # One commit: an edit that lands between the write and its ack must not be marked clean (C12 rc7).
    with backend.transaction():
        current = backend.get(entry.id, namespace=entry.namespace)
        if revision_of(current) != if_revision:
            return "conflict", f"{entry.id} changed since it was read; nothing was written, re-read and retry"
        backend.store(row)
        if vector is not None:
            backend.upsert_vector(row.id, vector, namespace=row.namespace, **proof)
        elif current is not None and f"{current.content} {current.detail}" != text:
            backend.delete_vector(row.id, namespace=row.namespace)
        if synced:
            DeltaTracker.mark_synced([entry.id], backend, namespace=entry.namespace)
    return "stored", ""
