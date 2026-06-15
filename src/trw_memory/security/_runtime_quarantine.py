"""Quarantine store + review-workflow helpers for the runtime path.

Belongs to ``security/runtime.py``. Re-exported there for back-compat.

9 helpers covering the SEC-001 quarantine subsystem:

- ``store_quarantined_entry`` — persist a quarantined entry into the
  per-config quarantine SQLite DB with ``quarantined=true`` metadata
  + a ``quarantined`` review-log entry.
- ``list_quarantined_entries`` — return quarantined entries filtered
  by namespace + actor with ``updated_at`` reverse sort.
- ``delete_quarantined_entries`` — delete matching quarantined entries
  by namespace + actor + optional ``memory_id``.
- ``review_quarantined_entry`` — approve/reject + log + move
  approved entries into the active backend.
- ``get_status_history`` — return the SEC-001 review-log rows for
  a learning id.
- ``open_quarantine_backend`` — open the quarantine SQLiteBackend
  with the SEC-001 recovery policy.
- ``append_review_log`` — INSERT a review row into the quarantine
  reviews table (creates table on first call).
- ``quarantine_namespace_dir`` — per-namespace dir path under the
  config's quarantine root.
- ``read_namespace_metadata`` — read ``namespace.txt`` if present.

The runtime-path functions defer a lookup of
``ensure_security_maintenance`` via ``runtime`` to break the import
cycle.

Extracted as PRD-DIST-245 Phase 3 batch 100.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import structlog

from trw_memory.exceptions import QuarantineUnreachableError
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.security.startup import resolve_security_path
from trw_memory.storage.interface import StorageBackend
from trw_memory.storage.sqlite_backend import SQLiteBackend

logger = structlog.get_logger(__name__)

NAMESPACE_METADATA_FILE = "namespace.txt"


def _ensure_maintenance(config: MemoryConfig) -> None:
    from trw_memory.security import runtime as _runtime

    _runtime.ensure_security_maintenance(config)


def open_quarantine_backend(config: MemoryConfig) -> SQLiteBackend:
    path = resolve_security_path(config, "quarantine_db_path", create_parent=True)
    return SQLiteBackend(
        db_path=path,
        dim=config.embedding_dim,
        recovery_policy=config.memory_recovery_policy,
        corrupt_backup_keep=config.memory_corrupt_backup_keep,
        rebuild_from_cold=config.memory_recovery_rebuild_from_cold,
        recovery_inline_max_bytes=config.memory_recovery_inline_max_bytes,
    )


def quarantine_namespace_dir(config: MemoryConfig, namespace: str) -> Path:
    return Path(config.quarantine_path) / namespace.replace(":", "_")


def read_namespace_metadata(namespace_dir: Path) -> str | None:
    metadata_path = namespace_dir / NAMESPACE_METADATA_FILE
    if not metadata_path.exists():
        return None
    # Fail open: a quarantine namespace whose ``namespace.txt`` is unreadable
    # (OSError) or non-UTF-8 (UnicodeDecodeError from a torn/partial write)
    # yields ``None`` ("metadata absent") rather than raising into the
    # quarantine review/discovery path. Mirrors the storage-side seam in
    # ``integrations/_backend._read_namespace_metadata``. Never log the decoded
    # text or raw bytes — the namespace string can carry sensitive project
    # identifiers; only the path + error class.
    try:
        namespace = metadata_path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError) as exc:
        logger.warning(
            "namespace_metadata_read_failed",
            path=str(metadata_path),
            error=type(exc).__name__,
        )
        return None
    return namespace or None


def store_quarantined_entry(config: MemoryConfig, entry: MemoryEntry) -> None:
    """Persist a quarantined entry in the SEC-001 quarantine SQLite store."""
    try:
        with open_quarantine_backend(config) as backend:
            backend.store(
                entry.model_copy(
                    update={
                        "metadata": {
                            **entry.metadata,
                            "quarantined": "true",
                            "quarantined_at": datetime.now(timezone.utc).isoformat(),
                        }
                    }
                )
            )
            append_review_log(config, entry.id, "quarantined", reviewer_id="system")
    except OSError as exc:
        raise QuarantineUnreachableError(f"quarantine DB unavailable: {exc}") from exc


def list_quarantined_entries(
    config: MemoryConfig,
    *,
    namespace: str | None = None,
    actor: str | None = None,
    limit: int = 100,
) -> list[MemoryEntry]:
    """Return quarantined entries filtered by namespace and actor."""
    _ensure_maintenance(config)
    entries: list[MemoryEntry] = []
    with open_quarantine_backend(config) as backend:
        # Over-fetch with a large bound (matching the delete path) so the
        # actor/quarantined Python-side filter cannot silently drop matching
        # entries that sort beyond a small ``limit * 5`` window — an audit
        # truncation hazard (closure re-audit #2). The final ``[:limit]``
        # slice is applied AFTER the updated_at sort, so the newest matches win.
        for entry in backend.list_entries(namespace=namespace, limit=10_000):
            if entry.metadata.get("quarantined") != "true":
                continue
            if actor is not None and entry.source_identity != actor:
                continue
            entries.append(entry)
    entries.sort(key=lambda item: item.updated_at, reverse=True)
    return entries[:limit]


def delete_quarantined_entries(
    config: MemoryConfig,
    *,
    namespace: str,
    actor: str | None = None,
    memory_id: str | None = None,
) -> int:
    """Delete matching quarantined entries and return the count removed."""
    _ensure_maintenance(config)
    deleted = 0
    with open_quarantine_backend(config) as backend:
        if memory_id is not None:
            # Closure re-audit #1 + #6: the quarantine DB is a single SQLite
            # store keyed on config (NOT per-namespace), so a blind
            # ``backend.delete(memory_id)`` would let a caller scoped to one
            # namespace delete another namespace's row by id — and would also
            # delete a non-quarantined row that happens to live in the
            # quarantine DB. Fetch first and gate on both namespace match and
            # the ``quarantined=true`` flag (same flag set by
            # ``store_quarantined_entry``).
            entry = backend.get(memory_id)
            if entry is None:
                return 0
            if entry.namespace != namespace:
                return 0
            if entry.metadata.get("quarantined") != "true":
                return 0
            return 1 if backend.delete(memory_id) else 0
        for entry in backend.list_entries(namespace=namespace, limit=10_000):
            if actor is not None and entry.source_identity != actor:
                continue
            if entry.metadata.get("quarantined") != "true":
                continue
            if backend.delete(entry.id):
                deleted += 1
    return deleted


def append_review_log(
    config: MemoryConfig,
    learning_id: str,
    decision: str,
    *,
    reviewer_id: str,
) -> None:
    with open_quarantine_backend(config) as backend:
        conn = getattr(backend, "_conn", None)
        if conn is None:
            raise QuarantineUnreachableError("quarantine DB connection unavailable")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS quarantine_reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                learning_id TEXT NOT NULL,
                decision TEXT NOT NULL,
                reviewer_id TEXT NOT NULL,
                reviewed_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "INSERT INTO quarantine_reviews (learning_id, decision, reviewer_id, reviewed_at) VALUES (?, ?, ?, ?)",
            (learning_id, decision, reviewer_id, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()


def get_status_history(config: MemoryConfig, learning_id: str) -> list[dict[str, str]]:
    with open_quarantine_backend(config) as backend:
        conn = getattr(backend, "_conn", None)
        if conn is None:
            return []
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS quarantine_reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                learning_id TEXT NOT NULL,
                decision TEXT NOT NULL,
                reviewer_id TEXT NOT NULL,
                reviewed_at TEXT NOT NULL
            )
            """
        )
        rows = conn.execute(
            "SELECT decision, reviewer_id, reviewed_at FROM quarantine_reviews WHERE learning_id = ? ORDER BY id ASC",
            (learning_id,),
        ).fetchall()
    return [
        {"status": str(decision), "reviewer_id": str(reviewer_id), "ts": str(reviewed_at)}
        for decision, reviewer_id, reviewed_at in rows
    ]


def review_quarantined_entry(
    config: MemoryConfig,
    *,
    active_backend: StorageBackend,
    learning_id: str,
    decision: str,
    reviewer_id: str,
) -> dict[str, str]:
    if decision not in {"approve", "reject"}:
        raise ValueError("decision must be approve or reject")
    existing_history = get_status_history(config, learning_id)
    resolved_status = next(
        (item["status"] for item in existing_history if item.get("status") in {"active", "obsolete_poisoned"}),
        "",
    )
    if resolved_status:
        return {"learning_id": learning_id, "status": "already_resolved", "resolved_status": resolved_status}
    with open_quarantine_backend(config) as quarantine_backend:
        entry = quarantine_backend.get(learning_id)
        if entry is None:
            return {"learning_id": learning_id, "status": "not_found"}
        append_review_log(
            config,
            learning_id,
            "active" if decision == "approve" else "obsolete_poisoned",
            reviewer_id=reviewer_id,
        )
        if decision == "approve":
            approved = entry.model_copy(
                update={
                    "metadata": {
                        **entry.metadata,
                        "quarantined": "false",
                        "reviewed_by": reviewer_id,
                        "review_decision": "approve",
                    }
                }
            )
            active_backend.store(approved)
            quarantine_backend.delete(learning_id)
            return {"learning_id": learning_id, "status": "approved"}
        rejected = entry.model_copy(
            update={
                "metadata": {
                    **entry.metadata,
                    "reviewed_by": reviewer_id,
                    "review_decision": "reject",
                    "security_status": "obsolete_poisoned",
                }
            }
        )
        quarantine_backend.store(rejected)
        return {"learning_id": learning_id, "status": "rejected"}
