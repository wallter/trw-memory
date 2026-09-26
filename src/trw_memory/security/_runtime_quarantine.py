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

The runtime-path functions defer a lookup of
``ensure_security_maintenance`` via ``runtime`` to break the import
cycle.

Extracted as PRD-DIST-245 Phase 3 batch 100.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone

import structlog

from trw_memory.exceptions import (
    PIIBlockError,
    QuarantineUnreachableError,
    SchemaValidationError,
    refuse_encryption_at_rest,
)
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.namespaces.validation import DEFAULT_NAMESPACE
from trw_memory.security._runtime_pii import apply_runtime_pii_policy
from trw_memory.security.poisoning import validate_entry_payload
from trw_memory.security.rbac import transport_grant
from trw_memory.security.startup import resolve_security_path
from trw_memory.storage.interface import StorageBackend
from trw_memory.storage.persistence import lock_for_rmw
from trw_memory.storage.sqlite_backend import SQLiteBackend

logger = structlog.get_logger(__name__)


def _ensure_maintenance(config: MemoryConfig) -> None:
    from trw_memory.security import runtime as _runtime

    _runtime.ensure_security_maintenance(config)


def open_quarantine_backend(config: MemoryConfig) -> SQLiteBackend:
    refuse_encryption_at_rest(config)
    path = resolve_security_path(config, "quarantine_db_path", create_parent=True)
    return SQLiteBackend(
        db_path=path,
        dim=config.embedding_dim,
        recovery_policy=config.memory_recovery_policy,
        corrupt_backup_keep=config.memory_corrupt_backup_keep,
        rebuild_from_cold=config.memory_recovery_rebuild_from_cold,
        recovery_inline_max_bytes=config.memory_recovery_inline_max_bytes,
    )


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
            append_review_log(config, entry.id, "quarantined", reviewer_id="system", namespace=entry.namespace)
    except OSError as exc:
        raise QuarantineUnreachableError(f"quarantine DB unavailable: {exc}") from exc


def list_quarantined_entries(
    config: MemoryConfig,
    *,
    namespace: str | None = None,
    actor: str | None = None,
    limit: int = 100,
    admits: Callable[[str], bool] | None = None,
) -> list[MemoryEntry]:
    """Return quarantined entries filtered by namespace and actor.

    Only namespaces inside the daemon token's grant, and that *admits* accepts
    when given, are ever read (PRD-CORE-298 FR02). Filtering fetched rows
    instead would pull other tenants' content into this process and let their
    newer rows starve ``limit``.
    """
    _ensure_maintenance(config)
    granted = transport_grant()
    entries: list[MemoryEntry] = []
    with open_quarantine_backend(config) as backend:
        scope = (
            [namespace]
            if namespace is not None
            else backend.list_namespaces(None if granted is None else sorted(granted))
        )
        for readable in scope:
            if (granted is not None and readable not in granted) or (admits is not None and not admits(readable)):
                continue
            # Over-fetch with a large bound (matching the delete path) so the
            # actor/quarantined Python-side filter cannot silently drop matching
            # entries that sort beyond a small ``limit * 5`` window — an audit
            # truncation hazard (closure re-audit #2). The final ``[:limit]``
            # slice is applied AFTER the updated_at sort, so the newest matches win.
            for entry in backend.list_entries(namespace=readable, limit=10_000):
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
            # store keyed on config (NOT per-namespace), so an unqualified
            # delete would let a caller scoped to one namespace delete another
            # namespace's row by id — and would also delete a non-quarantined
            # row that happens to live in the quarantine DB. Under PRD-CORE-245
            # FR03 the namespace predicate is carried by the read and the
            # delete themselves; the ``quarantined=true`` flag (same flag set
            # by ``store_quarantined_entry``) is still gated here.
            entry = backend.get(memory_id, namespace=namespace)
            if entry is None:
                return 0
            if entry.metadata.get("quarantined") != "true":
                return 0
            return 1 if backend.delete(memory_id, namespace=namespace) else 0
        for entry in backend.list_entries(namespace=namespace, limit=10_000):
            if actor is not None and entry.source_identity != actor:
                continue
            if entry.metadata.get("quarantined") != "true":
                continue
            if backend.delete(entry.id, namespace=entry.namespace):
                deleted += 1
    return deleted


#: DDL for the review-log table. ``namespace`` was added at schema 8 (Q3):
#: ``learning_id`` is caller-choosable and can collide across namespaces
#: (PRD-CORE-294), so a review keyed on ``learning_id`` alone let namespace A's
#: terminal decision block namespace B's own row with the same id, and leaked
#: A's reviewer identity to a caller asking about B's. See
#: ``trw_memory.storage._schema._migrate_v8_quarantine_review_namespace`` for
#: the forward migration that adds this column to a pre-existing table.
_CREATE_QUARANTINE_REVIEWS = """
CREATE TABLE IF NOT EXISTS quarantine_reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    learning_id TEXT NOT NULL,
    namespace TEXT NOT NULL DEFAULT '',
    decision TEXT NOT NULL,
    reviewer_id TEXT NOT NULL,
    reviewed_at TEXT NOT NULL
)
"""


def append_review_log(
    config: MemoryConfig,
    learning_id: str,
    decision: str,
    *,
    reviewer_id: str,
    namespace: str,
) -> None:
    with open_quarantine_backend(config) as backend:
        conn = getattr(backend, "_conn", None)
        if conn is None:
            raise QuarantineUnreachableError("quarantine DB connection unavailable")
        conn.execute(_CREATE_QUARANTINE_REVIEWS)
        conn.execute(
            "INSERT INTO quarantine_reviews (learning_id, namespace, decision, reviewer_id, reviewed_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (learning_id, namespace, decision, reviewer_id, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()


def get_status_history(config: MemoryConfig, learning_id: str, *, namespace: str) -> list[dict[str, str]]:
    with open_quarantine_backend(config) as backend:
        conn = getattr(backend, "_conn", None)
        if conn is None:
            return []
        conn.execute(_CREATE_QUARANTINE_REVIEWS)
        rows = conn.execute(
            "SELECT decision, reviewer_id, reviewed_at FROM quarantine_reviews "
            "WHERE learning_id = ? AND namespace = ? ORDER BY id ASC",
            (learning_id, namespace),
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
    namespace: str | None = None,
) -> dict[str, str]:
    if decision not in {"approve", "reject"}:
        raise ValueError("decision must be approve or reject")
    review_state_path = resolve_security_path(config, "quarantine_db_path", create_parent=True)
    with lock_for_rmw(review_state_path):
        return _review_quarantined_entry_locked(
            config,
            active_backend=active_backend,
            learning_id=learning_id,
            decision=decision,
            reviewer_id=reviewer_id,
            namespace=namespace,
        )


def _review_quarantined_entry_locked(
    config: MemoryConfig,
    *,
    active_backend: StorageBackend,
    learning_id: str,
    decision: str,
    reviewer_id: str,
    namespace: str | None,
) -> dict[str, str]:
    """Apply one review while the per-quarantine-store decision lock is held."""
    # PRD-CORE-245 FR03: ``None`` here means "the caller named no namespace",
    # which resolves to the default namespace rather than an unscoped read —
    # there is no unqualified read path left. The old pre-check that fetched
    # the ACTIVE row purely to reject a cross-namespace id is gone: the
    # quarantine read below is namespace-qualified, so it can no longer match a
    # foreign row in the first place.
    effective_namespace = namespace if namespace is not None else DEFAULT_NAMESPACE
    with open_quarantine_backend(config) as quarantine_backend:
        entry = quarantine_backend.get(learning_id, namespace=effective_namespace)
        existing_history = get_status_history(config, learning_id, namespace=effective_namespace)
        resolved_status = next(
            (item["status"] for item in existing_history if item.get("status") in {"active", "obsolete_poisoned"}),
            "",
        )
        if resolved_status:
            return {"learning_id": learning_id, "status": "already_resolved", "resolved_status": resolved_status}
        if entry is None:
            return {"learning_id": learning_id, "status": "not_found"}
        if decision == "approve":
            # Q1: a quarantined row that reached this queue via a caller-forged
            # ``metadata.quarantined`` flag (rather than a genuine trust-score
            # or anomaly hold) was never subjected to schema/PII checks. Re-run
            # both here so a reviewer's approval cannot promote unscanned
            # content — rate-limit and anomaly scoring are skipped: they need
            # server-side session context this delayed, out-of-band review does not
            # have, and re-running them against "now" would judge the entry by
            # a window it was never actually written in.
            try:
                validate_entry_payload(
                    entry,
                    max_chars=config.max_entry_chars,
                    min_evidence_items_for_verified=config.min_evidence_items_for_verified,
                )
                revalidated, _pii_matches = apply_runtime_pii_policy(entry, config)
            except (SchemaValidationError, PIIBlockError) as exc:
                append_review_log(
                    config, learning_id, "approve_blocked", reviewer_id=reviewer_id, namespace=effective_namespace
                )
                return {"learning_id": learning_id, "status": "blocked", "reason": str(exc)}
            # Adversarial audit 2026-09-24: a plain ``active_backend.store``
            # here would silently overwrite whatever now lives at
            # (namespace, id) — a legitimate write made to that id while this
            # row sat in the review queue, or a caller who chose a colliding
            # id specifically to land on approve. Refuse rather than clobber;
            # the reviewer can re-run under a different id once the conflict
            # is resolved. The check and the store share one write transaction
            # (BEGIN IMMEDIATE on SQLite), so no writer can land between them (C12).
            approved = revalidated.model_copy(
                update={
                    "metadata": {
                        **revalidated.metadata,
                        "quarantined": "false",
                        "reviewed_by": reviewer_id,
                        "review_decision": "approve",
                    }
                }
            )
            if getattr(type(active_backend), "transaction", None) is StorageBackend.transaction:
                # No atomic check-and-insert here (YAML): an approval could overwrite a racing write.
                return {"learning_id": learning_id, "status": "unsupported_backend"}
            with active_backend.transaction() as txn:
                conflict = txn.get(learning_id, namespace=effective_namespace) is not None
                if not conflict:
                    txn.store(approved)
            if conflict:
                append_review_log(
                    config, learning_id, "approve_conflict", reviewer_id=reviewer_id, namespace=effective_namespace
                )
                return {"learning_id": learning_id, "status": "conflict"}
            quarantine_backend.delete(learning_id, namespace=effective_namespace)
            append_review_log(config, learning_id, "active", reviewer_id=reviewer_id, namespace=effective_namespace)
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
        append_review_log(
            config, learning_id, "obsolete_poisoned", reviewer_id=reviewer_id, namespace=effective_namespace
        )
        return {"learning_id": learning_id, "status": "rejected"}
