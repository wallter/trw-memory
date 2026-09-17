"""MCP tool: memory_maintain -- daemon-side maintenance -- PRD-CORE-279 FR08.

Decay, consolidation and WAL checkpointing are triggered by a *session* ending
in the framework host (``trw_deliver`` and friends). A daemon serving an
application has no sessions, so a store that only the daemon touches is never
maintained (field report sub_nq63Eql-xQ8IWsdL). This tool is that trigger, and
nothing more: it calls the three existing passes and records when it ran.

Two truths the response and the tool description must carry, because getting
them wrong would be worse than not having the tool:

**Scope differs per pass.** Consolidation is namespace-scoped. The importance
decay pass and the WAL checkpoint act on the whole STORE -- and on the daemon,
one store holds every namespace. Calling this for five namespaces therefore
runs one namespace's consolidation five times and the store-wide passes five
times too.

**A returned value is not a success.** ``memory_consolidate_impl`` reports an
error by returning ``{"status": "error"}`` and ``checkpoint_wal`` reports one as
``mode="error"``; neither raises. ``last_maintained_at`` advances only when
every pass reported success, while ``last_attempted_at`` advances always. A
caller reading only the timestamp still learns the truth.

The decay pass keys every read and write on ``(namespace, id)``, the table's
primary key, so a store that holds one entry id in more than one namespace
decays only the rows that qualify (fixed 2026-09-17; the pass used to select
and update by bare ``id``, and this tool refused to run it on such a store).
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from trw_memory.exceptions import ConfigError, StorageError
from trw_memory.models.config import MemoryConfig
from trw_memory.namespaces.validation import validate_namespace
from trw_memory.security.rbac import Permission, require_namespace_permission
from trw_memory.storage.persistence import lock_for_rmw
from trw_memory.tools._types import McpServer

if TYPE_CHECKING:  # pragma: no cover - typing only
    from trw_memory.storage.interface import StorageBackend

logger = structlog.get_logger(__name__)

__all__ = ["MAINTENANCE_STATE_FILE", "memory_maintain_impl", "register_maintain_tool"]

#: Where the per-namespace stamps live: beside the store, because the stamp
#: describes THAT store. A store replaced underneath a stale file is why the
#: record carries the store path it was written for.
MAINTENANCE_STATE_FILE = "maintenance.json"

_OK = "ok"
_SKIPPED = "skipped"
_ERROR = "error"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _state_path(backend: StorageBackend) -> Path | None:
    """Return the stamp file beside the backend's database, if it has one."""
    db_path = getattr(backend, "db_path", None)
    if db_path is None:
        return None
    return Path(db_path).parent / MAINTENANCE_STATE_FILE


def _read_state(path: Path) -> dict[str, object]:
    """Read the stamp file, or raise when it exists and cannot be trusted.

    "Absent" and "unreadable" are different answers and must not collapse into
    one. An empty mapping means nobody has run maintenance here; if a corrupt
    file returned the same thing, the next run would report "never maintained",
    overwrite the file, and take every other namespace's stamp with it.
    """
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise StorageError(
            f"maintenance state at {path} exists but cannot be read ({type(exc).__name__}). "
            f"Inspect it and remove it if it is corrupt; maintenance will recreate it."
        ) from exc
    if not isinstance(raw, dict):
        raise StorageError(f"maintenance state at {path} is not a JSON object; inspect and remove it.")
    return raw


def _namespace_stamp(backend: StorageBackend, namespace: str) -> dict[str, object]:
    """Return the recorded stamps for *namespace*, or an empty mapping."""
    path = _state_path(backend)
    if path is None:
        return {}
    with lock_for_rmw(path) as locked:
        entry = _read_state(locked).get(namespace)
    return entry if isinstance(entry, dict) else {}


def _record_stamp(
    backend: StorageBackend,
    namespace: str,
    *,
    attempted_at: str,
    succeeded: bool,
    passes: dict[str, object],
) -> dict[str, object]:
    """Write the attempt (and, on full success, the completion) under a lock."""
    path = _state_path(backend)
    if path is None:
        return {}
    with lock_for_rmw(path) as locked:
        state = _read_state(locked)
        entry = state.get(namespace)
        record: dict[str, object] = dict(entry) if isinstance(entry, dict) else {}
        record["store"] = str(getattr(backend, "db_path", ""))
        record["last_attempted_at"] = attempted_at
        record["last_passes"] = passes
        if succeeded:
            record["last_maintained_at"] = attempted_at
        state[namespace] = record
        locked.write_text(json.dumps(state, indent=2, sort_keys=True))
    return record


@contextlib.contextmanager
def _optional(lock: object) -> Iterator[None]:
    """Hold *lock* when there is one; otherwise do nothing."""
    if lock is None or not hasattr(lock, "__enter__"):
        yield
        return
    with lock:  # type: ignore[attr-defined]
        yield


def _run_decay(backend: StorageBackend) -> dict[str, object]:
    """Run the store-wide importance decay pass, or say why it was skipped."""
    from trw_memory.graph import memory_decay_pass

    conn = getattr(backend, "_conn", None)
    if conn is None:
        return {"status": _SKIPPED, "reason": "backend_not_sqlite"}
    lock = getattr(backend, "_lock", None)
    try:
        result = memory_decay_pass(conn, lock=lock)
    except (sqlite3.Error, ValueError) as exc:
        logger.warning("maintenance_decay_failed", error=str(exc))
        return {"status": _ERROR, "reason": type(exc).__name__}
    return {"status": _OK, "scope": "store", **result}


def _run_consolidation(namespace: str, backend: StorageBackend, config: MemoryConfig) -> dict[str, object]:
    """Run one consolidation cycle for *namespace*."""
    from trw_memory.tools.consolidate import memory_consolidate_impl

    try:
        result = memory_consolidate_impl(namespace, backend=backend, config=config)
    except Exception as exc:  # justified: one failing pass must not abort the others
        logger.warning("maintenance_consolidation_failed", namespace=namespace, error=str(exc))
        return {"status": _ERROR, "reason": type(exc).__name__}
    reported = str(result.get("status", ""))
    # ``consolidate_cycle`` catches PER-CLUSTER failures and still returns a
    # completed status with a populated ``errors`` list. A run where every
    # cluster failed would otherwise be reported as a successful maintenance.
    errors = result.get("errors") or []
    failed = reported in {"error", "invalid"} or bool(errors)
    return {
        "status": _ERROR if failed else _OK,
        "scope": "namespace",
        "clusters_found": result.get("clusters_found", 0),
        "entries_consolidated": result.get("entries_consolidated", 0),
        **({"errors": errors} if errors else {}),
        **({"reason": str(result.get("error", reported) or "cluster_errors")} if failed else {}),
    }


def _run_checkpoint(backend: StorageBackend) -> dict[str, object]:
    """Checkpoint the write-ahead log for the whole store."""
    try:
        result = dict(backend.checkpoint_wal())
    except Exception as exc:  # justified: one failing pass must not abort the others
        logger.warning("maintenance_checkpoint_failed", error=str(exc))
        return {"status": _ERROR, "reason": type(exc).__name__}
    # ``checkpoint_wal`` reports failure in the payload, not by raising: a
    # busy database or an error mode is a real failure to checkpoint.
    busy = result.get("busy", 0)
    failed = str(result.get("mode", "")).lower() == "error" or str(busy or "0") not in {"0", "False"}
    return {"status": _ERROR if failed else _OK, "scope": "store", **result}


def memory_maintain_impl(
    namespace: str,
    *,
    backend: StorageBackend,
    config: MemoryConfig | None = None,
) -> dict[str, object]:
    """Run decay, consolidation and a WAL checkpoint, and record the attempt.

    Args:
        namespace: Namespace to consolidate and to stamp.
        backend: Storage backend for the store being maintained.
        config: Optional config; constructed when omitted.

    Returns:
        ``{"namespace", "passes", "last_attempted_at", "last_maintained_at",
        "previous_maintained_at", "status"}``. ``status`` is ``"ok"`` only when
        every pass reported success.
    """
    try:
        validate_namespace(namespace)
    except ConfigError as exc:
        return {"error": str(exc), "status": "invalid"}

    cfg = config or MemoryConfig()
    require_namespace_permission(cfg, namespace, Permission.WRITE, "maintain")

    previous = _namespace_stamp(backend, namespace).get("last_maintained_at", "")
    attempted_at = _now()
    passes: dict[str, object] = {
        "decay": _run_decay(backend),
        "consolidation": _run_consolidation(namespace, backend, cfg),
        "wal_checkpoint": _run_checkpoint(backend),
    }
    succeeded = all(str(p.get("status")) != _ERROR for p in passes.values() if isinstance(p, dict))
    record = _record_stamp(
        backend,
        namespace,
        attempted_at=attempted_at,
        succeeded=succeeded,
        passes=passes,
    )
    logger.info(
        "memory_maintain",
        namespace=namespace,
        status=_OK if succeeded else _ERROR,
        decay=passes["decay"],
        consolidation=passes["consolidation"],
    )
    return {
        "namespace": namespace,
        "status": _OK if succeeded else _ERROR,
        "passes": passes,
        "last_attempted_at": attempted_at,
        "last_maintained_at": record.get("last_maintained_at", ""),
        "previous_maintained_at": previous,
    }


def register_maintain_tool(mcp: McpServer) -> None:
    """Register memory_maintain with a FastMCP server instance.

    Args:
        mcp: FastMCP server instance (imported lazily to keep fastmcp optional).
    """
    from trw_memory.daemon._offload import run_offloaded
    from trw_memory.integrations._backend import create_backend_from_config

    @mcp.tool()
    async def memory_maintain(namespace: str = "project:default") -> dict[str, object]:
        """Run memory maintenance now: decay, consolidation and a WAL checkpoint.

        Intended for a long-lived server, which has no session end to hang
        maintenance off. Cost is proportional to the store: consolidation
        embeds and clusters the namespace's entries, and the checkpoint rewrites
        the write-ahead log. Run it on a schedule (hourly or nightly), not per
        request.

        Scope is NOT uniform. Consolidation applies to *namespace*. The decay
        pass and the WAL checkpoint apply to the whole store, which on the
        loopback daemon is every namespace. Decay is skipped, with a reason,
        when the store holds one entry id in more than one namespace.

        Args:
            namespace: Namespace to consolidate and stamp.

        Returns:
            {"namespace": str, "status": "ok" | "error", "passes": {...},
             "last_attempted_at": str, "last_maintained_at": str,
             "previous_maintained_at": str}. last_maintained_at advances only
            when every pass succeeded; last_attempted_at always advances.
        """

        def _run() -> dict[str, object]:
            cfg = MemoryConfig()
            with create_backend_from_config(cfg, namespace) as backend:
                return memory_maintain_impl(namespace, backend=backend, config=cfg)

        return await run_offloaded(_run)
