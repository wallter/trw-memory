"""MCP tool: memory_maintain -- daemon-side maintenance -- PRD-CORE-279 FR08.

Decay, consolidation and WAL checkpointing are triggered by a *session* ending
in the framework host (``trw_deliver`` and friends). A daemon serving an
application has no sessions, so a store that only the daemon touches is never
maintained (field report sub_nq63Eql-xQ8IWsdL). This tool is that trigger, and
nothing more: it calls the existing passes and records when it ran. The
verification pass (PRD-CORE-294 FR07(b)) is the same
``trw_memory.lifecycle.verification_pass.run_maintain_verify`` trw-mcp's
``maintain-verify`` calls, and runs only when ``project_root`` is configured.

Two truths the response and the tool description must carry, because getting
them wrong would be worse than not having the tool:

**Scope differs per pass.** Consolidation and decay are namespace-scoped; the
WAL checkpoint acts on the whole STORE -- and on the daemon, one store holds
every namespace. Calling this for five namespaces therefore runs one
namespace's consolidation and decay five times and the store-wide checkpoint
five times too. The decay pass narrows to *this call's* namespace, intersected
with the token's grant when one applies (PRD-CORE-298 FR02, narrowed from the
whole grant by PRD-CORE-307 FR05 so each namespace's maintain advances its own
resumable cursor rather than every namespace racing over the same store-wide
one -- breaking; see UPGRADE-NOTES-8.0.0.md). The WAL checkpoint is a file
operation that reads and changes no row.

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

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from trw_memory.exceptions import StorageError
from trw_memory.models.config import MemoryConfig
from trw_memory.security.rbac import Permission, transport_grant, transport_root
from trw_memory.storage.persistence import lock_for_rmw
from trw_memory.tools._types import McpServer

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable

    from trw_memory.lifecycle.consolidation import ClusterWrite
    from trw_memory.models.memory import MemoryEntry
    from trw_memory.storage.interface import StorageBackend

logger = structlog.get_logger(__name__)

__all__ = ["GRAPH_BACKFILL_PAGE_MAX", "MAINTENANCE_STATE_FILE", "register_maintain_tool"]

#: The most rows one ``memory_graph_backfill`` call reads (trw-mcp's list page).
GRAPH_BACKFILL_PAGE_MAX = 10_000

#: Where the per-namespace stamps live: beside the store, because the stamp
#: describes THAT store. A store replaced underneath a stale file is why the
#: record carries the store path it was written for.
MAINTENANCE_STATE_FILE = "maintenance.json"
#: The most unfinished verification sweeps a namespace's stamp keeps (one per root and verify settings).
VERIFY_SWEEPS_KEPT = 8

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
    attempted_at: str | None,
    succeeded: bool = False,
    passes: dict[str, object] | None = None,
    sweep: dict[str, object] | None = None,
    sweep_key: str = "",
    decay_next: list[str] | None = None,
    decay_ran: bool = False,
) -> dict[str, object]:
    """Write the attempt (and, on full success, the completion) under a lock, with the unfinished
    verification *sweep* under *sweep_key* to resume (``None`` once it finished). Without *attempted_at* (a
    ``memory_verify``) only the sweep is written. *decay_ran* (PRD-CORE-307 FR05) records *decay_next*,
    the decay pass's persisted keyset cursor (``None`` when it wrapped to the start)."""
    path = _state_path(backend)
    if path is None:
        return {}
    with lock_for_rmw(path) as locked:
        state = _read_state(locked)
        entry = state.get(namespace)
        record: dict[str, object] = dict(entry) if isinstance(entry, dict) else {}
        record["store"] = str(getattr(backend, "db_path", ""))
        if attempted_at is not None:
            record["last_attempted_at"], record["last_passes"] = attempted_at, passes
        if succeeded:
            record["last_maintained_at"] = attempted_at
        if decay_ran:
            record["decay_next"] = decay_next
        found = record.pop("verify_sweeps", None)
        sweeps = {key: value for key, value in (found if isinstance(found, dict) else {}).items() if key != sweep_key}
        if sweep is not None:
            sweeps[sweep_key] = {**sweep, "at": _now()}
        # The newest few unfinished sweeps are kept (one per root and settings); an abandoned one ages out.
        # The one just written sorts newest whatever the clock says (B71-92).
        if kept := sorted(
            sweeps.items(),
            key=lambda item: (item[0] == sweep_key, str(item[1].get("at", "")) if isinstance(item[1], dict) else ""),
        ):
            record["verify_sweeps"] = dict(kept[-VERIFY_SWEEPS_KEPT:])
        state[namespace] = record
        locked.write_text(json.dumps(state, indent=2, sort_keys=True))
    return record


def _run_decay(backend: StorageBackend, namespace: str, config: MemoryConfig | None = None) -> dict[str, object]:
    """Run the importance decay pass over *namespace* (intersected with the token's grant, if any),
    resuming its persisted keyset cursor and advancing it after this pass (PRD-CORE-307 FR05), or say
    why the pass was skipped. *config* supplies ``decay_cutoff_days``/``decay_batch_size``
    (PRD-CORE-331 FR10 B71-135h); ``None`` falls back to ``MemoryConfig()`` defaults (90d, 1000 rows)."""
    from trw_memory.graph import memory_decay_pass

    conn = getattr(backend, "_conn", None)
    if conn is None:
        return {"status": _SKIPPED, "reason": "backend_not_sqlite"}
    cfg = config or MemoryConfig()
    lock = getattr(backend, "_lock", None)
    granted = transport_grant()
    scope = {namespace} if granted is None else {namespace} & set(granted)
    raw_cursor = _namespace_stamp(backend, namespace).get("decay_next")
    cursor = (str(raw_cursor[0]), str(raw_cursor[1])) if isinstance(raw_cursor, list) and len(raw_cursor) == 2 else None
    try:
        result = memory_decay_pass(
            conn, cfg.decay_cutoff_days, cfg.decay_batch_size, lock=lock, namespaces=scope, cursor=cursor
        )
    except (sqlite3.Error, ValueError) as exc:
        logger.warning("maintenance_decay_failed", error=str(exc))
        return {"status": _ERROR, "reason": type(exc).__name__}
    next_cursor = result.get("next")
    _record_stamp(
        backend,
        namespace,
        attempted_at=None,
        decay_next=next_cursor if isinstance(next_cursor, list) else None,
        decay_ran=True,
    )
    return {"status": _OK, "scope": "namespace", **result}


def _run_consolidation(
    namespace: str, backend: StorageBackend, config: MemoryConfig, lane: Callable[[ClusterWrite], str] | None = None
) -> dict[str, object]:
    """Run one consolidation cycle for *namespace* (each cluster's writes on *lane*, if given)."""
    from trw_memory.tools.consolidate import memory_consolidate_impl

    try:
        result = memory_consolidate_impl(namespace, backend=backend, config=config, lane=lane)
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
        **({"clusters_skipped": result["clusters_skipped"]} if "clusters_skipped" in result else {}),
        **({"errors": errors} if errors else {}),
        **({"reason": str(result.get("error", reported) or "cluster_errors")} if failed else {}),
    }


def _run_security_maintenance() -> dict[str, object]:
    """Drain audit logs queued while ``security_maintenance_inline`` was ``False`` (B71-97: this
    queue previously had no drainer, so its bounded deque silently dropped compaction work once
    full)."""
    from trw_memory.security.runtime import drain_security_maintenance

    try:
        result = drain_security_maintenance()
    except Exception as exc:  # justified: one failing pass must not abort the others
        logger.warning("maintenance_security_drain_failed", error=str(exc))
        return {"status": _ERROR, "reason": type(exc).__name__}
    return {"status": _OK, "scope": "store", **result}


def _graph_backfill_lane_write(
    namespace: str,
) -> Callable[[MemoryEntry, list[float] | None, MemoryConfig | None], dict[str, object]]:
    """CORE-331 FR04: ``memory_graph_backfill`` serves off the write lane (``exclusive=False``), so its
    per-row edge write raced a forget/update/another writer touching the same row -- the off-lane
    writer census this closes. Each row's write goes through the lane instead, re-read fresh first
    so a row that changed since the page was listed is skipped (content-hashed, PRD-CORE-308's
    ``revision_of``) rather than overwritten from stale data."""
    from trw_memory import graph
    from trw_memory._client_store import _existing_entry_for_namespace
    from trw_memory.storage._shared import revision_of
    from trw_memory.tools.entry import lane_step

    def write(entry: MemoryEntry, embedding: list[float] | None, config: MemoryConfig | None) -> dict[str, object]:
        expected = revision_of(entry)

        def step(fresh_backend: StorageBackend) -> dict[str, object]:
            if revision_of(_existing_entry_for_namespace(fresh_backend, entry.id, namespace)) != expected:
                return {"status": "stale"}
            return {
                "status": "ok",
                "built": graph.update_entry_graph(entry, fresh_backend, embedding=embedding, config=config),
            }

        try:
            result = lane_step(namespace, "graph_backfill", step).result()
        except Exception as exc:  # justified: one row's failure must not abort the page
            return {"status": "error", "reason": type(exc).__name__}
        return result if isinstance(result, dict) else {"status": "error", "reason": "refused"}

    return write


def _run_snapshot(backend: StorageBackend, config: MemoryConfig, *, now: datetime | None = None) -> dict[str, object]:
    """Take the rolling snapshots: one daily per UTC day, and the weekly one on Sunday (UF-MEM-05).

    Snapshot rotation ran only from a manual CLI before; nothing took an automatic restore point. Store-wide, like
    the checkpoint, and never raises: a failing snapshot is reported in the pass (``status: error``), and like any failed
    pass it keeps this run from advancing ``last_maintained_at``; the other passes still run. A day (or Sunday's week)
    that already has its snapshot is not re-taken: a ``VACUUM INTO`` of the whole store on every maintain call would be
    the cost of a backup per call. Pruning follows ``memory_snapshot_daily_keep`` / ``memory_snapshot_weekly_keep``.

    The files land under ``<store dir>/memory/snapshots/`` (the layout ``backup restore`` reads); the default
    ``trw-memory snapshot`` CLI derives its directory from a namespace, so it does not list them yet (open).
    """
    from trw_memory.storage import _snapshot

    raw = str(getattr(backend, "db_path", "") or "")
    if not raw or raw == ":memory:":
        return {"status": _SKIPPED, "scope": "store", "reason": "no store file"}
    db_path = Path(raw)
    base_dir = db_path.parent  # the layout the snapshot CLI and `backup restore` read: <base>/memory/snapshots
    moment = now or datetime.now(timezone.utc)
    try:
        today = _snapshot.snapshots_base_dir(base_dir) / "daily" / f"{moment.strftime('%Y-%m-%d')}.db"
        if today.is_file():
            daily = "already_taken"
        else:
            _snapshot.take_daily_snapshot(base_dir, db_path, config.memory_snapshot_daily_keep, now=moment)
            daily = "taken"
        iso_year, iso_week, _ = moment.isocalendar()
        week_file = _snapshot.snapshots_base_dir(base_dir) / "weekly" / f"{iso_year:04d}-W{iso_week:02d}.db"
        weekly = (
            None
            if week_file.is_file()
            else _snapshot.take_weekly_snapshot(base_dir, db_path, config.memory_snapshot_weekly_keep, now=moment)
        )
    except Exception as exc:  # justified: one failing pass must not abort the others
        logger.warning("maintenance_snapshot_failed", error=str(exc))
        return {"status": _ERROR, "scope": "store", "reason": type(exc).__name__}
    return {"status": _OK, "scope": "store", "daily": daily, "weekly": "taken" if weekly is not None else "not_due"}


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


def _run_verification(
    namespace: str,
    backend: StorageBackend,
    config: MemoryConfig,
    *,
    after: tuple[str, str] | None = None,
    seconds: float | None = None,
    rows: int | None = None,
    **knobs: Any,
) -> dict[str, object]:
    """Verify *namespace*'s assertions/anchors against ``config.project_root``, resuming *after* a
    position and stopping after about *seconds* (``complete`` and ``next`` say where it stopped).

    Without a usable root nothing is checked: no entry becomes verified and
    any prior "verified" verdict is cleared, since it can no longer be re-checked.
    """
    from trw_memory.lifecycle import verification_pass

    root = Path(config.project_root) if config.project_root else None
    usable = root is not None and root.is_dir()
    try:
        # Runs even without a usable root: the sweep then clears any prior
        # "verified" verdict it can no longer re-check. An in-process caller's own root
        # is resolved once here, so a symlinked path (macOS /tmp) survives the no-follow
        # anchor walk. A grant's root was resolved at mint and is never re-resolved: that
        # would follow a root or ancestor swapped for a symlink after the grant (C12).
        if root is not None and usable and not transport_root()[0]:
            root = root.resolve()
        summary = verification_pass.run_maintain_verify(
            backend,
            project_root=root if usable else None,
            namespace=namespace,
            after=after,
            seconds=seconds,
            max_rows=rows,
            **knobs,
        )
    except Exception as exc:  # justified: one failing pass must not abort the others
        logger.warning("maintenance_verification_failed", namespace=namespace, error=str(exc))
        return {"status": _ERROR, "reason": type(exc).__name__}
    counts: dict[str, object] = {
        **summary.as_dict(),
        "complete": summary.resume_after is None,
        "next": None if summary.resume_after is None else list(summary.resume_after),
    }
    # Sweep failures outrank the no-root skip: the verdict-clearing sweep ran
    # either way, and a skip would report ok and advance last_maintained_at.
    # A configured root that is missing is a misconfiguration, not a skip.
    checks = (
        ("entry_failures", counts["entry_failures"]),
        ("persist_failures", counts["persist_failures"]),
        ("project_root_not_a_directory", root is not None and not usable),
        ("project_root_unwalkable", counts["root_unwalkable"]),
    )
    failure = next((reason for reason, failed in checks if failed), "")
    if root is None and not failure:
        return {"status": _SKIPPED, "reason": "no project_root", **counts}
    return {
        "status": _ERROR if failure else _OK,
        "scope": "namespace",
        **counts,
        **({"reason": failure} if failure else {}),
    }


class ConsolidationPolicy(BaseModel):
    """One project's consolidation settings, validated with the ranges ``MemoryConfig`` enforces."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool
    similarity_threshold: float = Field(ge=0.0, le=1.0)
    min_cluster: int = Field(ge=2)
    max_per_cycle: int = Field(gt=0)

    def config_fields(self) -> dict[str, object]:
        """The ``MemoryConfig`` fields this policy sets."""
        return {
            "consolidation_enabled": self.enabled,
            "consolidation_similarity_threshold": self.similarity_threshold,
            "consolidation_min_cluster": self.min_cluster,
            "consolidation_max_per_cycle": self.max_per_cycle,
        }


def maintain_config(consolidation: dict[str, object] | None) -> MemoryConfig | dict[str, object]:
    """The daemon's config with the caller's consolidation *policy*, verifying the root a grant
    records (never the daemon's own); or why the policy is refused."""
    cfg = MemoryConfig()
    if consolidation is not None:
        try:
            policy = ConsolidationPolicy.model_validate(consolidation)
        except ValidationError as exc:
            return {"error": f"invalid consolidation policy: {exc}", "status": "invalid"}
        cfg = cfg.model_copy(update=policy.config_fields())
    on_transport, granted_root = transport_root()
    if on_transport:
        cfg = cfg.model_copy(update={"project_root": granted_root or ""})
    return cfg


def register_maintain_tool(mcp: McpServer) -> None:
    """Register memory_maintain with a FastMCP server instance.

    Args:
        mcp: FastMCP server instance (imported lazily to keep fastmcp optional).
    """
    from trw_memory.tools.entry import serve_namespace

    @mcp.tool()
    async def memory_maintain(
        namespace: str = "project:default", consolidation: dict[str, object] | None = None
    ) -> dict[str, object]:
        """Run memory maintenance now: decay, consolidation, verification, WAL checkpoint.

        Intended for a long-lived server, which has no session end to hang
        maintenance off. Cost is proportional to the store: consolidation
        embeds and clusters the namespace's entries, and the checkpoint rewrites
        the write-ahead log. Run it on a schedule (hourly or nightly), not per
        request.

        Scope is NOT uniform. Consolidation applies to *namespace*, and so does
        decay (intersected with the token's grant, resuming its own persisted
        cursor); the WAL checkpoint applies to the whole store, which on the
        loopback daemon is every namespace. Verification re-checks the
        namespace's stored assertions against MEMORY_PROJECT_ROOT; without a
        root it is skipped and verdicts stay unknown.

        One call verifies a bounded part of a large namespace (about a minute or
        10,000 rows), then returns; call again to continue. While one maintain
        of a namespace runs, another returns {"status": "busy"}.

        Args:
            namespace: Namespace to consolidate and stamp.
            consolidation: The caller's project policy for the consolidation pass --
                ``enabled``, ``similarity_threshold``, ``min_cluster``, ``max_per_cycle``.
                The daemon's config is process-wide, so one project's policy travels
                with its request (PRD-CORE-302 FR03). Omitted: the daemon's defaults.

        Returns:
            {"namespace": str, "status": "ok" | "error", "passes": {...},
             "last_attempted_at": str, "last_maintained_at": str,
             "previous_maintained_at": str}. passes["verification"] carries
            "complete": bool and "next": [namespace, id] | null. While complete
            is false the sweep stopped at its bound; the next call resumes after
            "next". last_maintained_at advances only when every pass succeeded
            and the sweep completed; last_attempted_at always advances.
        """

        from trw_memory.tools._maintain_sweep import serve_maintain

        return await serve_maintain(namespace, consolidation)

    @mcp.tool()
    async def memory_graph_backfill(
        namespace: str,
        after: dict[str, str] | None = None,
        limit: int = 500,
        deadline_seconds: float | None = None,
    ) -> dict[str, object]:
        """Build the knowledge-graph edges of one page of *namespace*'s existing rows, listed after *after*.

        The resume point is the caller's: pass back ``next`` until ``complete``.
        *deadline_seconds* is a soft budget counted from before the page is read:
        once spent no further row starts, so at most one row's enrichment overruns
        it, and the next call resumes from ``next``.
        Returns {"status", "processed", "edges_built", "skipped", "failed", "next", "complete"}.
        """
        from trw_memory.graph import backfill_graph_page
        from trw_memory.storage.interface import EntryCursor

        try:
            cursor = EntryCursor(**after) if after else None
        except TypeError as exc:
            return {"error": f"invalid cursor: {exc}", "status": "invalid"}
        if not 1 <= limit <= GRAPH_BACKFILL_PAGE_MAX or (deadline_seconds is not None and deadline_seconds < 0):
            return {"error": f"invalid page: limit={limit}, deadline_seconds={deadline_seconds}", "status": "invalid"}
        return await serve_namespace(
            namespace,
            Permission.WRITE,
            "graph_backfill",
            lambda backend, config: {
                "status": _OK,
                **backfill_graph_page(
                    backend,
                    namespace,
                    after=cursor,
                    limit=limit,
                    deadline_seconds=deadline_seconds,
                    config=config,
                    write=_graph_backfill_lane_write(namespace),
                ),
            },
            exclusive=False,
        )
