"""MCP tools: namespace rename, merge and moved-checkout diagnosis.

PRD-CORE-253 FR05 (the two curate verbs) and FR01 (the detection that tells an
operator to run one). All three are served by the same loopback daemon and the
same token as every other tool, because a second surface would be a second
permission model.

Both write verbs check WRITE permission on **every** namespace they name,
before any row is touched -- a bulk re-key is exactly where a late permission
check turns into a half-completed move.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Callable, Iterator
from dataclasses import asdict
from pathlib import Path

import structlog

from trw_memory.daemon._lane import INTERACTIVE, MAINTENANCE, run_on_lane, run_slices
from trw_memory.exceptions import AuthorizationError, ConfigError, StorageError
from trw_memory.integrations._backend import (
    create_backend_from_config,
    resolve_backend_location,
)
from trw_memory.models.config import MemoryConfig
from trw_memory.namespaces.curate import (
    MoveProgress,
    NamespaceStores,
    detect_moved_checkout,
    merge_namespace,
    move_batch,
    rename_namespace,
    store_census,
)
from trw_memory.namespaces.identity import resolve_project_identity
from trw_memory.namespaces.validation import validate_namespace
from trw_memory.security.rbac import Permission, require_namespace_permission
from trw_memory.storage.interface import EntryCursor
from trw_memory.storage.persistence import lock_for_rmw
from trw_memory.tools._types import McpServer

__all__ = [
    "memory_namespace_diagnose_impl",
    "memory_namespace_merge_impl",
    "memory_namespace_rename_impl",
    "register_namespace_admin_tools",
]

logger = structlog.get_logger(__name__)


def _curate_impl(
    source: str,
    destination: str,
    *,
    merge: bool,
    config: MemoryConfig | None = None,
    batch: int | None = None,
) -> dict[str, object]:
    """Shared body: authorize both namespaces, then re-key the whole source, or its next *batch* rows."""
    cfg = config or MemoryConfig()
    try:
        validate_namespace(source)
        validate_namespace(destination)
        for namespace in (source, destination):
            require_namespace_permission(cfg, namespace, Permission.WRITE, "namespace curate")
    except ConfigError as exc:
        return {"error": str(exc), "status": "invalid"}
    except AuthorizationError as exc:
        return {"error": str(exc), "status": "forbidden"}

    operation = merge_namespace if merge else rename_namespace
    key = (merge, source, destination)
    try:
        with _open_stores(cfg, source, destination) as stores:
            if batch is None:
                return dict(operation(stores, source, destination).model_dump())
            starting = _MOVES.get(key) or _load_move_progress(stores, key) or MoveProgress(merge=merge)
            progress = _MOVES[key] = move_batch(stores, source, destination, starting, limit=batch)
            if progress.complete:
                _forget_move_progress(stores, key)
            else:
                _save_move_progress(stores, key, progress)
    except ConfigError as exc:
        return {"error": str(exc), "status": "invalid"}
    except StorageError as exc:
        logger.warning("namespace_curate_failed", source=source, destination=destination, error=type(exc).__name__)
        return {"error": str(exc), "status": "error"}
    if progress.complete:
        del _MOVES[key]
    return dict(progress.result(source, destination).model_dump())


#: The daemon's unfinished batched moves, by (merge, source, destination); touched only on the write lane.
#: An in-memory cache of the persisted state below -- CORE-331 FR03: a daemon restart used to forget
#: this dict entirely (a resumed merge rescanned from empty, a resumed rename was refused with a
#: pointer to merge); it now reloads from _MOVES_STATE_FILE on first use per key after a restart.
_MOVES: dict[tuple[bool, str, str], MoveProgress] = {}

#: Where a namespace's unfinished moves are persisted, beside the source store (maintain.py's
#: MAINTENANCE_STATE_FILE sibling pattern, one file per store rather than per move).
_MOVES_STATE_FILE = "namespace-moves.json"


def _move_key(key: tuple[bool, str, str]) -> str:
    merge, source, destination = key
    return f"{'merge' if merge else 'rename'}:{source}->{destination}"


def _moves_state_path(stores: NamespaceStores) -> Path | None:
    db_path = getattr(stores.source, "db_path", None)
    if db_path is None:
        return None
    return Path(db_path).parent / _MOVES_STATE_FILE


def _read_moves_state(path: Path) -> dict[str, object]:
    """Read the moves-state file, or raise when it exists and cannot be trusted (see maintain.py's
    ``_read_state``: "absent" and "unreadable" must not collapse into the same, silently-restarting
    answer)."""
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise StorageError(
            f"namespace-move state at {path} exists but cannot be read ({type(exc).__name__}). "
            f"Inspect it and remove it if it is corrupt; the next batch call will recreate it."
        ) from exc
    return raw if isinstance(raw, dict) else {}


def _as_int(value: object) -> int:
    return value if isinstance(value, int) else 0


def _progress_from_json(data: dict[str, object]) -> MoveProgress:
    cursor_raw = data.get("cursor")
    cursor = EntryCursor(**cursor_raw) if isinstance(cursor_raw, dict) else None
    return MoveProgress(
        merge=bool(data.get("merge", False)),
        started=bool(data.get("started", False)),
        complete=bool(data.get("complete", False)),
        source_rows=_as_int(data.get("source_rows")),
        moved=_as_int(data.get("moved")),
        skipped=_as_int(data.get("skipped")),
        pass_moved=_as_int(data.get("pass_moved")),
        cursor=cursor,
    )


def _load_move_progress(stores: NamespaceStores, key: tuple[bool, str, str]) -> MoveProgress | None:
    """The persisted progress for *key*, or ``None`` (nothing to resume): the daemon-restart path."""
    path = _moves_state_path(stores)
    if path is None:
        return None
    with lock_for_rmw(path) as locked:
        state = _read_moves_state(locked)
    raw = state.get(_move_key(key))
    return _progress_from_json(raw) if isinstance(raw, dict) else None


def _save_move_progress(stores: NamespaceStores, key: tuple[bool, str, str], progress: MoveProgress) -> None:
    path = _moves_state_path(stores)
    if path is None:
        return
    with lock_for_rmw(path) as locked:
        state = _read_moves_state(locked)
        state[_move_key(key)] = asdict(progress)
        locked.write_text(json.dumps(state, indent=2, sort_keys=True))


def _forget_move_progress(stores: NamespaceStores, key: tuple[bool, str, str]) -> None:
    """The move finished: drop its persisted entry so a restart never resumes a completed move."""
    path = _moves_state_path(stores)
    if path is None:
        return
    with lock_for_rmw(path) as locked:
        state = _read_moves_state(locked)
        if state.pop(_move_key(key), None) is not None:
            locked.write_text(json.dumps(state, indent=2, sort_keys=True))


@contextlib.contextmanager
def _open_stores(config: MemoryConfig, source: str, destination: str) -> Iterator[NamespaceStores]:
    """Open the source and destination stores, sharing one when they coincide.

    Under ``memory_single_store_path`` -- which the daemon always sets, and the
    daemon is what serves these verbs -- both namespaces resolve to one file and
    the whole move runs in one transaction. Opening that file twice would put two
    connections on a store whose WAL mitigation assumes one, so the shared case
    is DETECTED rather than assumed either way.

    The predicate is :func:`resolve_backend_location`, not the SQLite path: a
    YAML store keys a namespace on its entries DIRECTORY, so treating every
    non-SQLite config as "shared" made a cross-namespace YAML move read the
    destination's directory, find zero source rows and report a no-op. That was
    a silent wrong answer, which is the worst kind for a bulk re-key.
    """
    if resolve_backend_location(config, source) == resolve_backend_location(config, destination):
        with create_backend_from_config(config, destination) as shared:
            yield NamespaceStores.shared(shared)
        return
    with (
        create_backend_from_config(config, source) as source_store,
        create_backend_from_config(config, destination) as destination_store,
    ):
        yield NamespaceStores(source=source_store, destination=destination_store)


def memory_namespace_rename_impl(
    source: str,
    destination: str,
    *,
    config: MemoryConfig | None = None,
    batch: int | None = None,
) -> dict[str, object]:
    """Re-label every row of *source* onto *destination*.

    Refuses when the destination already holds rows -- that case is a merge and
    the caller has to say so, which is what stops an accidental silent union.

    Returns:
        ``{source, destination, source_rows, moved, skipped, status, complete}`` where
        status is ``renamed`` or ``noop`` (``moving`` with ``complete`` false while
        a *batch*-at-a-time move is unfinished), or ``{error, status}`` for an
        invalid or unauthorized request. Without *batch* the whole source moves.
    """
    return _curate_impl(source, destination, merge=False, config=config, batch=batch)


def memory_namespace_merge_impl(
    source: str,
    destination: str,
    *,
    config: MemoryConfig | None = None,
    batch: int | None = None,
) -> dict[str, object]:
    """Fold *source* into *destination*, keeping the destination on a conflict.

    Returns:
        As :func:`memory_namespace_rename_impl`, with status ``merged``.
        Skipped rows stay in the source: the
        merge never deletes a row it did not copy.
    """
    return _curate_impl(source, destination, merge=True, config=config, batch=batch)


def memory_namespace_diagnose_impl(
    namespace: str = "",
    *,
    config: MemoryConfig | None = None,
) -> dict[str, object]:
    """Report whether this checkout looks moved or renamed. Never writes.

    Args:
        namespace: Identity to check. Empty resolves the caller's FR01 project
            namespace from the working directory.
        config: Memory configuration.

    Returns:
        ``{"status": "ok", "namespace": str, "moved_checkout": null,
        "identity_source": str, "identity_degraded": null}`` when there is
        nothing to report, or the same shape with a
        ``MovedCheckoutObservation`` payload naming the populated same-slug
        siblings and the exact repair command.

        ``status`` is ``"degraded"`` when this checkout's canonical identity
        could not be established (git could not run AND the repository carried
        no readable on-disk evidence), with ``identity_degraded`` naming the
        reason. That case is reported rather than answered because the resolved
        namespace may not be the one this project's existing rows are under --
        which is precisely the moved/renamed confusion this tool exists to
        remove, so silently returning ``"ok"`` for it defeats the tool.
    """
    cfg = config or MemoryConfig()
    # Resolved unconditionally: whether THIS checkout's identity is trustworthy
    # is the diagnosis, independent of which namespace the caller asked about.
    identity = resolve_project_identity()
    resolved = namespace or identity.namespace
    try:
        validate_namespace(resolved)
        require_namespace_permission(cfg, resolved, Permission.READ, "namespace diagnose")
    except ConfigError as exc:
        return {"error": str(exc), "status": "invalid"}
    except AuthorizationError as exc:
        return {"error": str(exc), "status": "forbidden"}
    observation = detect_moved_checkout(resolved, store_census(cfg))
    return {
        "status": "degraded" if identity.degraded else "ok",
        "namespace": resolved,
        "moved_checkout": observation.model_dump() if observation is not None else None,
        "identity_source": identity.source,
        "identity_degraded": identity.degraded,
    }


async def _serve_move(impl: Callable[..., dict[str, object]], source: str, destination: str) -> dict[str, object]:
    """Move one batch per lane job (MAINTENANCE, the source as tenant), so other tenants' jobs run between
    batches, until the source drains or this call's ``CALL_SECONDS`` are spent; ``complete: false`` then
    says to call again, which resumes. Every batch re-authorizes both namespaces."""
    reply: dict[str, object] = {}

    def step(_: object) -> dict[str, object]:
        reply.clear()
        reply.update(impl(source, destination, batch=MAINTENANCE.rows))
        return {} if "complete" in reply else dict(reply)  # a reply without progress is a refusal

    refused = await run_slices(
        lambda body: run_on_lane(MAINTENANCE, source, body, None),
        step,
        lambda: reply.get("complete") is False,
        lambda: 0,  # bounded by the call's seconds; each batch by MAINTENANCE.rows
        max_rows=1,
    )
    return refused or reply


def register_namespace_admin_tools(mcp: McpServer) -> None:
    """Register the two curate verbs and the moved-checkout diagnosis."""

    @mcp.tool()
    async def memory_namespace_rename(source: str, destination: str) -> dict[str, object]:
        """Re-label every row of one namespace onto another, refusing a merge."""

        return await _serve_move(memory_namespace_rename_impl, source, destination)

    @mcp.tool()
    async def memory_namespace_merge(source: str, destination: str) -> dict[str, object]:
        """Fold one namespace into another, keeping the destination on conflicts."""

        return await _serve_move(memory_namespace_merge_impl, source, destination)

    @mcp.tool()
    async def memory_namespace_diagnose(namespace: str = "") -> dict[str, object]:
        """Report a moved or renamed checkout and the command that repairs it."""

        return await run_on_lane(INTERACTIVE, namespace, memory_namespace_diagnose_impl, namespace)
