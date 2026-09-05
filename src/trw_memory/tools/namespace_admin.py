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
from collections.abc import Iterator

import structlog

from trw_memory.exceptions import AuthorizationError, ConfigError, StorageError
from trw_memory.integrations._backend import (
    create_backend_from_config,
    resolve_backend_location,
)
from trw_memory.models.config import MemoryConfig
from trw_memory.namespaces.curate import (
    NamespaceStores,
    detect_moved_checkout,
    merge_namespace,
    rename_namespace,
    store_census,
)
from trw_memory.namespaces.identity import resolve_project_identity
from trw_memory.namespaces.validation import validate_namespace
from trw_memory.security.rbac import Permission, require_namespace_permission
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
) -> dict[str, object]:
    """Shared body: authorize both namespaces, then re-key."""
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
    try:
        with _open_stores(cfg, source, destination) as stores:
            result = operation(stores, source, destination)
    except ConfigError as exc:
        return {"error": str(exc), "status": "invalid"}
    except StorageError as exc:
        logger.warning("namespace_curate_failed", source=source, destination=destination, error=type(exc).__name__)
        return {"error": str(exc), "status": "error"}
    return dict(result.model_dump())


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
) -> dict[str, object]:
    """Re-label every row of *source* onto *destination*.

    Refuses when the destination already holds rows -- that case is a merge and
    the caller has to say so, which is what stops an accidental silent union.

    Returns:
        ``{source, destination, source_rows, moved, skipped, status}`` where
        status is ``renamed`` or ``noop``, or ``{error, status}`` for an
        invalid or unauthorized request.
    """
    return _curate_impl(source, destination, merge=False, config=config)


def memory_namespace_merge_impl(
    source: str,
    destination: str,
    *,
    config: MemoryConfig | None = None,
) -> dict[str, object]:
    """Fold *source* into *destination*, keeping the destination on a conflict.

    Returns:
        ``{source, destination, source_rows, moved, skipped, status}`` where
        status is ``merged`` or ``noop``. Skipped rows stay in the source: the
        merge never deletes a row it did not copy.
    """
    return _curate_impl(source, destination, merge=True, config=config)


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


def register_namespace_admin_tools(mcp: McpServer) -> None:
    """Register the two curate verbs and the moved-checkout diagnosis."""

    @mcp.tool()
    async def memory_namespace_rename(source: str, destination: str) -> dict[str, object]:
        """Re-label every row of one namespace onto another, refusing a merge."""

        return memory_namespace_rename_impl(source, destination)

    @mcp.tool()
    async def memory_namespace_merge(source: str, destination: str) -> dict[str, object]:
        """Fold one namespace into another, keeping the destination on conflicts."""

        return memory_namespace_merge_impl(source, destination)

    @mcp.tool()
    async def memory_namespace_diagnose(namespace: str = "") -> dict[str, object]:
        """Report a moved or renamed checkout and the command that repairs it."""

        return memory_namespace_diagnose_impl(namespace)
