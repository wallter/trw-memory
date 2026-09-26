"""MCP tool: memory_get -- one entry by ``(namespace, id)`` (PRD-CORE-298 FR01).

trw-mcp's store reads an entry by id over the daemon with this verb; a correction
goes through ``memory_update`` (``tools/update.py``, PRD-CORE-294 FR03). It goes
through the namespace grant like every other tool, and a cross-namespace id
answers exactly like a missing one.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

import structlog

from trw_memory._dir_trust import open_anchored_walk, open_component_fd
from trw_memory.exceptions import ConfigError, UntrustedDirectoryError
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.namespaces.validation import validate_namespace
from trw_memory.security.rbac import Permission, require_namespace_permission, transport_root
from trw_memory.storage.interface import StorageBackend
from trw_memory.tools._types import McpServer

logger = structlog.get_logger(__name__)


def _log_refused(operation: str, path: str | None, reason: str) -> None:
    """One structured log line naming the path and reason for a checkout-boundary refusal -- PRD-SEC-016 NFR04.

    NFR04's own text: "A refusal reply and its log line name the path and
    the reason." Before this helper existed, only ONE of this module's three
    ``refused()`` closures ever produced a log event at all -- the symlink-
    escape path, indirectly, via ``_dir_trust.py::_refuse``'s
    ``dir_open_refused`` -- so a ``..``-traversal or an outside-the-granted-
    checkout refusal returned straight away with NO log line. Every
    ``refused()`` closure in this module now calls here first, so NFR04
    holds for every refusal cause this module can produce, not only the one
    that happened to reach ``_dir_trust``. *reason* is always a pre-built
    string describing WHY (never the refused file's bytes or content).
    """
    logger.error(
        "checkout_boundary_refused", operation=operation, path=path if path is not None else "<checkout>", reason=reason
    )


def _refused(operation: str, path: str | None, reason: str) -> dict[str, object]:
    """The one ``{"status": "refused", ...}`` reply shape, logged first (NFR04)."""
    _log_refused(operation, path, reason)
    return {"error": f"{operation} refused: {reason}", "status": "refused"}


def _anchor_relative_parts(root: str, path: str, operation: str) -> tuple[int, tuple[str, ...]] | dict[str, object]:
    """*path*'s components relative to *root*, plus an open no-follow anchor fd on *root* itself.

    Shared front matter for :func:`open_checkout_file_fd` and
    :func:`open_checkout_parent_fd`: both walk from the SAME anchored,
    no-follow-verified root and both refuse the same three malformed-input
    shapes (a ``..`` component, a path outside the granted checkout, an empty
    path) the same way -- see either caller's own docstring for why each
    check exists. The anchor fd is caller-owned (``os.close()`` it, or let
    the caller's own walk close it as it unwinds).
    """
    granted = Path(root)
    wanted = Path(path)
    if ".." in wanted.parts:
        # See ``open_checkout_file_fd``'s docstring: a relative *path* skips the
        # ``relative_to`` containment check below, so ``..`` is checked up front.
        return _refused(operation, path, f"{path} contains a '..' component")
    try:
        relative = wanted.relative_to(granted) if wanted.is_absolute() else wanted
    except ValueError:
        return _refused(operation, path, f"{path} is not inside the granted checkout {root}")
    parts = relative.parts
    if not parts:
        return _refused(operation, path, "no path given")
    try:
        anchor = open_anchored_walk(granted)
    except (OSError, UntrustedDirectoryError) as exc:
        return _refused(operation, path, f"the checkout root {granted} could not be opened securely ({exc})")
    return anchor, parts


def refused_namespace(
    namespace: str, permission: Permission, operation: str, config: MemoryConfig
) -> dict[str, object] | None:
    """An ``invalid`` result for a malformed *namespace*; raises ``AuthorizationError`` outside the grant."""
    try:
        validate_namespace(namespace)
    except ConfigError as exc:
        return {"error": str(exc), "status": "invalid"}
    require_namespace_permission(config, namespace, permission, operation)
    return None


def checkout_path(path: str | None, operation: str, *, within: bool) -> str | None | dict[str, object]:
    """*path* as this request may use it, or a ``refused`` result.

    Over the transport a file-reading tool is bounded by the checkout its token
    was minted for: *within* admits a path inside it, otherwise only the checkout
    itself (``None`` means the checkout). A grant recording no checkout reaches
    no file. Off the transport -- the in-process SDK -- *path* is used as given.

    *root* (the grant's checkout) is used EXACTLY as the grant recorded it --
    never re-resolved here. ``daemon/_grants.py::mint_grant`` already resolved
    it ONCE, at mint time (``str(root.resolve())``), which is what makes a
    legitimately symlinked home keep working (macOS ``/var`` -> ``/private/var``,
    a symlinked ``~/.trw``) without this function re-walking the filesystem on
    every request. Re-resolving *root* here (PRD-SEC-016 round-2 finding 1) let
    a filesystem change AFTER mint time -- an ancestor of *root* swapped for a
    symlink by a DIFFERENT tenant's grant, if one checkout is nested inside
    another -- silently redirect where "the granted checkout" points, because
    ``Path.resolve()`` always follows symlinks with no boundary of its own.
    """
    on_transport, root = transport_root()
    if not on_transport:
        return path

    if root is None:
        return _refused(
            operation, path, "this token's grant records no checkout; run `trw-mcp memory token` to re-mint it"
        )
    granted = Path(root)
    if path is None:
        return None if within else str(granted)
    try:
        wanted = Path(path).resolve()
    except OSError as exc:
        # A component swapped mid-resolve: CPython's non-strict realpath lstat()s a
        # symlink, then readlink()s it unguarded, so one removed in between raises.
        return _refused(operation, path, f"{path} changed while it was being resolved ({type(exc).__name__})")
    if wanted.is_relative_to(granted) if within else wanted == granted:
        return str(wanted)
    return _refused(operation, path, f"{path} is not {'inside ' if within else ''}the granted checkout {root}")


def open_checkout_file_fd(root: str, path: str, operation: str) -> int | dict[str, object]:
    """An open, no-follow fd for *path*, walked component-by-component from *root* -- PRD-SEC-016 FR02/FR03/FR05.

    This is THE checkout-bound opener: every served tool that reads a caller-named
    filesystem path SHALL reach it only through here (enforced by a source scan,
    ``tests/test_tool_registration_interop.py``), never by handing a resolved
    string to ``open()``/``Path.read_text()``/``sqlite3.connect()`` directly.

    ``checkout_path()`` already proved *path* resolves inside *root* at
    validation time; this function does NOT trust that resolution -- it re-walks
    *path*'s components from a directory descriptor on *root*, opening each one
    with ``O_NOFOLLOW``. A component swapped for a symlink between the earlier
    check and this call (the TOCTOU race FR02 exists for) is refused here, not
    missed, because the walk consults the live filesystem object at each step,
    never a cached resolved string. The final component must be a regular file.

    *root* itself is walked the SAME no-follow, component-by-component way
    (:func:`trw_memory._dir_trust.open_anchored_walk`), not opened as one
    resolved path string: a single ``os.open(full_root_path, O_NOFOLLOW)``
    only refuses a symlink at *root*'s own final component, leaving every
    ancestor of *root* to resolve the ordinary way. That matters here because
    this daemon can serve nested checkouts for different tenants -- a request
    holding only an outer checkout's grant could otherwise swap an ancestor of
    an INNER checkout's root (a path inside the outer one, which its own grant
    permits writing to) and redirect where the inner root resolves.

    *root* is used EXACTLY as the grant recorded it -- ``Path(root)``, never
    re-resolved with a further ``.resolve()`` call. ``daemon/_grants.py::mint_grant``
    already resolved it once, at mint time; calling ``.resolve()`` again here
    (PRD-SEC-016 round-2 finding 1) re-walks *root*'s ENTIRE path through the
    kernel's ordinary (symlink-following) resolution BEFORE the protected
    ``open_anchored_walk`` below ever runs, so a swapped ancestor is silently
    accepted as part of "the granted checkout" and the no-follow walk then
    faithfully walks the WRONG, already-redirected destination. The mint-time
    resolve is what keeps a legitimately symlinked home working (macOS
    ``/var`` -> ``/private/var``); nothing after that point may resolve *root*
    again.

    Returns the open descriptor (caller-owned; close it with
    :func:`trw_memory._live_stores.close_reader_fd`, which releases its read lease) or a
    ``{"status": "refused", ...}`` reply naming the reason -- never raises.
    """

    # A relative *path* skips the ``relative_to`` containment check
    # (:func:`_anchor_relative_parts`) entirely -- ``..`` is a real, non-symlink
    # directory entry in every directory, so the no-follow walk would not
    # catch it either. Every existing caller only ever hands this function an
    # absolute, already-resolved string, so this never fires for them; it
    # exists for a caller (e.g. PRD-SEC-016 round-2 finding 2's anchor-file
    # reader) that passes a RELATIVE, less-trusted string straight through.
    result = _anchor_relative_parts(root, path, operation)
    if isinstance(result, dict):
        return result
    anchor, parts = result
    opened: list[int] = [anchor]
    try:
        for index, part in enumerate(parts):
            last = index == len(parts) - 1
            fd = open_component_fd(opened[-1], part, directory=not last)
            opened.append(fd)
        leaf = opened.pop()
        return leaf
    except UntrustedDirectoryError as exc:
        return _refused(operation, path, f"{path} could not be opened without following a symlink ({exc})")
    finally:
        for fd in opened:
            os.close(fd)


def open_checkout_parent_fd(root: str, path: str, operation: str) -> tuple[int, str] | dict[str, object]:
    """*path*'s parent directory, opened no-follow and anchored on *root* -- PRD-SEC-016 round-7 finding 2.

    A sibling-of-*path* existence probe (the import's WAL/SHM sidecar check,
    ``checkout_import.py::_private_checkout_copy``) must not reach *path*'s
    parent by re-resolving *path*'s own string a second time -- that reopens
    exactly the ancestor-swap window :func:`open_checkout_file_fd` closes for
    the leaf itself. This walks the same way (:func:`open_anchored_walk` from
    *root*, then :func:`open_component_fd` per remaining component), stopping
    one short of the leaf: the caller gets an open, no-follow-verified
    directory descriptor plus the leaf's bare name, and probes a SIBLING of
    that name directly off this fd, never through a path string rooted
    anywhere outside it.

    Returns ``(parent_fd, leaf_name)`` (caller-owned; ``os.close()`` the fd)
    or a ``{"status": "refused", ...}`` reply -- never raises.
    """

    result = _anchor_relative_parts(root, path, operation)
    if isinstance(result, dict):
        return result
    anchor, parts = result
    opened: list[int] = [anchor]
    try:
        for part in parts[:-1]:
            fd = open_component_fd(opened[-1], part, directory=True)
            opened.append(fd)
        parent_fd = opened.pop()
        return parent_fd, parts[-1]
    except UntrustedDirectoryError as exc:
        return _refused(operation, path, f"{path} could not be opened without following a symlink ({exc})")
    finally:
        for fd in opened:
            os.close(fd)


_Result = TypeVar("_Result")
_Run = Callable[[StorageBackend, MemoryConfig], _Result]


def in_namespace(
    namespace: str, permission: Permission, operation: str, run: _Run[_Result]
) -> _Result | dict[str, object]:
    """Validate and authorize *namespace*, then open its backend and *run*.

    A daemon tool serves through this, so a malformed or ungranted namespace is
    refused before any backend is created for it (PRD-CORE-298 FR02).
    """
    from trw_memory.integrations._backend import create_backend_from_config

    config = MemoryConfig()
    if refused := refused_namespace(namespace, permission, operation, config):
        return refused
    with create_backend_from_config(config, namespace) as backend:
        return run(backend, config)


async def serve_namespace(
    namespace: str, permission: Permission, operation: str, run: _Run[_Result], *, exclusive: bool = True
) -> _Result | dict[str, object]:
    """:func:`in_namespace` on the daemon's worker pool, the way every served tool body runs (C12 rc4).

    Each body reads SQLite, and some load the embedding model or read checkout files; on the event
    loop, one caller's slow body stalled every other tenant's request. *exclusive* keeps the bodies
    that loop ran one at a time mutually exclusive (see ``run_serialized``); ``False`` is for those
    that already ran on the pool.
    """
    from trw_memory.daemon._offload import run_offloaded, run_serialized

    submit = run_serialized if exclusive else run_offloaded
    return await submit(in_namespace, namespace, permission, operation, run)


def _scoped_entry(
    memory_id: str,
    namespace: str,
    permission: Permission,
    operation: str,
    backend: StorageBackend,
    config: MemoryConfig,
) -> MemoryEntry | dict[str, object]:
    if refused := refused_namespace(namespace, permission, operation, config):
        return refused
    entry = backend.get(memory_id, namespace=namespace)
    if entry is None or entry.namespace != namespace:
        return {"status": "not_found"}
    return entry


def memory_get_impl(
    memory_id: str, namespace: str, *, backend: StorageBackend, config: MemoryConfig | None = None
) -> dict[str, object]:
    """Return ``{"status": "ok", "entry": <json>}`` or ``{"status": "not_found"}``."""
    entry = _scoped_entry(memory_id, namespace, Permission.READ, "get", backend, config or MemoryConfig())
    if isinstance(entry, dict):
        return entry
    return {"status": "ok", "entry": entry.model_dump(mode="json")}


def memory_find_duplicate_impl(
    namespace: str, content: str, detail: str, *, backend: StorageBackend, config: MemoryConfig | None = None
) -> dict[str, object]:
    """``{"status": "ok", "entry_id": <id or None>}``: an ACTIVE row of *namespace* with exactly this content."""
    if refused := refused_namespace(namespace, Permission.READ, "find_duplicate", config or MemoryConfig()):
        return refused
    return {"status": "ok", "entry_id": backend.find_active_by_content(content, detail, namespace=namespace)}


def register_entry_tools(mcp: McpServer) -> None:
    """Register memory_get and memory_find_duplicate with a FastMCP server instance."""

    async def memory_get(memory_id: str, namespace: str) -> dict[str, object]:
        """Return one memory entry by id within *namespace*."""
        return await serve_namespace(
            namespace, Permission.READ, "get", lambda b, c: memory_get_impl(memory_id, namespace, backend=b, config=c)
        )

    async def memory_find_duplicate(namespace: str, content: str, detail: str) -> dict[str, object]:
        """Id of an ACTIVE entry in *namespace* whose content and detail match exactly, or null."""
        return await serve_namespace(
            namespace,
            Permission.READ,
            "find_duplicate",
            lambda b, c: memory_find_duplicate_impl(namespace, content, detail, backend=b, config=c),
        )

    mcp.tool()(memory_get)
    mcp.tool()(memory_find_duplicate)
