"""MCP tool: fold a checkout's project store into its namespace (PRD-CORE-280 FR03).

``trw-mcp memory migrate --to user`` never opens the user store while the daemon
serves it: it hands the daemon a working copy of the checkout's project store (a
file inside the checkout its token was minted for), and ``memory_import_checkout``
merges that copy's ``default`` rows, vectors and graph edges into the granted
namespace. The destination wins an id collision; one whose row or vector differs from
the copy's in anything the store did not assign, or holds a vector only one side has,
is refused by id, since the copy's row would not be imported. The compare and the merge share one transaction on each store,
so a refused import leaves the namespace, and the copy, as they were. An import that
outlives ``IMPORT_DEADLINE_SECONDS``, waiting for the write lock included, is rolled back
and answers ``busy``. The destination commits first: if the copy's commit then fails the
answer is ``uncertain`` (rerun it), never a claimed rollback. A rerun after a lost reply
or a killed caller moves nothing twice. The reply counts what the namespace now
holds of the migrated ids, which the caller checks before it cuts over.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import functools
import os
import secrets
import shutil
import sqlite3
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO, NamedTuple

from trw_memory._dir_trust import NOFOLLOW_SUPPORTED, create_private_file_fd
from trw_memory._inode_pin import Identity, current_identity, pinned_identity
from trw_memory._live_stores import close_reader_fd, sqlite_read_lock
from trw_memory.daemon._paths import IMPORT_TMP_SUBDIR, DaemonPaths, _harden_dir
from trw_memory.exceptions import UntrustedDirectoryError
from trw_memory.models.config import MemoryConfig
from trw_memory.security.rbac import Permission, transport_root
from trw_memory.storage._connection import mark_verified
from trw_memory.storage.interface import StorageBackend
from trw_memory.storage.sqlite_backend import SQLiteBackend
from trw_memory.tools._checkout_merge import plan_import, write_import
from trw_memory.tools._types import McpServer
from trw_memory.tools.entry import (
    checkout_path,
    in_namespace,
    open_checkout_file_fd,
    open_checkout_parent_fd,
    refused_namespace,
)

#: How long the import may hold the daemon's write lock before it gives up and rolls back.
IMPORT_DEADLINE_SECONDS = 10.0

#: The private copy's bounds (C12): a project store far past these is refused, not copied,
#: so an oversized or endless source cannot fill the daemon's filesystem. A checkout's store is its
#: own learnings (1.5 KiB of vector each at 384 dimensions): 512 MiB holds far more than one checkout
#: writes, and a local disk copies it in seconds (rc9: the budgets were 2 GiB and 120 s).
IMPORT_COPY_MAX_BYTES = 512 * 1024**2
IMPORT_COPY_DEADLINE_SECONDS = 30.0

#: How many ids one import may ask about: the reply counts what the namespace holds of them, one read
#: each. ``trw-mcp memory migrate`` sends every id of the checkout's store in one call.
IMPORT_MAX_IDS = 100_000

#: Runs the import's write step on the daemon's write lane, with a destination backend of its own.
Lane = Callable[[Callable[[SQLiteBackend], dict[str, object]]], dict[str, object]]
_COPY_CHUNK = 1024 * 1024


def memory_import_checkout_impl(
    namespace: str,
    source_path: str,
    ids: list[str],
    *,
    backend: StorageBackend,
    deadline_seconds: float = IMPORT_DEADLINE_SECONDS,
    lane: Lane | None = None,
) -> dict[str, object]:
    """``{"status": "ok", "moved", "skipped", "held": {"rows", "vectors", "edges"}}`` -- held among *ids* --
    plus ``vectors_not_carried``: the ids whose vectors are from another embedding space (dimension), which
    the namespace cannot hold; their rows moved, and ``memory_reembed`` rebuilds the vectors.

    Everything that reads the copy runs on the caller's thread: its schema check and open, the read of
    every row and the compare with the namespace. Only the write goes through *lane* (the daemon's
    one-thread write lane, called at most once with a destination backend of its own), so a crafted or
    large copy costs its own caller's time, not every tenant's (rc9). Each phase gets *deadline_seconds*.
    """
    if not Path(source_path).is_file():
        return {"error": f"no project store at {source_path}", "status": "invalid"}
    if not isinstance(backend, SQLiteBackend):
        return {"error": "memory_import_checkout needs a SQLite store to compare vectors", "status": "invalid"}
    if not hasattr(sqlite3.Connection, "setlimit"):  # Python 3.10: no SQLITE_LIMIT_LENGTH cap on the copy
        return {"error": "memory_import_checkout needs Python 3.11 or later", "status": "unsupported_runtime"}
    path = Path(source_path)
    try:
        plan = plan_import(path, backend, namespace, deadline_seconds)
        if isinstance(plan, dict):
            return plan
        step = functools.partial(write_import, path, namespace, plan, deadline_seconds)
        written = lane(step) if lane is not None else step(backend)
    finally:
        mark_verified(path, verified=False)  # each import's copy is a new file: keep no mark for it
    if written.get("status") != "ok":
        return written
    wanted = set(ids)  # each id once
    held = {
        "rows": sum(1 for entry_id in wanted if backend.get(entry_id, namespace=namespace) is not None),
        "vectors": len(backend.existing_vector_ids(namespace=namespace) & wanted) if backend.supports_vectors() else 0,
        "edges": sum(1 for edge in backend.graph_edges(namespace) if edge.source_id in wanted),
    }
    carried = {"vectors_not_carried": plan.other_space} if plan.other_space else {}  # another embedding space
    return {"status": "ok", "moved": written["moved"], "skipped": written["skipped"], "held": held, **carried}


#: The files a SQLite connection reopens alongside the main database: an uncheckpointed WAL
#: (and its index) holds writes the main file lacks, and a rollback journal is either hot
#: (the main file needs rolling back) or a writer is mid-transaction. Copying the main file
#: alone would silently drop or tear those writes, so any of them refuses the import (FR02,
#: C12). This is conservative: a clean PERSIST/TRUNCATE-mode journal is refused too.
_SIDECAR_SUFFIXES = ("-wal", "-shm", "-journal")

#: The private copy's file name inside its own per-import directory (see ``_private_checkout_copy``).
_COPY_NAME = "source.db"


def _sidecar_present_no_follow(parent_fd: int, name: str) -> bool:
    """Whether *name* exists directly inside *parent_fd*, without ever following a symlink there.

    PRD-SEC-016 round-7 finding 2: the old check, ``Path(source_path +
    suffix).exists()``, followed ordinary (symlink-following) path
    resolution -- a symlinked sidecar entry pointing OUTSIDE the checkout
    turned "does this sidecar exist" into "does <attacker-chosen path>
    exist," an existence oracle over any path the daemon user can stat, with
    no race required at all. This probes the same bare name anchored on the
    already no-follow-verified parent directory descriptor instead: a
    symlink sitting there is refused outright (never followed to learn
    whether ITS target exists), any other object there is reported present,
    and ``ENOENT`` (nothing there) is the only "not present" answer.

    NFR02: this platform's caller (``open_checkout_file_fd``) already refuses
    up front, before this ever runs, when ``NOFOLLOW_SUPPORTED`` is False --
    there is no silent downgrade to a following ``os.open`` here either.
    """
    if not NOFOLLOW_SUPPORTED:
        return True
    # A stat, not an open: closing a descriptor on a live store's sidecar would
    # release that store's SQLite locks in this process (C15).
    try:
        os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:  # trw-fail-silent-allow: ENOENT is the one legitimate "no sidecar" answer this probe distinguishes from "present" (symlink or regular file, handled by the OSError branch below); the caller's loop already treats every OTHER outcome as a refusal, so this is not a silently-swallowed error
        return False
    except OSError:
        # Any unexpected error is "present" -- the fail-closed answer (refuses
        # the import); "absent" would not be. A symlink stats as present too.
        return True
    return True


def _quiet_source_state(fd: int, parent_fd: int, leaf_name: str) -> tuple[int, int, int] | str:
    """*fd*'s (size, mtime_ns, ctime_ns), or why the source is not a quiet, single-file database.

    The sidecars are probed through *parent_fd*, which was walked separately from *fd*, so
    the name under it must still be *fd*'s own inode first, or the probes describe another file.
    """
    st = os.fstat(fd)
    try:
        named = os.stat(leaf_name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError:
        named = None
    if named is None or (named.st_dev, named.st_ino) != (st.st_dev, st.st_ino):
        return f"{leaf_name} is no longer the file that was opened"
    if st.st_nlink != 1:
        # SQLite keeps sidecars beside whichever name a writer opened, so another name's WAL
        # would be invisible from here.
        return f"{leaf_name} has {st.st_nlink} hard links; import a store that has exactly one name"
    for suffix in _SIDECAR_SUFFIXES:
        if _sidecar_present_no_follow(parent_fd, leaf_name + suffix):
            return f"{leaf_name}{suffix} sidecar present; checkpoint and close the source first"
    return st.st_size, st.st_mtime_ns, st.st_ctime_ns


class _PrivateCopy(NamedTuple):
    """A private import copy's path, its pinned identity, and its wall time.

    ``pin`` holds the identity (see :mod:`trw_memory._inode_pin`) until the caller
    closes it, so a file swapped in at ``path`` can never share it.
    """

    path: str
    identity: Identity
    copy_seconds: float
    pin: contextlib.ExitStack


def _private_checkout_copy(root: str, source_path: str, operation: str) -> _PrivateCopy | dict[str, object]:
    """A daemon-private (0700 dir, 0600 file) byte-copy of *source_path*, opened via the FR02 walk.

    *source_path* is never opened by name after this call returns: the merge
    that follows reads only the private copy. Refused when *source_path*
    carries a WAL, SHM or rollback-journal sidecar before or after the copy, when
    a writer holds its SQLite lock, when it changed during the copy (the copy runs
    under SQLite's own SHARED lock, so a rollback-mode writer cannot commit meanwhile),
    or when :func:`open_checkout_file_fd` refuses the walk itself
    (symlink component, missing file, non-regular leaf, or -- NFR02 -- a
    platform without ``dir_fd``/``O_NOFOLLOW`` support).

    The returned identity is pinned while the new file's descriptor is still
    open, and stays pinned until the caller closes ``pin`` -- the caller checks
    it again, by path, immediately before (and after) the later SQLite open,
    narrowing (never eliminating: SQLite opens by path, not by fd) the window
    between this function's own fd closing and that open (round-2 finding 3).

    C12-R: each copy gets its OWN directory under ``IMPORT_TMP_SUBDIR``, removed whole when the
    import ends. Opening the copy can write beside it -- a pre-migration schema snapshot under
    ``backups/``, WAL/SHM sidecars, a corrupt-store rotation -- and removing only the file kept
    each of those forever, one more per retry. The caller's own store is the backup that matters.
    """
    opened = open_checkout_file_fd(root, source_path, operation)
    if isinstance(opened, dict):
        return opened
    fd = opened
    dir_fd = -1
    dest_fd = -1
    work_dir: Path | None = None
    copied_ok = False
    source_parent_fd = -1
    pin = contextlib.ExitStack()
    try:
        if os.fstat(fd).st_size > IMPORT_COPY_MAX_BYTES:
            raise OSError(f"{source_path} is larger than the {IMPORT_COPY_MAX_BYTES}-byte import limit")
        # PRD-SEC-016 round-7 finding 2: the sidecar check is anchored on the
        # SAME no-follow-verified parent this leaf came from, not a re-resolved
        # path string -- see ``open_checkout_parent_fd`` and
        # ``_sidecar_present_no_follow`` for why a symlinked sidecar entry is
        # refused regardless of what it points at, without ever following it.
        parent_result = open_checkout_parent_fd(root, source_path, operation)
        if isinstance(parent_result, dict):
            return parent_result
        source_parent_fd, leaf_name = parent_result
        before = _quiet_source_state(fd, source_parent_fd, leaf_name)
        if isinstance(before, str):
            return {"error": f"{operation} refused: {before}", "status": "refused"}
        # Under the daemon's OWN user_memory_dir, never TMPDIR (shared with every other process
        # on the box; PRD-SEC-016 round-2 finding 3), in a directory unique to this call and
        # hardened to 0700 and owner-checked by _harden_dir; the file is created with
        # O_CREAT|O_EXCL|O_NOFOLLOW anchored on that directory's verified fd. The residual
        # window is the accepted G4 risk: the same user or root, who can open the store anyway.
        # Named for its owning process, so a starting daemon removes only a dead owner's copy.
        import_tmp = DaemonPaths.resolve(create=True).user_memory_dir / IMPORT_TMP_SUBDIR
        os.close(_harden_dir(import_tmp))  # the parent too: a symlinked import-tmp is refused, never followed
        work_dir = import_tmp / f"{os.getpid()}-{secrets.token_hex(16)}"
        dir_fd = _harden_dir(work_dir)
        dest_fd = create_private_file_fd(dir_fd, _COPY_NAME, mode=0o600)
        # Pinned while the new file is still open, so nothing can have swapped it yet.
        identity = pin.enter_context(pinned_identity(_COPY_NAME, dir_fd=dir_fd))
        created = os.fstat(dest_fd)
        if identity != (created.st_dev, created.st_ino):
            raise OSError(f"{work_dir / _COPY_NAME} was replaced while it was being created")
        copy_started = time.monotonic()
        # The source stays open (closefd=False) so it is closed through close_reader_fd,
        # which releases its read lease (C15).
        with sqlite_read_lock(fd):
            with os.fdopen(fd, "rb", closefd=False) as source_file, os.fdopen(dest_fd, "wb") as out:
                dest_fd = -1  # os.fdopen now owns the destination descriptor
                copied = _bounded_copy(source_file, out, copy_started)
            # Rechecked under the lock: a WAL writer, a new journal, or a writer that ignores
            # SQLite's locks would have changed one of these (C12).
            after = _quiet_source_state(fd, source_parent_fd, leaf_name)
        if after != before or copied != before[0]:
            reason = after if isinstance(after, str) else "the source changed during the copy; retry once it is idle"
            return {"error": f"{operation} refused: {reason}", "status": "refused"}
        copy_seconds = time.monotonic() - copy_started
        copied_ok = True
        return _PrivateCopy(str(work_dir / _COPY_NAME), identity, copy_seconds, pin.pop_all())
    except (OSError, UntrustedDirectoryError) as exc:
        return {"error": f"{operation} refused: could not create a private copy ({exc})", "status": "refused"}
    finally:
        pin.close()
        if dest_fd != -1:
            os.close(dest_fd)
        if work_dir is not None and not copied_ok:  # a failed copy never outlives its refusal (C12)
            shutil.rmtree(work_dir, ignore_errors=True)
        if dir_fd != -1:
            os.close(dir_fd)
        if fd != -1:
            close_reader_fd(fd)
        if source_parent_fd != -1:
            os.close(source_parent_fd)


def _bounded_copy(source: BinaryIO, out: BinaryIO, started: float) -> int:
    """Copy *source* to *out*, refusing past ``IMPORT_COPY_MAX_BYTES`` or ``IMPORT_COPY_DEADLINE_SECONDS``."""
    copied = 0
    while chunk := source.read(_COPY_CHUNK):
        copied += len(chunk)
        if copied > IMPORT_COPY_MAX_BYTES:
            raise OSError(f"the source grew past the {IMPORT_COPY_MAX_BYTES}-byte import limit")
        if time.monotonic() - started > IMPORT_COPY_DEADLINE_SECONDS:
            raise OSError(f"the copy took longer than {IMPORT_COPY_DEADLINE_SECONDS:.0f}s")
        out.write(chunk)
    return copied


def register_checkout_import_tools(mcp: McpServer) -> None:
    """Register memory_import_checkout with a FastMCP server instance."""

    async def memory_import_checkout(namespace: str, source_path: str, ids: list[str]) -> dict[str, object]:
        """Merge the project store at *source_path* (inside the granted checkout) into *namespace*.

        *ids* (at most ``IMPORT_MAX_IDS``, 100,000, counted as sent) are the ids the reply's ``held``
        counts, each once. The cap is that high because ``trw-mcp memory migrate`` sends every id of the
        checkout's store in one call, and the reads it costs run off the write lane, on this caller's time.
        """
        from trw_memory.daemon._offload import run_offloaded

        # Off the loop and off the write lane: the copy, its checks, its reads and the compare are the
        # untrusted work, and they cost this caller's time alone (rc9).
        lane = _serialized_lane(namespace, asyncio.get_running_loop())
        return await run_offloaded(_import_checkout, namespace, source_path, ids, lane)

    def _import_checkout(namespace: str, source_path: str, ids: list[str], lane: Lane) -> dict[str, object]:
        if refused := refused_namespace(namespace, Permission.WRITE, "import_checkout", MemoryConfig()):
            return refused
        source = checkout_path(source_path, "memory_import_checkout", within=True)
        if not isinstance(source, str):
            return source or {"error": "memory_import_checkout needs a source_path", "status": "invalid"}

        on_transport, root = transport_root()
        if not on_transport or root is None:
            # In-process SDK path: no token, no boundary to defend (PRD-SEC-016 non-goal). A
            # rootless grant on transport was already refused by checkout_path() above; `root
            # is None` here is unreachable in that case but keeps this branch total for mypy.
            return in_namespace(
                namespace,
                Permission.WRITE,
                "import_checkout",
                lambda backend, _config: memory_import_checkout_impl(
                    namespace, source, ids, backend=backend, lane=lane
                ),
            )

        copied = _private_checkout_copy(root, source, "memory_import_checkout")
        if isinstance(copied, dict):
            return copied
        try:
            # Narrows (does not eliminate: SQLite opens by path, not fd) the
            # window between this function's own descriptors closing and the
            # SQLite open inside memory_import_checkout_impl -- the same pinned
            # before/after check connect_registered gives the daemon's own store
            # open (round-2 finding 4); this is its FR02 sibling.
            if current_identity(copied.path) != copied.identity:
                return {
                    "error": "memory_import_checkout refused: the private copy was replaced before it could be used",
                    "status": "refused",
                }
            result = in_namespace(
                namespace,
                Permission.WRITE,
                "import_checkout",
                lambda backend, _config: memory_import_checkout_impl(
                    namespace, copied.path, ids, backend=backend, lane=lane
                ),
            )
            if current_identity(copied.path) != copied.identity:
                return {
                    "error": "memory_import_checkout: the private copy was replaced during use; rerun -- "
                    "the import is idempotent",
                    "status": "uncertain",
                }
            # PRD-SEC-016 NFR01: "The reply reports the copy's wall time." `result`
            # is always a dict here -- memory_import_checkout_impl's own return
            # type is dict[str, object] -- so this never clobbers a non-dict reply.
            result["copy_seconds"] = copied.copy_seconds
            return result
        finally:
            copied.pin.close()
            # The copy's own directory, never the shared IMPORT_TMP_SUBDIR: everything opening the
            # copy wrote beside it goes too (C12-R). The SQLite handles closed inside the impl.
            shutil.rmtree(Path(copied.path).parent, ignore_errors=True)

    mcp.tool()(memory_import_checkout)


#: How long an import's write may wait for the write lane before it answers ``busy`` instead.
IMPORT_QUEUE_SECONDS = 30.0


def _serialized_lane(
    namespace: str, loop: asyncio.AbstractEventLoop, *, queue_seconds: float = IMPORT_QUEUE_SECONDS
) -> Lane:
    """Only the write takes the one lane every learning-row writer shares (C12 rc7), and waits for it at
    most *queue_seconds*. Past that, a step the lane has not started is abandoned: it writes nothing when
    the lane reaches it, and the caller answers ``busy``. A step already started is waited for (its own
    deadline bounds it), so the caller never answers, or removes the copy, while its write still runs."""
    from trw_memory.daemon._offload import run_serialized

    def lane(step: Callable[[SQLiteBackend], dict[str, object]]) -> dict[str, object]:
        busy: dict[str, object] = {
            "error": f"the write lane stayed busy past {queue_seconds:g}s: retry",
            "status": "busy",
        }
        phase = ["queued"]
        claim = threading.Lock()

        def started(backend: SQLiteBackend) -> dict[str, object]:
            with claim:
                if phase[0] == "abandoned":
                    return busy
                phase[0] = "running"
            return step(backend)

        write = functools.partial(in_namespace, namespace, Permission.WRITE, "import_checkout", _on(started))
        future = asyncio.run_coroutine_threadsafe(run_serialized(write), loop)
        try:
            return future.result(timeout=queue_seconds)
        except concurrent.futures.TimeoutError:
            with claim:
                if phase[0] == "queued":
                    phase[0] = "abandoned"
                    return busy
            return future.result()

    return lane


def _on(step: Callable[[SQLiteBackend], dict[str, object]]) -> Callable[[object, object], dict[str, object]]:
    """*step* as an ``in_namespace`` body: it needs the SQLite store the impl already checked it has."""

    def run(backend: object, _config: object) -> dict[str, object]:
        if not isinstance(backend, SQLiteBackend):
            return {"error": "memory_import_checkout needs a SQLite store to compare vectors", "status": "invalid"}
        return step(backend)

    return run


__all__ = ["memory_import_checkout_impl", "register_checkout_import_tools"]
