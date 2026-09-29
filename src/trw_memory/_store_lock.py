"""Who may use a store file, and who may replace it (PRD-CORE-306, ``docs/sprint-mcp7/DESIGN-B71-00-store-protocol.md``).

Every connection to a store holds ``open`` and the daemon holds ``serve`` for its
whole life; both are SHARED. An operation that destroys or replaces the store
holds an EXCLUSIVE op for one explicit scope (no nesting, no upgrade), which no
other process's hold can coexist with. The lock is POSIX ``fcntl`` ``F_SETLK``
over the whole of the permanent ``<db>.oplock``: the kernel drops it when its
process dies, and read/write conversions are atomic. Waits poll every 50 ms.

One store, one lock file: the path is resolved (symlinks included) and a store
with more than one hard link is refused, before the lock and again after the
connect, since each name would have its own lock file.

``_live_stores.FD_LOCK`` guards the descriptor table: one descriptor per lock
file, closed only once no hold uses it (closing ANY descriptor on the file drops
every lock this process holds on it). ``_R`` guards the hold counts and the
kernel conversion; it is taken alone or inside FD_LOCK, never around it. A GC
finalizer queues its release and applies it only if ``_R`` is free; whoever
holds ``_R`` drains the queue while waiting and on exit. So a collected
connection's kernel lock goes promptly, and its idle descriptor is closed by the
next acquire or release (never by the finalizer, which may run inside a scan
holding FD_LOCK). A forked child starts empty: it inherits no lock.
"""

from __future__ import annotations

import errno
import fcntl
import os
import stat
import threading
import time
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from trw_memory import _live_stores
from trw_memory._dir_trust import open_or_create_at
from trw_memory._inode_pin import Identity, current_identity
from trw_memory.exceptions import StoreBusyError, UnsupportedStorageError

StoreOp = Literal["open", "serve", "recover", "restore", "migrate", "import", "reembed", "snapshot"]
#: The SHARED ops; every other op is EXCLUSIVE, and a quiescing one also waits out this process's other connections.
_SHARED, _QUIESCING = ("open", "serve"), ("recover", "restore", "migrate")
#: Seconds an op waits for a conflicting holder (none if unlisted); an open waits as long as SQLite's busy timeout.
WAITS = {"open": 30.0, "serve": 30.0, "recover": 10.0}
_HOW = {fcntl.F_RDLCK: fcntl.LOCK_SH | fcntl.LOCK_NB, fcntl.F_WRLCK: fcntl.LOCK_EX | fcntl.LOCK_NB}
_POLL = 0.05
_SUFFIX = ".oplock"
_OPEN_FLAGS = (
    os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_CLOEXEC", 0)
)


@dataclass(eq=False)
class _File:
    """This process's one descriptor on a lock file; every count is under ``_R``."""

    name: str
    fd: int
    identity: Identity
    users: int = 0  # holds taken or being taken: the descriptor stays open while any
    shared: int = 0  # open and serve holds
    opens: int = 0
    exclusive: _Hold | None = None
    kernel: int = fcntl.F_UNLCK
    closed: bool = False  # a forked child's inherited entry: its holds release nothing


@dataclass(eq=False)
class _Hold:
    file: _File
    db: str
    op: str
    released: bool = False


_R = threading.Lock()
_COND = threading.Condition(_R)
_QUEUE: deque[_Hold] = deque()
#: Lock-file name -> its entry; guarded by FD_LOCK.
_FILES: dict[str, _File] = {}
#: Holds the current context took (never ``serve``: the daemon's copied request contexts must not own it).
_OWNED: ContextVar[tuple[_Hold, ...]] = ContextVar("trw_memory_store_holds", default=())


def check_one_name(real: str) -> None:
    """Refuse a store with more than one hard link: each name would get its own lock file."""
    try:
        links = os.lstat(real).st_nlink
    except FileNotFoundError:  # trw-fail-silent-allow: a store not created yet has no second name
        return
    if links > 1:
        raise UnsupportedStorageError(f"{real} has more than one name ({links} links); remove the others.", path=real)


# --- the descriptor table (FD_LOCK) ---------------------------------------------------------


def _pin(real: str, *, read_only: bool) -> _File | None:
    """The lock file's entry for the store at *real*, with one more user; ``None`` when there is nothing to lock."""
    name = real + _SUFFIX
    with _live_stores._locked():
        _sweep()
        file = _known(name, real) or _admit(real, name, read_only=read_only)
        if file is not None:
            with _registry():
                file.users += 1
        return file


def _unpin(file: _File) -> None:
    with _registry():
        file.users -= 1
    with _live_stores._locked():
        _sweep()


def _sweep() -> None:
    """Close every descriptor no hold uses (caller holds FD_LOCK); with no hold, its kernel lock is gone."""
    with _registry():
        idle = [file for file in _FILES.values() if not file.users]
    for file in idle:
        del _FILES[file.name]
        os.close(file.fd)
        _live_stores.track_lock_file(file.identity, open_=False)


def _known(name: str, real: str) -> _File | None:
    """The entry this process already has for *name*, refused if the name now names another file."""
    known = _FILES.get(name)
    if known is not None and current_identity(name) != known.identity:
        raise UnsupportedStorageError(
            f"{name} was replaced while this process held it. Stop every process using {real}, then retry.", path=name
        )
    return known


def _admit(real: str, name: str, *, read_only: bool) -> _File | None:
    try:
        dir_fd = os.open(os.path.dirname(name), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except FileNotFoundError:  # trw-fail-silent-allow: no directory, no store; the connect that follows raises
        return None
    leaf = os.path.basename(name)
    try:
        while (leased := current_identity(leaf, dir_fd=dir_fd)) is not None and _live_stores._live_leases(leased):
            _live_stores._LEASE_RELEASED.wait(_live_stores._LEASE_POLL_SECONDS)  # releases FD_LOCK
            if (known := _known(name, real)) is not None:
                return known  # another thread admitted it meanwhile: a second descriptor would drop its locks
        fd = open_or_create_at(dir_fd, leaf, _OPEN_FLAGS, 0o600)
    except OSError as exc:
        if exc.errno == errno.EROFS and read_only:
            return None  # nothing can write a read-only mount
        hint = {
            errno.EACCES: f"{name} is not yours (left by sudo?). Stop every process using the store, then `sudo rm {name}`."
            if os.path.lexists(name)
            else f"{name} cannot be created: {os.path.dirname(name)} is not writable.",
            errno.EROFS: f"{real} is on a read-only filesystem; open it read-only.",
        }.get(exc.errno or 0, f"{name} cannot be used as the store's lock file: {exc}")
        raise UnsupportedStorageError(hint, path=name) from exc
    finally:
        os.close(dir_fd)
    st = os.fstat(fd)
    identity = (st.st_dev, st.st_ino)
    checks = {
        "is not a regular file": not stat.S_ISREG(st.st_mode),
        "has more than one name": st.st_nlink != 1,
        f"is owned by uid {st.st_uid}": st.st_uid != os.geteuid(),
        "is already locked under another name": any(f.identity == identity for f in _FILES.values()),
    }
    if problem := next((text for text, failed in checks.items() if failed), None):
        if _live_stores.admit_reader_fd(fd):  # a descriptor on a live inode is parked, never closed
            _live_stores.close_reader_fd(fd)
        raise UnsupportedStorageError(f"{name} {problem}; refusing to use it as the store's lock file.", path=name)
    file = _FILES[name] = _File(name, fd, identity)
    _live_stores.track_lock_file(identity, open_=True)
    return file


# --- the hold registry (_R) -----------------------------------------------------------------


@contextmanager
def _registry() -> Iterator[None]:
    try:
        with _R:
            yield
    finally:
        _drain_if_free()


def _drain() -> None:
    while _QUEUE:
        _release_locked(_QUEUE.popleft())


def _drain_if_free() -> None:
    """Apply releases queued while ``_R`` was held; whoever holds it next does the same."""
    while _QUEUE and _R.acquire(blocking=False):
        try:
            _drain()
        finally:
            _R.release()


def _kernel(file: _File, kind: int) -> bool:
    """Set this process's lock on *file* to *kind* without waiting; False when another process conflicts."""
    try:  # F_SETLK over the whole file
        fcntl.lockf(file.fd, _HOW.get(kind, fcntl.LOCK_UN))
    except OSError as exc:
        if exc.errno in (errno.EACCES, errno.EAGAIN):
            return False
        if exc.errno in (errno.ENOLCK, errno.EOPNOTSUPP):
            raise UnsupportedStorageError("stores need a local filesystem with advisory locks", path=file.name) from exc
        raise
    file.kernel = kind
    return True


def _acquire_locked(hold: _Hold) -> None:
    file = hold.file
    exclusive, quiesce = hold.op not in _SHARED, hold.op in _QUIESCING
    owned = [h for h in _OWNED.get() if h.file is file and not h.released]
    if exclusive and owned:
        raise RuntimeError(f"{hold.op} on {hold.db}: this context already holds {owned[0].op}; release it first")
    deadline = time.monotonic() + WAITS.get(hold.op, 0.0)
    while True:
        ex = file.exclusive
        if exclusive and (ex is not None or (quiesce and file.opens)):
            blocker = f"this process ({ex.op if ex else 'its open connections'})"
        elif not exclusive and ex is not None and ex.op in _QUIESCING and not owned:
            blocker = f"this process ({ex.op})"  # a quiescing op is replacing the store
        elif (file.kernel != fcntl.F_UNLCK and not exclusive) or _kernel(
            file, fcntl.F_WRLCK if exclusive else fcntl.F_RDLCK
        ):
            break
        else:
            blocker = "another process (the memory daemon or an MCP session?)"
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            fix = "stop it, then retry" if "another" in blocker else "retry once it finishes"
            raise StoreBusyError(
                f"{hold.op}: {hold.db} is in use by {blocker}. Nothing was changed; {fix}.", path=hold.db
            )
        _COND.wait(min(_POLL, remaining))
        _drain()
    if exclusive:
        file.exclusive = hold
    else:
        file.shared += 1
        file.opens += hold.op == "open"
    if hold.op != "serve":
        _OWNED.set((*(h for h in _OWNED.get() if not h.released), hold))


def _release_locked(hold: _Hold) -> None:
    file = hold.file
    if hold.released or file.closed:
        return
    hold.released = True
    if file.exclusive is hold:
        file.exclusive = None
    else:
        file.shared -= 1
        file.opens -= hold.op == "open"
    file.users -= 1
    target = fcntl.F_WRLCK if file.exclusive else fcntl.F_RDLCK if file.shared else fcntl.F_UNLCK
    if target != file.kernel:
        _kernel(file, target)  # a downgrade or an unlock never conflicts
    _COND.notify_all()


# --- the interface --------------------------------------------------------------------------


def acquire(db: Path | str, op: StoreOp, *, read_only: bool = False) -> _Hold | None:
    """Take *op* on the store at *db*; ``None`` when there is nothing to lock (no directory, or a
    read-only mount opened read-only). Raises :class:`StoreBusyError` once ``op``'s wait runs out."""
    real = os.path.realpath(db)
    check_one_name(real)
    file = _pin(real, read_only=read_only)
    if file is None:
        return None
    hold = _Hold(file, real, op)
    try:
        with _registry():
            _acquire_locked(hold)
    except BaseException:
        _unpin(file)
        raise
    return hold


def release(hold: _Hold | None) -> None:
    """Release *hold* (idempotent)."""
    if hold is not None:
        with _registry():
            _release_locked(hold)
        with _live_stores._locked():
            _sweep()


def release_soon(hold: _Hold | None) -> None:
    """Release *hold* from a GC finalizer: never blocks on ``_R``, which this thread may hold."""
    if hold is not None:
        _QUEUE.append(hold)
        _drain_if_free()


@contextmanager
def store_access(db: Path | str, op: StoreOp) -> Iterator[None]:
    """Hold *op* on the store at *db* for the ``with`` block."""
    hold = acquire(db, op)
    try:
        yield
    finally:
        release(hold)


def _after_fork_in_child() -> None:
    """POSIX locks are not inherited: start empty, and let no inherited hold release anything."""
    global _R, _COND
    _R = threading.Lock()
    _COND = threading.Condition(_R)
    _QUEUE.clear()
    for file in _FILES.values():
        file.closed = True
        os.close(file.fd)  # the child holds no lock on it, so the close drops nothing
        _live_stores.track_lock_file(file.identity, open_=False)
    _FILES.clear()


os.register_at_fork(after_in_child=_after_fork_in_child)
