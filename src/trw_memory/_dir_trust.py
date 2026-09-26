"""Shared TOCTOU-hardened directory trust checks -- PRD-SEC-016.

``trw-memory`` writes secrets (tokens, grants, the SQLite store itself) under a
few user-space directories (``~/.trw/memory`` by default, or wherever
``TRW_USER_DIR``/``XDG_DATA_HOME`` point). Every one of those call sites used
to resolve a path with ``Path.resolve()`` or ``.is_symlink()`` and only later
``open()``/``chmod()`` it -- a check-then-use window in which a local attacker
who can write anywhere under the parent (a shared ``/tmp``-like mount, a
misconfigured home directory, a race with another process) can swap a plain
directory for a symlink and redirect the write, or swap a file for a symlink
between the existence check and the chmod that widens (or narrows) its mode.

This module is the ONE place that closes that window, for every caller:

* :func:`open_verified_dir_fd` opens a directory with ``O_NOFOLLOW`` (refusing
  a symlink outright) and returns the descriptor -- callers use it as a
  ``dir_fd`` anchor for further opens, so nothing after this point
  re-resolves the path and nothing after this point can be redirected by a
  swap. The caller owns the fd and must ``os.close()`` it.
* :func:`verify_and_harden_dir_fd` ``fstat``s that SAME descriptor (never a
  fresh path stat) and refuses a group/world-writable directory owned by
  someone else; a group/world-writable directory we own is auto-hardened to
  ``0700`` via ``os.fchmod`` on the fd, which -- like the open -- cannot be
  redirected by a later swap because it operates on the already-open object.

Residual window: SQLite itself (`sqlite3.connect`) opens the database file BY
PATH, not by fd, so there is an unavoidable gap between the last identity
check this module performs and the driver's own ``open()``. That gap is
checked where every file-backed SQLite open funnels through,
``trw_memory._live_stores.connect_registered``: the store's inode is pinned
(:mod:`trw_memory._inode_pin`) before the connect and compared with the path's
identity after it, so a swap in between is refused -- including on Linux,
where an unpinned before/after ``stat`` would be fooled by the replacement
getting the freed inode number back.
That narrows, but does not eliminate, the window: a race precisely inside
SQLite's own C-level ``open()`` call is out of Python's control. The residual
risk is recorded in PRD-SEC-016.
"""

from __future__ import annotations

import errno as _errno
import os
import stat
from pathlib import Path

import structlog

from trw_memory._live_stores import FD_LOCK, admit_reader_fd, is_known_live
from trw_memory.exceptions import UntrustedDirectoryError

__all__ = [
    "DIR_FD_SUPPORTED",
    "NOFOLLOW_SUPPORTED",
    "create_private_file_fd",
    "make_private_dirs",
    "open_anchored_walk",
    "open_component_fd",
    "open_or_create_at",
    "open_verified_dir_fd",
    "verify_ancestor_chain_trusted",
    "verify_and_harden_dir_fd",
]

logger = structlog.get_logger(__name__)

#: ``O_NOFOLLOW``/``O_DIRECTORY`` are POSIX-only; on Windows they are absent
#: and ``os.open`` ignores the 0 flag, so the calls below degrade to a plain
#: open on that platform rather than raising ``AttributeError``.
_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_O_DIRECTORY = getattr(os, "O_DIRECTORY", 0)
_CLOEXEC = getattr(os, "O_CLOEXEC", 0)
_DIR_OPEN_FLAGS = os.O_RDONLY | _O_DIRECTORY | _CLOEXEC | _NOFOLLOW

#: dir_fd-relative opens require ``openat(2)``; absent on some platforms
#: (older Windows Python builds). Resolved once so call sites stay branch-free.
DIR_FD_SUPPORTED = os.open in os.supports_dir_fd

#: ``_NOFOLLOW`` degrading to ``0`` (an absent ``O_NOFOLLOW`` constant) is
#: indistinguishable, at the flags level, from a caller who simply chose not
#: to ask for it -- a symlink is then followed SILENTLY rather than refused.
#: Checkout-boundary callers (:func:`open_component_fd`, :func:`open_anchored_walk`,
#: :func:`create_private_file_fd`) must refuse outright when this is False,
#: never combine ``0`` into their flags and proceed (PRD-SEC-016 NFR02/FR05
#: round-2 finding 5 -- a missing capability is a refusal, not a downgrade).
NOFOLLOW_SUPPORTED = _NOFOLLOW != 0


def _require_symlink_safe_open(name_or_path: str) -> None:
    """Refuse up front when this platform cannot guarantee a no-follow open -- never a silent downgrade."""
    if DIR_FD_SUPPORTED and NOFOLLOW_SUPPORTED:
        return
    missing = []
    if not DIR_FD_SUPPORTED:
        missing.append("dir_fd")
    if not NOFOLLOW_SUPPORTED:
        missing.append("O_NOFOLLOW")
    raise UntrustedDirectoryError(
        f"this platform lacks {' and '.join(missing)}; refusing to open {name_or_path!r} without it "
        "rather than falling back to a by-name, symlink-following open",
        path=name_or_path,
    )


def make_private_dirs(path: Path, mode: int = 0o700) -> None:
    """``mkdir -p`` *path*, creating EVERY missing component with *mode*.

    ``Path.mkdir(parents=True, mode=...)`` applies *mode* to the leaf only; the
    parents get the umask default -- 0775 under the 0002 umask stock Ubuntu gives
    its users -- and :func:`verify_ancestor_chain_trusted` then refuses the
    group-writable ``~/.trw`` this package created itself.
    """
    missing: list[Path] = []
    while not path.exists():
        missing.append(path)
        path = path.parent
    for component in reversed(missing):
        try:
            os.mkdir(component, mode)
        except (
            FileExistsError
        ):  # trw-fail-silent-allow: a concurrent creator won; the caller's no-follow open still verifies what is there
            continue


def open_verified_dir_fd(path: Path, *, create: bool, mode: int = 0o700) -> int:
    """Open *path* as a directory fd, refusing to follow a symlink.

    When *create* is True and the directory is absent, it is created first
    (:func:`make_private_dirs`, every component *mode*) and then re-opened the same way
    -- the open, not the mkdir, is the trust boundary, so a symlink planted
    between the mkdir and the open is still refused.

    Returns the open descriptor; the caller must ``os.close()`` it. Raises
    ``UntrustedDirectoryError`` (wrapping ``OSError``) when the path is a
    symlink or otherwise cannot be opened securely; ``FileNotFoundError``
    when *create* is False and the directory is absent (mirrors the historic
    "never raises on a missing directory" contract for presence probes).
    """
    try:
        return os.open(path, _DIR_OPEN_FLAGS)
    except FileNotFoundError:
        if not create:
            raise
        make_private_dirs(path, mode)
        try:
            return os.open(path, _DIR_OPEN_FLAGS)
        except OSError as exc:
            raise _refuse(path, exc) from exc
    except OSError as exc:
        raise _refuse(path, exc) from exc


def verify_and_harden_dir_fd(
    fd: int, path_for_error: Path, *, target_mode: int = 0o700, force: bool = False
) -> os.stat_result:
    """``fstat`` the OPEN descriptor *fd* and refuse an untrusted directory.

    A directory that is group- or world-writable and owned by someone else is
    refused outright -- another user could have swapped it, or could still
    swap a file inside it out from under a later write. A directory we own
    that happens to be group/world-writable is auto-hardened to
    *target_mode* via ``fchmod`` on the SAME fd (no re-resolution), mirroring
    the historic best-effort ``chmod(0700)`` this replaces. A directory with
    the sticky bit set (``/tmp``-style) is treated as safe regardless of
    writability, matching standard POSIX practice -- UNLESS *force* is True,
    for callers (secrets-only directories that are never meant to be shared,
    e.g. the daemon's token/grants directory) that want *target_mode*
    enforced unconditionally rather than only when the current mode is
    already unsafe.

    Every check operates on *fd*, never on a fresh ``path.stat()`` -- that is
    the whole point: the object being trusted is the object already open, so
    nothing between "check" and "use" can substitute a different one.
    """
    st = os.fstat(fd)
    geteuid = getattr(os, "geteuid", None)
    if not callable(geteuid):
        return st  # Windows: POSIX owner/mode bits do not apply.
    world_or_group_writable = bool(st.st_mode & 0o022) and not bool(st.st_mode & stat.S_ISVTX)
    if world_or_group_writable and st.st_uid != geteuid():
        raise UntrustedDirectoryError(
            f"{path_for_error} is group/world-writable and owned by uid {st.st_uid}, "
            f"not the current user (uid {geteuid()}) -- refusing to trust it",
            path=str(path_for_error),
        )
    if not force and not world_or_group_writable:
        return st
    try:
        os.fchmod(fd, target_mode)
    except OSError as exc:
        raise UntrustedDirectoryError(
            f"Cannot harden {path_for_error} permissions: {exc}", path=str(path_for_error)
        ) from exc
    hardened = os.fstat(fd)
    if stat.S_IMODE(hardened.st_mode) != target_mode:
        raise UntrustedDirectoryError(
            f"Permission hardening of {path_for_error} did not take effect", path=str(path_for_error)
        )
    logger.info(
        "dir_permissions_hardened",
        path=str(path_for_error),
        old_mode=oct(stat.S_IMODE(st.st_mode)),
        new_mode=oct(stat.S_IMODE(hardened.st_mode)),
    )
    return hardened


def verify_ancestor_chain_trusted(path: Path) -> None:
    """Refuse *path* if IT or ANY ancestor is untrusted -- PRD-SEC-016 FR04.

    Unlike :func:`verify_and_harden_dir_fd` (which auto-hardens a directory
    WE own), an ancestor above the directory we manage is never ours to
    ``chmod`` -- ``/home``, ``/Users``, a parent a sysadmin owns. There is
    nothing to harden, so the only two outcomes here are "trusted" and
    "refuse and name the reason": an ancestor owned by a uid that is
    neither ours nor root's, OR group/world-writable without the sticky
    bit, refuses regardless of who owns it (a self-owned-but-group-writable
    ancestor is still a directory another local principal in that group
    could rewrite).

    Uses ``path.stat()`` (a path-based, not fd-based, walk) because a
    directory has an unbounded number of ancestors and opening every one as
    a ``dir_fd`` anchor for the remainder of the walk is not a meaningful
    hardening -- a hostile ancestor mid-chain can still be swapped for a
    symlink pointing anywhere, in which case ``.stat()`` (which follows
    symlinks) reports the identity of whatever it now points at, and that
    reported identity is exactly what gets judged. This is a STARTUP gate,
    not a per-write TOCTOU close: see the residual-window note in
    ``trw_memory.storage._permissions`` for what happens between this check
    and the store's own open.
    """
    geteuid = getattr(os, "geteuid", None)
    if not callable(geteuid):
        return  # Windows: POSIX owner/mode bits do not apply; NFR02 covers refusal elsewhere.
    my_uid = geteuid()
    start = path if path.is_dir() else path.parent
    for ancestor in (start, *start.parents):
        try:
            st = ancestor.stat()
        except OSError:  # trw-fail-silent-allow: an unstat'able ancestor is neither trustable nor a positive threat signal; every OTHER ancestor is still checked, so the walk stays fail-closed overall
            continue
        writable_no_sticky = bool(st.st_mode & 0o022) and not bool(st.st_mode & stat.S_ISVTX)
        foreign_owner = st.st_uid not in (my_uid, 0)
        if not (foreign_owner or writable_no_sticky):
            continue
        reason = (
            f"owned by uid {st.st_uid}, not the daemon's uid ({my_uid}) or root"
            if foreign_owner
            else "group/world-writable without the sticky bit"
        )
        logger.error(
            "store_ancestor_untrusted",
            path=str(ancestor),
            owner_uid=st.st_uid,
            mode=oct(stat.S_IMODE(st.st_mode)),
            reason=reason,
        )
        raise UntrustedDirectoryError(
            f"refusing to serve: {ancestor} is {reason} (mode {oct(stat.S_IMODE(st.st_mode))}, owner uid {st.st_uid})",
            path=str(ancestor),
        )


def open_component_fd(dir_fd: int, name: str, *, directory: bool) -> int:
    """Open *name* directly inside the already-open directory *dir_fd* (``openat``), no-follow.

    This is the per-component primitive PRD-SEC-016 FR02/FR03/FR05 build a
    descriptor walk out of: a caller resolves a checkout-relative path into its
    individual components, then opens each one in turn, anchored on the fd
    returned by the PREVIOUS open rather than by re-resolving the path from its
    string form. Nothing between two calls can redirect the walk, because each
    open only ever consults the single component name against a descriptor the
    walk itself already trusts -- an ancestor swapped for a symlink after an
    earlier ``Path.resolve()``-based check is caught here, not missed by it.

    Raises ``UntrustedDirectoryError`` when *name* is a symlink, absent, a
    store this process has open (or its sidecar), or the wrong kind (not a directory when *directory* is True; not a regular file
    when it is False). Raises ``UntrustedDirectoryError`` up front, without
    attempting a by-name fallback, when the platform lacks ``dir_fd`` OR
    ``O_NOFOLLOW`` support (NFR02) -- there is deliberately no silent
    downgrade in either case.
    The caller owns the returned descriptor and must close it with
    ``os.close()`` for a directory, and with
    :func:`trw_memory._live_stores.close_reader_fd` for a file (C15).
    """
    _require_symlink_safe_open(name)
    if directory:
        return _open_component(dir_fd, name, directory=True)
    # The name check, the open and the admission run as one FD_LOCK section, so the
    # parked-descriptor cap holds under concurrent readers (C15).
    with FD_LOCK:
        if is_known_live(dir_fd, name):
            # Refused by name before opening, so no descriptor is parked.
            raise UntrustedDirectoryError(f"{name} is a memory store (or its sidecar) this process has open", path=name)
        fd = _open_component(dir_fd, name, directory=False)
        if not admit_reader_fd(fd):
            # The descriptor ACTUALLY opened is a live store or its sidecar (a name
            # swapped after the check): closing it would release that store's SQLite
            # locks, so it is parked, not closed.
            raise UntrustedDirectoryError(f"{name} is a memory store (or its sidecar) this process has open", path=name)
        return fd


def _open_component(dir_fd: int, name: str, *, directory: bool) -> int:
    # O_NONBLOCK on a leaf: a FIFO would otherwise block this open while FD_LOCK is
    # held; it changes nothing for a regular file, and a FIFO is refused below.
    flags = os.O_RDONLY | _CLOEXEC | _NOFOLLOW | (_O_DIRECTORY if directory else getattr(os, "O_NONBLOCK", 0))
    try:
        fd = os.open(name, flags, dir_fd=dir_fd)
    except OSError as exc:
        raise _refuse(Path(name), exc) from exc
    if not directory:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            os.close(fd)
            raise UntrustedDirectoryError(f"{name} is not a regular file", path=name)
    return fd


def open_anchored_walk(root: Path) -> int:
    """Open *root* ITSELF via a component-by-component no-follow walk from the filesystem root.

    :func:`open_verified_dir_fd` opens *root* with a single ``os.open()`` call
    on its full path string -- ``O_NOFOLLOW`` there refuses a symlink only at
    *root*'s own FINAL path component; every ancestor of *root* is still
    resolved the ordinary way (following symlinks), because that is how POSIX
    path resolution treats every component except the last one. For most
    ``open_verified_dir_fd`` callers (the daemon's own secrets/store
    directories) that gap is out of this PRD's threat model -- their
    ancestors sit outside any tenant's checkout.

    The checkout boundary is different: this daemon can serve MULTIPLE
    tenants at once, and nothing stops one tenant's checkout from being
    nested inside another's (checkout B's root is some path under checkout
    A's root). A request holding only A's grant can then write anywhere
    inside A -- including an ancestor of B's root -- and swap it for a
    symlink, redirecting where B's root resolves for the NEXT request that
    walks it. That is a cross-tenant sandbox escape, not the "same OS user"
    case this PRD explicitly declines to defend (CONSTITUTION non-goal): B's
    own token never authorized reading anything outside B's root.

    This closes that gap by walking every component of *root* itself --
    not just the file path below it -- with ``O_NOFOLLOW``, anchored at the
    filesystem root (``root.anchor``, e.g. ``"/"``, which by construction
    cannot itself be a symlink). :func:`open_component_fd` does the
    per-component work; this only supplies the anchor and the component
    list. Raises ``UntrustedDirectoryError`` on any symlinked component
    anywhere in *root*'s path, a missing component, or -- NFR02 -- a
    platform lacking ``dir_fd``/``O_NOFOLLOW`` support. The caller owns the
    returned descriptor and must ``os.close()`` it.
    """
    if not root.is_absolute():
        raise UntrustedDirectoryError(
            f"{root} is not an absolute path; refusing to anchor a walk on it", path=str(root)
        )
    _require_symlink_safe_open(str(root))
    try:
        anchor_fd = os.open(root.anchor, _DIR_OPEN_FLAGS)
    except OSError as exc:
        raise _refuse(Path(root.anchor), exc) from exc
    opened = [anchor_fd]
    try:
        for part in root.parts[1:]:
            fd = open_component_fd(opened[-1], part, directory=True)
            opened.append(fd)
        return opened.pop()
    finally:
        for fd in opened:
            os.close(fd)


def create_private_file_fd(dir_fd: int, name: str, *, mode: int = 0o600) -> int:
    """Create and open *name* inside *dir_fd*, refusing to reuse or follow an existing object.

    ``O_CREAT | O_EXCL`` makes this atomic: the call fails if *name* already
    exists -- as a regular file, a directory, or a symlink -- rather than
    opening whatever is already there. That closes the by-name TOCTOU a plain
    ``mkdtemp()`` followed by a later ``open(dest, "wb")`` leaves open: if the
    directory that ``dir_fd`` anchors sits under a shared, non-sticky-writable
    TMPDIR, an attacker who deletes and replaces the daemon's freshly created
    temp directory (or a file inside it) between the two calls could redirect
    the write. Anchoring on the already-open ``dir_fd`` and using ``O_EXCL``
    here removes that window: nothing about *name*'s resolution happens by
    path after the caller's own directory-fd check. ``O_NOFOLLOW`` on top of
    ``O_EXCL`` is redundant in practice (a symlink would already trip
    ``O_EXCL``) but keeps this opener symmetric with every other one in this
    module. Raises ``UntrustedDirectoryError`` on any failure, including the
    platform-capability gap NFR02 requires refusing rather than silently
    downgrading. The caller owns the returned descriptor.
    """
    _require_symlink_safe_open(name)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | _NOFOLLOW | _CLOEXEC
    try:
        return os.open(name, flags, mode, dir_fd=dir_fd)
    except OSError as exc:
        raise _refuse(Path(name), exc) from exc


#: Exclusive-create / plain-open rounds :func:`open_or_create_at` tries. Each
#: lost round means another process created or removed *name* in between; a
#: store's files are created once, so more than one lost round is already rare.
_OPEN_OR_CREATE_ROUNDS = 8


def open_or_create_at(dir_fd: int, name: str, flags: int, mode: int) -> int:
    """Open *name* inside *dir_fd*, creating it when absent, even while other openers race to create it.

    A plain ``openat(dir_fd, name, O_CREAT)`` is NOT safe to race on macOS:
    when two callers create the same new name at once, the loser gets
    ``ENOENT`` (measured on macOS 26.5: about 7 in 10 contended pairs), although
    the directory exists and the name is being created. A path-based
    ``open(O_CREAT)`` does not show it, which is why the dir_fd hardening of
    PRD-SEC-016 turned concurrent first opens of one store into
    ``Secure SQLite open failed: [Errno 2]``.

    So the create is split in two, each with one well-defined race outcome:
    ``O_CREAT|O_EXCL`` either creates the file or fails ``EEXIST``, and the
    open without ``O_CREAT`` either opens it or fails ``ENOENT`` because it
    was removed in between -- and the loop goes round again. Both opens are
    relative to the same *dir_fd* with the caller's flags, so ``O_NOFOLLOW``
    still refuses a symlink: it fails the exclusive create (``EEXIST``) and
    then the plain open (``ELOOP``), which is raised. The caller owns the
    returned descriptor.
    """
    plain = flags & ~(os.O_CREAT | os.O_EXCL)
    last: OSError = FileNotFoundError(_errno.ENOENT, os.strerror(_errno.ENOENT), name)
    for _ in range(_OPEN_OR_CREATE_ROUNDS):
        try:
            return os.open(name, plain | os.O_CREAT | os.O_EXCL, mode, dir_fd=dir_fd)
        except FileExistsError as exc:
            last = exc
        try:
            return os.open(name, plain, dir_fd=dir_fd)
        except FileNotFoundError as exc:
            last = exc
    raise last


def _refuse(path: Path, exc: OSError) -> UntrustedDirectoryError:
    hint = ""
    if getattr(exc, "errno", None) == _errno.ELOOP or path.is_symlink():
        hint = " (the path is a symlink; refusing to follow it)"
    logger.error("dir_open_refused", path=str(path), error=type(exc).__name__)
    return UntrustedDirectoryError(f"Cannot securely open directory {path}: {exc}{hint}", path=str(path))
