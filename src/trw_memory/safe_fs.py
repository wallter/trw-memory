"""Write beneath a root directory without following a symlink -- PRD-CORE-337.

A project checkout is not trusted content: a hostile branch or template can plant a symlink at
``.cursor/hooks.json`` or make ``.codex/`` itself a symlink, and a plain ``Path.write_text`` then
writes wherever the link points. This module is the one way trw-mcp and trw-memory write into a
checkout. Two calls:

* :func:`write_beneath` replaces ``root/rel_path`` with *data* as a whole: the new bytes go to a
  fresh temp file and are published by ``os.replace``, so a reader sees the old file or the new
  one, never a partial write.
* :func:`append_beneath` appends *data* to ``root/rel_path``, creating it when absent.

Both create missing parent directories (like ``mkdir(parents=True)``; the umask applies) and both
refuse, with :class:`~trw_memory.exceptions.UnsafeWriteError`, rather than follow a symlink.

**POSIX (race-safe).** *root* is opened as a directory fd that refuses a symlink at its final
component (:func:`trw_memory._dir_trust.open_verified_dir_fd`). Every parent component of
*rel_path* is then opened with ``O_DIRECTORY | O_NOFOLLOW`` relative to the fd of the component
before it, so no step re-resolves a path by name and a swap between two steps cannot redirect the
walk: the walk holds the directories it already opened. The temp file is created relative to the
held parent fd with ``O_CREAT | O_EXCL | O_NOFOLLOW`` and its final *mode* passed to that same
open -- there is no later ``chmod``, so the file is never visible at a wider mode than *mode*
(the process umask can only narrow it). The bytes are written and fsynced, then published with
``os.replace(tmp, leaf, src_dir_fd=parent, dst_dir_fd=parent)`` and the parent directory is
fsynced. A symlink found at the leaf is refused; one planted there after that check is
replaced as a directory entry, never written through. An append opens the leaf itself with
``O_APPEND | O_NOFOLLOW`` relative to the held parent fd. A POSIX platform that lacks ``dir_fd``
or ``O_NOFOLLOW`` support is refused outright (reason ``unsupported_platform``); there is no
by-name fallback.

**Windows (best effort, NOT race-safe).** Windows Python has neither ``dir_fd`` nor ``O_NOFOLLOW``.
There, each component is checked with ``lstat`` (a symlink or junction is refused) and the write
then proceeds by path: a temp file beside the leaf and ``os.replace``, or an append open. A symlink
swapped in between the last ``lstat`` and the write is NOT caught. This is a deliberate,
documented degradation, not a refusal: a missing capability never refuses a write on Windows,
so a Windows install that wrote a path before keeps writing it. The branch is chosen once, at
import.

**What is a refusal.** ``UnsafeWriteError`` (a subclass of ``UntrustedDirectoryError``, never of
``OSError``, so a broad ``except OSError`` cannot swallow it) carries ``path`` -- the component
that was refused -- and ``reason``, one of: ``escapes_root`` (*rel_path* is absolute, empty or
contains ``..``), ``root_untrusted`` (*root* is a symlink or cannot be opened as a directory),
``symlink_component``, ``not_a_directory`` (a parent component is some other kind of file),
``symlink_leaf``, ``leaf_not_regular_file`` and ``unsupported_platform``. A missing *root* is a
plain ``FileNotFoundError``, and a failure of the write itself (a full disk, a permission denied)
is the ``OSError`` the operating system raised: those are failures, not refusals. Nothing this
module logs or raises ever includes *data*.
"""

from __future__ import annotations

import errno
import os
import secrets
import stat
import sys
from pathlib import Path, PurePath

import structlog

from trw_memory._dir_trust import DIR_FD_SUPPORTED, NOFOLLOW_SUPPORTED, open_or_create_at, open_verified_dir_fd
from trw_memory.exceptions import UnsafeWriteError, UntrustedDirectoryError

__all__ = [
    "UnsafeWriteError",
    "anchored_removal_supported",
    "append_beneath",
    "check_parent_beneath",
    "open_parent_beneath",
    "write_beneath",
]

logger = structlog.get_logger(__name__)

_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_O_DIRECTORY = getattr(os, "O_DIRECTORY", 0)
_CLOEXEC = getattr(os, "O_CLOEXEC", 0)
_NONBLOCK = getattr(os, "O_NONBLOCK", 0)  # a FIFO leaf fails fast instead of blocking the append open
_BINARY = getattr(os, "O_BINARY", 0)  # Windows: no newline translation
_DIR_FLAGS = os.O_RDONLY | _O_DIRECTORY | _NOFOLLOW | _CLOEXEC
_TEMP_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_EXCL | _NOFOLLOW | _CLOEXEC | _BINARY
_APPEND_FLAGS = os.O_WRONLY | os.O_APPEND | _NOFOLLOW | _CLOEXEC | _NONBLOCK | _BINARY
_DIR_MODE = 0o777  # what Path.mkdir passes; the umask narrows it
_TEMP_NAME_ROUNDS = 8
#: The errno an ``O_NOFOLLOW`` open fails with on a symlink: ``ELOOP`` (POSIX), ``EMLINK`` (FreeBSD/DragonFly).
#: It is translated into a refusal on its own -- never via a follow-up ``stat``, which a symlink removed in
#: between would answer with "absent" and let the ``OSError`` escape as a non-refusal.
_SYMLINK_ERRNOS = frozenset(
    {errno.ELOOP, errno.EMLINK} if sys.platform.startswith(("freebsd", "dragonfly")) else {errno.ELOOP}
)


def _select_branch(os_name: str, dir_fd_supported: bool, nofollow_supported: bool) -> str:
    """``posix`` (race-safe), ``windows`` (best effort) or ``unsupported`` (a POSIX platform that must refuse)."""
    if os_name == "nt":
        return "windows"
    return "posix" if dir_fd_supported and nofollow_supported else "unsupported"


_BRANCH = _select_branch(os.name, DIR_FD_SUPPORTED, NOFOLLOW_SUPPORTED)


def write_beneath(root: Path, rel_path: str | PurePath, data: bytes, *, mode: int, exact_mode: bool = False) -> None:
    """Replace ``root/rel_path`` with *data*, atomically, refusing any symlinked component.

    The file is created at *mode* (narrowed by the umask, never widened). With *exact_mode* the
    published file carries *mode* exactly -- for a caller keeping the bits of the file it replaces,
    which the umask must not narrow a second time (the temp is ``fchmod``-ed before the replace). Raises
    ``UnsafeWriteError`` on a refusal; see the module docstring for the reasons and the
    platform contract.
    """
    dirs, leaf = _split(root, rel_path)
    if _BRANCH == "windows":
        _write_best_effort(_walk_best_effort(root, dirs), leaf, data, mode, exact_mode=exact_mode)
        return
    parent_fd = _open_parent(root, dirs)
    try:
        _publish_at(parent_fd, root.joinpath(*dirs, leaf), leaf, data, mode, exact_mode=exact_mode)
    finally:
        os.close(parent_fd)


def append_beneath(root: Path, rel_path: str | PurePath, data: bytes, *, mode: int, sync: bool = False) -> None:
    """Append *data* to ``root/rel_path``, creating it at *mode* when absent; a symlinked leaf is refused.

    With *sync* the file is ``fsync``-ed before the call returns -- for the last append of a copy that
    ``write_beneath`` would have synced whole.
    """
    dirs, leaf = _split(root, rel_path)
    if _BRANCH == "windows":
        _append_best_effort(_walk_best_effort(root, dirs), leaf, data, mode, sync=sync)
        return
    parent_fd = _open_parent(root, dirs)
    try:
        _append_at(parent_fd, root.joinpath(*dirs, leaf), leaf, data, mode, sync=sync)
    finally:
        os.close(parent_fd)


def anchored_removal_supported() -> bool:
    """Whether :func:`open_parent_beneath` can run here (the descriptor-anchored POSIX branch)."""
    return _BRANCH == "posix"


def open_parent_beneath(root: Path, rel_path: str | PurePath) -> tuple[int, str]:
    """``(fd of rel_path's parent, leaf name)`` for a caller that removes ``root/rel_path`` (POSIX only).

    Every component is opened no-follow and none is created: a symlinked ancestor is refused
    (``UnsafeWriteError``) and an absent one raises ``FileNotFoundError``. The caller closes the fd and
    acts on the leaf with ``dir_fd=`` so a link swapped in above it after this call cannot redirect it.
    """
    dirs, leaf = _split(root, rel_path)
    return _open_parent(root, dirs, create=False), leaf


def check_parent_beneath(root: Path, rel_path: str | PurePath) -> tuple[Path, str]:
    """``(parent path, leaf name)`` after an ``lstat`` check that no existing ancestor is a link (Windows, best effort).

    Not race-safe, like the rest of the Windows branch. An absent ancestor raises ``FileNotFoundError``.
    """
    dirs, leaf = _split(root, rel_path)
    if _is_link(os.lstat(root)):
        raise _refusal(root, "root_untrusted")
    current = root
    for name in dirs:
        current = current / name
        if _is_link(os.lstat(current)):
            raise _refusal(current, "symlink_component")
    return current, leaf


def _split(root: Path, rel_path: str | PurePath) -> tuple[tuple[str, ...], str]:
    relative = PurePath(rel_path)
    parts = relative.parts
    if relative.anchor or not parts or ".." in parts:
        raise _refusal(root / relative, "escapes_root")
    return parts[:-1], parts[-1]


def _refusal(path: Path, reason: str) -> UnsafeWriteError:
    logger.warning("safe_write_refused", path=str(path), reason=reason)
    return UnsafeWriteError(f"refusing to write {path}: {reason}", path=str(path), reason=reason)


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        view = view[os.write(fd, view) :]


def _temp_name(leaf: str) -> str:
    return f".{leaf[:64]}.{secrets.token_hex(8)}.tmp"


# --- POSIX: descriptor-anchored ------------------------------------------------------------------


def _open_parent(root: Path, dirs: tuple[str, ...], *, create: bool = True) -> int:
    """The fd of ``root/dirs``, reached one no-follow component at a time; the caller closes it.

    With ``create=False`` a missing component raises ``FileNotFoundError`` instead of being made.
    """
    if _BRANCH != "posix":
        raise _refusal(root, "unsupported_platform")
    try:
        fd = open_verified_dir_fd(root, create=False)
    except UntrustedDirectoryError as exc:
        raise _refusal(root, "root_untrusted") from exc
    try:
        for depth, name in enumerate(dirs, start=1):
            child = _open_dir_at(fd, name, root.joinpath(*dirs[:depth]), create=create)
            os.close(fd)
            fd = child
    except BaseException:
        os.close(fd)
        raise
    return fd


def _open_dir_at(dir_fd: int, name: str, shown: Path, *, create: bool = True) -> int:
    if create:
        try:
            os.mkdir(name, _DIR_MODE, dir_fd=dir_fd)
        except FileExistsError:  # trw-fail-silent-allow: an existing entry is the common case; the no-follow open below decides whether it is a directory we may use
            pass
    try:
        return os.open(name, _DIR_FLAGS, dir_fd=dir_fd)
    except OSError as exc:
        if exc.errno in _SYMLINK_ERRNOS:
            raise _refusal(shown, "symlink_component") from exc
        kind = _kind_at(dir_fd, name)
        if kind == stat.S_IFLNK:
            raise _refusal(shown, "symlink_component") from exc
        if exc.errno == errno.ENOTDIR or (kind is not None and kind != stat.S_IFDIR):
            raise _refusal(shown, "not_a_directory") from exc
        raise


def _kind_at(dir_fd: int, name: str) -> int | None:
    """The ``S_IFMT`` of *name* inside *dir_fd* without following it; ``None`` when it is absent."""
    try:
        return stat.S_IFMT(os.stat(name, dir_fd=dir_fd, follow_symlinks=False).st_mode)
    except FileNotFoundError:  # trw-fail-silent-allow: absence is an answer here (None), not an error; every caller treats None as "nothing is in the way"
        return None


def _publish_at(parent_fd: int, shown: Path, leaf: str, data: bytes, mode: int, *, exact_mode: bool = False) -> None:
    kind = _kind_at(parent_fd, leaf)
    if kind == stat.S_IFLNK:
        raise _refusal(shown, "symlink_leaf")
    if kind is not None and kind != stat.S_IFREG:
        raise _refusal(shown, "leaf_not_regular_file")
    tmp_name, tmp_fd = _create_temp_at(parent_fd, leaf, mode)
    try:
        try:
            if exact_mode:
                os.fchmod(tmp_fd, mode)  # on the descriptor: the temp it names is ours (O_EXCL|O_NOFOLLOW)
            _write_all(tmp_fd, data)
            os.fsync(tmp_fd)
        finally:
            os.close(tmp_fd)
        os.replace(tmp_name, leaf, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
    except BaseException:
        _discard_temp_at(parent_fd, tmp_name)
        raise
    os.fsync(parent_fd)


def _create_temp_at(parent_fd: int, leaf: str, mode: int) -> tuple[str, int]:
    for _ in range(_TEMP_NAME_ROUNDS):
        name = _temp_name(leaf)
        try:
            return name, os.open(name, _TEMP_FLAGS, mode, dir_fd=parent_fd)
        except FileExistsError:  # trw-fail-silent-allow: a random-name collision; the next round draws a fresh name, and exhausting every round raises below
            continue
    raise FileExistsError(f"could not create a unique temp file beside {leaf!r}")


def _discard_temp_at(parent_fd: int, tmp_name: str) -> None:
    try:
        os.unlink(tmp_name, dir_fd=parent_fd)
    except OSError:  # trw-fail-silent-allow: cleanup on a failure path; the original error is re-raised by the caller, and a stray dot-temp file is harmless
        logger.warning("safe_write_temp_not_removed", temp=tmp_name)


def _append_at(parent_fd: int, shown: Path, leaf: str, data: bytes, mode: int, *, sync: bool = False) -> None:
    try:
        fd = open_or_create_at(parent_fd, leaf, _APPEND_FLAGS, mode)
    except OSError as exc:
        if exc.errno in _SYMLINK_ERRNOS:
            raise _refusal(shown, "symlink_leaf") from exc
        kind = _kind_at(parent_fd, leaf)
        if kind == stat.S_IFLNK:
            raise _refusal(shown, "symlink_leaf") from exc
        if exc.errno in {errno.EISDIR, errno.ENXIO} or (kind is not None and kind != stat.S_IFREG):
            raise _refusal(shown, "leaf_not_regular_file") from exc
        raise
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise _refusal(shown, "leaf_not_regular_file")
        _write_all(fd, data)
        if sync:
            os.fsync(fd)
    finally:
        os.close(fd)


# --- Windows: lstat per component, NOT race-safe (see the module docstring) -----------------------


def _is_link(st: os.stat_result) -> bool:
    reparse = getattr(st, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
    return stat.S_ISLNK(st.st_mode) or bool(reparse)


def _walk_best_effort(root: Path, dirs: tuple[str, ...]) -> Path:
    """``root/dirs``, each component ``lstat``-checked and created when absent. Refuses only a detected link."""
    if _is_link(os.lstat(root)):
        raise _refusal(root, "root_untrusted")
    current = root
    for name in dirs:
        current = current / name
        try:
            os.mkdir(current)
        except (
            FileExistsError
        ):  # trw-fail-silent-allow: an existing entry is the common case; the lstat below judges it
            pass
        if _is_link(os.lstat(current)):
            raise _refusal(current, "symlink_component")
    return current


def _refuse_linked_leaf(leaf_path: Path) -> None:
    try:
        st = os.lstat(leaf_path)
    except FileNotFoundError:  # trw-fail-silent-allow: no leaf yet is the create case; there is nothing to refuse
        return
    if _is_link(st):
        raise _refusal(leaf_path, "symlink_leaf")


def _write_best_effort(parent: Path, leaf: str, data: bytes, mode: int, *, exact_mode: bool = False) -> None:
    target = parent / leaf
    _refuse_linked_leaf(target)
    tmp = parent / _temp_name(leaf)
    fd = os.open(tmp, _TEMP_FLAGS, mode)
    try:
        try:
            if exact_mode and hasattr(
                os, "fchmod"
            ):  # Windows has no fchmod; its permission bits are best effort anyway
                os.fchmod(fd, mode)
            _write_all(fd, data)
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp, target)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:  # trw-fail-silent-allow: cleanup on a failure path; the original error is re-raised below
            logger.warning("safe_write_temp_not_removed", temp=str(tmp))
        raise


def _append_best_effort(parent: Path, leaf: str, data: bytes, mode: int, *, sync: bool = False) -> None:
    target = parent / leaf
    _refuse_linked_leaf(target)
    fd = os.open(target, os.O_WRONLY | os.O_APPEND | os.O_CREAT | _BINARY, mode)
    try:
        _write_all(fd, data)
        if sync:
            os.fsync(fd)
    finally:
        os.close(fd)
