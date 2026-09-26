"""Machine-local (user-space) memory path resolution -- PRD-CORE-253 FR01.

The user-space memory tier lives OUTSIDE any project's ``.trw`` directory so a
single machine-local store is shared by every checkout on the box. This module
is the single source of truth for resolving that directory.

Resolution precedence (highest wins), unchanged from PRD-CORE-185 FR01 (D1):

1. ``TRW_USER_DIR`` env var   -> ``<TRW_USER_DIR>/memory``
2. ``XDG_DATA_HOME`` env var  -> ``<XDG_DATA_HOME>/trw/memory``
3. fallback                   -> ``<home>/.trw/memory``

PRD-CORE-253 FR01 **promotes** this resolver from
``trw_mcp.state._user_paths`` into trw-memory so the loopback daemon (FR03) can
resolve its own store, token, lock and discovery files without importing
trw-mcp -- trw-mcp depends on trw-memory, never the reverse. ``trw-mcp``'s
``resolve_user_memory_dir`` now delegates here, so there is exactly ONE
resolver, not two.

The resolver is cross-platform: it relies only on ``os.environ`` and
``Path.home()``. It creates the directory lazily and never raises on a missing
directory.
"""

from __future__ import annotations

import os
from pathlib import Path

import structlog

from trw_memory._dir_trust import make_private_dirs, open_verified_dir_fd, verify_and_harden_dir_fd
from trw_memory.exceptions import UnsupportedPlatformError

__all__ = ["USER_MEMORY_SUBDIR", "resolve_user_memory_dir"]

logger = structlog.get_logger(__name__)

#: Subdirectory (under the resolved user base) that holds the memory store,
#: mirroring the project layout ``<trw_dir>/memory/memory.db``.
USER_MEMORY_SUBDIR = "memory"
#: XDG application directory under ``$XDG_DATA_HOME``.
_XDG_APP_DIR = "trw"
#: Home fallback base directory name.
_HOME_TRW_DIR = ".trw"


#: The refusal native Windows gets (C12, 7.0): no-follow directory descriptors do not exist there.
UNSUPPORTED_PLATFORM_MESSAGE = (
    "trw-memory 4.0 supports macOS and Linux (glibc); native Windows is not supported in this release; use WSL2"
)


def require_supported_platform() -> None:
    """Refuse native Windows with a named error before any directory is opened or created."""
    if os.name == "nt":
        raise UnsupportedPlatformError(UNSUPPORTED_PLATFORM_MESSAGE)


def resolve_user_memory_dir(*, create: bool = True) -> Path:
    """Resolve the machine-local user-space memory directory.

    Precedence: ``TRW_USER_DIR`` > ``$XDG_DATA_HOME`` > ``~/.trw``.

    Args:
        create: When True (default) ensure the directory exists
            (every created component 0700). When False, create nothing (used
            by presence probes and by the daemon discovery read). Either way an
            EXISTING self-owned trw root and ``memory`` dir that are
            group/world-writable are hardened to 0700 in place (fchmod on a
            no-follow fd), and a foreign-owned one is refused.

    Returns:
        Absolute path to the user-space ``memory`` directory. The user-space
        ``memory.db`` lives at ``<returned>/memory.db``.
    """
    require_supported_platform()
    user_dir = os.environ.get("TRW_USER_DIR")
    if user_dir:
        base = Path(user_dir) / USER_MEMORY_SUBDIR
        source = "trw_user_dir"
    else:
        xdg = os.environ.get("XDG_DATA_HOME")
        if xdg:
            base = Path(xdg) / _XDG_APP_DIR / USER_MEMORY_SUBDIR
            source = "xdg_data_home"
        else:
            base = Path.home() / _HOME_TRW_DIR / USER_MEMORY_SUBDIR
            source = "home_fallback"

    resolved = base.resolve()
    if create:
        make_private_dirs(resolved)
        _verify_trusted(resolved.parent)
        _verify_trusted(resolved)
    elif resolved.exists():
        # A presence probe / discovery read: do not create anything, but a
        # directory that DOES already exist is still verified before any
        # caller trusts it -- ``.resolve()`` above already followed any
        # symlink in the base path, so this is the check on the real,
        # final directory a swapped ``~/.trw`` (or ``TRW_USER_DIR``) would
        # have redirected to (PRD-SEC-016).
        _verify_trusted(resolved.parent)
        _verify_trusted(resolved)
    logger.debug("user_memory_dir_resolved", path=str(resolved), source=source, created=create)
    return resolved


def _verify_trusted(resolved: Path) -> None:
    """Refuse *resolved* if it is a symlink, or group/world-writable and not ours; else harden it to 0700.

    Called for the ``memory`` leaf AND its parent, the trw-owned root (``~/.trw``,
    ``$XDG_DATA_HOME/trw`` or ``TRW_USER_DIR``): 3.x created that root with
    ``mkdir(parents=True)``, so under a 0002 umask it is 0775 and the daemon's
    :func:`~trw_memory._dir_trust.verify_ancestor_chain_trusted` would refuse to
    start. Nothing above the trw root is ever chmodded.
    """
    fd = open_verified_dir_fd(resolved, create=False)
    try:
        verify_and_harden_dir_fd(fd, resolved)
    finally:
        os.close(fd)
