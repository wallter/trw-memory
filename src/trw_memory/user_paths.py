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

__all__ = ["USER_MEMORY_SUBDIR", "resolve_user_memory_dir"]

logger = structlog.get_logger(__name__)

#: Subdirectory (under the resolved user base) that holds the memory store,
#: mirroring the project layout ``<trw_dir>/memory/memory.db``.
USER_MEMORY_SUBDIR = "memory"
#: XDG application directory under ``$XDG_DATA_HOME``.
_XDG_APP_DIR = "trw"
#: Home fallback base directory name.
_HOME_TRW_DIR = ".trw"


def resolve_user_memory_dir(*, create: bool = True) -> Path:
    """Resolve the machine-local user-space memory directory.

    Precedence: ``TRW_USER_DIR`` > ``$XDG_DATA_HOME`` > ``~/.trw``.

    Args:
        create: When True (default) ensure the directory exists
            (``mkdir(parents=True, exist_ok=True)``). When False, resolve the
            path without touching the filesystem (used by presence probes and
            by the daemon discovery read, which must not create anything).

    Returns:
        Absolute path to the user-space ``memory`` directory. The user-space
        ``memory.db`` lives at ``<returned>/memory.db``.
    """
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
        resolved.mkdir(parents=True, exist_ok=True)
    logger.debug("user_memory_dir_resolved", path=str(resolved), source=source, created=create)
    return resolved
