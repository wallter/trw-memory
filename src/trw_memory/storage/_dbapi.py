"""SQLite driver preference shim — prefer ``pysqlite3`` over stdlib ``sqlite3``.

Why this exists
---------------
Python's stdlib ``sqlite3`` is bound to whatever SQLite version was bundled
when Python itself was built. Many Python releases ship SQLite versions that
predate the **WAL-reset bug fix** landed in SQLite 3.51.3 (2026-03-13;
backports 3.44.6, 3.50.7). Under WAL mode with concurrent writers, the
pre-fix versions can leave the WAL-index header in an inconsistent state,
causing a later checkpoint to skip a committed transaction and producing
``database disk image is malformed``.

The fix is to use a SQLite build at or beyond 3.51.3. ``pysqlite3-binary``
bundles a recent SQLite as a wheel, independent of the Python interpreter's
own ``sqlite3``. By installing the wheel we get the fix on every platform.

How it works
------------
Importing this module performs a one-time, idempotent swap:

    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

After the swap, every subsequent ``import sqlite3`` resolves to the
``pysqlite3`` module — which keeps the stdlib API surface (so the 64+
``except sqlite3.Error`` callsites continue to work) but ships modern
SQLite under the hood.

When ``pysqlite3`` is not installed (or the swap has already happened),
this module is a no-op and the stdlib driver remains in use.

This module MUST be imported before any other code imports ``sqlite3`` so
that the stdlib driver isn't already cached. ``trw_memory/__init__.py``
imports it as its first statement; ``trw_mcp/__init__.py`` does the same
for the MCP server. Tests can import it directly at the top of any
``conftest.py``.

Operational notes
-----------------
- The swap is silent on the happy path; debug log records which driver was
  selected so ops can confirm via structured logs.
- ``pysqlite3`` supports ``conn.enable_load_extension(True)`` on every
  platform we ship to, so sqlite-vec continues to load.
- Exception classes raised by ``pysqlite3`` are caught by
  ``except sqlite3.Error`` after the swap because both names resolve to
  the same module object in ``sys.modules``.
"""

from __future__ import annotations

import logging
import sys

logger = logging.getLogger(__name__)

# Avoid a second swap (e.g., when trw-mcp imports trw-memory which already
# performed the swap) or a no-op when stdlib has been deliberately pinned.
_ALREADY_SWAPPED_FLAG = "_trw_pysqlite3_active"


def _install_pysqlite3_if_available() -> tuple[str, str]:
    """Swap stdlib sqlite3 with pysqlite3 when the wheel is installed.

    Returns ``(backend_name, sqlite_version)`` for observability.
    """
    stdlib_sqlite = sys.modules.get("sqlite3")
    if stdlib_sqlite is not None and getattr(stdlib_sqlite, _ALREADY_SWAPPED_FLAG, False):
        return ("pysqlite3", stdlib_sqlite.sqlite_version)

    try:
        import pysqlite3  # type: ignore[import-untyped]
    except ImportError:
        # Fallback path: stdlib sqlite3 (may carry the WAL-reset bug on older
        # interpreter builds). We do not warn — this is a soft preference.
        if stdlib_sqlite is None:
            import sqlite3 as _sqlite3

            stdlib_sqlite = _sqlite3
        return ("sqlite3", stdlib_sqlite.sqlite_version)

    sys.modules["sqlite3"] = pysqlite3
    sys.modules["sqlite3.dbapi2"] = pysqlite3.dbapi2
    pysqlite3._trw_pysqlite3_active = True
    logger.debug(
        "pysqlite3_active sqlite_version=%s threadsafety=%s",
        pysqlite3.sqlite_version,
        pysqlite3.threadsafety,
    )
    return ("pysqlite3", pysqlite3.sqlite_version)


BACKEND, SQLITE_VERSION = _install_pysqlite3_if_available()


def backend() -> str:
    """Return ``'pysqlite3'`` if the wheel was loaded, else ``'sqlite3'``."""
    return BACKEND


def sqlite_version() -> str:
    """SQLite version string of the active driver (e.g. ``'3.52.0'``)."""
    return SQLITE_VERSION


def is_wal_reset_safe() -> bool:
    """``True`` when the active driver carries the WAL-reset bug fix."""
    parts = SQLITE_VERSION.split(".")
    try:
        major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
    except (ValueError, IndexError):
        return False
    if (major, minor) >= (3, 51) and (major, minor, patch) >= (3, 51, 3):
        return True
    # Backports: 3.44.6 and 3.50.7
    if (major, minor) == (3, 44) and patch >= 6:
        return True
    return (major, minor) == (3, 50) and patch >= 7
