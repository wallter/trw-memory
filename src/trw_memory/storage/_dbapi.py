"""SQLite engine POLICY — two supported engines, one explicit choice.

TRW supports the interpreter's own stdlib ``sqlite3`` (the default, and the only
one the package depends on) and an optionally-installed ``pysqlite3`` wheel that
bundles its own SQLite (the ``[sqlite-fix]`` extra). This module decides which
one the process runs and reports the answer. Everything pysqlite3-SPECIFIC —
import, ``sys.modules`` swap, eviction — is isolated in
``_pysqlite3_shim.py``, which carries its own removal instructions.

Why a policy and not a preference
---------------------------------
Python's stdlib ``sqlite3`` is bound to whatever SQLite was compiled into the
interpreter. Many interpreters ship a SQLite predating the **WAL-reset bug fix**
(3.51.3, 2026-03-13; backports 3.44.6 and 3.50.7): under WAL mode with concurrent
writers those versions can leave the WAL-index header inconsistent, so a later
checkpoint skips a committed transaction and the store reports ``database disk
image is malformed``.

The old rule was "swap pysqlite3 in whenever it imports". Measured 2026-09-16,
every published wheel bundles SQLite **3.51.1** — below the fix — while Homebrew
CPython 3.14 ships **3.53.4** above it. So the old rule could DOWNGRADE a capable
interpreter to an incapable wheel in the name of a fix the wheel does not carry.

The selection rule
------------------
Candidates rank on ``(wal_reset_safe_version(v), version_tuple(v))``; the highest
rank wins, and ``pysqlite3`` must STRICTLY outrank the stdlib to be installed.
Ranking on the safety flag FIRST is load-bearing, not cosmetic: 3.50.7 carries the
backported fix and 3.51.1 does not, so a plain version comparison would replace a
safe engine with an unsafe one — the exact class of mistake this policy exists to
prevent.

The interpreter's own version is read from the top-level ``_sqlite3`` extension
module rather than from ``sqlite3``: ``pysqlite3`` ships its C extension as
``pysqlite3._sqlite3`` and never shadows the stdlib one, so ``_sqlite3`` stays
truthful even while ``sys.modules["sqlite3"]`` is the wheel.

Operational notes
-----------------
- Selection is silent on the happy path; a debug log records the outcome.
- ``pysqlite3`` supports ``conn.enable_load_extension(True)`` on every platform we
  ship to, so sqlite-vec continues to load when it is selected.
- An ENCRYPTED store opens its own DB-API module (``sqlcipher3``) and is therefore
  NOT described by ``backend()`` / ``sqlite_version()`` here. ``SQLiteBackend``
  derives its own ``wal_reset_safe`` from the driver it opened, via
  :func:`wal_reset_safe_version`.
"""

from __future__ import annotations

import logging

from trw_memory.storage import _pysqlite3_shim

logger = logging.getLogger(__name__)

#: The WAL-reset fix landed in 3.51.3, with backports onto the 3.44.x and 3.50.x
#: maintenance series at 3.44.6 and 3.50.7.
_WAL_RESET_FIX = (3, 51, 3)


def version_tuple(version: str) -> tuple[int, ...]:
    """Parse ``"3.51.3"`` into ``(3, 51, 3)``; ``()`` when unparseable."""
    try:
        return tuple(int(part) for part in version.split(".")[:3])
    except ValueError:
        return ()


def wal_reset_safe_version(version: str) -> bool:
    """``True`` when *version* carries the SQLite WAL-reset bug fix.

    Applies to ANY version string, not just this process's driver — the doctor
    row uses it on versions read out of other interpreters, and ``SQLiteBackend``
    uses it on the driver it actually opened.
    """
    parsed = version_tuple(version)
    if len(parsed) < 3:
        return False
    if parsed >= _WAL_RESET_FIX:
        return True
    major, minor, patch = parsed
    if (major, minor) == (3, 44):
        return patch >= 6
    return (major, minor) == (3, 50) and patch >= 7


def _driver_rank(version: str) -> tuple[bool, tuple[int, ...]]:
    """Rank an engine: carrying the WAL-reset fix outranks being newer."""
    return (wal_reset_safe_version(version), version_tuple(version))


def stdlib_sqlite_version() -> str:
    """The interpreter's OWN SQLite version, read past any active swap.

    ``_sqlite3`` is the stdlib C accelerator; ``pysqlite3`` ships its own as
    ``pysqlite3._sqlite3`` and never replaces this name. An interpreter built
    without SQLite has no ``_sqlite3`` at all, which reports as ``""`` — an
    unparseable, unsafe version, so any real candidate outranks it.
    """
    try:
        import _sqlite3
    except ImportError:  # trw-fail-silent-allow: an interpreter with no _sqlite3 has no stdlib engine to rank, and "" ranks BELOW every real version, so the candidate wins rather than the absence being mistaken for a capable engine  # pragma: no cover
        return ""
    return str(getattr(_sqlite3, "sqlite_version", ""))


def select_driver() -> tuple[str, str]:
    """Install and report the highest-ranked SQLite engine available.

    Returns ``(driver_name, sqlite_version)`` for the engine every later
    ``import sqlite3`` in this process resolves to.
    """
    stdlib_version = stdlib_sqlite_version()
    candidate = _pysqlite3_shim.candidate_version()
    if candidate and _driver_rank(candidate) > _driver_rank(stdlib_version) and _pysqlite3_shim.install():
        logger.debug(
            "sqlite_driver_selected driver=pysqlite3 version=%s stdlib=%s",
            candidate,
            stdlib_version,
        )
        return ("pysqlite3", candidate)

    # The stdlib wins (or is all there is). Undo the provisional swap that
    # ``trw_memory/__init__.py`` performs before any policy can run, while
    # nothing has captured the rejected module yet.
    _pysqlite3_shim.evict_active_swap()
    logger.debug(
        "sqlite_driver_selected driver=sqlite3 version=%s rejected_pysqlite3=%s",
        stdlib_version,
        candidate or "absent",
    )
    return ("sqlite3", stdlib_version)


BACKEND, SQLITE_VERSION = select_driver()


def backend() -> str:
    """Return ``'pysqlite3'`` when the wheel was selected, else ``'sqlite3'``."""
    return BACKEND


def sqlite_version() -> str:
    """SQLite version string of the SELECTED engine (e.g. ``'3.53.4'``)."""
    return SQLITE_VERSION


def is_wal_reset_safe() -> bool:
    """``True`` when the selected engine carries the WAL-reset bug fix."""
    return wal_reset_safe_version(SQLITE_VERSION)
