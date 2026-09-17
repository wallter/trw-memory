"""Everything about the optional ``pysqlite3`` engine lives here, and nowhere else.

TRW supports TWO SQLite engines: the interpreter's own stdlib ``sqlite3``, and an
optionally-installed ``pysqlite3`` wheel that bundles its own SQLite. The stdlib
is the default and the only one the package depends on. This module is the whole
of the second path — the import, the ``sys.modules`` swap that makes it answer to
the name ``sqlite3``, and the eviction that undoes that swap. ``_dbapi.py`` owns
the POLICY (which engine wins); this module owns the MECHANISM (how the winner is
installed).

Why the wheel is optional, and why it usually loses
--------------------------------------------------
It was a hard Linux dependency until 2026-09-16, on the belief that it delivered
the SQLite 3.51.3 WAL-reset fix. Three measurements said otherwise:

* ``pysqlite3-binary`` 0.5.4.post2 publishes exactly ONE wheel
  (``manylinux2014_x86_64``), so requiring it made ``pip install`` fail outright
  on aarch64 Linux — Graviton, Ampere, Apple-Silicon containers, Raspberry Pi.
* That wheel bundles SQLite **3.51.1**, below the fix it was there to provide.
* The macOS ``pysqlite3`` 0.6.0 wheel bundles 3.51.1 too, while Homebrew CPython
  3.14 ships stdlib SQLite 3.53.4 — so installing the wheel there DOWNGRADED the
  engine in the name of a safety fix.

It is kept, rather than deleted, because a bundled engine is the only lever on an
interpreter whose own SQLite is old and cannot be changed (a locked-down base
image, a vendor Python). Hence: two supported engines, one explicit policy.

How to remove this shim (one commit)
------------------------------------
When every supported interpreter ships a SQLite at or beyond the floor
``_dbapi`` enforces, the second engine stops earning its keep. Removing it is a
single, mechanical commit: (1) delete this module and
``trw-memory/tests/test_pysqlite3_shim.py``; (2) delete the ``[sqlite-fix]``
extra from ``trw-memory/pyproject.toml`` and ``trw-mcp/pyproject.toml`` and
re-lock both; (3) in ``_dbapi.py`` delete the ``_pysqlite3_shim`` import and the
three calls into it (``candidate_version``, ``install``, ``evict_active_swap``),
leaving ``select_driver`` as ``return ("sqlite3", stdlib_sqlite_version())``, and
delete its ``TestSelectDriver`` cases that synthesise a wheel; (4) delete the
provisional swap block at the top of
``trw_memory/__init__.py`` and the ``_dbapi``-first import comment in
``trw_memory/storage/__init__.py`` that exists to beat it; (5) drop the
``pysqlite3`` entry from the mypy override list. Nothing outside those files
imports this module, and no on-disk state depends on it.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

logger = logging.getLogger(__name__)

#: Marks a ``pysqlite3`` module that THIS process installed under the name
#: ``sqlite3``. Lets a later call recognise its own work — including the
#: provisional swap ``trw_memory/__init__.py`` performs before any policy can
#: run — instead of guessing from module identity.
ACTIVE_FLAG = "_trw_pysqlite3_active"


def candidate_version() -> str:
    """SQLite version bundled by an installed ``pysqlite3``; ``""`` when absent.

    Importing ``pysqlite3`` does not install it as ``sqlite3``; that is
    :func:`install` and it happens only when the policy in ``_dbapi`` says so.
    """
    module = _import_pysqlite3()
    return "" if module is None else str(module.sqlite_version)


def install() -> bool:
    """Make ``import sqlite3`` resolve to ``pysqlite3``; ``False`` when absent.

    The swap keeps the stdlib API surface — the 64+ ``except sqlite3.Error``
    call sites keep working, because both names resolve to the same module
    object — with a different SQLite underneath.
    """
    module = _import_pysqlite3()
    if module is None:
        return False
    sys.modules["sqlite3"] = module
    sys.modules["sqlite3.dbapi2"] = module.dbapi2
    setattr(module, ACTIVE_FLAG, True)
    return True


def evict_active_swap() -> bool:
    """Undo a swap this process installed; ``True`` when one was removed.

    ``trw_memory/__init__.py`` swaps ``pysqlite3`` in unconditionally, before any
    submodule can load, because the policy lives in a submodule and would
    otherwise run too late. That swap is PROVISIONAL: when the policy rejects the
    candidate, this puts the stdlib back. It is only effective while nothing has
    captured ``sqlite3`` in its own namespace yet, which is why
    ``trw_memory/storage/__init__.py`` imports ``_dbapi`` before anything else.
    """
    active = sys.modules.get("sqlite3")
    if active is None or not getattr(active, ACTIVE_FLAG, False):
        return False
    del sys.modules["sqlite3"]
    if sys.modules.get("sqlite3.dbapi2") is getattr(active, "dbapi2", None):
        del sys.modules["sqlite3.dbapi2"]
    setattr(active, ACTIVE_FLAG, False)
    # Re-import EAGERLY rather than leaving the name unbound. Deleting the entry
    # alone would leave a window in which ``sqlite3`` is absent from
    # ``sys.modules``, and a concurrent import in another thread would race the
    # first caller to re-create it.
    import sqlite3  # noqa: F401 — restores the stdlib module under its own name

    logger.debug("pysqlite3_swap_evicted version=%s", getattr(active, "sqlite_version", "?"))
    return True


def _import_pysqlite3() -> Any | None:
    try:
        import pysqlite3
    except ImportError:  # trw-fail-silent-allow: absence is the ANSWER here -- the wheel is an optional extra and None routes the caller to the stdlib engine, which is the supported default
        return None
    return pysqlite3


__all__ = ["ACTIVE_FLAG", "candidate_version", "evict_active_swap", "install"]
