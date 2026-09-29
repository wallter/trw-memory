"""Skip helpers for tests that need sqlite-vec.

The isolated release gate installs only ``trw-memory[dev]``. A test whose
behavior needs an extra must say so and skip there, not fail -- and it must
not pass there by accident either, which is why the markers name the extra.

Two different absences, two helpers:

- ``requires_sqlite_vec``: the package is not installed (an import-time skip).
- ``vec_unavailable(reason)``: the package imports but the extension did not
  LOAD (a system SQLite without extension loading, a musl build). It skips by
  default and fails under ``TRW_REQUIRE_SQLITE_VEC=1``, which the CI test job
  sets, so a broken load fails the gate instead of turning ~25 vector tests
  into quiet skips.
"""

from __future__ import annotations

import os
from importlib.util import find_spec
from typing import NoReturn

import pytest

requires_sqlite_vec = pytest.mark.skipif(
    find_spec("sqlite_vec") is None,
    reason="needs sqlite-vec: the SQLite backend reports supports_vectors() False without it; it is a base dependency, so reinstall trw-memory",
)


def vec_unavailable(reason: str) -> NoReturn:
    """Skip (default) or fail (``TRW_REQUIRE_SQLITE_VEC=1``) on a sqlite-vec load failure."""
    if os.environ.get("TRW_REQUIRE_SQLITE_VEC") == "1":
        pytest.fail(reason)
    pytest.skip(reason)  # skip-category: optional-dependency


def refuse_missing_sqlite_vec_when_required() -> str | None:
    """Under ``TRW_REQUIRE_SQLITE_VEC=1``, the reason the session must stop if sqlite-vec cannot import.

    Checked once at configure time, so an import failure fails the gate instead of
    reaching the ``requires_sqlite_vec`` / ``importorskip("sqlite_vec")`` sites as skips.
    """
    if os.environ.get("TRW_REQUIRE_SQLITE_VEC") != "1":
        return None
    try:
        import sqlite_vec  # noqa: F401
    except Exception as exc:  # trw-fail-silent-allow: returned as the refusal reason, which stops the session
        return f"TRW_REQUIRE_SQLITE_VEC=1 but sqlite-vec does not import: {type(exc).__name__}: {exc}"
    return None
