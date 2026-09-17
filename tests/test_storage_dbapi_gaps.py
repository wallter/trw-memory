"""Engine POLICY and the WAL-reset version predicate (PRD-INFRA-185 FR02).

``_dbapi`` chooses between the two supported engines; ``_pysqlite3_shim`` (tested
in ``test_pysqlite3_shim.py``) is the mechanism that installs the optional one.
The interesting branch cannot be reached by installing anything: on this
repository's own interpreter (CPython 3.14, stdlib SQLite 3.53.4) no published
pysqlite3 wheel is newer, and on an old interpreter no wheel is older, so both
directions are driven with a synthetic ``pysqlite3`` module.
"""

from __future__ import annotations

import sys
from types import ModuleType

import pytest

import trw_memory.storage._dbapi as dbapi


def _fake_pysqlite3(version: str) -> ModuleType:
    """A stand-in with the attributes the policy reads.

    Kept local rather than imported from ``test_pysqlite3_shim``: these files sit
    in a flat, non-package test tree, so a cross-file import depends on pytest's
    rootdir insertion and breaks when either file is run alone.
    """
    module = ModuleType("pysqlite3")
    module.sqlite_version = version  # type: ignore[attr-defined]
    module.threadsafety = 1  # type: ignore[attr-defined]
    module.dbapi2 = ModuleType("pysqlite3.dbapi2")  # type: ignore[attr-defined]
    return module


class TestWalResetSafeVersion:
    """The 3.51.3 threshold and its two maintenance backports."""

    @pytest.mark.parametrize(
        ("version", "expected"),
        [
            ("3.52.0", True),
            ("3.51.3", True),
            ("3.51.2", False),
            ("3.51.1", False),
            ("3.44.6", True),
            ("3.44.5", False),
            ("3.50.7", True),
            ("3.50.6", False),
            ("3.45.0", False),
            ("4.0.0", True),
            ("not-a-version", False),
            ("3.51", False),
            ("", False),
        ],
    )
    def test_threshold(self, version: str, expected: bool) -> None:
        assert dbapi.wal_reset_safe_version(version) is expected

    def test_is_wal_reset_safe_reads_the_selected_version(self) -> None:
        original = dbapi.SQLITE_VERSION
        try:
            dbapi.SQLITE_VERSION = "3.50.6"
            assert dbapi.is_wal_reset_safe() is False
            dbapi.SQLITE_VERSION = "3.50.7"
            assert dbapi.is_wal_reset_safe() is True
        finally:
            dbapi.SQLITE_VERSION = original


@pytest.fixture
def sqlite_modules_restored() -> object:
    """Snapshot and restore every sys.modules entry select_driver may touch."""
    names = ("sqlite3", "sqlite3.dbapi2", "pysqlite3")
    saved = {name: sys.modules.get(name) for name in names}
    yield None
    for name, module in saved.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


class TestSelectDriver:
    """``select_driver`` ranks on (carries the fix, version) — never version alone."""

    def test_no_pysqlite3_reports_the_stdlib(self, sqlite_modules_restored: object) -> None:
        sys.modules.pop("pysqlite3", None)
        sys.modules["pysqlite3"] = None  # type: ignore[assignment]  # blocks the import
        name, version = dbapi.select_driver()
        assert name == "sqlite3"
        import sqlite3

        assert version == sqlite3.sqlite_version

    def test_a_newer_pysqlite3_still_wins(self, sqlite_modules_restored: object) -> None:
        sys.modules["pysqlite3"] = _fake_pysqlite3("9.9.9")
        name, version = dbapi.select_driver()
        assert (name, version) == ("pysqlite3", "9.9.9")
        assert sys.modules["sqlite3"] is sys.modules["pysqlite3"]

    def test_an_older_pysqlite3_loses_and_an_eager_swap_is_reverted(
        self, sqlite_modules_restored: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The defect this FR exists for: the wheel bundles 3.51.1, the stdlib is newer.

        The stdlib version is pinned, as the sibling tests pin it: on the public
        CI's Ubuntu runner the interpreter's own SQLite is 3.45, older than the
        wheel, so the wheel correctly wins there and the unpinned form of this
        test failed on the v0.19.0 tag (2026-09-17).
        """
        monkeypatch.setattr(dbapi, "stdlib_sqlite_version", lambda: "3.53.4")
        stale = _fake_pysqlite3("3.51.1")
        sys.modules["pysqlite3"] = stale
        # Simulate the eager, unconditional swap performed by trw_memory/__init__.py
        # BEFORE this module gets a chance to judge the candidate.
        sys.modules["sqlite3"] = stale
        sys.modules["sqlite3.dbapi2"] = stale.dbapi2  # type: ignore[attr-defined]
        stale._trw_pysqlite3_active = True  # type: ignore[attr-defined]

        name, version = dbapi.select_driver()

        assert name == "sqlite3"
        assert version == "3.53.4"
        assert sys.modules["sqlite3"] is not stale
        assert sys.modules["sqlite3"].sqlite_version != "3.51.1"

    def test_a_safe_backport_outranks_a_newer_unsafe_wheel(
        self, sqlite_modules_restored: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """3.50.7 carries the backported fix; 3.51.1 does not, despite being newer."""
        monkeypatch.setattr(dbapi, "stdlib_sqlite_version", lambda: "3.50.7")
        sys.modules["pysqlite3"] = _fake_pysqlite3("3.51.1")
        name, _ = dbapi.select_driver()
        assert name == "sqlite3"

    def test_a_newer_safe_wheel_beats_an_unsafe_stdlib(
        self, sqlite_modules_restored: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(dbapi, "stdlib_sqlite_version", lambda: "3.46.1")
        sys.modules["pysqlite3"] = _fake_pysqlite3("3.53.4")
        assert dbapi.select_driver() == ("pysqlite3", "3.53.4")

    def test_an_unparseable_stdlib_version_loses_to_any_candidate(
        self, sqlite_modules_restored: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An interpreter built without _sqlite3 reports "" — unsafe and unranked."""
        monkeypatch.setattr(dbapi, "stdlib_sqlite_version", lambda: "")
        sys.modules["pysqlite3"] = _fake_pysqlite3("3.46.1")
        assert dbapi.select_driver() == ("pysqlite3", "3.46.1")

    def test_selection_is_idempotent(self, sqlite_modules_restored: object) -> None:
        sys.modules["pysqlite3"] = _fake_pysqlite3("9.9.9")
        first = dbapi.select_driver()
        second = dbapi.select_driver()
        assert first == second


class TestDriverReporting:
    """The module-level report every consumer reads."""

    def test_backend_names_one_of_the_two_drivers(self) -> None:
        assert dbapi.backend() in ("sqlite3", "pysqlite3")

    def test_sqlite_version_looks_like_a_version(self) -> None:
        assert len(dbapi.version_tuple(dbapi.sqlite_version())) == 3

    def test_the_selected_module_is_the_one_import_sqlite3_resolves_to(self) -> None:
        import sqlite3

        assert sqlite3.sqlite_version == dbapi.sqlite_version()
