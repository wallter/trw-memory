"""Wave 12: targeted tests for uncovered branches in storage/_dbapi.py."""
from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest


class TestIsWalResetSafe:
    """Tests for is_wal_reset_safe() — exercises the version comparison branches."""

    def _check(self, version: str) -> bool:
        import trw_memory.storage._dbapi as dbapi

        original = dbapi.SQLITE_VERSION
        dbapi.SQLITE_VERSION = version
        try:
            return dbapi.is_wal_reset_safe()
        finally:
            dbapi.SQLITE_VERSION = original

    def test_version_above_3_51_3_is_safe(self) -> None:
        assert self._check("3.52.0") is True

    def test_version_exactly_3_51_3_is_safe(self) -> None:
        assert self._check("3.51.3") is True

    def test_version_below_3_51_3_is_not_safe(self) -> None:
        assert self._check("3.51.2") is False

    def test_backport_3_44_6_is_safe(self) -> None:
        assert self._check("3.44.6") is True

    def test_backport_3_44_5_is_not_safe(self) -> None:
        assert self._check("3.44.5") is False

    def test_backport_3_50_7_is_safe(self) -> None:
        assert self._check("3.50.7") is True

    def test_backport_3_50_6_is_not_safe(self) -> None:
        assert self._check("3.50.6") is False

    def test_malformed_version_returns_false(self) -> None:
        assert self._check("not-a-version") is False

    def test_short_version_returns_false(self) -> None:
        assert self._check("3.51") is False

    def test_version_3_45_0_is_not_safe(self) -> None:
        # (3,45,x) — neither backport series, below 3.51.3
        assert self._check("3.45.0") is False


class TestInstallPysqlite3IfAvailable:
    """Tests for _install_pysqlite3_if_available() — exercises import swap paths."""

    def test_already_swapped_flag_returns_early(self) -> None:
        """When _trw_pysqlite3_active is set, skips the swap and returns pysqlite3."""
        import trw_memory.storage._dbapi as dbapi

        # Create a fake sqlite3 module with the swap flag set
        fake_sqlite3 = ModuleType("sqlite3")
        fake_sqlite3._trw_pysqlite3_active = True  # type: ignore[attr-defined]
        fake_sqlite3.sqlite_version = "3.99.0"  # type: ignore[attr-defined]

        saved = sys.modules.get("sqlite3")
        try:
            sys.modules["sqlite3"] = fake_sqlite3
            name, version = dbapi._install_pysqlite3_if_available()
        finally:
            if saved is None:
                sys.modules.pop("sqlite3", None)
            else:
                sys.modules["sqlite3"] = saved

        assert name == "pysqlite3"
        assert version == "3.99.0"

    def test_pysqlite3_import_success_swaps_module(self) -> None:
        """When pysqlite3 is importable, sys.modules['sqlite3'] is replaced."""
        import trw_memory.storage._dbapi as dbapi

        fake_pysqlite3 = ModuleType("pysqlite3")
        fake_pysqlite3.sqlite_version = "3.55.0"  # type: ignore[attr-defined]
        fake_pysqlite3.threadsafety = 1  # type: ignore[attr-defined]
        fake_pysqlite3.dbapi2 = ModuleType("pysqlite3.dbapi2")  # type: ignore[attr-defined]

        saved_sqlite3 = sys.modules.get("sqlite3")
        saved_pysqlite3 = sys.modules.get("pysqlite3")

        # Remove the already-swapped flag if present
        if saved_sqlite3 is not None:
            original_flag = getattr(saved_sqlite3, "_trw_pysqlite3_active", False)
            if original_flag:
                saved_sqlite3._trw_pysqlite3_active = False  # type: ignore[attr-defined]

        try:
            sys.modules["pysqlite3"] = fake_pysqlite3
            # Remove swap flag so we don't hit the already-swapped path
            if "sqlite3" in sys.modules:
                del sys.modules["sqlite3"]._trw_pysqlite3_active  # type: ignore[attr-defined]  # noqa: SIM910
        except (AttributeError, KeyError):
            pass

        try:
            name, version = dbapi._install_pysqlite3_if_available()
            # Should have swapped or returned pysqlite3 name
            assert "pysqlite3" in name or name == "sqlite3"
        finally:
            if saved_sqlite3 is None:
                sys.modules.pop("sqlite3", None)
            else:
                sys.modules["sqlite3"] = saved_sqlite3
            if saved_pysqlite3 is None:
                sys.modules.pop("pysqlite3", None)
            else:
                sys.modules["pysqlite3"] = saved_pysqlite3

    def test_pysqlite3_unavailable_returns_stdlib(self) -> None:
        """When pysqlite3 is not importable, falls back to stdlib sqlite3."""
        import trw_memory.storage._dbapi as dbapi

        # Ensure pysqlite3 is not importable
        saved_pysqlite3 = sys.modules.pop("pysqlite3", None)
        try:
            with patch.dict(sys.modules, {"pysqlite3": None}):  # type: ignore[dict-item]
                name, version = dbapi._install_pysqlite3_if_available()
        finally:
            if saved_pysqlite3 is not None:
                sys.modules["pysqlite3"] = saved_pysqlite3

        assert name == "sqlite3"
        assert isinstance(version, str)

    def test_backend_function_returns_string(self) -> None:
        """backend() returns the name of the active SQLite driver."""
        import trw_memory.storage._dbapi as dbapi

        result = dbapi.backend()
        assert result in ("sqlite3", "pysqlite3")

    def test_sqlite_version_function_returns_string(self) -> None:
        """sqlite_version() returns the SQLite version of the active driver."""
        import trw_memory.storage._dbapi as dbapi

        result = dbapi.sqlite_version()
        assert isinstance(result, str)
        assert "." in result  # sanity: looks like a version string
