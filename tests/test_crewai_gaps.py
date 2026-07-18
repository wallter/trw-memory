"""Wave 15: coverage gap-fill for integrations/crewai.py.

Target lines: 29-30, 44, 51-52, 192.
"""

# ruff: noqa: F811  # pytest fixture imports are consumed by same-named parameters
from __future__ import annotations

import importlib
import importlib.metadata
import sys
from unittest.mock import patch

import pytest

from ._test_integrations_support import (
    _import_crewai_adapter,
    _make_crewai_mocks,
    _purge_modules,
    tmp_backend,  # noqa: F401  # imported fixture is discovered by pytest
)

_MODULE = "trw_memory.integrations.crewai"


# ---------------------------------------------------------------------------
# line 44: _parse_version break on non-digit part
# ---------------------------------------------------------------------------


class TestParseVersion:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.mocks = _make_crewai_mocks()
        self.mod = _import_crewai_adapter(self.mocks)

    def test_alpha_suffix_stops_at_non_digit_part(self) -> None:
        """_parse_version('1.2.alpha') → break when part has no digits (line 44)."""
        result = self.mod._parse_version("1.2.alpha")
        assert result == (1, 2)

    def test_empty_version_string(self) -> None:
        """_parse_version('') → empty tuple (no parts → loop body never fires)."""
        result = self.mod._parse_version("")
        assert result == ()

    def test_version_hyphen_separator(self) -> None:
        """_parse_version('1-2-0') → hyphens replaced by dots (line 41)."""
        result = self.mod._parse_version("1-2-0")
        assert result == (1, 2, 0)


# ---------------------------------------------------------------------------
# line 192: reset() early return when deleted == 0
# ---------------------------------------------------------------------------


class TestTRWCrewStorageReset:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.mocks = _make_crewai_mocks()
        self.mod = _import_crewai_adapter(self.mocks)

    def test_reset_empty_namespace_early_return(self, tmp_backend) -> None:
        """reset() with no entries → deleted==0 → early return at line 192."""
        storage = self.mod.TRWCrewStorage(namespace="project:crew-empty", backend=tmp_backend)
        storage.reset()  # Should complete without error, returns None

    def test_reset_with_entries_does_not_early_return(self, tmp_backend) -> None:
        """reset() with entries → deleted>0 → does not hit early return."""
        storage = self.mod.TRWCrewStorage(namespace="project:crew-reset", backend=tmp_backend)
        storage.save("content to delete")
        storage.reset()
        entries = tmp_backend.list_entries(namespace="project:crew-reset", limit=10)
        assert len(entries) == 0


# ---------------------------------------------------------------------------
# lines 29-30: find_spec raises ValueError → _crewai_spec = None → ImportError
# ---------------------------------------------------------------------------


class TestModuleLevelFindSpecFailure:
    def test_find_spec_value_error_triggers_import_error(self) -> None:
        """find_spec raises ValueError → _crewai_spec = None (line 30) → ImportError raised."""
        _purge_modules(_MODULE)
        orig_version = importlib.metadata.version

        def _patched_version(name: str) -> str:
            if name == "crewai":
                return "0.74.0"
            return orig_version(name)

        with patch("importlib.util.find_spec", side_effect=ValueError("bad spec")):
            with patch("importlib.metadata.version", side_effect=_patched_version):
                with pytest.raises(ImportError, match="crewai is required"):
                    importlib.import_module(_MODULE)

        _purge_modules(_MODULE)


# ---------------------------------------------------------------------------
# lines 51-52: PackageNotFoundError → ImportError raised
# ---------------------------------------------------------------------------


class TestModuleLevelPackageNotFoundError:
    def test_metadata_not_found_triggers_import_error(self) -> None:
        """importlib.metadata.version raises PackageNotFoundError (lines 51-52) → ImportError."""
        _purge_modules(_MODULE)
        mocks = _make_crewai_mocks()

        def _raise_not_found(name: str) -> str:
            if name == "crewai":
                raise importlib.metadata.PackageNotFoundError("crewai")
            return importlib.metadata.version(name)

        with patch.dict(sys.modules, mocks):
            with patch("importlib.metadata.version", side_effect=_raise_not_found):
                with pytest.raises(ImportError, match="crewai metadata is required"):
                    importlib.import_module(_MODULE)

        _purge_modules(_MODULE)
