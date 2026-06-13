"""Wave 14: coverage gap-fill for security/startup.py (lines 21-34, 45-60, 65-81)."""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from trw_memory.exceptions import SecurityDefaultUnresolvableError
from trw_memory.models.config import MemoryConfig
from trw_memory.security.startup import _discover_anchor, resolve_security_path, verify_defaults


class TestDiscoverAnchorTrwDir:
    def test_explicit_trw_dir_is_returned_resolved(self, tmp_path: Path) -> None:
        """explicit trw_dir → returned as resolved Path (line 21)."""
        cfg = MemoryConfig()
        result = _discover_anchor(cfg, trw_dir=tmp_path)
        assert result == tmp_path.resolve()

    def test_env_trw_dir_is_used_when_set(self, tmp_path: Path) -> None:
        """TRW_DIR env var → used as anchor (lines 23-24)."""
        cfg = MemoryConfig()
        with patch.dict(os.environ, {"TRW_DIR": str(tmp_path)}):
            result = _discover_anchor(cfg)
        assert result == tmp_path.resolve()

    def test_absolute_storage_path_uses_parent(self, tmp_path: Path) -> None:
        """absolute storage_path → parent is the anchor (lines 25-27)."""
        storage_path = tmp_path / "memory" / "store.db"
        storage_path.parent.mkdir(parents=True)
        cfg = MemoryConfig(storage_path=str(storage_path))
        result = _discover_anchor(cfg)
        assert result == storage_path.parent.resolve()

    def test_cwd_walk_finds_trw_dir_when_exists(self) -> None:
        """CWD walk hits .trw discovery branch (lines 28-32) when .trw exists in parent tree."""
        cfg = MemoryConfig(storage_path="relative/path.db")
        # /tmp/.trw exists in this environment; the walk from /tmp will find it
        with patch.dict(os.environ, {"TRW_DIR": ""}):
            with patch("trw_memory.security.startup.Path.cwd", return_value=Path("/tmp")):
                result = _discover_anchor(cfg)
        assert isinstance(result, Path)

    def test_raises_when_no_trw_dir_found(self) -> None:
        """All .trw checks return False → SecurityDefaultUnresolvableError (lines 33-34)."""
        cfg = MemoryConfig(storage_path="relative/path.db")
        with patch.dict(os.environ, {"TRW_DIR": ""}):
            with patch("trw_memory.security.startup.Path.cwd", return_value=Path("/tmp")):
                with patch("trw_memory.security.startup.Path.exists", return_value=False):
                    with pytest.raises(SecurityDefaultUnresolvableError, match="unable to resolve"):
                        _discover_anchor(cfg)


class TestResolveSecurityPath:
    def test_empty_field_value_raises(self) -> None:
        """Empty field value → SecurityDefaultUnresolvableError (lines 45-46)."""
        cfg = MemoryConfig()
        with patch.object(cfg, "canary_fixtures_path", ""):
            with pytest.raises(SecurityDefaultUnresolvableError, match="empty"):
                resolve_security_path(cfg, "canary_fixtures_path")

    def test_package_prefix_resolves_from_package(self) -> None:
        """'package:' prefix → resolved from package resources (lines 47-48)."""
        cfg = MemoryConfig()
        with patch.object(cfg, "canary_fixtures_path", "package:fixtures"):
            result = resolve_security_path(cfg, "canary_fixtures_path")
        assert isinstance(result, Path)

    def test_absolute_path_resolved_directly(self, tmp_path: Path) -> None:
        """Absolute path → resolved directly (lines 51-53)."""
        abs_path = tmp_path / "key.pem"
        cfg = MemoryConfig()
        with patch.object(cfg, "provenance_signing_key_path", str(abs_path)):
            result = resolve_security_path(cfg, "provenance_signing_key_path")
        assert result == abs_path.resolve()

    def test_create_parent_makes_directory(self, tmp_path: Path) -> None:
        """create_parent=True → parent dir created (line 55)."""
        abs_path = tmp_path / "subdir" / "key.pem"
        cfg = MemoryConfig()
        with patch.object(cfg, "provenance_signing_key_path", str(abs_path)):
            resolve_security_path(cfg, "provenance_signing_key_path", create_parent=True)
        assert abs_path.parent.exists()


class TestVerifyDefaults:
    def test_verify_defaults_passes_when_paths_exist(self, tmp_path: Path) -> None:
        """verify_defaults completes when all paths resolve correctly (lines 65-81)."""
        canary_dir = tmp_path / "fixtures"
        canary_dir.mkdir()
        quarantine_db = tmp_path / "quarantine.db"
        signing_key = tmp_path / "signing.pem"

        cfg = MemoryConfig()
        with (
            patch.object(cfg, "canary_fixtures_path", str(canary_dir)),
            patch.object(cfg, "quarantine_db_path", str(quarantine_db)),
            patch.object(cfg, "provenance_signing_key_path", str(signing_key)),
        ):
            verify_defaults(cfg)

    def test_verify_defaults_raises_when_canary_dir_missing(self, tmp_path: Path) -> None:
        """Missing canary fixtures dir → SecurityDefaultUnresolvableError (lines 66-67)."""
        canary_dir = tmp_path / "nonexistent_fixtures"
        quarantine_db = tmp_path / "quarantine.db"
        signing_key = tmp_path / "signing.pem"

        cfg = MemoryConfig()
        with (
            patch.object(cfg, "canary_fixtures_path", str(canary_dir)),
            patch.object(cfg, "quarantine_db_path", str(quarantine_db)),
            patch.object(cfg, "provenance_signing_key_path", str(signing_key)),
        ):
            with pytest.raises(SecurityDefaultUnresolvableError, match="canary fixtures"):
                verify_defaults(cfg)

    def test_oserror_on_quarantine_mkdir_raises(self, tmp_path: Path) -> None:
        """OSError creating quarantine DB parent → SecurityDefaultUnresolvableError (lines 72-73)."""
        canary_dir = tmp_path / "fixtures"
        canary_dir.mkdir()
        # quarantine_db parent already exists → resolve_security_path mkdir is no-op
        quarantine_parent = tmp_path / "qdb"
        quarantine_parent.mkdir()
        quarantine_db = quarantine_parent / "quarantine.db"
        signing_key = tmp_path / "signing.pem"

        cfg = MemoryConfig()
        call_count = {"n": 0}

        def _mkdir_fail_on_second(*_args: object, **_kwargs: object) -> None:
            call_count["n"] += 1
            if call_count["n"] >= 2:
                raise OSError("no space")

        with (
            patch.object(cfg, "canary_fixtures_path", str(canary_dir)),
            patch.object(cfg, "quarantine_db_path", str(quarantine_db)),
            patch.object(cfg, "provenance_signing_key_path", str(signing_key)),
            patch("trw_memory.security.startup.Path.mkdir", side_effect=_mkdir_fail_on_second),
        ):
            with pytest.raises(SecurityDefaultUnresolvableError, match="quarantine DB path"):
                verify_defaults(cfg)

    def test_oserror_on_signing_key_mkdir_raises(self, tmp_path: Path) -> None:
        """OSError creating signing key parent → SecurityDefaultUnresolvableError (lines 78-79)."""
        canary_dir = tmp_path / "fixtures"
        canary_dir.mkdir()
        quarantine_parent = tmp_path / "qdb"
        quarantine_parent.mkdir()
        quarantine_db = quarantine_parent / "quarantine.db"
        signing_parent = tmp_path / "keys"
        signing_parent.mkdir()
        signing_key = signing_parent / "signing.pem"

        cfg = MemoryConfig()
        call_count = {"n": 0}

        def _mkdir_fail_on_fourth(*_args: object, **_kwargs: object) -> None:
            call_count["n"] += 1
            if call_count["n"] >= 4:
                raise OSError("permission denied")

        with (
            patch.object(cfg, "canary_fixtures_path", str(canary_dir)),
            patch.object(cfg, "quarantine_db_path", str(quarantine_db)),
            patch.object(cfg, "provenance_signing_key_path", str(signing_key)),
            patch("trw_memory.security.startup.Path.mkdir", side_effect=_mkdir_fail_on_fourth),
        ):
            with pytest.raises(SecurityDefaultUnresolvableError, match="provenance key"):
                verify_defaults(cfg)
