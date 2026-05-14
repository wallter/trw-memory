# ruff: noqa: F401
"""Tests for ``trw_memory.security.keys.store_master_key``."""

from __future__ import annotations

import stat
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from trw_memory.exceptions import ConfigError
from trw_memory.security import generate_master_key, get_master_key, store_master_key

from ._test_keys_support import _make_config, clear_master_key_cache_fixture


class TestStoreMasterKeyFile:
    def test_stores_key_to_file(self, tmp_path: Path) -> None:
        key = generate_master_key()
        key_file = tmp_path / "stored.key"
        config = _make_config(key_source="file", key_file_path=str(key_file), encryption_enabled=False)
        store_master_key(key, config)
        assert key_file.exists()
        assert key_file.read_bytes() == key

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        key = generate_master_key()
        key_file = tmp_path / "nested" / "dir" / "master.key"
        config = _make_config(key_source="file", key_file_path=str(key_file), encryption_enabled=False)
        store_master_key(key, config)
        assert key_file.exists()

    def test_file_permissions_owner_only(self, tmp_path: Path) -> None:
        key = generate_master_key()
        key_file = tmp_path / "perms.key"
        config = _make_config(key_source="file", key_file_path=str(key_file), encryption_enabled=False)
        store_master_key(key, config)
        if sys.platform != "win32":
            mode = key_file.stat().st_mode
            assert not (mode & stat.S_IRGRP), "Group should not have read access"
            assert not (mode & stat.S_IROTH), "Others should not have read access"

    def test_stored_key_can_be_retrieved(self, tmp_path: Path) -> None:
        key = generate_master_key()
        key_file = tmp_path / "roundtrip.key"
        config = _make_config(key_source="file", key_file_path=str(key_file), encryption_enabled=False)
        store_master_key(key, config)
        retrieved = get_master_key(config)
        assert retrieved == key

    def test_overwrites_existing_key_file(self, tmp_path: Path) -> None:
        old_key = generate_master_key()
        new_key = generate_master_key()
        key_file = tmp_path / "overwrite.key"
        config = _make_config(key_source="file", key_file_path=str(key_file), encryption_enabled=False)
        store_master_key(old_key, config)
        store_master_key(new_key, config)
        assert key_file.read_bytes() == new_key

    def test_rejects_short_key(self, tmp_path: Path) -> None:
        key_file = tmp_path / "bad.key"
        config = _make_config(key_source="file", key_file_path=str(key_file), encryption_enabled=False)
        with pytest.raises(ConfigError, match="Master key must be 32 bytes"):
            store_master_key(b"tooshort", config)

    def test_rejects_long_key(self, tmp_path: Path) -> None:
        key_file = tmp_path / "bad.key"
        config = _make_config(key_source="file", key_file_path=str(key_file), encryption_enabled=False)
        with pytest.raises(ConfigError, match="Master key must be 32 bytes"):
            store_master_key(b"x" * 64, config)

    def test_rejects_empty_key(self, tmp_path: Path) -> None:
        key_file = tmp_path / "empty.key"
        config = _make_config(key_source="file", key_file_path=str(key_file), encryption_enabled=False)
        with pytest.raises(ConfigError, match="Master key must be 32 bytes"):
            store_master_key(b"", config)

    @pytest.mark.parametrize("bad_length", [0, 1, 16, 31, 33, 48, 64])
    def test_parametrized_bad_key_lengths_rejected(self, tmp_path: Path, bad_length: int) -> None:
        key_file = tmp_path / f"bad_{bad_length}.key"
        config = _make_config(key_source="file", key_file_path=str(key_file), encryption_enabled=False)
        with pytest.raises(ConfigError):
            store_master_key(b"x" * bad_length, config)


class TestStoreMasterKeyEnv:
    def test_env_source_raises_config_error(self) -> None:
        key = generate_master_key()
        config = _make_config(key_source="env")
        with pytest.raises(ConfigError, match="Cannot persist key to env var"):
            store_master_key(key, config)


class TestStoreMasterKeyKeyring:
    def test_stores_key_to_keyring(self) -> None:
        key = generate_master_key()
        mock_keyring = MagicMock()

        with (
            patch("trw_memory.security.keys._keyring", mock_keyring),
            patch("trw_memory.security.keys._KEYRING_AVAILABLE", True),
        ):
            config = _make_config(key_source="keyring")
            store_master_key(key, config)

        mock_keyring.set_password.assert_called_once_with("trw-memory", "master", key.hex())

    def test_raises_config_error_when_keyring_unavailable(self) -> None:
        key = generate_master_key()
        with (
            patch("trw_memory.security.keys._KEYRING_AVAILABLE", False),
            patch("trw_memory.security.keys._keyring", None),
        ):
            config = _make_config(key_source="keyring")
            with pytest.raises(ConfigError, match="keyring package not installed"):
                store_master_key(key, config)

    def test_raises_config_error_when_keyring_set_password_fails(self) -> None:
        key = generate_master_key()
        mock_keyring = MagicMock()
        mock_keyring.set_password.side_effect = RuntimeError("keyring error")

        with (
            patch("trw_memory.security.keys._keyring", mock_keyring),
            patch("trw_memory.security.keys._KEYRING_AVAILABLE", True),
        ):
            config = _make_config(key_source="keyring")
            with pytest.raises(ConfigError, match="Failed to store key in keyring"):
                store_master_key(key, config)
