# ruff: noqa: F401
"""Tests for ``trw_memory.security.keys.get_master_key``."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from trw_memory.exceptions import ConfigError, MasterKeyNotFoundError
from trw_memory.models.config import MemoryConfig
from trw_memory.security import generate_master_key, get_master_key

from ._test_keys_support import (
    _KEY_LENGTH,
    _make_config,
    clear_master_key_cache_fixture,
)


class TestGetMasterKeyEnv:
    def test_reads_key_from_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        key = generate_master_key()
        monkeypatch.setenv("MEMORY_MASTER_KEY", key.hex())
        config = _make_config(key_source="env")
        result = get_master_key(config)
        assert result == key

    def test_env_key_returns_32_bytes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        key = generate_master_key()
        monkeypatch.setenv("MEMORY_MASTER_KEY", key.hex())
        config = _make_config(key_source="env")
        result = get_master_key(config)
        assert len(result) == _KEY_LENGTH

    def test_env_key_not_set_raises_master_key_not_found_without_keyring(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MEMORY_MASTER_KEY", raising=False)
        with (
            patch("trw_memory.security.keys._KEYRING_AVAILABLE", False),
            patch("trw_memory.security.keys._keyring", None),
        ):
            config = _make_config(key_source="env", auto_generate_key=False)
            with pytest.raises(MasterKeyNotFoundError, match="No master key found"):
                get_master_key(config)

    def test_env_key_raises_config_error_for_wrong_hex_length(self, monkeypatch: pytest.MonkeyPatch) -> None:
        short_key = bytes(16)
        monkeypatch.setenv("MEMORY_MASTER_KEY", short_key.hex())
        config = _make_config(key_source="env")
        with pytest.raises(ConfigError, match="must decode to"):
            get_master_key(config)

    def test_env_key_raises_config_error_for_invalid_hex(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MEMORY_MASTER_KEY", "not-valid-hex!!")
        config = _make_config(key_source="env")
        with pytest.raises(ConfigError, match="Invalid hex"):
            get_master_key(config)

    def test_no_sources_available_raises_master_key_not_found(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.delenv("MEMORY_MASTER_KEY", raising=False)
        nonexistent_file = tmp_path / "nonexistent.key"
        config = _make_config(
            key_source="env",
            key_file_path=str(nonexistent_file),
            auto_generate_key=False,
        )
        with (
            patch("trw_memory.security.keys._KEYRING_AVAILABLE", False),
            patch("trw_memory.security.keys._keyring", None),
            pytest.raises(MasterKeyNotFoundError, match="No master key found"),
        ):
            get_master_key(config)

    def test_env_mode_falls_through_to_keyring_when_env_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MEMORY_MASTER_KEY", raising=False)
        key = generate_master_key()
        mock_keyring = MagicMock()
        mock_keyring.get_password.return_value = key.hex()

        with (
            patch("trw_memory.security.keys._keyring", mock_keyring),
            patch("trw_memory.security.keys._KEYRING_AVAILABLE", True),
        ):
            config = _make_config(key_source="env", auto_generate_key=False)
            assert get_master_key(config) == key

        mock_keyring.get_password.assert_called_once_with("trw-memory", "master")

    def test_auto_generate_requires_keyring_when_env_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MEMORY_MASTER_KEY", raising=False)

        with patch("trw_memory.security.keys._KEYRING_AVAILABLE", False):
            config = _make_config(key_source="env", auto_generate_key=True)
            with pytest.raises(ConfigError, match="OS keyring unavailable for auto-generated master key"):
                get_master_key(config)


class TestGetMasterKeyFile:
    def test_reads_key_from_file(self, tmp_path: Path) -> None:
        key = generate_master_key()
        key_file = tmp_path / "master.key"
        key_file.write_bytes(key)
        config = _make_config(key_source="file", key_file_path=str(key_file), encryption_enabled=False)
        result = get_master_key(config)
        assert result == key

    def test_file_not_found_raises_master_key_not_found(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.delenv("MEMORY_MASTER_KEY", raising=False)
        nonexistent = tmp_path / "no.key"
        config = _make_config(
            key_source="file",
            key_file_path=str(nonexistent),
            auto_generate_key=False,
            encryption_enabled=False,
        )
        with pytest.raises(MasterKeyNotFoundError, match="No master key found"):
            get_master_key(config)

    def test_file_wrong_size_raises_config_error(self, tmp_path: Path) -> None:
        key_file = tmp_path / "bad.key"
        key_file.write_bytes(b"tooshort")
        config = _make_config(key_source="file", key_file_path=str(key_file), encryption_enabled=False)
        with pytest.raises(ConfigError, match="must contain exactly"):
            get_master_key(config)

    def test_file_too_long_raises_config_error(self, tmp_path: Path) -> None:
        key_file = tmp_path / "toolong.key"
        key_file.write_bytes(b"x" * 64)
        config = _make_config(key_source="file", key_file_path=str(key_file), encryption_enabled=False)
        with pytest.raises(ConfigError, match="must contain exactly"):
            get_master_key(config)

    def test_home_dir_tilde_expansion(self, tmp_path: Path) -> None:
        key = generate_master_key()
        key_file = tmp_path / "master.key"
        key_file.write_bytes(key)
        config = _make_config(key_source="file", key_file_path=str(key_file), encryption_enabled=False)
        result = get_master_key(config)
        assert result == key

    def test_file_source_rejected_when_encryption_enabled(self, tmp_path: Path) -> None:
        key_file = tmp_path / "master.key"
        key_file.write_bytes(generate_master_key())
        config = _make_config(key_source="file", key_file_path=str(key_file), encryption_enabled=True)
        with pytest.raises(ConfigError, match="key_source='file' is unsupported"):
            get_master_key(config)


class TestGetMasterKeyUnknownSource:
    def test_unknown_key_source_raises_config_error(self) -> None:
        config = MemoryConfig(key_source="env")
        object.__setattr__(config, "key_source", "unknown-source")
        with pytest.raises(ConfigError, match="Unknown key_source"):
            get_master_key(config)


class TestGetMasterKeyKeyring:
    def test_reads_key_from_keyring_when_available(self, monkeypatch: pytest.MonkeyPatch) -> None:
        key = generate_master_key()
        mock_keyring = MagicMock()
        mock_keyring.get_password.return_value = key.hex()

        with (
            patch("trw_memory.security.keys._keyring", mock_keyring),
            patch("trw_memory.security.keys._KEYRING_AVAILABLE", True),
        ):
            config = _make_config(key_source="keyring")
            result = get_master_key(config)

        assert result == key
        mock_keyring.get_password.assert_called_once_with("trw-memory", "master")

    def test_keyring_read_is_cached_across_calls(self) -> None:
        key = generate_master_key()
        mock_keyring = MagicMock()
        mock_keyring.get_password.return_value = key.hex()

        with (
            patch("trw_memory.security.keys._keyring", mock_keyring),
            patch("trw_memory.security.keys._KEYRING_AVAILABLE", True),
        ):
            config = _make_config(key_source="keyring")
            assert get_master_key(config) == key
            assert get_master_key(config) == key

        mock_keyring.get_password.assert_called_once_with("trw-memory", "master")

    def test_env_key_wins_without_consulting_keyring(self, monkeypatch: pytest.MonkeyPatch) -> None:
        env_key = generate_master_key()
        monkeypatch.setenv("MEMORY_MASTER_KEY", env_key.hex())
        mock_keyring = MagicMock()
        mock_keyring.get_password.return_value = generate_master_key().hex()

        with (
            patch("trw_memory.security.keys._keyring", mock_keyring),
            patch("trw_memory.security.keys._KEYRING_AVAILABLE", True),
        ):
            config = _make_config(key_source="keyring")
            result = get_master_key(config)

        assert result == env_key
        mock_keyring.get_password.assert_not_called()

    def test_legacy_keyring_account_still_reads(self) -> None:
        key = generate_master_key()
        mock_keyring = MagicMock()
        mock_keyring.get_password.side_effect = [None, key.hex()]

        with (
            patch("trw_memory.security.keys._keyring", mock_keyring),
            patch("trw_memory.security.keys._KEYRING_AVAILABLE", True),
        ):
            config = _make_config(key_source="keyring")
            result = get_master_key(config)

        assert result == key
        assert mock_keyring.get_password.call_args_list[0].args == ("trw-memory", "master")
        assert mock_keyring.get_password.call_args_list[1].args == ("trw-memory", "master-key")

    def test_falls_through_to_env_when_keyring_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        key = generate_master_key()
        monkeypatch.setenv("MEMORY_MASTER_KEY", key.hex())
        mock_keyring = MagicMock()
        mock_keyring.get_password.return_value = None

        with (
            patch("trw_memory.security.keys._keyring", mock_keyring),
            patch("trw_memory.security.keys._KEYRING_AVAILABLE", True),
        ):
            config = _make_config(key_source="keyring")
            result = get_master_key(config)

        assert result == key
        mock_keyring.get_password.assert_not_called()

    def test_keyring_without_env_or_keyring_entry_raises_when_unavailable(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("MEMORY_MASTER_KEY", raising=False)
        with (
            patch("trw_memory.security.keys._KEYRING_AVAILABLE", False),
            patch("trw_memory.security.keys._keyring", None),
        ):
            config = _make_config(key_source="keyring", auto_generate_key=False)
            with pytest.raises(MasterKeyNotFoundError, match="No master key found"):
                get_master_key(config)

    def test_keyring_get_password_exception_falls_through(self, monkeypatch: pytest.MonkeyPatch) -> None:
        key = generate_master_key()
        monkeypatch.setenv("MEMORY_MASTER_KEY", key.hex())
        mock_keyring = MagicMock()
        mock_keyring.get_password.side_effect = RuntimeError("keyring broken")

        with (
            patch("trw_memory.security.keys._keyring", mock_keyring),
            patch("trw_memory.security.keys._KEYRING_AVAILABLE", True),
        ):
            config = _make_config(key_source="keyring")
            result = get_master_key(config)

        assert result == key

    def test_generates_new_key_in_keyring_when_enabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MEMORY_MASTER_KEY", raising=False)
        mock_keyring = MagicMock()
        mock_keyring.get_password.return_value = None

        with (
            patch("trw_memory.security.keys._keyring", mock_keyring),
            patch("trw_memory.security.keys._KEYRING_AVAILABLE", True),
        ):
            config = _make_config(key_source="keyring", auto_generate_key=True)
            result = get_master_key(config)

        assert len(result) == _KEY_LENGTH
        mock_keyring.set_password.assert_called_once_with("trw-memory", "master", result.hex())
