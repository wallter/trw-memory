"""Tests for ``trw_memory.security.keys.store_master_key``."""

from __future__ import annotations

import stat
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from trw_memory.exceptions import ConfigError
from trw_memory.security import generate_master_key, get_master_key, store_master_key

from ._test_keys_support import _make_config


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


class TestKeyringReadFailureNeverAutoGenerates:
    """A keyring that cannot be READ must never be treated as EMPTY.

    ``_read_key_from_keyring`` returned ``None`` on any keyring exception, and
    ``None`` flows to ``config.auto_generate_key`` — default True — which calls
    ``store_master_key`` and OVERWRITES the entry with a fresh key. A locked
    keyring, a backend that raised, or a corrupt hex payload therefore destroyed
    the real key and permanently orphaned every SQLCipher-encrypted memory. The
    failure was silent, happened on the next start, and was unrecoverable.

    Found by a cross-family audit 2026-09-12.
    """

    @staticmethod
    def _keyring(monkeypatch: pytest.MonkeyPatch, get_password: Any) -> list[tuple[str, str, str]]:
        """Install a fake keyring; return the list that records every WRITE."""
        from trw_memory.security import keys as keys_mod

        writes: list[tuple[str, str, str]] = []

        class _FakeKeyring:
            @staticmethod
            def get_password(service: str, account: str) -> str | None:
                return get_password(service, account)  # type: ignore[no-any-return]

            @staticmethod
            def set_password(service: str, account: str, value: str) -> None:
                writes.append((service, account, value))

        monkeypatch.setattr(keys_mod, "_KEYRING_AVAILABLE", True)
        monkeypatch.setattr(keys_mod, "_keyring", _FakeKeyring)
        return writes

    def test_a_raising_keyring_refuses_rather_than_overwriting(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from trw_memory.exceptions import MasterKeyUnreadableError
        from trw_memory.security import keys as keys_mod

        def _locked(_service: str, _account: str) -> str | None:
            raise OSError("keyring is locked")

        writes = self._keyring(monkeypatch, _locked)
        with pytest.raises(MasterKeyUnreadableError):
            keys_mod._read_key_from_keyring()
        assert writes == [], "a failed READ must never cause a WRITE"

    def test_a_corrupt_stored_key_refuses_rather_than_overwriting(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The worst case: the key IS there, merely malformed, and the old code
        replaced it. ``bytes.fromhex`` raises ``ValueError``, which took the same
        silent path as a missing entry."""
        from trw_memory.exceptions import MasterKeyUnreadableError
        from trw_memory.security import keys as keys_mod

        writes = self._keyring(monkeypatch, lambda _s, _a: "not-hex-at-all")
        with pytest.raises(MasterKeyUnreadableError):
            keys_mod._read_key_from_keyring()
        assert writes == []

    def test_one_bad_account_still_tries_the_legacy_accounts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The handler ``return``ed instead of ``continue``ing, so an error on the
        primary account aborted the legacy fallback a real user's key may live in."""
        from trw_memory.security import keys as keys_mod

        good = "ab" * 32
        seen: list[str] = []

        def _primary_fails(_service: str, account: str) -> str | None:
            seen.append(account)
            if account == keys_mod._KEY_ACCOUNT:
                raise RuntimeError("backend exploded")
            return good

        self._keyring(monkeypatch, _primary_fails)
        assert keys_mod._read_key_from_keyring() == bytes.fromhex(good)
        assert len(seen) > 1, "the legacy accounts were never consulted"

    def test_a_genuinely_absent_key_is_still_absent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Non-vacuity partner. ``None`` must stay reachable, or auto-generation —
        the legitimate first-run path — would be broken by this fix."""
        from trw_memory.security import keys as keys_mod

        self._keyring(monkeypatch, lambda _s, _a: None)
        assert keys_mod._read_key_from_keyring() is None
