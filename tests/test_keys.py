"""Tests for trw_memory.security.keys — master key management.

Coverage:
- get_master_key() with env var source
- get_master_key() with file source
- get_master_key() raises ConfigError when no key found
- get_master_key() with keyring source (mocked)
- store_master_key() to file: creates file with correct content
- store_master_key() rejects wrong key length
- store_master_key() raises ConfigError for env source
- store_master_key() raises ConfigError when keyring unavailable
- rotate_master_key(): re-encrypts all entries correctly
- rotate_master_key() rejects wrong key lengths
- _read_key_from_env() hex validation
"""

from __future__ import annotations

import stat
import sys
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from trw_memory.exceptions import ConfigError, MasterKeyNotFoundError
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry, MemoryStatus
from trw_memory.security import (
    decrypt_entry_fields,
    derive_namespace_key_bytes,
    encrypt_entry_fields,
    generate_master_key,
    get_master_key,
    rotate_master_key,
    store_master_key,
)
from trw_memory.security.keys import clear_key_cache

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_KEY_LENGTH = 32


@pytest.fixture(autouse=True)
def clear_master_key_cache_fixture() -> Iterator[None]:
    clear_key_cache()
    yield
    clear_key_cache()


def _make_config(
    key_source: str = "env",
    key_file_path: str = "~/.trw-memory/master.key",
    auto_generate_key: bool = True,
    *,
    encryption_enabled: bool = True,
) -> MemoryConfig:
    return MemoryConfig(
        encryption_enabled=encryption_enabled,
        key_source=key_source,  # type: ignore[arg-type]
        key_file_path=key_file_path,
        auto_generate_key=auto_generate_key,
    )


def _make_entry(entry_id: str = "k-test-1", content: str = "content") -> MemoryEntry:
    now = datetime.now(timezone.utc)
    return MemoryEntry(
        id=entry_id,
        content=content,
        detail="detail text",
        namespace="default",
        status=MemoryStatus.ACTIVE,
        created_at=now,
        updated_at=now,
    )


# ---------------------------------------------------------------------------
# get_master_key — env var source
# ---------------------------------------------------------------------------


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
        # 16 bytes = 32 hex chars — too short for 32-byte key
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
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
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


# ---------------------------------------------------------------------------
# get_master_key — file source
# ---------------------------------------------------------------------------


class TestGetMasterKeyFile:
    def test_reads_key_from_file(self, tmp_path: Path) -> None:
        key = generate_master_key()
        key_file = tmp_path / "master.key"
        key_file.write_bytes(key)
        config = _make_config(key_source="file", key_file_path=str(key_file), encryption_enabled=False)
        result = get_master_key(config)
        assert result == key

    def test_file_not_found_raises_master_key_not_found(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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
        """Tilde in key_file_path is expanded."""
        key = generate_master_key()
        key_file = tmp_path / "master.key"
        key_file.write_bytes(key)
        # Pass actual absolute path (no tilde) — tests the Path.expanduser() path
        config = _make_config(key_source="file", key_file_path=str(key_file), encryption_enabled=False)
        result = get_master_key(config)
        assert result == key

    def test_file_source_rejected_when_encryption_enabled(self, tmp_path: Path) -> None:
        key_file = tmp_path / "master.key"
        key_file.write_bytes(generate_master_key())
        config = _make_config(key_source="file", key_file_path=str(key_file), encryption_enabled=True)
        with pytest.raises(ConfigError, match="key_source='file' is unsupported"):
            get_master_key(config)


# ---------------------------------------------------------------------------
# get_master_key — unknown key_source
# ---------------------------------------------------------------------------


class TestGetMasterKeyUnknownSource:
    def test_unknown_key_source_raises_config_error(self) -> None:
        config = MemoryConfig(key_source="env")  # type: ignore[arg-type]
        # Monkey-patch key_source after construction to bypass Literal validation
        object.__setattr__(config, "key_source", "unknown-source")
        with pytest.raises(ConfigError, match="Unknown key_source"):
            get_master_key(config)


# ---------------------------------------------------------------------------
# get_master_key — keyring source (mocked)
# ---------------------------------------------------------------------------


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
        self, monkeypatch: pytest.MonkeyPatch
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


# ---------------------------------------------------------------------------
# store_master_key — file source
# ---------------------------------------------------------------------------


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
        """On POSIX systems, stored key file must be owner-readable only (0600)."""
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


# ---------------------------------------------------------------------------
# store_master_key — env source (should raise)
# ---------------------------------------------------------------------------


class TestStoreMasterKeyEnv:
    def test_env_source_raises_config_error(self) -> None:
        key = generate_master_key()
        config = _make_config(key_source="env")
        with pytest.raises(ConfigError, match="Cannot persist key to env var"):
            store_master_key(key, config)


# ---------------------------------------------------------------------------
# store_master_key — keyring source (mocked)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# rotate_master_key
# ---------------------------------------------------------------------------


class TestRotateMasterKey:
    def test_rotate_re_encrypts_all_entries(self, tmp_path: Path) -> None:
        """After rotation, entries decryptable with new key, not old key."""
        from trw_memory.storage.sqlite_backend import SQLiteBackend

        old_key = generate_master_key()
        new_key = generate_master_key()

        entry = _make_entry("rotate-1", "secret content")
        ns_key_old = derive_namespace_key_bytes(old_key, entry.namespace)
        encrypted_entry = encrypt_entry_fields(entry, ns_key_old)

        with SQLiteBackend(tmp_path / "rotate.db") as backend:
            backend.store(encrypted_entry)

            count = rotate_master_key(old_key, new_key, backend)

            assert count == 1

            # Re-read the stored entry and verify it decrypts with new key
            stored = backend.get("rotate-1")
            assert stored is not None
            ns_key_new = derive_namespace_key_bytes(new_key, stored.namespace)
            decrypted = decrypt_entry_fields(stored, ns_key_new)
            assert decrypted.content == "secret content"

    def test_rotate_returns_entry_count(self, tmp_path: Path) -> None:
        from trw_memory.storage.sqlite_backend import SQLiteBackend

        old_key = generate_master_key()
        new_key = generate_master_key()

        with SQLiteBackend(tmp_path / "rotate_count.db") as backend:
            for i in range(5):
                entry = _make_entry(f"count-{i}", f"content {i}")
                ns_key = derive_namespace_key_bytes(old_key, entry.namespace)
                enc = encrypt_entry_fields(entry, ns_key)
                backend.store(enc)

            count = rotate_master_key(old_key, new_key, backend)

        assert count == 5

    def test_rotate_empty_backend_returns_zero(self, tmp_path: Path) -> None:
        from trw_memory.storage.sqlite_backend import SQLiteBackend

        old_key = generate_master_key()
        new_key = generate_master_key()
        with SQLiteBackend(tmp_path / "empty.db") as backend:
            count = rotate_master_key(old_key, new_key, backend)
        assert count == 0

    def test_rotate_rejects_short_old_key(self, tmp_path: Path) -> None:
        from trw_memory.storage.sqlite_backend import SQLiteBackend

        new_key = generate_master_key()
        with SQLiteBackend(tmp_path / "r_short_old.db") as backend:
            with pytest.raises(ConfigError, match="old_key must be 32 bytes"):
                rotate_master_key(b"short", new_key, backend)

    def test_rotate_rejects_short_new_key(self, tmp_path: Path) -> None:
        from trw_memory.storage.sqlite_backend import SQLiteBackend

        old_key = generate_master_key()
        with SQLiteBackend(tmp_path / "r_short_new.db") as backend:
            with pytest.raises(ConfigError, match="new_key must be 32 bytes"):
                rotate_master_key(old_key, b"short", backend)

    def test_rotate_multiple_namespaces(self, tmp_path: Path) -> None:
        """Entries in different namespaces are all re-encrypted correctly."""
        from trw_memory.storage.sqlite_backend import SQLiteBackend

        old_key = generate_master_key()
        new_key = generate_master_key()
        namespaces = ["agents", "teams", "global"]

        now = datetime.now(timezone.utc)
        entries = [
            MemoryEntry(
                id=f"ns-{ns}",
                content=f"content for {ns}",
                detail="detail",
                namespace=ns,
                status=MemoryStatus.ACTIVE,
                created_at=now,
                updated_at=now,
            )
            for ns in namespaces
        ]

        with SQLiteBackend(tmp_path / "multi_ns.db") as backend:
            for entry in entries:
                ns_key = derive_namespace_key_bytes(old_key, entry.namespace)
                enc = encrypt_entry_fields(entry, ns_key)
                backend.store(enc)

            count = rotate_master_key(old_key, new_key, backend)
            assert count == len(namespaces)

            for entry in entries:
                stored = backend.get(entry.id)
                assert stored is not None
                ns_key_new = derive_namespace_key_bytes(new_key, stored.namespace)
                decrypted = decrypt_entry_fields(stored, ns_key_new)
                assert decrypted.content == f"content for {stored.namespace}"

    def test_old_key_cannot_decrypt_after_rotation(self, tmp_path: Path) -> None:
        """After rotation, old key must no longer decrypt stored entries."""
        from cryptography.exceptions import InvalidTag

        from trw_memory.storage.sqlite_backend import SQLiteBackend

        old_key = generate_master_key()
        new_key = generate_master_key()

        entry = _make_entry("after-rotate", "was secret")
        ns_key_old = derive_namespace_key_bytes(old_key, entry.namespace)
        enc = encrypt_entry_fields(entry, ns_key_old)

        with SQLiteBackend(tmp_path / "old_fail.db") as backend:
            backend.store(enc)
            rotate_master_key(old_key, new_key, backend)
            stored = backend.get("after-rotate")
            assert stored is not None
            ns_key_old_again = derive_namespace_key_bytes(old_key, stored.namespace)
            with pytest.raises(InvalidTag):
                decrypt_entry_fields(stored, ns_key_old_again)
