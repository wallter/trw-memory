"""Key management for memory encryption.

Key source hierarchy (highest priority first):
1. OS keyring (requires ``keyring`` package — optional dependency)
2. Environment variable ``MEMORY_MASTER_KEY`` (hex-encoded)
3. File at ``~/.trw-memory/master.key`` (raw 32 bytes)
"""

from __future__ import annotations

import contextlib
import os
import stat
from pathlib import Path

import structlog

from trw_memory.exceptions import ConfigError
from trw_memory.models.config import MemoryConfig
from trw_memory.security.encryption import (
    decrypt_entry_fields,
    derive_namespace_key,
    encrypt_entry_fields,
)
from trw_memory.storage.interface import StorageBackend

try:
    import keyring as _keyring
    _KEYRING_AVAILABLE = True
except ImportError:
    _keyring = None  # type: ignore[assignment]
    _KEYRING_AVAILABLE = False

logger = structlog.get_logger()

_SERVICE_NAME = "trw-memory"
_KEY_ACCOUNT = "master-key"
_KEY_LENGTH = 32
_ENV_VAR = "MEMORY_MASTER_KEY"


def _read_key_from_keyring() -> bytes | None:
    """Attempt to read the master key from the OS keyring."""
    if not _KEYRING_AVAILABLE or _keyring is None:
        return None
    try:
        stored: str | None = _keyring.get_password(_SERVICE_NAME, _KEY_ACCOUNT)
        if stored is None:
            return None
        return bytes.fromhex(stored)
    except Exception:
        logger.debug("keyring_read_failed", exc_info=True)
        return None


def _read_key_from_env() -> bytes | None:
    """Read the master key from the ``MEMORY_MASTER_KEY`` env var (hex)."""
    raw = os.environ.get(_ENV_VAR)
    if not raw:
        return None
    try:
        key = bytes.fromhex(raw)
        if len(key) != _KEY_LENGTH:
            raise ConfigError(
                f"{_ENV_VAR} must decode to {_KEY_LENGTH} bytes, "
                f"got {len(key)}"
            )
        return key
    except ValueError as exc:
        raise ConfigError(f"Invalid hex in {_ENV_VAR}: {exc}") from exc


def _key_file_path(config: MemoryConfig) -> Path:
    """Resolve the key file path from config."""
    return Path(config.key_file_path).expanduser()


def _read_key_from_file(config: MemoryConfig) -> bytes | None:
    """Read the master key from a file on disk."""
    path = _key_file_path(config)
    if not path.exists():
        return None
    data = path.read_bytes()
    if len(data) != _KEY_LENGTH:
        raise ConfigError(
            f"Key file {path} must contain exactly {_KEY_LENGTH} bytes, "
            f"got {len(data)}"
        )
    return data


def get_master_key(config: MemoryConfig) -> bytes:
    """Retrieve the master key using the configured key source hierarchy.

    The lookup order depends on ``config.key_source``:

    - ``"keyring"`` — OS keyring first, then env var, then file
    - ``"env"`` — env var first, then file
    - ``"file"`` — file only

    Args:
        config: Memory configuration with key source settings.

    Returns:
        The 32-byte master key.

    Raises:
        ConfigError: If no key is found in any configured source.
    """
    sources: list[str] = []

    if config.key_source == "keyring":
        sources = ["keyring", "env", "file"]
    elif config.key_source == "env":
        sources = ["env", "file"]
    elif config.key_source == "file":
        sources = ["file"]
    else:
        raise ConfigError(f"Unknown key_source: {config.key_source!r}")

    for source in sources:
        key: bytes | None = None
        if source == "keyring":
            key = _read_key_from_keyring()
        elif source == "env":
            key = _read_key_from_env()
        elif source == "file":
            key = _read_key_from_file(config)

        if key is not None:
            logger.debug("master_key_loaded", source=source)
            return key

    raise ConfigError(
        "No master key found. Set MEMORY_MASTER_KEY env var (hex), "
        f"place a 32-byte key at {_key_file_path(config)}, "
        "or install keyring and store via store_master_key()."
    )


def store_master_key(key: bytes, config: MemoryConfig) -> None:
    """Persist the master key to the configured key source.

    Args:
        key: The 32-byte master key to store.
        config: Memory configuration with key source settings.

    Raises:
        ConfigError: If the key cannot be stored (wrong length, write error).
    """
    if len(key) != _KEY_LENGTH:
        raise ConfigError(
            f"Master key must be {_KEY_LENGTH} bytes, got {len(key)}"
        )

    if config.key_source == "keyring":
        if not _KEYRING_AVAILABLE or _keyring is None:
            raise ConfigError(
                "keyring package not installed — "
                "install with: pip install keyring"
            )
        try:
            _keyring.set_password(_SERVICE_NAME, _KEY_ACCOUNT, key.hex())
            logger.info("master_key_stored", target="keyring")
            return
        except Exception as exc:
            raise ConfigError(f"Failed to store key in keyring: {exc}") from exc

    if config.key_source == "env":
        raise ConfigError(
            "Cannot persist key to env var — set MEMORY_MASTER_KEY "
            "in your shell profile or use key_source='file'"
        )

    # file source
    path = _key_file_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(key)
    # Restrict permissions to owner-only (0600)
    # Windows doesn't support Unix permissions — best effort
    with contextlib.suppress(OSError):
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    logger.info("master_key_stored", target=str(path))


def rotate_master_key(
    old_key: bytes,
    new_key: bytes,
    backend: StorageBackend,
) -> int:
    """Re-encrypt all entries from *old_key* to *new_key*.

    Iterates through every entry in the backend, decrypts fields with the
    old key, re-encrypts with the new key, and stores the updated entry.

    Args:
        old_key: The current 32-byte master key.
        new_key: The new 32-byte master key to rotate to.
        backend: Storage backend containing entries to re-encrypt.

    Returns:
        Number of entries re-encrypted.

    Raises:
        ConfigError: If either key is not 32 bytes.
    """
    if len(old_key) != _KEY_LENGTH:
        raise ConfigError(
            f"old_key must be {_KEY_LENGTH} bytes, got {len(old_key)}"
        )
    if len(new_key) != _KEY_LENGTH:
        raise ConfigError(
            f"new_key must be {_KEY_LENGTH} bytes, got {len(new_key)}"
        )

    entries = backend.list_entries(limit=100_000)
    count = 0

    for entry in entries:
        ns_key_old = derive_namespace_key(old_key, entry.namespace)
        ns_key_new = derive_namespace_key(new_key, entry.namespace)

        decrypted = decrypt_entry_fields(entry, ns_key_old)
        re_encrypted = encrypt_entry_fields(decrypted, ns_key_new)
        backend.store(re_encrypted)
        count += 1

    logger.info("key_rotation_complete", entries_rotated=count)
    return count
