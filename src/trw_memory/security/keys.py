"""Key management for memory encryption.

Secure source hierarchy (highest priority first):
1. Environment variable ``MEMORY_MASTER_KEY`` (hex-encoded)
2. OS keyring entry ``("trw-memory", "master")``
"""

from __future__ import annotations

import contextlib
import os
import secrets
import stat
from pathlib import Path
from typing import Any

import structlog

try:
    from nacl.signing import SigningKey as _SigningKey

    _NACL_AVAILABLE = True
except ImportError:  # pragma: no cover
    _NACL_AVAILABLE = False
    _SigningKey = Any  # type: ignore[misc,assignment]

try:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey as _CryptoEd25519PrivateKey

    _CRYPTO_ED25519_AVAILABLE = True
except ImportError:  # pragma: no cover
    _CRYPTO_ED25519_AVAILABLE = False
    _CryptoEd25519PrivateKey = Any  # type: ignore[misc,assignment]

from trw_memory.exceptions import ConfigError, KeyRotationError, MasterKeyNotFoundError
from trw_memory.models.config import MemoryConfig
from trw_memory.security.encryption import (
    decrypt_entry_fields,
    derive_namespace_key_bytes,
    encrypt_entry_fields,
    generate_master_key,
)
from trw_memory.storage.interface import StorageBackend

try:
    import keyring as _keyring

    _KEYRING_AVAILABLE = True
except ImportError:
    _keyring = None  # type: ignore[assignment]
    _KEYRING_AVAILABLE = False

logger = structlog.get_logger(__name__)

_SERVICE_NAME = "trw-memory"
_KEY_ACCOUNT = "master"
_LEGACY_KEY_ACCOUNTS = ("master-key",)
_KEY_LENGTH = 32
# Extra rows fetched beyond the live count() during key rotation so entries
# inserted in the count->fetch window are still re-encrypted (coverage is
# re-verified against count() afterward).
_ROTATION_FETCH_HEADROOM = 1000
_ENV_VAR = "MEMORY_MASTER_KEY"
_CACHED_MASTER_KEY: bytes | None = None
_CACHED_SOURCE: str | None = None
_CACHED_ENV_HEX: str | None = None
_CACHED_FILE_PATH: str | None = None
_INSECURE_FILE_SOURCE_MESSAGE = (
    "key_source='file' is unsupported when memory_encryption_enabled=True; "
    "use MEMORY_MASTER_KEY or key_source='keyring'."
)


def clear_key_cache() -> None:
    """Reset the in-process master-key cache."""
    global _CACHED_MASTER_KEY, _CACHED_SOURCE, _CACHED_ENV_HEX, _CACHED_FILE_PATH
    _CACHED_MASTER_KEY = None
    _CACHED_SOURCE = None
    _CACHED_ENV_HEX = None
    _CACHED_FILE_PATH = None


def _cache_master_key(
    key: bytes,
    source: str,
    *,
    env_hex: str | None = None,
    file_path: str | None = None,
) -> bytes:
    global _CACHED_MASTER_KEY, _CACHED_SOURCE, _CACHED_ENV_HEX, _CACHED_FILE_PATH
    _CACHED_MASTER_KEY = key
    _CACHED_SOURCE = source
    _CACHED_ENV_HEX = env_hex
    _CACHED_FILE_PATH = file_path
    return key


def _target_config_for_generated_key(config: MemoryConfig) -> MemoryConfig:
    """Choose a secure writable target for auto-generated keys."""
    if _KEYRING_AVAILABLE and _keyring is not None:
        return config.model_copy(update={"key_source": "keyring"})
    raise ConfigError("OS keyring unavailable for auto-generated master key; install keyring or set MEMORY_MASTER_KEY.")


def _ensure_secure_key_source(config: MemoryConfig) -> None:
    if config.encryption_enabled and config.key_source == "file":
        raise ConfigError(_INSECURE_FILE_SOURCE_MESSAGE)


def _read_key_from_keyring() -> bytes | None:
    """Attempt to read the master key from the OS keyring."""
    if not _KEYRING_AVAILABLE or _keyring is None:
        return None
    for account in (_KEY_ACCOUNT, *_LEGACY_KEY_ACCOUNTS):
        try:
            stored: str | None = _keyring.get_password(_SERVICE_NAME, account)
            if stored is None:
                continue
            return bytes.fromhex(stored)
        except (ValueError, OSError, RuntimeError):
            logger.debug("keyring_read_failed", account=account, exc_info=True)
            return None
    return None


def _read_key_from_env() -> bytes | None:
    """Read the master key from the ``MEMORY_MASTER_KEY`` env var (hex)."""
    raw = os.environ.get(_ENV_VAR)
    if not raw:
        return None
    try:
        key = bytes.fromhex(raw)
        if len(key) != _KEY_LENGTH:
            raise ConfigError(f"{_ENV_VAR} must decode to {_KEY_LENGTH} bytes, got {len(key)}")
        return key
    except ValueError as exc:
        raise ConfigError(f"Invalid hex in {_ENV_VAR}: {exc}") from exc


def _validate_key_path(path: Path) -> Path:
    """Validate a key file path against traversal attacks.

    Args:
        path: Raw path (may contain ``~`` or ``..``).

    Returns:
        Resolved, validated path.

    Raises:
        ConfigError: If the path contains ``..`` traversal components.
    """
    if ".." in path.parts:
        raise ConfigError(f"Path traversal detected in key_file_path: {path}")
    return path.expanduser().resolve()


def _key_file_path(config: MemoryConfig) -> Path:
    """Resolve the key file path from config."""
    return _validate_key_path(Path(config.key_file_path))


def _read_key_from_file(config: MemoryConfig) -> bytes | None:
    """Read the master key from a file on disk."""
    path = _key_file_path(config)
    if not path.exists():
        return None
    data = path.read_bytes()
    if len(data) != _KEY_LENGTH:
        raise ConfigError(f"Key file {path} must contain exactly {_KEY_LENGTH} bytes, got {len(data)}")
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
    _ensure_secure_key_source(config)
    raw_env = os.environ.get(_ENV_VAR)
    if raw_env:
        if _CACHED_SOURCE == "env" and _CACHED_MASTER_KEY is not None and raw_env == _CACHED_ENV_HEX:
            logger.debug("master_key_loaded", source="env", cached=True)
            return _CACHED_MASTER_KEY
        env_key = _read_key_from_env()
        if env_key is not None:
            logger.debug("master_key_loaded", source="env")
            return _cache_master_key(env_key, "env", env_hex=raw_env)

    key_path = str(_key_file_path(config))
    if config.key_source == "keyring" and _CACHED_SOURCE == "keyring" and _CACHED_MASTER_KEY is not None:
        logger.debug("master_key_loaded", source="keyring", cached=True)
        return _CACHED_MASTER_KEY
    if (
        config.key_source in {"env", "file"}
        and _CACHED_SOURCE == "file"
        and key_path == _CACHED_FILE_PATH
        and _CACHED_MASTER_KEY is not None
    ):
        logger.debug("master_key_loaded", source="file", cached=True)
        return _CACHED_MASTER_KEY

    sources: list[str]
    if (config.encryption_enabled and config.key_source != "file") or config.key_source == "keyring":
        sources = ["keyring"]
    elif config.key_source == "env":
        sources = []
    elif config.key_source == "file":
        sources = ["file"]
    else:
        raise ConfigError(f"Unknown key_source: {config.key_source!r}")

    for source in sources:
        key: bytes | None = None
        if source == "keyring":
            key = _read_key_from_keyring()
        elif source == "file":
            key = _read_key_from_file(config)

        if key is not None:
            logger.debug("master_key_loaded", source=source)
            return _cache_master_key(key, source, file_path=key_path if source == "file" else None)

    if config.auto_generate_key:
        key = generate_master_key()
        target_config = _target_config_for_generated_key(config)
        store_master_key(key, target_config)
        logger.info("Generated and stored new master key in OS keyring.")
        target_path = str(_key_file_path(target_config)) if target_config.key_source == "file" else None
        return _cache_master_key(key, target_config.key_source, file_path=target_path)

    raise MasterKeyNotFoundError(
        "No master key found. Set MEMORY_MASTER_KEY env var or store key in OS keyring "
        "(service='trw-memory', username='master')."
    )


def store_master_key(key: bytes, config: MemoryConfig) -> None:
    """Persist the master key to the configured key source.

    Args:
        key: The 32-byte master key to store.
        config: Memory configuration with key source settings.

    Raises:
        ConfigError: If the key cannot be stored (wrong length, write error).
    """
    _ensure_secure_key_source(config)
    if len(key) != _KEY_LENGTH:
        raise ConfigError(f"Master key must be {_KEY_LENGTH} bytes, got {len(key)}")

    if config.key_source == "keyring":
        if not _KEYRING_AVAILABLE or _keyring is None:
            raise ConfigError("keyring package not installed — install with: pip install keyring")
        try:
            _keyring.set_password(_SERVICE_NAME, _KEY_ACCOUNT, key.hex())
            logger.info("master_key_stored", target="keyring")
            _cache_master_key(key, "keyring")
            return
        except (OSError, RuntimeError) as exc:
            raise ConfigError(f"Failed to store key in keyring: {exc}") from exc

    if config.key_source == "env":
        raise ConfigError(
            "Cannot persist key to env var — set MEMORY_MASTER_KEY in your shell profile or use key_source='keyring'"
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
    _cache_master_key(key, "file", file_path=str(path))


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
        KeyRotationError: If, after paginating, not every entry was
            re-encrypted (e.g. a concurrent insert outran the rotation).
    """
    if len(old_key) != _KEY_LENGTH:
        raise ConfigError(f"old_key must be {_KEY_LENGTH} bytes, got {len(old_key)}")
    if len(new_key) != _KEY_LENGTH:
        raise ConfigError(f"new_key must be {_KEY_LENGTH} bytes, got {len(new_key)}")

    # Re-encrypt EVERY entry — never a silent cap. ``store`` preserves the
    # entry's ``updated_at`` (INSERT OR REPLACE), so the ``ORDER BY updated_at
    # DESC`` window is stable across re-encryption and a single fetch sized to
    # the live count covers all rows. The previous hardcoded ``limit=100_000``
    # silently left any surplus entries encrypted under the OLD key.
    total = backend.count()
    # Headroom absorbs entries inserted between count() and the fetch.
    fetch_limit = total + _ROTATION_FETCH_HEADROOM
    entries = backend.list_entries(limit=fetch_limit)

    processed_ids: set[str] = set()
    for entry in entries:
        if entry.id in processed_ids:
            continue
        ns_key_old = derive_namespace_key_bytes(old_key, entry.namespace)
        ns_key_new = derive_namespace_key_bytes(new_key, entry.namespace)

        decrypted = decrypt_entry_fields(entry, ns_key_old)
        re_encrypted = encrypt_entry_fields(decrypted, ns_key_new)
        backend.store(re_encrypted)
        processed_ids.add(entry.id)

    count = len(processed_ids)
    if count < total:
        raise KeyRotationError(
            f"key rotation re-encrypted {count} of {total} entries; "
            "not all data was rotated to the new key",
        )

    logger.info("key_rotation_complete", entries_rotated=count)
    return count


# ---------------------------------------------------------------------------
# Ed25519 provenance-signing keys (PRD-SEC-001 FR-002, Sprint-96 carry-forward-b)
# ---------------------------------------------------------------------------

_ED25519_SEED_LENGTH = 32
_ED25519_KEY_FILENAME = "ed25519_signing_key.bin"


def generate_ed25519_signing_key() -> bytes:
    """Return a fresh 32-byte seed suitable for :class:`nacl.signing.SigningKey`.

    Uses :func:`secrets.token_bytes`, which is acceptable whether or not
    PyNaCl is installed.
    """
    return secrets.token_bytes(_ED25519_SEED_LENGTH)


def load_ed25519_signing_key(path: Path) -> Any:
    """Load a SigningKey from a 32-byte seed file.

    Returns ``None`` when PyNaCl is unavailable so callers can degrade
    gracefully. Raises :class:`ConfigError` on malformed/missing files
    when PyNaCl IS available.
    """
    if not path.exists():
        raise ConfigError(f"Ed25519 key file not found: {path}")
    data = path.read_bytes()
    if len(data) != _ED25519_SEED_LENGTH:
        raise ConfigError(f"Ed25519 seed must be {_ED25519_SEED_LENGTH} bytes, got {len(data)}")
    if _NACL_AVAILABLE:
        return _SigningKey(data)
    if _CRYPTO_ED25519_AVAILABLE:
        return _CryptoEd25519PrivateKey.from_private_bytes(data)
    logger.warning("ed25519_runtime_unavailable", path=str(path))
    return None


def get_or_create_ed25519_key(trw_dir: Path) -> Any:
    """Return an Ed25519 :class:`SigningKey` for *trw_dir*, creating one if needed.

    Idempotent. Writes the seed to
    ``<trw_dir>/memory/security/ed25519_signing_key.bin`` with chmod 0600.
    When PyNaCl is unavailable, writes the seed for later use but returns
    ``None`` and logs a warning — callers must fall back to SHA-256-only
    provenance chains.
    """
    key_path = trw_dir / "memory" / "security" / _ED25519_KEY_FILENAME
    return get_or_create_ed25519_key_at_path(key_path)


def get_or_create_ed25519_key_at_path(key_path: Path) -> Any:
    """Return an Ed25519 signing key stored exactly at *key_path*."""
    key_dir = key_path.parent
    if key_path.exists():
        if _NACL_AVAILABLE or _CRYPTO_ED25519_AVAILABLE:
            try:
                return load_ed25519_signing_key(key_path)
            except ConfigError:
                logger.warning("ed25519_key_load_failed", path=str(key_path), exc_info=True)
                return None
        logger.warning("ed25519_runtime_unavailable", path=str(key_path))
        return None

    # Create new key
    key_dir.mkdir(parents=True, exist_ok=True)
    seed = generate_ed25519_signing_key()
    key_path.write_bytes(seed)
    with contextlib.suppress(OSError):
        key_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    logger.info("ed25519_key_generated", path=str(key_path))

    if not (_NACL_AVAILABLE or _CRYPTO_ED25519_AVAILABLE):
        logger.warning("ed25519_runtime_unavailable_after_write", path=str(key_path))
        return None
    return load_ed25519_signing_key(key_path)
