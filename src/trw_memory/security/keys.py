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
from trw_memory.storage.persistence import lock_for_rmw

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
# Bound on the convergence sweep in rotate_master_key. Each sweep re-reads the
# live backend and re-encrypts only newly-seen rows; a quiet (or quiescing)
# store converges in 1-2 passes. The cap stops an adversary inserting faster
# than we rotate from spinning the loop forever.
_ROTATION_MAX_SWEEPS = 100
_ENV_VAR = "MEMORY_MASTER_KEY"
_CACHED_MASTER_KEY: bytes | None = None
_CACHED_SOURCE: str | None = None
_CACHED_ENV_HEX: str | None = None
_CACHED_FILE_PATH: str | None = None
# Keyring cache identity. Captures the (service, account) the cached key was
# loaded from so a keyring cache hit is validated against the live identity
# rather than the bare ``source == "keyring"`` flag — preventing two configs
# that resolve to different keyring identities from colliding on the cache.
_CACHED_KEYRING_ID: tuple[str, str] | None = None
_INSECURE_FILE_SOURCE_MESSAGE = (
    "key_source='file' is unsupported when memory_encryption_enabled=True; "
    "use MEMORY_MASTER_KEY or key_source='keyring'."
)


def _keyring_identity() -> tuple[str, str]:
    """Return the (service, account) tuple the keyring master key lives under."""
    return (_SERVICE_NAME, _KEY_ACCOUNT)


def clear_key_cache() -> None:
    """Reset the in-process master-key cache."""
    global _CACHED_MASTER_KEY, _CACHED_SOURCE, _CACHED_ENV_HEX, _CACHED_FILE_PATH
    global _CACHED_KEYRING_ID
    _CACHED_MASTER_KEY = None
    _CACHED_SOURCE = None
    _CACHED_ENV_HEX = None
    _CACHED_FILE_PATH = None
    _CACHED_KEYRING_ID = None


def _cache_master_key(
    key: bytes,
    source: str,
    *,
    env_hex: str | None = None,
    file_path: str | None = None,
) -> bytes:
    global _CACHED_MASTER_KEY, _CACHED_SOURCE, _CACHED_ENV_HEX, _CACHED_FILE_PATH
    global _CACHED_KEYRING_ID
    _CACHED_MASTER_KEY = key
    _CACHED_SOURCE = source
    _CACHED_ENV_HEX = env_hex
    _CACHED_FILE_PATH = file_path
    _CACHED_KEYRING_ID = _keyring_identity() if source == "keyring" else None
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
    if (
        config.key_source == "keyring"
        and _CACHED_SOURCE == "keyring"
        and _CACHED_MASTER_KEY is not None
        and _keyring_identity() == _CACHED_KEYRING_ID
    ):
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
        KeyRotationError: If, after the bounded convergence sweep, entries are
            still being inserted faster than they can be re-encrypted.
    """
    if len(old_key) != _KEY_LENGTH:
        raise ConfigError(f"old_key must be {_KEY_LENGTH} bytes, got {len(old_key)}")
    if len(new_key) != _KEY_LENGTH:
        raise ConfigError(f"new_key must be {_KEY_LENGTH} bytes, got {len(new_key)}")

    # Re-encrypt EVERY entry — never a silent cap, and never against a stale
    # PRE-rotation snapshot. The prior implementation fetched once and compared
    # the processed count against a count() taken BEFORE re-encryption, so any
    # entry inserted concurrently DURING the loop was never fetched and escaped
    # re-encryption while leaving the snapshot check satisfied (count == total)
    # — those rows stayed on the OLD key.
    #
    # Fix: sweep repeatedly, each pass re-reading the LIVE backend and processing
    # only IDs not yet seen, until a pass discovers no new entries. ``store``
    # preserves ``updated_at`` (INSERT OR REPLACE), so re-encrypted rows do not
    # churn the ORDER BY window; only genuinely-new concurrent inserts appear in
    # a later pass. The sweep is bounded so a writer inserting faster than we can
    # rotate cannot spin forever.
    processed_ids: set[str] = set()
    for _ in range(_ROTATION_MAX_SWEEPS):
        live_count = backend.count()
        fetch_limit = live_count + _ROTATION_FETCH_HEADROOM
        entries = backend.list_entries(limit=fetch_limit)

        new_this_pass = 0
        for entry in entries:
            if entry.id in processed_ids:
                continue
            ns_key_old = derive_namespace_key_bytes(old_key, entry.namespace)
            ns_key_new = derive_namespace_key_bytes(new_key, entry.namespace)

            decrypted = decrypt_entry_fields(entry, ns_key_old)
            re_encrypted = encrypt_entry_fields(decrypted, ns_key_new)
            backend.store(re_encrypted)
            processed_ids.add(entry.id)
            new_this_pass += 1

        # Converged: a full pass over the LIVE backend found nothing new, AND the
        # live count is fully covered by what we processed. Re-read the count
        # AFTER the pass so a row inserted mid-pass forces another sweep.
        if new_this_pass == 0 and backend.count() <= len(processed_ids):
            count = len(processed_ids)
            logger.info("key_rotation_complete", entries_rotated=count)
            return count

    raise KeyRotationError(
        f"key rotation did not converge after {_ROTATION_MAX_SWEEPS} sweeps "
        f"({len(processed_ids)} entries re-encrypted); entries are being inserted "
        "faster than they can be rotated — quiesce writes and retry",
    )


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
    key_dir.mkdir(mode=0o700, parents=True, exist_ok=True)

    with lock_for_rmw(key_path):
        if key_path.is_symlink():
            logger.warning("ed25519_key_symlink_rejected", path=str(key_path))
            return None
        if not key_path.exists():
            seed = generate_ed25519_signing_key()
            temp_path = key_dir / f".{key_path.name}.{secrets.token_hex(16)}.tmp"
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
            fd = os.open(temp_path, flags, stat.S_IRUSR | stat.S_IWUSR)
            try:
                fchmod = getattr(os, "fchmod", None)
                if fchmod is not None:
                    fchmod(fd, stat.S_IRUSR | stat.S_IWUSR)
                else:  # pragma: no cover - Windows fallback
                    os.chmod(temp_path, stat.S_IRUSR | stat.S_IWUSR)
                remaining = memoryview(seed)
                while remaining:
                    written = os.write(fd, remaining)
                    if written <= 0:
                        raise OSError("failed to write Ed25519 seed")
                    remaining = remaining[written:]
                os.fsync(fd)
            except BaseException:
                with contextlib.suppress(OSError):
                    os.close(fd)
                with contextlib.suppress(OSError):
                    temp_path.unlink()
                raise
            else:
                os.close(fd)
            try:
                os.link(temp_path, key_path, follow_symlinks=False)
            except FileExistsError:
                if key_path.is_symlink():
                    logger.warning("ed25519_key_symlink_rejected", path=str(key_path))
                    return None
            else:
                logger.info("ed25519_key_generated", path=str(key_path))
            finally:
                with contextlib.suppress(OSError):
                    temp_path.unlink()

        if _NACL_AVAILABLE or _CRYPTO_ED25519_AVAILABLE:
            try:
                return load_ed25519_signing_key(key_path)
            except ConfigError:
                logger.warning("ed25519_key_load_failed", path=str(key_path), exc_info=True)
                return None
        logger.warning("ed25519_runtime_unavailable", path=str(key_path))
        return None
