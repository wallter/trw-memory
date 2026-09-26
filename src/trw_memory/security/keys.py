"""Ed25519 provenance-signing keys.

The master-key half of this module (``get_master_key``, ``store_master_key``,
``MEMORY_MASTER_KEY`` and the keyring entry) served encryption at rest, which
trw-memory 4.1 removed.
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

from trw_memory.exceptions import ConfigError
from trw_memory.storage.persistence import lock_for_rmw

logger = structlog.get_logger(__name__)


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
