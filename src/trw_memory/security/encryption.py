"""Field-level AES-256-GCM encryption and SQLCipher key rotation helpers."""

from __future__ import annotations

import base64
import hashlib
import os
import shutil
from pathlib import Path

import structlog
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from trw_memory.exceptions import KeyRotationError
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.namespaces.validation import validate_namespace

logger = structlog.get_logger(__name__)

_KEY_LENGTH = 32
_NONCE_LENGTH = 12
_NAMESPACE_KEY_INFO = b"trw-memory-namespace-key-v1"


def generate_master_key() -> bytes:
    """Generate a cryptographically random 256-bit master key."""
    return os.urandom(_KEY_LENGTH)


def derive_namespace_key_bytes(master_key: bytes, namespace: str) -> bytes:
    """Derive a unique 256-bit AEAD key for *namespace* using HKDF-SHA256."""
    if len(master_key) != _KEY_LENGTH:
        raise ValueError(f"master_key must be {_KEY_LENGTH} bytes, got {len(master_key)}")
    namespace_bytes = namespace.encode("utf-8")
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=_KEY_LENGTH,
        salt=hashlib.sha256(namespace_bytes).digest(),
        info=_NAMESPACE_KEY_INFO,
    )
    return bytes(hkdf.derive(master_key))


def derive_namespace_key(master_key: bytes, namespace: str) -> str:
    """Return the SQLCipher-ready 64-character lowercase hex key for *namespace*."""
    return derive_namespace_key_bytes(master_key, namespace).hex()


def encrypt_field(plaintext: str, key: bytes) -> str:
    """Encrypt a plaintext string with AES-256-GCM."""
    if len(key) != _KEY_LENGTH:
        raise ValueError(f"key must be {_KEY_LENGTH} bytes, got {len(key)}")
    nonce = os.urandom(_NONCE_LENGTH)
    aesgcm = AESGCM(key)
    payload = nonce + aesgcm.encrypt(nonce, plaintext.encode("utf-8"), None)
    return base64.b64encode(payload).decode("ascii")


def decrypt_field(ciphertext_b64: str, key: bytes) -> str:
    """Decrypt a base64-encoded AES-256-GCM payload."""
    if len(key) != _KEY_LENGTH:
        raise ValueError(f"key must be {_KEY_LENGTH} bytes, got {len(key)}")
    payload = base64.b64decode(ciphertext_b64)
    if len(payload) < _NONCE_LENGTH + 16:
        raise ValueError("Encrypted payload too short")
    nonce = payload[:_NONCE_LENGTH]
    ct_with_tag = payload[_NONCE_LENGTH:]
    aesgcm = AESGCM(key)
    plaintext_bytes = bytes(aesgcm.decrypt(nonce, ct_with_tag, None))
    return plaintext_bytes.decode("utf-8")


def encrypt_entry_fields(entry: MemoryEntry, key: bytes) -> MemoryEntry:
    """Return a copy of *entry* with ``content`` and ``detail`` encrypted."""
    data = entry.model_dump()
    data["content"] = encrypt_field(entry.content, key)
    if entry.detail:
        data["detail"] = encrypt_field(entry.detail, key)
    return MemoryEntry.model_validate(data, strict=False)


def decrypt_entry_fields(entry: MemoryEntry, key: bytes) -> MemoryEntry:
    """Return a copy of *entry* with ``content`` and ``detail`` decrypted."""
    data = entry.model_dump()
    data["content"] = decrypt_field(entry.content, key)
    if entry.detail:
        data["detail"] = decrypt_field(entry.detail, key)
    return MemoryEntry.model_validate(data, strict=False)


def _namespace_db_path(config: MemoryConfig, namespace: str) -> Path:
    ns_dir = namespace.replace(":", "_")
    return Path(config.storage_path) / ns_dir / config.sqlite_db_name


def _coerce_master_key_input(new_passphrase: str) -> bytes:
    if not new_passphrase:
        raise ValueError("new_passphrase must not be empty")
    if len(new_passphrase) == 64:
        try:
            decoded = bytes.fromhex(new_passphrase)
        except ValueError:
            decoded = b""
        if len(decoded) == _KEY_LENGTH:
            return decoded
    return hashlib.sha256(new_passphrase.encode("utf-8")).digest()


def _remove_sqlite_sidecars(db_path: Path) -> None:
    for suffix in ("-wal", "-shm"):
        sidecar = Path(f"{db_path}{suffix}")
        if sidecar.exists():
            sidecar.unlink()


def _restore_database_from_backup(db_path: Path, backup_path: Path) -> None:
    shutil.copy2(backup_path, db_path)
    _remove_sqlite_sidecars(db_path)


def _persist_rotated_master_key(new_master_key: bytes, config: MemoryConfig) -> None:
    from trw_memory.security.keys import store_master_key

    keyring_config = config.model_copy(update={"key_source": "keyring"})
    store_master_key(new_master_key, keyring_config)


def rotate_key(namespace: str, new_passphrase: str, config: MemoryConfig | None = None) -> None:
    """Rotate the SQLCipher key for a namespace-local SQLite store."""
    from trw_memory.security.keys import get_master_key
    from trw_memory.security.rbac import Permission, require_namespace_permission
    from trw_memory.storage.sqlite_backend import (
        SQLiteBackend,
        _apply_sqlcipher_pragmas,
        _import_sqlcipher_driver,
    )

    effective_config = config or MemoryConfig()
    validate_namespace(namespace)
    require_namespace_permission(effective_config, namespace, Permission.ADMIN, "rotate_key")

    if effective_config.storage_backend != "sqlite":
        raise KeyRotationError("rotate_key() only supports sqlite storage backends")
    if not effective_config.encryption_enabled:
        raise KeyRotationError("rotate_key() requires encryption_enabled=True")

    db_path = _namespace_db_path(effective_config, namespace)
    if not db_path.exists():
        raise KeyRotationError(f"Encrypted database not found for namespace {namespace!r}", path=str(db_path))

    current_master_key = get_master_key(effective_config)
    new_master_key = _coerce_master_key_input(new_passphrase)
    current_sqlcipher_key_hex = derive_namespace_key(current_master_key, namespace)
    new_sqlcipher_key_hex = derive_namespace_key(new_master_key, namespace)
    backup_path = Path(f"{db_path}.bak")
    dbapi = _import_sqlcipher_driver()

    checkpoint_conn = SQLiteBackend._connect(
        db_path,
        dbapi=dbapi,
        timeout=30.0,
        check_same_thread=False,
        cached_statements=0,
        sqlcipher_key_hex=current_sqlcipher_key_hex,
    )
    try:
        checkpoint_conn.execute("PRAGMA busy_timeout = 30000")
        checkpoint_conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        checkpoint_conn.close()

    if not effective_config.key_rotation_backup:
        logger.warning("key_rotation_backup_forced_enabled", reason="required_for_safe_sqlcipher_rotation")
    shutil.copy2(db_path, backup_path)

    rotation_conn = SQLiteBackend._connect(
        db_path,
        dbapi=dbapi,
        timeout=30.0,
        check_same_thread=False,
        cached_statements=0,
        sqlcipher_key_hex=current_sqlcipher_key_hex,
    )
    try:
        rotation_conn.execute("PRAGMA busy_timeout = 30000")
        _apply_sqlcipher_pragmas(rotation_conn)
        rotation_conn.execute(f'PRAGMA rekey = "x\'{new_sqlcipher_key_hex}\'"')  # noqa: S608

        rows = rotation_conn.execute("PRAGMA integrity_check").fetchall()
        if len(rows) != 1 or rows[0][0] != "ok":
            detail = rows[0][0] if rows else "empty"
            raise KeyRotationError(f"Key rotation integrity check failed: {detail}", path=str(db_path))
    except Exception as exc:
        rotation_conn.close()
        _restore_database_from_backup(db_path, backup_path)
        if isinstance(exc, KeyRotationError):
            raise
        raise KeyRotationError(f"Key rotation failed: {exc}", path=str(db_path)) from exc
    else:
        rotation_conn.close()

    try:
        _persist_rotated_master_key(new_master_key, effective_config)
    except Exception as exc:
        _restore_database_from_backup(db_path, backup_path)
        raise KeyRotationError(f"Key rotation persisted new database key but failed to store it: {exc}") from exc

    logger.info("key_rotation_complete", namespace=namespace, db=str(db_path), backup=str(backup_path))
