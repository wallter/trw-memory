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


def encrypt_field(plaintext: str, key: bytes, *, aad: bytes | None = None) -> str:
    """Encrypt a plaintext string with AES-256-GCM.

    *aad* is Additional Authenticated Data bound into the GCM tag.  Passing
    AAD (e.g. ``b'{entry_id}:{namespace}:{field_name}'``) prevents ciphertext
    transplant attacks where an adversary swaps the encrypted ``content`` blob
    of one entry into another entry's ``detail`` field and the decryption still
    succeeds.  The AAD is NOT stored in the ciphertext — the caller must supply
    the same value at decrypt time.
    """
    if len(key) != _KEY_LENGTH:
        raise ValueError(f"key must be {_KEY_LENGTH} bytes, got {len(key)}")
    nonce = os.urandom(_NONCE_LENGTH)
    aesgcm = AESGCM(key)
    payload = nonce + aesgcm.encrypt(nonce, plaintext.encode("utf-8"), aad)
    return base64.b64encode(payload).decode("ascii")


def decrypt_field(ciphertext_b64: str, key: bytes, *, aad: bytes | None = None) -> str:
    """Decrypt a base64-encoded AES-256-GCM payload.

    *aad* must match the value supplied to :func:`encrypt_field` exactly.
    """
    if len(key) != _KEY_LENGTH:
        raise ValueError(f"key must be {_KEY_LENGTH} bytes, got {len(key)}")
    payload = base64.b64decode(ciphertext_b64)
    if len(payload) < _NONCE_LENGTH + 16:
        raise ValueError("Encrypted payload too short")
    nonce = payload[:_NONCE_LENGTH]
    ct_with_tag = payload[_NONCE_LENGTH:]
    aesgcm = AESGCM(key)
    plaintext_bytes = bytes(aesgcm.decrypt(nonce, ct_with_tag, aad))
    return plaintext_bytes.decode("utf-8")


def encrypt_entry_fields(entry: MemoryEntry, key: bytes) -> MemoryEntry:
    """Return a copy of *entry* with ``content`` and ``detail`` encrypted.

    AAD is derived from ``{entry_id}:{namespace}:{field_name}`` so that a
    ciphertext copied from one field or entry cannot be decrypted as another
    field (ciphertext transplant prevention — GCM tag binds the AAD).
    """
    ns = entry.namespace or "default"
    data = entry.model_dump()
    data["content"] = encrypt_field(entry.content, key, aad=f"{entry.id}:{ns}:content".encode())
    if entry.detail:
        data["detail"] = encrypt_field(entry.detail, key, aad=f"{entry.id}:{ns}:detail".encode())
    return MemoryEntry.model_validate(data, strict=False)


def decrypt_entry_fields(entry: MemoryEntry, key: bytes) -> MemoryEntry:
    """Return a copy of *entry* with ``content`` and ``detail`` decrypted."""
    ns = entry.namespace or "default"
    data = entry.model_dump()
    data["content"] = decrypt_field(entry.content, key, aad=f"{entry.id}:{ns}:content".encode())
    if entry.detail:
        data["detail"] = decrypt_field(entry.detail, key, aad=f"{entry.id}:{ns}:detail".encode())
    return MemoryEntry.model_validate(data, strict=False)


def _namespace_db_path(config: MemoryConfig, namespace: str) -> Path:
    ns_dir = namespace.replace(":", "_")
    return Path(config.storage_path) / ns_dir / config.sqlite_db_name


_KDF_SALT = b"trw-memory-key-rotation-v1"  # static per-purpose salt; callers supply the domain
_KDF_ITERATIONS = 600_000  # NIST SP 800-132 § 5.2 (2023) recommended minimum for PBKDF2-SHA256


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
    # Passphrases: stretch with PBKDF2-HMAC-SHA256 so dictionary attacks on
    # recovered ciphertexts require per-guess hashing rather than a single SHA-256.
    return hashlib.pbkdf2_hmac(
        "sha256",
        new_passphrase.encode("utf-8"),
        _KDF_SALT,
        _KDF_ITERATIONS,
        dklen=_KEY_LENGTH,
    )


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
    from trw_memory.storage import _dbapi as _driver
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
        # The WAL must be fully flushed before the re-key copy below. TRUNCATE
        # does the most thorough flush, but on an engine WITHOUT the WAL-reset
        # fix (SQLite < 3.51.3, pre-backport 3.44.x/3.50.x) a TRUNCATE racing
        # another connection is itself the corruption trigger
        # (sqlite.org/wal.html §walresetbug — the documented memory.db incident
        # class). On a WAL-unsafe engine fall back to PASSIVE, which is safe on
        # any engine: it never resets the WAL, and if it cannot fully flush
        # (because another connection holds the WAL) it reports busy != 0, which
        # the exclusivity guard below converts into an abort. Key rotation on a
        # WAL-unsafe engine therefore requires all other connections closed.
        checkpoint_mode = "TRUNCATE" if _driver.is_wal_reset_safe() else "PASSIVE"
        checkpoint_row = checkpoint_conn.execute(f"PRAGMA wal_checkpoint({checkpoint_mode})").fetchone()
        # Enforce exclusivity MECHANICALLY: busy == 0 means the checkpoint had
        # exclusive access and fully flushed. ANY other result — busy != 0 (a
        # writer holds the WAL) OR a None row (an abnormal/empty PRAGMA response)
        # — must abort. A destructive key rotation cannot proceed on an
        # ambiguous flush; the safe default for a missing result is failure
        # (memory-storage-6).
        if checkpoint_row is None or checkpoint_row[0] != 0:
            busy = "unknown (no PRAGMA result)" if checkpoint_row is None else checkpoint_row[0]
            raise KeyRotationError(
                "key rotation requires exclusive database access, but the WAL "
                f"checkpoint did not complete cleanly (wal_checkpoint busy={busy})",
                path=str(db_path),
            )
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
        # SQLCipher's PRAGMA rekey does NOT support bound parameters — the key
        # hex MUST be embedded in the SQL text. To stop the key from ever
        # leaking through a raised exception or SQL error log, run the rekey in
        # an isolated try/except and re-raise a sanitized error that drops the
        # original exception's message (and, via `from None`, its chained
        # context) — both of which may echo the SQL containing the key.
        try:
            rotation_conn.execute(f"PRAGMA rekey = \"x'{new_sqlcipher_key_hex}'\"")
        except Exception:
            # Raise OUTSIDE the except handler so the original (key-bearing)
            # exception is not retained on `__context__`. `from None` alone only
            # sets __suppress_context__ for display; the object is still
            # reachable. Building the error here guarantees no key material can
            # be recovered from the raised exception's chain.
            rekey_error: KeyRotationError | None = KeyRotationError(
                "Key rotation failed while applying the new database key "
                "(error details suppressed to protect key material)",
                path=str(db_path),
            )
        else:
            rekey_error = None
        if rekey_error is not None:
            raise rekey_error

        rows = rotation_conn.execute("PRAGMA integrity_check").fetchall()
        if len(rows) != 1 or rows[0][0] != "ok":
            detail = rows[0][0] if rows else "empty"
            raise KeyRotationError(f"Key rotation integrity check failed: {detail}", path=str(db_path))
    except Exception as exc:
        rotation_conn.close()
        _restore_database_from_backup(db_path, backup_path)
        if isinstance(exc, KeyRotationError):
            raise
        # The only key-bearing statement (PRAGMA rekey) is sanitized in its own
        # inner block above, so any exception reaching here cannot carry key
        # material — it is safe to surface for debugging.
        raise KeyRotationError(f"Key rotation failed: {exc}", path=str(db_path)) from exc
    else:
        rotation_conn.close()

    try:
        _persist_rotated_master_key(new_master_key, effective_config)
    except Exception as exc:
        _restore_database_from_backup(db_path, backup_path)
        raise KeyRotationError(f"Key rotation persisted new database key but failed to store it: {exc}") from exc

    logger.info("key_rotation_complete", namespace=namespace, db=str(db_path), backup=str(backup_path))
