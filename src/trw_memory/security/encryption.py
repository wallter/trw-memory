"""Field-level AES-256-GCM encryption for memory entries.

Provides encrypt/decrypt operations using AES-256-GCM with per-namespace
key derivation via HKDF-SHA256.  Encrypted payloads are base64-encoded
strings containing ``nonce || ciphertext || tag``.
"""

from __future__ import annotations

import base64
import os

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from trw_memory.models.memory import MemoryEntry

# AES-256 key length in bytes
_KEY_LENGTH = 32

# GCM nonce length (96 bits as recommended by NIST SP 800-38D)
_NONCE_LENGTH = 12


def generate_master_key() -> bytes:
    """Generate a cryptographically random 256-bit master key.

    Returns:
        32 random bytes suitable for AES-256.
    """
    return os.urandom(_KEY_LENGTH)


def derive_namespace_key(master_key: bytes, namespace: str) -> bytes:
    """Derive a unique 256-bit key for *namespace* using HKDF-SHA256.

    Args:
        master_key: The 32-byte master key.
        namespace: Namespace string used as HKDF info parameter.

    Returns:
        32-byte derived key unique to the master key + namespace pair.

    Raises:
        ValueError: If *master_key* is not 32 bytes.
    """
    if len(master_key) != _KEY_LENGTH:
        raise ValueError(f"master_key must be {_KEY_LENGTH} bytes, got {len(master_key)}")
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=_KEY_LENGTH,
        salt=None,
        info=namespace.encode("utf-8"),
    )
    return bytes(hkdf.derive(master_key))


def encrypt_field(plaintext: str, key: bytes) -> str:
    """Encrypt a plaintext string with AES-256-GCM.

    Args:
        plaintext: UTF-8 string to encrypt.
        key: 32-byte AES key.

    Returns:
        Base64-encoded string containing ``nonce || ciphertext || tag``.

    Raises:
        ValueError: If *key* is not 32 bytes.
    """
    if len(key) != _KEY_LENGTH:
        raise ValueError(f"key must be {_KEY_LENGTH} bytes, got {len(key)}")
    nonce = os.urandom(_NONCE_LENGTH)
    aesgcm = AESGCM(key)
    # AESGCM.encrypt returns ciphertext + 16-byte tag appended
    ct_with_tag = aesgcm.encrypt(nonce, plaintext.encode("utf-8"), None)
    # Combine: nonce || ciphertext_with_tag
    payload = nonce + ct_with_tag
    return base64.b64encode(payload).decode("ascii")


def decrypt_field(ciphertext_b64: str, key: bytes) -> str:
    """Decrypt a base64-encoded AES-256-GCM payload.

    Args:
        ciphertext_b64: Base64-encoded ``nonce || ciphertext || tag``.
        key: 32-byte AES key (must match the key used for encryption).

    Returns:
        The decrypted plaintext string.

    Raises:
        ValueError: If *key* is not 32 bytes or the payload is malformed.
        cryptography.exceptions.InvalidTag: If decryption fails (wrong key
            or tampered data).
    """
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
    """Return a copy of *entry* with ``content`` and ``detail`` encrypted.

    Args:
        entry: The memory entry to encrypt.
        key: 32-byte AES key.

    Returns:
        A new :class:`MemoryEntry` with encrypted ``content`` and ``detail``.
    """
    data = entry.model_dump()
    data["content"] = encrypt_field(entry.content, key)
    if entry.detail:
        data["detail"] = encrypt_field(entry.detail, key)
    return MemoryEntry.model_validate(data, strict=False)


def decrypt_entry_fields(entry: MemoryEntry, key: bytes) -> MemoryEntry:
    """Return a copy of *entry* with ``content`` and ``detail`` decrypted.

    Args:
        entry: The memory entry to decrypt.
        key: 32-byte AES key.

    Returns:
        A new :class:`MemoryEntry` with decrypted ``content`` and ``detail``.
    """
    data = entry.model_dump()
    data["content"] = decrypt_field(entry.content, key)
    if entry.detail:
        data["detail"] = decrypt_field(entry.detail, key)
    return MemoryEntry.model_validate(data, strict=False)
