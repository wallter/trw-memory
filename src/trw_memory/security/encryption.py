"""Field-level AES-256-GCM encryption helpers."""

from __future__ import annotations

import base64
import hashlib
import os

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from trw_memory.models.memory import MemoryEntry

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
