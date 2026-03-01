"""Tests for trw_memory.security.encryption — AES-256-GCM field encryption.

Coverage:
- generate_master_key()
- derive_namespace_key()
- encrypt_field() / decrypt_field()
- encrypt_entry_fields() / decrypt_entry_fields()
- Error handling: wrong key length, short payload, invalid base64, wrong key
"""

from __future__ import annotations

import base64
from datetime import datetime, timezone

import pytest
from cryptography.exceptions import InvalidTag

from trw_memory.models.memory import MemoryEntry, MemoryStatus
from trw_memory.security import (
    decrypt_entry_fields,
    decrypt_field,
    derive_namespace_key,
    encrypt_entry_fields,
    encrypt_field,
    generate_master_key,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_KEY_LENGTH = 32


def _make_entry(
    entry_id: str = "enc-test-1",
    content: str = "test content",
    detail: str = "",
    namespace: str = "default",
) -> MemoryEntry:
    now = datetime.now(timezone.utc)
    return MemoryEntry(
        id=entry_id,
        content=content,
        detail=detail,
        namespace=namespace,
        status=MemoryStatus.ACTIVE,
        created_at=now,
        updated_at=now,
    )


# ---------------------------------------------------------------------------
# generate_master_key
# ---------------------------------------------------------------------------


class TestGenerateMasterKey:
    def test_returns_32_bytes(self) -> None:
        key = generate_master_key()
        assert isinstance(key, bytes)
        assert len(key) == _KEY_LENGTH

    def test_each_call_is_unique(self) -> None:
        keys = {generate_master_key() for _ in range(20)}
        # All 20 should be unique (probability of collision is negligible)
        assert len(keys) == 20

    def test_returns_bytes_type(self) -> None:
        key = generate_master_key()
        assert type(key) is bytes

    def test_key_is_not_all_zeros(self) -> None:
        key = generate_master_key()
        assert key != b"\x00" * _KEY_LENGTH


# ---------------------------------------------------------------------------
# derive_namespace_key
# ---------------------------------------------------------------------------


class TestDeriveNamespaceKey:
    def test_deterministic_for_same_inputs(self) -> None:
        master = generate_master_key()
        k1 = derive_namespace_key(master, "agents")
        k2 = derive_namespace_key(master, "agents")
        assert k1 == k2

    def test_different_namespaces_produce_different_keys(self) -> None:
        master = generate_master_key()
        k_agents = derive_namespace_key(master, "agents")
        k_projects = derive_namespace_key(master, "projects")
        assert k_agents != k_projects

    def test_different_master_keys_produce_different_derived_keys(self) -> None:
        master1 = generate_master_key()
        master2 = generate_master_key()
        k1 = derive_namespace_key(master1, "namespace")
        k2 = derive_namespace_key(master2, "namespace")
        assert k1 != k2

    def test_derived_key_is_32_bytes(self) -> None:
        master = generate_master_key()
        key = derive_namespace_key(master, "test-ns")
        assert len(key) == _KEY_LENGTH

    def test_rejects_short_master_key(self) -> None:
        with pytest.raises(ValueError, match="master_key must be 32 bytes"):
            derive_namespace_key(b"too_short", "namespace")

    def test_rejects_long_master_key(self) -> None:
        with pytest.raises(ValueError, match="master_key must be 32 bytes"):
            derive_namespace_key(b"x" * 64, "namespace")

    def test_empty_namespace_works(self) -> None:
        master = generate_master_key()
        key = derive_namespace_key(master, "")
        assert len(key) == _KEY_LENGTH

    def test_unicode_namespace_works(self) -> None:
        master = generate_master_key()
        key = derive_namespace_key(master, "namespace-\u00e9toile")
        assert len(key) == _KEY_LENGTH

    @pytest.mark.parametrize(
        "namespace",
        ["agents", "teams", "orgs", "global", "project-xyz", "a" * 200],
    )
    def test_parametrized_namespace_variants(self, namespace: str) -> None:
        master = generate_master_key()
        key = derive_namespace_key(master, namespace)
        assert isinstance(key, bytes)
        assert len(key) == _KEY_LENGTH


# ---------------------------------------------------------------------------
# encrypt_field / decrypt_field — happy path
# ---------------------------------------------------------------------------


class TestEncryptDecryptField:
    def test_round_trip_preserves_plaintext(self) -> None:
        key = generate_master_key()
        plaintext = "hello, world!"
        ciphertext = encrypt_field(plaintext, key)
        result = decrypt_field(ciphertext, key)
        assert result == plaintext

    def test_round_trip_empty_string(self) -> None:
        key = generate_master_key()
        ciphertext = encrypt_field("", key)
        result = decrypt_field(ciphertext, key)
        assert result == ""

    def test_round_trip_unicode(self) -> None:
        key = generate_master_key()
        plaintext = "Ünïcödé stríng with emojis: \U0001f600\U0001f44d"
        ciphertext = encrypt_field(plaintext, key)
        result = decrypt_field(ciphertext, key)
        assert result == plaintext

    def test_round_trip_multiline_content(self) -> None:
        key = generate_master_key()
        plaintext = "line one\nline two\nline three"
        ciphertext = encrypt_field(plaintext, key)
        result = decrypt_field(ciphertext, key)
        assert result == plaintext

    def test_round_trip_large_content(self) -> None:
        key = generate_master_key()
        plaintext = "A" * 100_000
        ciphertext = encrypt_field(plaintext, key)
        result = decrypt_field(ciphertext, key)
        assert result == plaintext

    def test_encrypt_returns_base64_string(self) -> None:
        key = generate_master_key()
        ciphertext = encrypt_field("test", key)
        assert isinstance(ciphertext, str)
        # Must be valid base64
        decoded = base64.b64decode(ciphertext)
        assert len(decoded) > 0

    def test_different_encryptions_of_same_text_are_different(self) -> None:
        """Random nonce means each encryption of the same plaintext differs."""
        key = generate_master_key()
        plaintext = "same plaintext"
        ct1 = encrypt_field(plaintext, key)
        ct2 = encrypt_field(plaintext, key)
        assert ct1 != ct2

    def test_ciphertext_is_not_plaintext(self) -> None:
        key = generate_master_key()
        plaintext = "sensitive data"
        ciphertext = encrypt_field(plaintext, key)
        assert plaintext not in ciphertext
        assert plaintext.encode() not in base64.b64decode(ciphertext)


# ---------------------------------------------------------------------------
# encrypt_field / decrypt_field — error handling
# ---------------------------------------------------------------------------


class TestEncryptDecryptFieldErrors:
    def test_encrypt_rejects_short_key(self) -> None:
        with pytest.raises(ValueError, match="key must be 32 bytes"):
            encrypt_field("data", b"shortkey")

    def test_encrypt_rejects_long_key(self) -> None:
        with pytest.raises(ValueError, match="key must be 32 bytes"):
            encrypt_field("data", b"x" * 64)

    def test_decrypt_rejects_short_key(self) -> None:
        key = generate_master_key()
        ciphertext = encrypt_field("data", key)
        with pytest.raises(ValueError, match="key must be 32 bytes"):
            decrypt_field(ciphertext, b"shortkey")

    def test_decrypt_wrong_key_raises_invalid_tag(self) -> None:
        key1 = generate_master_key()
        key2 = generate_master_key()
        ciphertext = encrypt_field("secret", key1)
        with pytest.raises(InvalidTag):
            decrypt_field(ciphertext, key2)

    def test_decrypt_tampered_ciphertext_raises_invalid_tag(self) -> None:
        key = generate_master_key()
        ciphertext = encrypt_field("original", key)
        # Decode, flip a byte in the ciphertext body, re-encode
        payload = bytearray(base64.b64decode(ciphertext))
        payload[20] ^= 0xFF  # flip a bit in ciphertext area
        tampered = base64.b64encode(bytes(payload)).decode("ascii")
        with pytest.raises(InvalidTag):
            decrypt_field(tampered, key)

    def test_decrypt_short_payload_raises_value_error(self) -> None:
        """Payload too short to contain nonce + tag."""
        key = generate_master_key()
        # 12-byte nonce + 16-byte tag = 28 minimum; give only 10 bytes
        short_payload = base64.b64encode(b"x" * 10).decode("ascii")
        with pytest.raises(ValueError, match="too short"):
            decrypt_field(short_payload, key)

    def test_decrypt_invalid_base64_raises(self) -> None:
        key = generate_master_key()
        with pytest.raises(Exception):  # base64.b64decode raises binascii.Error
            decrypt_field("not!!valid!!base64===", key)

    def test_decrypt_empty_string_raises(self) -> None:
        key = generate_master_key()
        with pytest.raises(Exception):
            decrypt_field("", key)

    @pytest.mark.parametrize("bad_key_len", [0, 1, 16, 31, 33, 64])
    def test_encrypt_parametrized_bad_key_lengths(self, bad_key_len: int) -> None:
        with pytest.raises(ValueError, match="key must be 32 bytes"):
            encrypt_field("data", b"x" * bad_key_len)

    @pytest.mark.parametrize("bad_key_len", [0, 1, 16, 31, 33, 64])
    def test_decrypt_parametrized_bad_key_lengths(self, bad_key_len: int) -> None:
        key = generate_master_key()
        ciphertext = encrypt_field("data", key)
        with pytest.raises(ValueError, match="key must be 32 bytes"):
            decrypt_field(ciphertext, b"x" * bad_key_len)


# ---------------------------------------------------------------------------
# encrypt_entry_fields / decrypt_entry_fields — happy path
# ---------------------------------------------------------------------------


class TestEncryptDecryptEntryFields:
    """Tests for encrypt_entry_fields / decrypt_entry_fields."""

    def test_round_trip_preserves_entry_content(self) -> None:
        key = generate_master_key()
        entry = _make_entry(content="Important knowledge", detail="More detail here")
        encrypted = encrypt_entry_fields(entry, key)
        decrypted = decrypt_entry_fields(encrypted, key)
        assert decrypted.content == entry.content
        assert decrypted.detail == entry.detail

    def test_encrypted_content_is_not_plaintext(self) -> None:
        key = generate_master_key()
        entry = _make_entry(content="plaintext content")
        encrypted = encrypt_entry_fields(entry, key)
        assert encrypted.content != entry.content
        assert "plaintext content" not in encrypted.content

    def test_encrypted_detail_is_not_plaintext(self) -> None:
        key = generate_master_key()
        entry = _make_entry(content="content", detail="plaintext detail")
        encrypted = encrypt_entry_fields(entry, key)
        assert encrypted.detail != "plaintext detail"

    def test_empty_detail_not_encrypted(self) -> None:
        """Empty detail field should remain empty (not encrypted)."""
        key = generate_master_key()
        entry = _make_entry(content="content only", detail="")
        encrypted = encrypt_entry_fields(entry, key)
        assert encrypted.detail == ""

    def test_round_trip_empty_detail(self) -> None:
        key = generate_master_key()
        entry = _make_entry(content="content only", detail="")
        encrypted = encrypt_entry_fields(entry, key)
        decrypted = decrypt_entry_fields(encrypted, key)
        assert decrypted.content == "content only"
        assert decrypted.detail == ""

    def test_non_content_fields_preserved(self) -> None:
        key = generate_master_key()
        entry = _make_entry(
            entry_id="preserve-me",
            content="content",
            detail="detail",
            namespace="custom-ns",
        )
        encrypted = encrypt_entry_fields(entry, key)
        decrypted = decrypt_entry_fields(encrypted, key)
        assert decrypted.id == "preserve-me"
        assert decrypted.namespace == "custom-ns"
        assert decrypted.importance == entry.importance
        assert decrypted.status == entry.status

    def test_encrypt_returns_new_entry_object(self) -> None:
        key = generate_master_key()
        entry = _make_entry(content="data")
        encrypted = encrypt_entry_fields(entry, key)
        assert encrypted is not entry

    def test_decrypt_returns_new_entry_object(self) -> None:
        key = generate_master_key()
        entry = _make_entry(content="data")
        encrypted = encrypt_entry_fields(entry, key)
        decrypted = decrypt_entry_fields(encrypted, key)
        assert decrypted is not encrypted

    def test_round_trip_unicode_content(self) -> None:
        key = generate_master_key()
        entry = _make_entry(
            content="\u00e9toile \U0001f600",
            detail="détail avec accents",
        )
        encrypted = encrypt_entry_fields(entry, key)
        decrypted = decrypt_entry_fields(encrypted, key)
        assert decrypted.content == entry.content
        assert decrypted.detail == entry.detail

    def test_wrong_key_on_decrypt_raises(self) -> None:
        key1 = generate_master_key()
        key2 = generate_master_key()
        entry = _make_entry(content="secret")
        encrypted = encrypt_entry_fields(entry, key1)
        with pytest.raises(InvalidTag):
            decrypt_entry_fields(encrypted, key2)

    def test_multiple_round_trips_stable(self) -> None:
        """Encrypt then decrypt three times — result must still match."""
        key = generate_master_key()
        entry = _make_entry(content="stable content", detail="stable detail")
        result = entry
        for _ in range(3):
            enc = encrypt_entry_fields(result, key)
            result = decrypt_entry_fields(enc, key)
        assert result.content == entry.content
        assert result.detail == entry.detail

    @pytest.mark.parametrize(
        "content,detail",
        [
            ("short", ""),
            ("medium length content string here", "some detail"),
            ("A" * 5000, "B" * 5000),
            ("contains\nnewlines\nand\ttabs", "more\nlines"),
        ],
    )
    def test_parametrized_content_sizes(self, content: str, detail: str) -> None:
        key = generate_master_key()
        entry = _make_entry(content=content, detail=detail)
        encrypted = encrypt_entry_fields(entry, key)
        decrypted = decrypt_entry_fields(encrypted, key)
        assert decrypted.content == content
        assert decrypted.detail == detail


# ---------------------------------------------------------------------------
# HKDF namespace key isolation
# ---------------------------------------------------------------------------


class TestHkdfNamespaceIsolation:
    def test_same_master_different_namespaces_cannot_cross_decrypt(self) -> None:
        """Data encrypted with ns-A key must not decrypt with ns-B key."""
        master = generate_master_key()
        key_a = derive_namespace_key(master, "namespace-A")
        key_b = derive_namespace_key(master, "namespace-B")

        ciphertext = encrypt_field("secret for A only", key_a)
        with pytest.raises(InvalidTag):
            decrypt_field(ciphertext, key_b)

    def test_derived_keys_are_distinct_for_all_standard_namespaces(self) -> None:
        master = generate_master_key()
        namespaces = ["default", "agents", "teams", "orgs", "global"]
        keys = [derive_namespace_key(master, ns) for ns in namespaces]
        # All must be unique
        assert len(set(keys)) == len(namespaces)

    def test_long_namespace_string_derives_valid_key(self) -> None:
        master = generate_master_key()
        long_ns = "a" * 1000
        key = derive_namespace_key(master, long_ns)
        assert len(key) == _KEY_LENGTH
        # Must be usable for encrypt/decrypt
        ct = encrypt_field("data", key)
        assert decrypt_field(ct, key) == "data"

    def test_namespace_key_derivation_is_independent_of_content(self) -> None:
        """Two different data values produce different ciphertexts under the same key."""
        master = generate_master_key()
        key = derive_namespace_key(master, "test-ns")
        ct1 = encrypt_field("content A", key)
        ct2 = encrypt_field("content B", key)
        assert ct1 != ct2
        assert decrypt_field(ct1, key) == "content A"
        assert decrypt_field(ct2, key) == "content B"
