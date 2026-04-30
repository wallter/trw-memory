"""Field and entry encryption tests for trw_memory.security.encryption."""

from __future__ import annotations

import base64

import pytest
from cryptography.exceptions import InvalidTag

from trw_memory.security import (
    decrypt_entry_fields,
    decrypt_field,
    encrypt_entry_fields,
    encrypt_field,
    generate_master_key,
)

from ._test_encryption_support import _make_entry, clear_master_key_cache_fixture


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
        decoded = base64.b64decode(ciphertext)
        assert len(decoded) > 0

    def test_different_encryptions_of_same_text_are_different(self) -> None:
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
        payload = bytearray(base64.b64decode(ciphertext))
        payload[20] ^= 0xFF
        tampered = base64.b64encode(bytes(payload)).decode("ascii")
        with pytest.raises(InvalidTag):
            decrypt_field(tampered, key)

    def test_decrypt_short_payload_raises_value_error(self) -> None:
        key = generate_master_key()
        short_payload = base64.b64encode(b"x" * 10).decode("ascii")
        with pytest.raises(ValueError, match="too short"):
            decrypt_field(short_payload, key)

    def test_decrypt_invalid_base64_raises(self) -> None:
        key = generate_master_key()
        with pytest.raises(Exception):
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


class TestEncryptDecryptEntryFields:
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
