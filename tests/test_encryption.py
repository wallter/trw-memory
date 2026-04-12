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
import hashlib
import sqlite3
import time
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from trw_memory.exceptions import AuthorizationError, KeyRotationError
from trw_memory.integrations._backend import create_backend_from_config
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry, MemoryStatus
from trw_memory.tools.recall import memory_recall_impl
from trw_memory.security import (
    decrypt_entry_fields,
    decrypt_field,
    derive_namespace_key,
    derive_namespace_key_bytes,
    encrypt_entry_fields,
    encrypt_field,
    generate_master_key,
    rotate_key,
)
from trw_memory.security.keys import clear_key_cache

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_KEY_LENGTH = 32


@pytest.fixture(autouse=True)
def clear_master_key_cache_fixture() -> Iterator[None]:
    clear_key_cache()
    yield
    clear_key_cache()


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


class _StaticCursor:
    def __init__(self, rows: list[tuple[object, ...]]) -> None:
        self._rows = rows

    def fetchall(self) -> list[tuple[object, ...]]:
        return self._rows

    def fetchone(self) -> tuple[object, ...] | None:
        return self._rows[0] if self._rows else None


class _RotatingSQLCipherConnection:
    def __init__(
        self,
        conn: sqlite3.Connection,
        statements: list[str],
        db_path: Path,
        *,
        integrity_result: str = "ok",
        mutate_on_rekey: bytes | None = None,
    ) -> None:
        object.__setattr__(self, "_conn", conn)
        object.__setattr__(self, "_statements", statements)
        object.__setattr__(self, "_db_path", db_path)
        object.__setattr__(self, "_integrity_result", integrity_result)
        object.__setattr__(self, "_mutate_on_rekey", mutate_on_rekey)

    def __getattr__(self, name: str) -> object:
        return getattr(self._conn, name)

    def __setattr__(self, name: str, value: object) -> None:
        setattr(self._conn, name, value)

    def execute(self, sql: str, *args: object) -> sqlite3.Cursor | _StaticCursor:
        self._statements.append(sql)
        normalized = sql.strip().upper()
        if normalized.startswith("PRAGMA REKEY") and self._mutate_on_rekey is not None:
            with self._db_path.open("ab") as handle:
                handle.write(self._mutate_on_rekey)
            return _StaticCursor([])
        if normalized.startswith("PRAGMA INTEGRITY_CHECK"):
            return _StaticCursor([(self._integrity_result,)])
        return self._conn.execute(sql, *args)


class _RotatingSQLCipherDBAPI:
    Error = sqlite3.Error
    DatabaseError = sqlite3.DatabaseError

    def __init__(
        self,
        statements: list[str],
        *,
        integrity_result: str = "ok",
        mutate_on_rekey: bytes | None = None,
    ) -> None:
        self._statements = statements
        self._integrity_result = integrity_result
        self._mutate_on_rekey = mutate_on_rekey

    def connect(self, database: str, **kwargs: object) -> _RotatingSQLCipherConnection:
        conn = sqlite3.connect(database, **kwargs)
        return _RotatingSQLCipherConnection(
            conn,
            self._statements,
            Path(database),
            integrity_result=self._integrity_result,
            mutate_on_rekey=self._mutate_on_rekey,
        )


def _load_real_sqlcipher_driver_or_skip() -> object:
    from trw_memory.storage.sqlite_backend import _import_sqlcipher_driver

    try:
        return _import_sqlcipher_driver()
    except Exception as exc:
        pytest.skip(f"real SQLCipher driver unavailable: {exc}")


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

    def test_derived_key_is_64_char_hex(self) -> None:
        master = generate_master_key()
        key = derive_namespace_key(master, "test-ns")
        assert len(key) == 64
        assert key == key.lower()
        assert set(key) <= set("0123456789abcdef")

    def test_rejects_short_master_key(self) -> None:
        with pytest.raises(ValueError, match="master_key must be 32 bytes"):
            derive_namespace_key(b"too_short", "namespace")

    def test_rejects_long_master_key(self) -> None:
        with pytest.raises(ValueError, match="master_key must be 32 bytes"):
            derive_namespace_key(b"x" * 64, "namespace")

    def test_empty_namespace_works(self) -> None:
        master = generate_master_key()
        key = derive_namespace_key(master, "")
        assert len(key) == 64

    def test_unicode_namespace_works(self) -> None:
        master = generate_master_key()
        key = derive_namespace_key(master, "namespace-\u00e9toile")
        assert len(key) == 64

    @pytest.mark.parametrize(
        "namespace",
        ["agents", "teams", "orgs", "global", "project-xyz", "a" * 200],
    )
    def test_parametrized_namespace_variants(self, namespace: str) -> None:
        master = generate_master_key()
        key = derive_namespace_key(master, namespace)
        assert isinstance(key, str)
        assert len(key) == 64

    def test_matches_prd_hkdf_parameters(self) -> None:
        master = bytes(range(32))
        namespace = "global"
        key = derive_namespace_key(master, namespace)
        reference = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=hashlib.sha256(namespace.encode("utf-8")).digest(),
            info=b"trw-memory-namespace-key-v1",
        ).derive(master)

        assert key == reference.hex()
        assert derive_namespace_key_bytes(master, namespace) == reference


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
        key_a = derive_namespace_key_bytes(master, "namespace-A")
        key_b = derive_namespace_key_bytes(master, "namespace-B")

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
        key = derive_namespace_key_bytes(master, long_ns)
        assert len(key) == _KEY_LENGTH
        # Must be usable for encrypt/decrypt
        ct = encrypt_field("data", key)
        assert decrypt_field(ct, key) == "data"

    def test_namespace_key_derivation_is_independent_of_content(self) -> None:
        """Two different data values produce different ciphertexts under the same key."""
        master = generate_master_key()
        key = derive_namespace_key_bytes(master, "test-ns")
        ct1 = encrypt_field("content A", key)
        ct2 = encrypt_field("content B", key)
        assert ct1 != ct2
        assert decrypt_field(ct1, key) == "content A"
        assert decrypt_field(ct2, key) == "content B"


class TestRotateKey:
    def test_rotate_key_rekeys_db_and_persists_keyring_entry(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        old_key = generate_master_key()
        new_key = generate_master_key()
        monkeypatch.setenv("MEMORY_MASTER_KEY", old_key.hex())
        mock_keyring = MagicMock()

        config = MemoryConfig(
            storage_backend="sqlite",
            storage_path=str(tmp_path / "storage"),
            encryption_enabled=True,
            key_source="keyring",
            auto_generate_key=False,
            rbac_enabled=True,
        )
        statements: list[str] = []
        monkeypatch.setattr(
            "trw_memory.storage.sqlite_backend._import_sqlcipher_driver",
            lambda: _RotatingSQLCipherDBAPI(statements),
        )

        with (
            patch("trw_memory.security.keys._keyring", mock_keyring),
            patch("trw_memory.security.keys._KEYRING_AVAILABLE", True),
        ):
            backend = create_backend_from_config(config, "default")
            backend.store(_make_entry(content="rotation target", detail="keep me"))
            backend.close()

            rotate_key("default", new_key.hex(), config)

        db_path = Path(config.storage_path) / "default" / config.sqlite_db_name
        assert Path(f"{db_path}.bak").exists()
        mock_keyring.set_password.assert_called_with("trw-memory", "master", new_key.hex())
        assert "PRAGMA wal_checkpoint(TRUNCATE)" in statements
        assert "PRAGMA integrity_check" in statements
        assert "PRAGMA cipher = 'aes-256-cbc'" in statements
        assert "PRAGMA cipher_page_size = 4096" in statements
        assert "PRAGMA kdf_iter = 256000" in statements
        assert any(statement.startswith('PRAGMA rekey = "x\'') for statement in statements)

    def test_rotate_key_restores_backup_on_integrity_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        old_key = generate_master_key()
        new_key = generate_master_key()
        monkeypatch.setenv("MEMORY_MASTER_KEY", old_key.hex())

        config = MemoryConfig(
            storage_backend="sqlite",
            storage_path=str(tmp_path / "storage"),
            encryption_enabled=True,
            key_source="keyring",
            auto_generate_key=False,
            rbac_enabled=True,
        )
        monkeypatch.setattr(
            "trw_memory.storage.sqlite_backend._import_sqlcipher_driver",
            lambda: _RotatingSQLCipherDBAPI([]),
        )

        backend = create_backend_from_config(config, "default")
        backend.store(_make_entry(content="rotation target", detail="keep me"))
        backend.close()

        db_path = Path(config.storage_path) / "default" / config.sqlite_db_name
        original_bytes = db_path.read_bytes()
        failure_statements: list[str] = []
        monkeypatch.setattr(
            "trw_memory.storage.sqlite_backend._import_sqlcipher_driver",
            lambda: _RotatingSQLCipherDBAPI(
                failure_statements,
                integrity_result="disk image malformed",
                mutate_on_rekey=b"corruption-marker",
            ),
        )

        with pytest.raises(KeyRotationError, match="integrity check failed: disk image malformed"):
            rotate_key("default", new_key.hex(), config)

        assert Path(f"{db_path}.bak").exists()
        assert db_path.read_bytes() == original_bytes
        assert any(statement.startswith('PRAGMA rekey = "x\'') for statement in failure_statements)

    def test_rotate_key_writes_keyring_when_configured(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        old_key = generate_master_key()
        new_key = generate_master_key()
        monkeypatch.setenv("MEMORY_MASTER_KEY", old_key.hex())
        mock_keyring = MagicMock()
        statements: list[str] = []
        config = MemoryConfig(
            storage_backend="sqlite",
            storage_path=str(tmp_path / "storage"),
            encryption_enabled=True,
            key_source="env",
            auto_generate_key=False,
            rbac_enabled=True,
        )
        monkeypatch.setattr(
            "trw_memory.storage.sqlite_backend._import_sqlcipher_driver",
            lambda: _RotatingSQLCipherDBAPI(statements),
        )

        with (
            patch("trw_memory.security.keys._keyring", mock_keyring),
            patch("trw_memory.security.keys._KEYRING_AVAILABLE", True),
        ):
            backend = create_backend_from_config(config, "default")
            backend.store(_make_entry(content="rotation target"))
            backend.close()
            rotate_key("default", new_key.hex(), config)

        mock_keyring.set_password.assert_called_with("trw-memory", "master", new_key.hex())

    @pytest.mark.parametrize("role", ["reader", "writer", "none"])
    def test_rotate_key_rejects_non_admin_roles(self, role: str, tmp_path: Path) -> None:
        config = MemoryConfig(
            storage_backend="sqlite",
            storage_path=str(tmp_path / "storage"),
            encryption_enabled=True,
            key_source="env",
            auto_generate_key=False,
            rbac_enabled=True,
            default_role=role,  # type: ignore[arg-type]
        )

        with pytest.raises(AuthorizationError, match=f"Role '{role}' does not have rotate_key permission"):
            rotate_key("default", generate_master_key().hex(), config)

    def test_real_sqlcipher_driver_reports_cipher_version_and_rejects_plain_open(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from trw_memory.storage.sqlite_backend import _apply_sqlcipher_pragmas

        driver = _load_real_sqlcipher_driver_or_skip()
        master_key = generate_master_key()
        monkeypatch.setenv("MEMORY_MASTER_KEY", master_key.hex())

        config = MemoryConfig(
            storage_backend="sqlite",
            storage_path=str(tmp_path / "storage"),
            encryption_enabled=True,
            key_source="env",
            auto_generate_key=False,
        )

        backend = create_backend_from_config(config, "default")
        backend.store(_make_entry(content="real sqlcipher"))
        backend.close()

        db_path = Path(config.storage_path) / "default" / config.sqlite_db_name
        plain_conn = sqlite3.connect(str(db_path))
        try:
            with pytest.raises(sqlite3.DatabaseError):
                plain_conn.execute("SELECT count(*) FROM sqlite_master").fetchone()
        finally:
            plain_conn.close()

        key_hex = derive_namespace_key(master_key, "default")
        conn = driver.connect(str(db_path))
        try:
            conn.execute(f'PRAGMA key = "x\'{key_hex}\'"')
            _apply_sqlcipher_pragmas(conn)
            version = conn.execute("PRAGMA cipher_version").fetchone()[0]
            cipher = conn.execute("PRAGMA cipher").fetchone()[0]
            kdf_iter = conn.execute("PRAGMA kdf_iter").fetchone()[0]
            assert version
            assert str(cipher).lower() == "aes-256-cbc"
            assert int(kdf_iter) == 256000
            assert conn.execute("SELECT count(*) FROM sqlite_master").fetchone()[0] > 0
        finally:
            conn.close()

    def test_real_sqlcipher_rotation_rejects_old_key_after_success(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from trw_memory.storage.sqlite_backend import _apply_sqlcipher_pragmas

        driver = _load_real_sqlcipher_driver_or_skip()
        old_key = generate_master_key()
        new_key = generate_master_key()
        monkeypatch.setenv("MEMORY_MASTER_KEY", old_key.hex())
        mock_keyring = MagicMock()

        config = MemoryConfig(
            storage_backend="sqlite",
            storage_path=str(tmp_path / "storage"),
            encryption_enabled=True,
            key_source="keyring",
            auto_generate_key=False,
            rbac_enabled=True,
        )

        with (
            patch("trw_memory.security.keys._keyring", mock_keyring),
            patch("trw_memory.security.keys._KEYRING_AVAILABLE", True),
        ):
            backend = create_backend_from_config(config, "default")
            backend.store(_make_entry(content="rotation target"))
            backend.close()
            rotate_key("default", new_key.hex(), config)

        db_path = Path(config.storage_path) / "default" / config.sqlite_db_name
        old_conn = driver.connect(str(db_path))
        try:
            old_conn.execute(f'PRAGMA key = "x\'{derive_namespace_key(old_key, "default")}\'"')
            _apply_sqlcipher_pragmas(old_conn)
            with pytest.raises(sqlite3.DatabaseError):
                old_conn.execute("SELECT count(*) FROM sqlite_master").fetchone()
        finally:
            old_conn.close()

        new_conn = driver.connect(str(db_path))
        try:
            new_conn.execute(f'PRAGMA key = "x\'{derive_namespace_key(new_key, "default")}\'"')
            _apply_sqlcipher_pragmas(new_conn)
            assert new_conn.execute("SELECT count(*) FROM sqlite_master").fetchone()[0] > 0
        finally:
            new_conn.close()

    def test_recall_encrypted_vs_unencrypted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _load_real_sqlcipher_driver_or_skip()
        monkeypatch.setattr("trw_memory.tools.recall.get_local_embedder", lambda **_: None)

        def _seed_and_measure(config: MemoryConfig) -> float:
            with create_backend_from_config(config, "default") as backend:
                for index in range(500):
                    backend.store(_make_entry(entry_id=f"recall-{index}", content=f"needle {index}", detail="payload"))
                durations: list[float] = []
                for _ in range(1000):
                    start = time.perf_counter()
                    result = memory_recall_impl(
                        query="needle",
                        namespace="default",
                        backend=backend,
                        config=config,
                    )
                    assert result["total_matches"] >= 1
                    durations.append(time.perf_counter() - start)
                durations.sort()
                return durations[int(len(durations) * 0.95)]

        plain_config = MemoryConfig(storage_backend="sqlite", storage_path=str(tmp_path / "plain"))
        plain_time = _seed_and_measure(plain_config)

        encrypted_key = generate_master_key()
        monkeypatch.setenv("MEMORY_MASTER_KEY", encrypted_key.hex())
        encrypted_config = MemoryConfig(
            storage_backend="sqlite",
            storage_path=str(tmp_path / "encrypted"),
            encryption_enabled=True,
            key_source="env",
            auto_generate_key=False,
        )
        encrypted_time = _seed_and_measure(encrypted_config)

        assert encrypted_time <= plain_time * 1.10

    def test_rotate_key_100mb_database_completes_under_30s(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _load_real_sqlcipher_driver_or_skip()
        old_key = generate_master_key()
        new_key = generate_master_key()
        monkeypatch.setenv("MEMORY_MASTER_KEY", old_key.hex())
        mock_keyring = MagicMock()
        payload = "x" * 5_000_000

        config = MemoryConfig(
            storage_backend="sqlite",
            storage_path=str(tmp_path / "rotation-100mb"),
            encryption_enabled=True,
            key_source="keyring",
            auto_generate_key=False,
            rbac_enabled=True,
        )

        with (
            patch("trw_memory.security.keys._keyring", mock_keyring),
            patch("trw_memory.security.keys._KEYRING_AVAILABLE", True),
        ):
            with create_backend_from_config(config, "default") as backend:
                for index in range(20):
                    backend.store(_make_entry(entry_id=f"large-{index}", content=payload))

            start = time.perf_counter()
            rotate_key("default", new_key.hex(), config)
            elapsed = time.perf_counter() - start

        assert elapsed < 30.0
