"""Tests for ``trw_memory.security.keys.rotate_master_key``."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from trw_memory.exceptions import ConfigError
from trw_memory.models.memory import MemoryEntry, MemoryStatus
from trw_memory.security import (
    decrypt_entry_fields,
    derive_namespace_key_bytes,
    encrypt_entry_fields,
    generate_master_key,
    rotate_master_key,
)

from ._test_keys_support import _make_entry, clear_master_key_cache_fixture


class TestRotateMasterKey:
    def test_rotate_re_encrypts_all_entries(self, tmp_path: Path) -> None:
        from trw_memory.storage.sqlite_backend import SQLiteBackend

        old_key = generate_master_key()
        new_key = generate_master_key()

        entry = _make_entry("rotate-1", "secret content")
        ns_key_old = derive_namespace_key_bytes(old_key, entry.namespace)
        encrypted_entry = encrypt_entry_fields(entry, ns_key_old)

        with SQLiteBackend(tmp_path / "rotate.db") as backend:
            backend.store(encrypted_entry)

            count = rotate_master_key(old_key, new_key, backend)

            assert count == 1

            stored = backend.get("rotate-1")
            assert stored is not None
            ns_key_new = derive_namespace_key_bytes(new_key, stored.namespace)
            decrypted = decrypt_entry_fields(stored, ns_key_new)
            assert decrypted.content == "secret content"

    def test_rotate_returns_entry_count(self, tmp_path: Path) -> None:
        from trw_memory.storage.sqlite_backend import SQLiteBackend

        old_key = generate_master_key()
        new_key = generate_master_key()

        with SQLiteBackend(tmp_path / "rotate_count.db") as backend:
            for i in range(5):
                entry = _make_entry(f"count-{i}", f"content {i}")
                ns_key = derive_namespace_key_bytes(old_key, entry.namespace)
                enc = encrypt_entry_fields(entry, ns_key)
                backend.store(enc)

            count = rotate_master_key(old_key, new_key, backend)

        assert count == 5

    def test_rotate_empty_backend_returns_zero(self, tmp_path: Path) -> None:
        from trw_memory.storage.sqlite_backend import SQLiteBackend

        old_key = generate_master_key()
        new_key = generate_master_key()
        with SQLiteBackend(tmp_path / "empty.db") as backend:
            count = rotate_master_key(old_key, new_key, backend)
        assert count == 0

    def test_rotate_rejects_short_old_key(self, tmp_path: Path) -> None:
        from trw_memory.storage.sqlite_backend import SQLiteBackend

        new_key = generate_master_key()
        with SQLiteBackend(tmp_path / "r_short_old.db") as backend:
            with pytest.raises(ConfigError, match="old_key must be 32 bytes"):
                rotate_master_key(b"short", new_key, backend)

    def test_rotate_rejects_short_new_key(self, tmp_path: Path) -> None:
        from trw_memory.storage.sqlite_backend import SQLiteBackend

        old_key = generate_master_key()
        with SQLiteBackend(tmp_path / "r_short_new.db") as backend:
            with pytest.raises(ConfigError, match="new_key must be 32 bytes"):
                rotate_master_key(old_key, b"short", backend)

    def test_rotate_multiple_namespaces(self, tmp_path: Path) -> None:
        from trw_memory.storage.sqlite_backend import SQLiteBackend

        old_key = generate_master_key()
        new_key = generate_master_key()
        namespaces = ["agents", "teams", "global"]

        now = datetime.now(timezone.utc)
        entries = [
            MemoryEntry(
                id=f"ns-{ns}",
                content=f"content for {ns}",
                detail="detail",
                namespace=ns,
                status=MemoryStatus.ACTIVE,
                created_at=now,
                updated_at=now,
            )
            for ns in namespaces
        ]

        with SQLiteBackend(tmp_path / "multi_ns.db") as backend:
            for entry in entries:
                ns_key = derive_namespace_key_bytes(old_key, entry.namespace)
                enc = encrypt_entry_fields(entry, ns_key)
                backend.store(enc)

            count = rotate_master_key(old_key, new_key, backend)
            assert count == len(namespaces)

            for entry in entries:
                stored = backend.get(entry.id)
                assert stored is not None
                ns_key_new = derive_namespace_key_bytes(new_key, stored.namespace)
                decrypted = decrypt_entry_fields(stored, ns_key_new)
                assert decrypted.content == f"content for {stored.namespace}"

    def test_old_key_cannot_decrypt_after_rotation(self, tmp_path: Path) -> None:
        from cryptography.exceptions import InvalidTag

        from trw_memory.storage.sqlite_backend import SQLiteBackend

        old_key = generate_master_key()
        new_key = generate_master_key()

        entry = _make_entry("after-rotate", "was secret")
        ns_key_old = derive_namespace_key_bytes(old_key, entry.namespace)
        enc = encrypt_entry_fields(entry, ns_key_old)

        with SQLiteBackend(tmp_path / "old_fail.db") as backend:
            backend.store(enc)
            rotate_master_key(old_key, new_key, backend)
            stored = backend.get("after-rotate")
            assert stored is not None
            ns_key_old_again = derive_namespace_key_bytes(old_key, stored.namespace)
            with pytest.raises(InvalidTag):
                decrypt_entry_fields(stored, ns_key_old_again)
