# ruff: noqa: F401
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

    def test_rotate_re_encrypts_all_entries_beyond_legacy_cap(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every entry must be re-encrypted with no silent cap.

        The old implementation hardcoded ``list_entries(limit=100_000)``. We
        prove coverage is now count-driven by forcing a tiny fetch headroom and
        confirming the rotation still touches a number of entries that would
        have straddled any fixed cap boundary.
        """
        from trw_memory.storage.sqlite_backend import SQLiteBackend

        # Force headroom to 0 so the fetch limit equals exactly count() — if the
        # code fell back to any smaller fixed cap, entries would be missed.
        monkeypatch.setattr("trw_memory.security.keys._ROTATION_FETCH_HEADROOM", 0)

        old_key = generate_master_key()
        new_key = generate_master_key()
        n_entries = 250

        with SQLiteBackend(tmp_path / "beyond_cap.db") as backend:
            for i in range(n_entries):
                entry = _make_entry(f"bulk-{i:04d}", f"content {i}")
                ns_key = derive_namespace_key_bytes(old_key, entry.namespace)
                backend.store(encrypt_entry_fields(entry, ns_key))

            count = rotate_master_key(old_key, new_key, backend)
            assert count == n_entries

            # Spot-check first and last: both decrypt under the NEW key only.
            for entry_id, expected in (
                ("bulk-0000", "content 0"),
                (f"bulk-{n_entries - 1:04d}", f"content {n_entries - 1}"),
            ):
                stored = backend.get(entry_id)
                assert stored is not None
                ns_key_new = derive_namespace_key_bytes(new_key, stored.namespace)
                assert decrypt_entry_fields(stored, ns_key_new).content == expected

    def test_rotate_raises_when_coverage_incomplete(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """If the backend perpetually reports more entries than it returns,
        rotation must RAISE (never converge) rather than silently leave data
        under the old key."""
        from trw_memory.exceptions import KeyRotationError
        from trw_memory.storage.sqlite_backend import SQLiteBackend

        # Keep the sweep bound small so the non-convergence path is fast.
        monkeypatch.setattr("trw_memory.security.keys._ROTATION_MAX_SWEEPS", 3)

        old_key = generate_master_key()
        new_key = generate_master_key()

        with SQLiteBackend(tmp_path / "incomplete.db") as backend:
            for i in range(3):
                entry = _make_entry(f"short-{i}", f"content {i}")
                ns_key = derive_namespace_key_bytes(old_key, entry.namespace)
                backend.store(encrypt_entry_fields(entry, ns_key))

            # count() reports more than list_entries ever returns → never converges.
            object.__setattr__(backend, "count", lambda namespace=None: 99)

            with pytest.raises(KeyRotationError, match="did not converge"):
                rotate_master_key(old_key, new_key, backend)

    def test_rotate_re_encrypts_entry_inserted_during_rotation(self, tmp_path: Path) -> None:
        """A row inserted concurrently DURING the sweep must still be rotated.

        Regression (v0.9.2): the prior single-fetch + PRE-rotation count snapshot
        let a concurrent insert escape re-encryption while the count check stayed
        satisfied. The convergence sweep must pick it up on a later pass and leave
        it decryptable ONLY under the new key.
        """
        from cryptography.exceptions import InvalidTag

        from trw_memory.storage.sqlite_backend import SQLiteBackend

        old_key = generate_master_key()
        new_key = generate_master_key()

        with SQLiteBackend(tmp_path / "concurrent.db") as backend:
            base = _make_entry("base-0", "base content")
            ns_key_old = derive_namespace_key_bytes(old_key, base.namespace)
            backend.store(encrypt_entry_fields(base, ns_key_old))

            # Wrap list_entries so the FIRST pass simulates a concurrent writer
            # inserting a brand-new row (encrypted under the OLD key) right after
            # the rotation read it — exactly the row the old code missed.
            real_list = backend.list_entries
            injected = {"done": False}

            def list_with_injection(*args: object, **kwargs: object) -> list[MemoryEntry]:
                result = real_list(*args, **kwargs)  # type: ignore[arg-type]
                if not injected["done"]:
                    injected["done"] = True
                    late = _make_entry("late-1", "late content")
                    late_key_old = derive_namespace_key_bytes(old_key, late.namespace)
                    backend.store(encrypt_entry_fields(late, late_key_old))
                return result

            object.__setattr__(backend, "list_entries", list_with_injection)

            count = rotate_master_key(old_key, new_key, backend)
            assert count == 2  # base + the concurrently-inserted row

            stored = backend.get("late-1")
            assert stored is not None
            ns_key_new = derive_namespace_key_bytes(new_key, stored.namespace)
            assert decrypt_entry_fields(stored, ns_key_new).content == "late content"
            # And it must NO LONGER decrypt under the old key.
            late_old_again = derive_namespace_key_bytes(old_key, stored.namespace)
            with pytest.raises(InvalidTag):
                decrypt_entry_fields(stored, late_old_again)

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
