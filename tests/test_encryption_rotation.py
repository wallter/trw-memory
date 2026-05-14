# ruff: noqa: F401
"""Rotation and SQLCipher integration tests for trw_memory.security.encryption."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Literal
from unittest.mock import MagicMock, patch

import pytest

from trw_memory.exceptions import AuthorizationError, KeyRotationError
from trw_memory.integrations._backend import create_backend_from_config
from trw_memory.models.config import MemoryConfig
from trw_memory.security import derive_namespace_key, generate_master_key, rotate_key
from trw_memory.tools.recall import memory_recall_impl

from ._test_encryption_support import (
    _load_real_sqlcipher_driver_or_skip,
    _make_entry,
    _RotatingSQLCipherDBAPI,
    clear_master_key_cache_fixture,
)


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
        assert any(statement.startswith("PRAGMA rekey = \"x'") for statement in statements)

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
        assert any(statement.startswith("PRAGMA rekey = \"x'") for statement in failure_statements)

    def test_rotate_key_writes_keyring_when_configured(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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
    def test_rotate_key_rejects_non_admin_roles(
        self,
        role: Literal["reader", "writer", "none"],
        tmp_path: Path,
    ) -> None:
        config = MemoryConfig(
            storage_backend="sqlite",
            storage_path=str(tmp_path / "storage"),
            encryption_enabled=True,
            key_source="env",
            auto_generate_key=False,
            rbac_enabled=True,
            default_role=role,
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
            conn.execute(f"PRAGMA key = \"x'{key_hex}'\"")
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
            old_conn.execute(f"PRAGMA key = \"x'{derive_namespace_key(old_key, 'default')}'\"")
            _apply_sqlcipher_pragmas(old_conn)
            with pytest.raises(sqlite3.DatabaseError):
                old_conn.execute("SELECT count(*) FROM sqlite_master").fetchone()
        finally:
            old_conn.close()

        new_conn = driver.connect(str(db_path))
        try:
            new_conn.execute(f"PRAGMA key = \"x'{derive_namespace_key(new_key, 'default')}'\"")
            _apply_sqlcipher_pragmas(new_conn)
            assert new_conn.execute("SELECT count(*) FROM sqlite_master").fetchone()[0] > 0
        finally:
            new_conn.close()

    def test_recall_encrypted_vs_unencrypted(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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
