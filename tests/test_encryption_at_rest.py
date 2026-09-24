"""SQLCipher at-rest encryption coverage for ``encryption_enabled=True``.

Split out of ``test_encryption_rotation.py`` when the whole key-rotation
surface (``security.encryption.rotate_key``, its backup/checkpoint/rekey
helpers, and ``KeyRotationError``) was retired per operator ruling: the
operator does not use key rotation. At-rest encryption itself
(``encryption_enabled`` / SQLCipher / the keychain-stored master key) stays —
it is independently reachable via config (``MemoryConfig.encryption_enabled``)
and the ``integrations/_backend.py`` write path, with no dependency on
rotation. These three tests keep that reachability and correctness pinned.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest

from trw_memory.integrations._backend import create_backend_from_config
from trw_memory.models.config import MemoryConfig
from trw_memory.security import derive_namespace_key, generate_master_key
from trw_memory.tools.recall import memory_recall_impl

from ._test_encryption_support import _load_real_sqlcipher_driver_or_skip, _make_entry
from ._timing import assert_budget


class TestEncryptionAtRest:
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

    def test_recall_encrypted_vs_unencrypted(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Correctness twin: recall on an encrypted backend still finds the needle."""
        _load_real_sqlcipher_driver_or_skip()
        monkeypatch.setattr("trw_memory.tools.recall.get_local_embedder", lambda **_: None)

        encrypted_key = generate_master_key()
        monkeypatch.setenv("MEMORY_MASTER_KEY", encrypted_key.hex())
        encrypted_config = MemoryConfig(
            storage_backend="sqlite",
            storage_path=str(tmp_path / "encrypted"),
            encryption_enabled=True,
            key_source="env",
            auto_generate_key=False,
        )
        with create_backend_from_config(encrypted_config, "default") as backend:
            backend.store(_make_entry(entry_id="recall-0", content="needle 0", detail="payload"))
            result = memory_recall_impl(
                query="needle",
                namespace="default",
                backend=backend,
                config=encrypted_config,
            )

        assert result["total_matches"] >= 1

        # Reopen a fresh backend (new SQLCipher connection, same on-disk file) to
        # prove the entry round-trips through real encryption at rest, not just
        # within the connection that wrote it.
        with create_backend_from_config(encrypted_config, "default") as reopened:
            reread = reopened.get("recall-0", namespace="default")

        assert reread is not None
        assert reread.id == "recall-0"
        assert reread.content == "needle 0"

    @pytest.mark.requires_local_timing
    def test_recall_encrypted_vs_unencrypted_budget(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _load_real_sqlcipher_driver_or_skip()
        monkeypatch.setattr("trw_memory.tools.recall.get_local_embedder", lambda **_: None)

        def _seed_and_measure(config: MemoryConfig) -> float:
            with create_backend_from_config(config, "default") as backend:
                for index in range(500):
                    backend.store(_make_entry(entry_id=f"recall-{index}", content=f"needle {index}", detail="payload"))
                durations: list[float] = []
                for _ in range(1000):
                    start = time.perf_counter()
                    memory_recall_impl(
                        query="needle",
                        namespace="default",
                        backend=backend,
                        config=config,
                    )
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

        assert_budget("encrypted_recall_p95_vs_plain", encrypted_time, plain_time * 1.10, "s")
