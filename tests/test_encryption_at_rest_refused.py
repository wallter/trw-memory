"""4.0 refuses ``encryption_enabled`` everywhere it can enter (7.0 freeze, C12).

The SQLCipher path never worked with a real driver (not on 3.1.0 either), so a
store that asked for encryption at rest must fail loudly with one named error
rather than silently run unencrypted or break at the first query.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from trw_memory.exceptions import (
    ENCRYPTION_AT_REST_UNSUPPORTED,
    ConfigError,
    EncryptionAtRestUnsupportedError,
)
from trw_memory.models.config import MemoryConfig

pytestmark = pytest.mark.integration


def _refused() -> Any:
    return pytest.raises(EncryptionAtRestUnsupportedError, match=f"^{re.escape(ENCRYPTION_AT_REST_UNSUPPORTED)}$")


def test_the_message_names_the_field_and_points_to_disk_encryption() -> None:
    assert issubclass(EncryptionAtRestUnsupportedError, ConfigError)
    assert "remove encryption_enabled" in ENCRYPTION_AT_REST_UNSUPPORTED
    assert "full-disk encryption" in ENCRYPTION_AT_REST_UNSUPPORTED


@pytest.mark.parametrize("field", ["encryption_enabled", "memory_encryption_enabled"])
def test_config_load_refuses_either_field_name(field: str) -> None:
    with _refused():
        MemoryConfig(**{field: True})  # type: ignore[arg-type]
    assert MemoryConfig(**{field: False}).encryption_enabled is False  # type: ignore[arg-type]


def test_config_load_refuses_the_env_var_and_the_project_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MEMORY_ENCRYPTION_ENABLED", "true")
    with _refused():
        MemoryConfig()

    monkeypatch.delenv("MEMORY_ENCRYPTION_ENABLED")
    (tmp_path / ".trw").mkdir()
    (tmp_path / ".trw" / "config.yaml").write_text("memory_encryption_enabled: true\n", encoding="utf-8")
    with _refused():
        MemoryConfig()


@pytest.mark.parametrize("single_store", [False, True])
def test_backend_creation_refuses_a_config_mutated_past_validation(tmp_path: Path, single_store: bool) -> None:
    """A validated model can still be mutated; the backend re-checks before touching disk."""
    from trw_memory.integrations._backend import create_backend_from_config, discover_namespace_backends

    storage = tmp_path / "storage"
    single = str(storage / "memory.db") if single_store else ""
    config = MemoryConfig(storage_path=str(storage), memory_single_store_path=single)
    config = config.model_copy(update={"encryption_enabled": True})

    with _refused():
        create_backend_from_config(config, "project:enc-aaaaaaaa")
    with _refused():
        with discover_namespace_backends(config) as stores:
            list(stores)
    assert not storage.exists(), "a refused config must leave nothing on disk"


def test_the_sdk_client_refuses(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from trw_memory.client import MemoryClient

    monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "s"))
    monkeypatch.setenv("MEMORY_ENCRYPTION_ENABLED", "true")
    with _refused():
        MemoryClient(namespace="default", mode="local")
    assert not (tmp_path / "s").exists()


def test_stdio_serving_refuses_before_it_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    import trw_memory.server as server_mod

    ran: list[bool] = []
    monkeypatch.setattr(server_mod.mcp, "run", lambda *_a, **_k: ran.append(True))
    monkeypatch.setenv("MEMORY_ENCRYPTION_ENABLED", "true")
    with _refused():
        server_mod.main([])
    assert ran == []

    # The preflight holds for a config that got past validation, too.
    monkeypatch.delenv("MEMORY_ENCRYPTION_ENABLED")
    mutated = MemoryConfig().model_copy(update={"encryption_enabled": True})
    with _refused():
        server_mod._preflight(mutated)


def test_the_daemon_refuses_before_it_claims_anything(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import trw_memory.server as server_mod
    from trw_memory.daemon import DaemonPaths

    monkeypatch.setenv("TRW_USER_DIR", str(tmp_path / "userhome"))
    monkeypatch.setenv("MEMORY_ENCRYPTION_ENABLED", "true")
    with _refused():
        server_mod._serve_http(None, None)

    paths = DaemonPaths.resolve(create=False)
    assert not paths.discovery.exists()
    assert not paths.token.exists()


def test_every_other_store_open_refuses_a_mutated_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """C12 sol review: the tier, quarantine, YAML-discovery and direct-open paths re-check too,
    before any disk write."""
    from trw_memory.integrations._backend import (
        NamespaceStoreLocation,
        discover_namespace_backends,
        open_namespace_store,
    )
    from trw_memory.lifecycle.tiers._manager_io import open_canonical_backend
    from trw_memory.security._runtime_quarantine import open_quarantine_backend

    storage = tmp_path / "storage"
    storage.mkdir()
    (storage / "project_a" / "entries").mkdir(parents=True)
    (storage / "memory.db").write_bytes(b"")
    quarantine = tmp_path / "q" / "quarantine.db"
    base = MemoryConfig(storage_path=str(storage), quarantine_db_path=str(quarantine))
    sqlite = base.model_copy(update={"encryption_enabled": True})
    yaml = base.model_copy(update={"encryption_enabled": True, "storage_backend": "yaml"})
    before = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*"))

    with _refused():
        open_canonical_backend(storage, storage / "entries", "project:a", sqlite)
    with _refused():
        open_quarantine_backend(sqlite)
    with _refused():
        open_namespace_store(sqlite, NamespaceStoreLocation(storage / "memory.db"))
    with _refused():
        with discover_namespace_backends(yaml) as stores:
            list(stores)
    assert sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*")) == before
