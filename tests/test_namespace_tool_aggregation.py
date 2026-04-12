"""Integration coverage for namespace-aggregating tool behavior."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from trw_memory.exceptions import EncryptionUnavailableError
from trw_memory.integrations._backend import create_backend_from_config, discover_namespace_backends, make_entry
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryStatus
from trw_memory.tools.recall import memory_recall_impl
from trw_memory.tools.status import memory_status_impl


def test_memory_recall_include_namespaces_reads_extra_namespace_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cross-namespace recall must reopen each requested namespace store."""
    cfg = MemoryConfig(storage_path=str(tmp_path))

    primary_entry = make_entry("shared primary context", namespace="project:aaa")
    extra_entry = make_entry("shared extra context", namespace="project:bbb")

    with create_backend_from_config(cfg, "project:aaa") as primary_backend:
        primary_backend.store(primary_entry)

    with create_backend_from_config(cfg, "project:bbb") as extra_backend:
        extra_backend.store(extra_entry)

    monkeypatch.setattr(
        "trw_memory.tools.recall.hybrid_search",
        lambda **kwargs: kwargs["entries"],
    )

    with create_backend_from_config(cfg, "project:aaa") as primary_backend:
        result = memory_recall_impl(
            query="shared",
            namespace="project:aaa",
            backend=primary_backend,
            namespace_backend_factory=lambda ns: create_backend_from_config(cfg, ns),
            include_namespaces=["project:bbb"],
            config=cfg,
        )

    memories = cast(list[dict[str, object]], result["memories"])
    namespaces = {str(memory["namespace"]) for memory in memories}
    assert namespaces == {"project:aaa", "project:bbb"}
    assert result["total_matches"] == 2


def test_memory_status_global_aggregates_all_namespace_stores(tmp_path: Path) -> None:
    """Global status should inspect every namespace store, not just the default one."""
    cfg = MemoryConfig(storage_path=str(tmp_path))

    active_primary = make_entry("primary active", namespace="project:aaa")
    archived_primary = make_entry("primary archived", namespace="project:aaa")
    archived_primary.status = MemoryStatus.ARCHIVED
    active_global = make_entry("global active", namespace="global")

    with create_backend_from_config(cfg, "project:aaa") as primary_backend:
        primary_backend.store(active_primary)
        primary_backend.store(archived_primary)

    with create_backend_from_config(cfg, "global") as global_backend:
        global_backend.store(active_global)

    with create_backend_from_config(cfg, "default") as default_backend:
        result = memory_status_impl(None, backend=default_backend, config=cfg)

    assert result["total_entries"] == 3
    assert result["namespaces"] == {
        "project:aaa": 2,
        "global": 1,
        "__active__": 2,
    }


def test_discover_namespace_backends_raises_when_encryption_requested(tmp_path: Path) -> None:
    """Namespace discovery must fail closed when encrypted sqlite is requested."""
    cfg = MemoryConfig(storage_path=str(tmp_path), encryption_enabled=True)

    with pytest.raises(
        EncryptionUnavailableError,
        match=r"SQLCipher is required when memory_encryption_enabled=True\. Install with: pip install trw-memory\[encryption\]",
    ):
        with discover_namespace_backends(cfg):
            pytest.fail("discover_namespace_backends should fail closed before yielding stores")
