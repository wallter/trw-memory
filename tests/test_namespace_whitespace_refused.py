"""PRD-CORE-308 S4 (B71-27 M6/M7): one spelling per namespace, refused before any effect.

``validate_namespace`` used to strip, and most callers discarded its return, so
``" team:x"`` passed the grant check as ``team:x`` while the row was stored
under ``" team:x"``. A padded name is now refused everywhere, which makes an
accepted name identical to its validated form at every call site.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from trw_memory.exceptions import ConfigError
from trw_memory.integrations._backend import create_backend_from_config
from trw_memory.models.config import MemoryConfig
from trw_memory.namespaces import NamespaceManager, validate_namespace
from trw_memory.storage.interface import StorageBackend
from trw_memory.tools.consolidate import memory_consolidate_impl
from trw_memory.tools.listing import memory_list_page_impl
from trw_memory.tools.recall import memory_recall_impl
from trw_memory.tools.status import memory_status_impl
from trw_memory.tools.store import memory_store_impl

PADDED = [" team:x", "team:x ", "team:x\n", "\tglobal"]


@pytest.fixture
def backend(tmp_path: Path) -> Iterator[StorageBackend]:
    with create_backend_from_config(MemoryConfig(storage_path=str(tmp_path)), "team:x") as opened:
        yield opened


@pytest.mark.parametrize("ns", PADDED)
def test_validate_namespace_refuses_padding(ns: str) -> None:
    with pytest.raises(ConfigError):
        validate_namespace(ns)


TOOLS: dict[str, Callable[[str, StorageBackend], dict[str, object]]] = {
    "store": lambda ns, b: memory_store_impl("a fact", ns, backend=b),
    "recall": lambda ns, b: memory_recall_impl("fact", ns, backend=b),
    "list": lambda ns, b: memory_list_page_impl(ns, 10, None, backend=b),
    "status": lambda ns, b: memory_status_impl(ns, backend=b),
    "consolidate": lambda ns, b: memory_consolidate_impl(ns, backend=b, dry_run=True),
}


@pytest.mark.parametrize("tool", sorted(TOOLS))
@pytest.mark.parametrize("ns", PADDED)
def test_every_tool_refuses_a_padded_namespace(tool: str, ns: str, backend: StorageBackend) -> None:
    result = TOOLS[tool](ns, backend)
    assert result.get("status") == "invalid", result
    assert backend.list_namespaces() == []  # nothing was stored under either spelling


@pytest.mark.parametrize(
    "call",
    [
        lambda m, ns: m.register(ns),
        lambda m, ns: m.delete(ns),
        lambda m, ns: m.count(ns),
        lambda m, ns: m.ensure_team_namespace(ns),
        lambda m, ns: m.mark_team_namespace_completed(ns),
        lambda m, ns: m.team_namespace_completed(ns),
        lambda m, ns: m.team_namespace_expired(ns),
    ],
)
def test_manager_api_refuses_a_padded_namespace(
    call: Callable[[NamespaceManager, str], object], backend: StorageBackend
) -> None:
    with pytest.raises(ConfigError):
        call(NamespaceManager(backend), " team:x")


@pytest.mark.parametrize("single_store", [False, True])
@pytest.mark.parametrize("ns", ["project:../../escape", "bogus", " team:x"])
def test_backend_factory_refuses_before_touching_disk(tmp_path: Path, ns: str, single_store: bool) -> None:
    root = tmp_path / "store"
    config = MemoryConfig(
        storage_path=str(root),
        memory_single_store_path=str(root / "one" / "memory.db") if single_store else "",
    )
    with pytest.raises(ConfigError):
        create_backend_from_config(config, ns)
    assert not root.exists()
