"""Tests for ``list_org_shared_entries`` storage-layer pre-filtering.

Regression for memory-retrieval-graph-4: the function previously loaded up to
10,000 full ``MemoryEntry`` objects per sibling namespace and then discarded
the low-importance ones in Python. The fix pushes ``min_importance`` (and the
already-present ``status``) into ``backend.list_entries`` so only the rows the
caller can keep are hydrated.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry, MemoryStatus

from .conftest import make_entry


class _SpyBackend:
    """Records the kwargs passed to ``list_entries`` and applies them honestly."""

    def __init__(self, entries: list[MemoryEntry]) -> None:
        self._entries = entries
        self.list_entries_calls: list[dict[str, Any]] = []

    def list_entries(
        self,
        *,
        status: MemoryStatus | None = None,
        namespace: str | None = None,
        min_importance: float = 0.0,
        limit: int = 100,
    ) -> list[MemoryEntry]:
        self.list_entries_calls.append(
            {
                "status": status,
                "namespace": namespace,
                "min_importance": min_importance,
                "limit": limit,
            }
        )
        results = list(self._entries)
        if status is not None:
            results = [e for e in results if e.status == status]
        if namespace is not None:
            results = [e for e in results if e.namespace == namespace]
        if min_importance > 0.0:
            results = [e for e in results if e.importance >= min_importance]
        return results[:limit]


def _xv(entry: MemoryEntry) -> MemoryEntry:
    return entry.model_copy(update={"cross_validated": True})


def test_list_org_shared_pushes_min_importance_into_backend(monkeypatch: Any) -> None:
    """The storage-layer pre-filter (status + min_importance) must reach list_entries."""
    from trw_memory import graph

    sibling_ns = "project:sibling"
    sibling_entries = [
        _xv(make_entry(entry_id="M-001", content="high", namespace=sibling_ns, importance=0.95)),
        _xv(make_entry(entry_id="M-002", content="also high", namespace=sibling_ns, importance=0.85)),
        _xv(make_entry(entry_id="M-003", content="too low", namespace=sibling_ns, importance=0.40)),
    ]
    spy = _SpyBackend(sibling_entries)

    @contextmanager
    def _fake_discover(_config: MemoryConfig) -> Any:
        yield [([sibling_ns], spy)]

    monkeypatch.setattr(
        "trw_memory.integrations._backend.discover_namespace_backends",
        _fake_discover,
    )

    config = MemoryConfig()
    result = graph.list_org_shared_entries(config, "project:current", min_importance=0.8)

    # The min_importance threshold and ACTIVE status must be pushed into the
    # storage layer so the low-importance row is never hydrated/returned.
    assert len(spy.list_entries_calls) == 1
    call = spy.list_entries_calls[0]
    assert call["min_importance"] == 0.8
    assert call["status"] == MemoryStatus.ACTIVE

    # Semantics unchanged: only the two cross-validated high-importance entries
    # survive, sorted by importance descending.
    returned = [r.content for r in result]
    assert returned == ["high", "also high"]


def test_list_org_shared_filters_below_threshold(monkeypatch: Any) -> None:
    """An entry below min_importance must not be returned even if cross-validated."""
    from trw_memory import graph

    sibling_ns = "project:sibling"
    spy = _SpyBackend([_xv(make_entry(content="weak", namespace=sibling_ns, importance=0.5))])

    @contextmanager
    def _fake_discover(_config: MemoryConfig) -> Any:
        yield [([sibling_ns], spy)]

    monkeypatch.setattr(
        "trw_memory.integrations._backend.discover_namespace_backends",
        _fake_discover,
    )

    result = graph.list_org_shared_entries(MemoryConfig(), "project:current", min_importance=0.8)
    assert result == []


def test_list_org_shared_omits_sibling_without_read_permission(monkeypatch: Any) -> None:
    from trw_memory import graph

    secret_ns = "project:secret"
    spy = _SpyBackend([_xv(make_entry(content="secret", namespace=secret_ns, importance=0.95))])

    @contextmanager
    def _fake_discover(_config: MemoryConfig) -> Any:
        yield [([secret_ns], spy)]

    monkeypatch.setattr("trw_memory.integrations._backend.discover_namespace_backends", _fake_discover)
    config = MemoryConfig(
        rbac_enabled=True,
        namespace_roles={"project:current": "reader", secret_ns: "none"},
    )

    assert graph.list_org_shared_entries(config, "project:current") == []
    assert spy.list_entries_calls == []
