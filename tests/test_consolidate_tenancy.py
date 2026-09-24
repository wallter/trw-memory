"""PRD-CORE-298 FR06: promotion writes into the caller's granted project namespace.

Promotion used to write every promoted team memory into a hard-coded
``project:default``. Over the daemon each token is granted its own
``project:<slug>`` namespace, so a promotion must land there, and a grant
that does not name exactly one project namespace is refused before any row
moves.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from trw_memory.exceptions import AuthorizationError
from trw_memory.integrations._backend import create_backend_from_config
from trw_memory.models.config import MemoryConfig
from trw_memory.namespaces.manager import NamespaceManager
from trw_memory.storage.interface import StorageBackend
from trw_memory.tools import consolidate
from trw_memory.tools.consolidate import memory_consolidate_impl

from ._test_team_memory_support import _InMemoryBackend, _make_entry


def _grant(monkeypatch: pytest.MonkeyPatch, namespaces: set[str] | None) -> None:
    grant = None if namespaces is None else frozenset(namespaces)
    monkeypatch.setattr(consolidate, "transport_grant", lambda: grant, raising=False)
    monkeypatch.setattr("trw_memory.security.rbac.transport_grant", lambda: grant)


def _promoted_namespaces(backend: _InMemoryBackend) -> list[str]:
    return sorted(e.namespace for e in backend.list_entries(limit=100) if e.id.startswith("promoted-"))


@pytest.mark.unit
def test_promotion_lands_in_the_granted_project_namespace(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _InMemoryBackend()
    backend.store(_make_entry("e1", importance=0.9))
    _grant(monkeypatch, {"team:sprint-37", "project:acme-1a2b", "user:local"})

    result = memory_consolidate_impl("team:sprint-37", backend=backend, config=MemoryConfig())

    assert result["promoted_count"] == 1
    assert _promoted_namespaces(backend) == ["project:acme-1a2b"]


@pytest.mark.unit
@pytest.mark.parametrize(
    "grant",
    [
        {"team:sprint-37", "user:local"},
        {"team:sprint-37", "project:a-1111", "project:b-2222"},
        set(),
    ],
    ids=["no-project", "two-projects", "empty-grant"],
)
def test_an_ambiguous_or_missing_target_is_refused_before_any_row_moves(
    monkeypatch: pytest.MonkeyPatch, grant: set[str]
) -> None:
    backend = _InMemoryBackend()
    backend.store(_make_entry("e1", importance=0.9))
    _grant(monkeypatch, grant)

    with pytest.raises(AuthorizationError, match="exactly one project namespace"):
        memory_consolidate_impl("team:sprint-37", backend=backend, config=MemoryConfig())

    assert _promoted_namespaces(backend) == []


@pytest.mark.unit
def test_the_in_process_sdk_without_a_grant_keeps_project_default(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _InMemoryBackend()
    backend.store(_make_entry("e1", importance=0.9))
    _grant(monkeypatch, None)

    memory_consolidate_impl("team:sprint-37", backend=backend, config=MemoryConfig())

    assert _promoted_namespaces(backend) == ["project:default"]


def _yaml_stores(tmp_path: Path) -> tuple[MemoryConfig, Callable[[str], StorageBackend]]:
    cfg = MemoryConfig(storage_backend="yaml", storage_path=str(tmp_path))
    return cfg, lambda ns: create_backend_from_config(cfg, ns)


def _rows(open_store: Callable[[str], StorageBackend], namespace: str) -> list[tuple[str, str]]:
    with open_store(namespace) as store:
        return sorted((e.id, e.namespace) for e in store.list_entries(limit=100))


def _wildcard(cfg: MemoryConfig, open_store: Callable[[str], StorageBackend]) -> dict[str, object]:
    with open_store("default") as default:
        return memory_consolidate_impl("team:*", backend=default, config=cfg, namespace_backend_factory=open_store)


@pytest.mark.integration
def test_the_wildcard_path_promotes_into_the_granted_project_store(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg, open_store = _yaml_stores(tmp_path)
    with open_store("team:sprint-37") as team:
        team.store(_make_entry("e1", importance=0.9, namespace="team:sprint-37"))
    _grant(monkeypatch, {"team:sprint-37", "project:acme-1a2b"})

    result = _wildcard(cfg, open_store)

    assert result["promoted_count"] == 1
    assert _rows(open_store, "project:acme-1a2b") == [("promoted-e1", "project:acme-1a2b")]
    assert _rows(open_store, "project:default") == []


@pytest.mark.integration
@pytest.mark.parametrize(
    "grant",
    [{"team:sprint-37", "user:local"}, {"team:sprint-37", "project:a-1111", "project:b-2222"}],
    ids=["no-project", "two-projects"],
)
def test_the_wildcard_path_refuses_an_ambiguous_grant_and_moves_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, grant: set[str]
) -> None:
    cfg, open_store = _yaml_stores(tmp_path)
    with open_store("team:sprint-37") as team:
        team.store(_make_entry("e1", importance=0.9, namespace="team:sprint-37"))
    before = {ns: _rows(open_store, ns) for ns in ("team:sprint-37", "project:default", *sorted(grant))}
    _grant(monkeypatch, grant)

    with pytest.raises(AuthorizationError, match="exactly one project namespace"):
        _wildcard(cfg, open_store)

    assert {ns: _rows(open_store, ns) for ns in before} == before
    with open_store("team:sprint-37") as team:
        assert not NamespaceManager(team).team_namespace_completed("team:sprint-37")


@pytest.mark.integration
def test_the_wildcard_path_refuses_an_ambiguous_grant_even_with_no_team_to_promote(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg, open_store = _yaml_stores(tmp_path)
    _grant(monkeypatch, {"project:a-1111", "project:b-2222"})

    with pytest.raises(AuthorizationError, match="exactly one project namespace"):
        _wildcard(cfg, open_store)
