"""PRD-CORE-298 FR02 -- the whole-store read paths stay inside the token's grant.

``require_namespace_permission`` guards every tool that NAMES a namespace. The
tools below also read namespaces the caller never named -- a status breakdown,
the moved-checkout census, the importance decay pass -- so each must narrow to
the grant itself. With no access token (the in-process SDK) nothing changes.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from mcp.server.auth.middleware.auth_context import auth_context_var
from mcp.server.auth.middleware.bearer_auth import AuthenticatedUser
from mcp.server.auth.provider import AccessToken

from tests.conftest import make_entry
from trw_memory.integrations._backend import create_backend_from_config
from trw_memory.models.config import MemoryConfig
from trw_memory.namespaces.curate import store_census
from trw_memory.tools.maintain import memory_maintain_impl
from trw_memory.tools.status import memory_status_impl

_ALPHA = "project:alpha-11111111"
_BETA = "project:beta-22222222"
_OLD = "2020-01-01T00:00:00+00:00"


@pytest.fixture
def config(tmp_path: Path) -> MemoryConfig:
    config = MemoryConfig(storage_path=str(tmp_path), memory_single_store_path=str(tmp_path / "memory.db"))
    with create_backend_from_config(config, _ALPHA) as backend:
        for namespace in (_ALPHA, _BETA):
            backend.store(make_entry(entry_id=f"M-{namespace[8:12]}", namespace=namespace, importance=0.8))
        backend._conn.execute("UPDATE memories SET created_at = ?, last_accessed_at = ?", (_OLD, _OLD))  # type: ignore[attr-defined]
        backend._conn.commit()  # type: ignore[attr-defined]
    return config


@pytest.fixture
def alpha_token() -> Iterator[None]:
    reset = auth_context_var.set(AuthenticatedUser(AccessToken(token="t", client_id="c", scopes=[f"ns:{_ALPHA}"])))
    yield
    auth_context_var.reset(reset)


def test_status_without_a_namespace_counts_only_the_grant(config: MemoryConfig, alpha_token: None) -> None:
    with create_backend_from_config(config, _ALPHA) as backend:
        result = memory_status_impl(None, backend=backend, config=config)

    assert _BETA not in str(result)
    assert result["total_entries"] == 1
    assert result["namespaces"] == {_ALPHA: 1, "__active__": 1}


def test_status_naming_an_ungranted_namespace_is_refused(config: MemoryConfig, alpha_token: None) -> None:
    with create_backend_from_config(config, _ALPHA) as backend:
        result = memory_status_impl(_BETA, backend=backend, config=config)

    assert result["status"] == "forbidden"
    assert "total_entries" not in result


def test_the_census_holds_only_granted_namespaces(config: MemoryConfig, alpha_token: None) -> None:
    assert store_census(config) == {_ALPHA: 1}


def test_maintain_decays_only_granted_rows(config: MemoryConfig, alpha_token: None) -> None:
    with create_backend_from_config(config, _ALPHA) as backend:
        memory_maintain_impl(_ALPHA, backend=backend, config=config)
        importance = dict(
            backend._conn.execute("SELECT namespace, importance FROM memories").fetchall()  # type: ignore[attr-defined]
        )

    assert importance[_ALPHA] < 0.8
    assert importance[_BETA] == 0.8


def test_without_a_token_every_namespace_is_visible(config: MemoryConfig) -> None:
    assert store_census(config) == {_ALPHA: 1, _BETA: 1}
    with create_backend_from_config(config, _ALPHA) as backend:
        assert memory_status_impl(None, backend=backend, config=config)["total_entries"] == 2


class _ReadRecorder:
    """Wraps the quarantine store and records which namespace every read named."""

    def __init__(self, inner: object, reads: list[tuple[str, object]]) -> None:
        self._inner = inner
        self._reads = reads

    def __enter__(self) -> _ReadRecorder:
        return self

    def __exit__(self, *exc: object) -> None:
        self._inner.close()  # type: ignore[attr-defined]

    def __getattr__(self, name: str) -> object:
        attr = getattr(self._inner, name)
        if name not in {"list_entries", "get", "search", "count", "list_namespaces"}:
            return attr

        def record(*args: object, **kwargs: object) -> object:
            scope = args[0] if name == "list_namespaces" and args else kwargs.get("namespace", args[1:2] or None)
            self._reads.append((name, scope))
            return attr(*args, **kwargs)

        return record


@pytest.fixture
def quarantine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[MemoryConfig, list[tuple[str, object]]]]:
    """One alpha row, then three newer beta rows, behind a read recorder."""
    from trw_memory.security import _runtime_quarantine
    from trw_memory.security.runtime import store_quarantined_entry

    config = MemoryConfig(storage_path=str(tmp_path / "store"), quarantine_db_path=str(tmp_path / "q.db"))
    store_quarantined_entry(config, make_entry(entry_id="Q-alpha", namespace=_ALPHA))
    for index in range(3):
        store_quarantined_entry(config, make_entry(entry_id=f"Q-beta-{index}", namespace=_BETA))
    reads: list[tuple[str, object]] = []
    real = _runtime_quarantine.open_quarantine_backend
    monkeypatch.setattr(_runtime_quarantine, "open_quarantine_backend", lambda cfg: _ReadRecorder(real(cfg), reads))
    yield config, reads


def _foreign(reads: list[tuple[str, object]]) -> list[tuple[str, object]]:
    return [read for read in reads if read[1] not in (_ALPHA, [_ALPHA])]


def test_the_quarantine_list_never_reads_an_ungranted_namespace(
    quarantine: tuple[MemoryConfig, list[tuple[str, object]]], alpha_token: None
) -> None:
    from trw_memory.tools.review import memory_quarantine_list_impl

    config, reads = quarantine
    listed = memory_quarantine_list_impl(config=config)

    assert reads
    assert _foreign(reads) == []
    assert listed["namespaces"] == [_ALPHA]
    assert memory_quarantine_list_impl(_BETA, config=config)["status"] == "forbidden"


def test_newer_ungranted_rows_cannot_starve_the_granted_page(
    quarantine: tuple[MemoryConfig, list[tuple[str, object]]], alpha_token: None
) -> None:
    from trw_memory.tools.review import memory_quarantine_list_impl

    listed = memory_quarantine_list_impl(limit=1, config=quarantine[0])

    assert [row["id"] for row in listed["entries"]] == ["Q-alpha"]  # type: ignore[index, union-attr]


def test_status_counts_only_the_granted_quarantine(
    quarantine: tuple[MemoryConfig, list[tuple[str, object]]], alpha_token: None
) -> None:
    config, reads = quarantine
    with create_backend_from_config(config, _ALPHA) as backend:
        result = memory_status_impl(_ALPHA, backend=backend, config=config)

    assert result["security_posture"]["quarantine_count"] == 1  # type: ignore[index]
    assert reads
    assert _foreign(reads) == []


def test_team_wildcard_consolidation_never_names_an_ungranted_team(config: MemoryConfig, alpha_token: None) -> None:
    from trw_memory.tools.consolidate import memory_consolidate_impl

    with create_backend_from_config(config, _ALPHA) as backend:
        backend.store(make_entry(entry_id="T-1", namespace="team:theirs", importance=0.9))
        result = memory_consolidate_impl("team:*", backend=backend, config=config)

    assert "team:theirs" not in str(result)


def test_status_under_a_grant_is_blind_to_the_process_wide_maintenance_queue(
    config: MemoryConfig, alpha_token: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The maintenance queue spans every tenant the daemon serves, so a scoped token sees none of it."""
    from trw_memory.tools import status

    def status_with(queued: int) -> dict[str, object]:
        busy = {"queued": queued, "processed": queued * 7, "bounded": True}
        monkeypatch.setattr(status, "security_maintenance_status", lambda: busy)
        with create_backend_from_config(config, _ALPHA) as backend:
            return memory_status_impl(_ALPHA, backend=backend, config=config)

    busy, idle = status_with(5), status_with(0)

    assert busy == idle
    assert "maintenance" not in busy.get("security_posture", {})  # type: ignore[operator]
