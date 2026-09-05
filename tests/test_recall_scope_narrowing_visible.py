"""W14 — a recall computed over fewer namespaces than asked for says so.

``authorize_namespaces`` already counted the namespaces it refused; the count
died at its return statement. ``memory_recall_impl`` then ranked over whatever
survived and returned the ordinary complete-looking payload, so "I searched the
three namespaces you named" and "I was cleared for one of them" were the same
answer. Same for a team namespace skipped mid-loop because it had expired.

The scope now carries ``denied``, and recall surfaces ``partial`` +
``namespaces_omitted`` counts (never the refused names -- a namespace label is
operator-chosen but still user data, PRD-CORE-245 NFR03).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.namespaces.manager import NamespaceManager
from trw_memory.security.namespace_scope import authorize_namespaces
from trw_memory.security.rbac import Permission
from trw_memory.storage.sqlite_backend import SQLiteBackend
from trw_memory.tools.recall import memory_recall_impl

pytestmark = pytest.mark.integration

_NAMESPACE = "project:scope-visible"


def _seeded(path: Path) -> SQLiteBackend:
    backend = SQLiteBackend(path)
    backend.store(MemoryEntry(id="M-1", content="retrieval notes", namespace=_NAMESPACE, tags=["x"]))
    return backend


def test_authorizer_reports_how_many_it_refused() -> None:
    """The count survives the return statement it used to die at."""
    scope = authorize_namespaces(
        MemoryConfig(),
        [_NAMESPACE, "not a namespace!", "also bad!!"],
        Permission.READ,
        "recall",
    )
    assert scope.namespaces == frozenset({_NAMESPACE})
    assert scope.denied == 2
    # Narrowing an authorized scope must not forget what was already refused.
    assert scope.without(_NAMESPACE).denied == 2


def test_recall_marks_a_refused_namespace_as_partial(tmp_path: Path) -> None:
    """An invalid additional namespace makes the answer explicitly incomplete."""
    backend = _seeded(tmp_path / "denied.db")
    try:
        result = memory_recall_impl(
            "retrieval",
            _NAMESPACE,
            backend=backend,
            include_namespaces=["not a namespace!"],
            include_org_memories=False,
            config=MemoryConfig(),
        )
    finally:
        backend.close()

    assert result["partial"] is True
    assert result["namespaces_omitted"] == {"denied": 1, "expired": 0}
    # The surviving namespace is still ranked -- this reports narrowing, it does
    # not turn a partial answer into no answer.
    assert result["total_matches"] >= 1


def test_recall_marks_an_expired_team_namespace_as_partial(tmp_path: Path) -> None:
    """A team namespace skipped mid-loop is counted, not silently dropped."""
    backend = _seeded(tmp_path / "expired.db")
    team = "team:sunset"
    try:
        manager = NamespaceManager(backend)
        manager.ensure_team_namespace(team)
        past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        with backend._lock:
            backend._conn.execute("UPDATE memory_namespaces SET expires_at = ? WHERE namespace_id = ?", (past, team))
            backend._conn.commit()
        assert manager.team_namespace_expired(team) is True

        result = memory_recall_impl(
            "retrieval",
            _NAMESPACE,
            backend=backend,
            include_namespaces=[team],
            include_org_memories=False,
            config=MemoryConfig(),
        )
    finally:
        backend.close()

    assert result["partial"] is True
    assert result["namespaces_omitted"] == {"denied": 0, "expired": 1}


def test_complete_recall_keeps_its_shape(tmp_path: Path) -> None:
    """Nothing omitted means no new keys -- the marker is evidence, not decoration."""
    backend = _seeded(tmp_path / "complete.db")
    try:
        result = memory_recall_impl(
            "retrieval",
            _NAMESPACE,
            backend=backend,
            include_org_memories=False,
            config=MemoryConfig(),
        )
    finally:
        backend.close()

    assert "partial" not in result
    assert "namespaces_omitted" not in result
