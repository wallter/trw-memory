"""PRD-CORE-253 FR06 — the quarantine LIST verb.

``list_quarantined_entries`` has been exported since SEC-001 with no tool
calling it, so the review queue was a hole rows fell into: ``memory_review``
can only resolve an id the maintainer already has. These tests drive the real
quarantine SQLite store through the registered impl.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import make_entry
from trw_memory.models.config import MemoryConfig
from trw_memory.security.runtime import store_quarantined_entry
from trw_memory.tools.review import memory_quarantine_list_impl

MINE = "team:mine"
THEIRS = "team:theirs"


def _config(tmp_path: Path, **overrides: object) -> MemoryConfig:
    return MemoryConfig(
        storage_path=str(tmp_path / "store"),
        quarantine_db_path=str(tmp_path / "quarantine.db"),
        **overrides,
    )


def _quarantine(config: MemoryConfig, namespace: str, entry_id: str) -> None:
    store_quarantined_entry(
        config,
        make_entry(entry_id=entry_id, namespace=namespace, content=f"suspicious {entry_id}"),
    )


def test_quarantine_list_is_namespace_scoped_and_rbac_checked(tmp_path: Path) -> None:
    """FR06: a maintainer of one namespace must not enumerate another's rows."""
    config = _config(
        tmp_path,
        rbac_enabled=True,
        default_role="none",
        namespace_roles={MINE: "admin"},
    )
    _quarantine(config, MINE, "M-mine-1")
    _quarantine(config, MINE, "M-mine-2")
    _quarantine(config, THEIRS, "M-theirs-1")

    listed = memory_quarantine_list_impl(config=config)

    assert listed["status"] == "ok"
    assert listed["namespaces"] == [MINE]
    assert {row["id"] for row in listed["entries"]} == {"M-mine-1", "M-mine-2"}
    assert listed["count"] == 2


def test_an_explicit_namespace_the_caller_lacks_admin_on_is_refused(tmp_path: Path) -> None:
    """Naming a forbidden namespace is an error, not a silent empty list.

    Filtering is right for an unscoped list; a caller who NAMED a namespace has
    to be told they may not read it, or the empty result reads as 'nothing is
    quarantined there'.
    """
    config = _config(tmp_path, rbac_enabled=True, default_role="none", namespace_roles={MINE: "admin"})
    _quarantine(config, THEIRS, "M-theirs-1")

    refused = memory_quarantine_list_impl(THEIRS, config=config)

    assert refused["status"] == "forbidden"
    assert THEIRS in str(refused["error"])
    assert "entries" not in refused, "a refusal must not look like an empty result"


def test_a_writer_without_admin_sees_nothing(tmp_path: Path) -> None:
    """The permission is ADMIN, the same one ``memory_review`` already requires."""
    config = _config(tmp_path, rbac_enabled=True, default_role="none", namespace_roles={MINE: "writer"})
    _quarantine(config, MINE, "M-mine-1")

    listed = memory_quarantine_list_impl(config=config)

    assert listed["count"] == 0
    assert listed["entries"] == []


def test_an_empty_quarantine_returns_an_empty_result_not_an_error(tmp_path: Path) -> None:
    config = _config(tmp_path)

    listed = memory_quarantine_list_impl(config=config)

    assert listed == {"entries": [], "count": 0, "namespaces": [], "status": "ok"}


def test_a_resolved_row_leaves_the_next_list(tmp_path: Path) -> None:
    """The discovery half and the resolution half compose: list, review, re-list."""
    from trw_memory.integrations._backend import create_backend_from_config
    from trw_memory.tools.review import memory_review_impl

    config = _config(tmp_path)
    _quarantine(config, MINE, "M-pending")
    with create_backend_from_config(config, MINE) as backend:
        backend.store(make_entry(entry_id="M-pending", namespace=MINE, content="suspicious M-pending"))

    before = memory_quarantine_list_impl(config=config)
    assert [row["id"] for row in before["entries"]] == ["M-pending"]

    memory_review_impl("M-pending", decision="approve", reviewer_id="maintainer", namespace=MINE, config=config)

    after = memory_quarantine_list_impl(config=config)
    assert [row["id"] for row in after["entries"]] == []


def test_listed_rows_carry_what_a_reviewer_decides_on(tmp_path: Path) -> None:
    """A queue entry has to be judgeable without a second lookup."""
    config = _config(tmp_path)
    _quarantine(config, MINE, "M-1")

    row = memory_quarantine_list_impl(config=config)["entries"][0]

    assert set(row) == {"id", "namespace", "content", "source_identity", "quarantined_at", "updated_at"}
    assert row["namespace"] == MINE
    assert row["content"] == "suspicious M-1"
    assert row["quarantined_at"], "the queue must show when the row arrived"


def test_the_list_verb_is_registered_on_the_published_surface() -> None:
    """FR06 takes the registered set from twelve to thirteen (plus FR05's two)."""
    pytest.importorskip("fastmcp")
    from trw_memory.server import REGISTERED_TOOL_NAMES

    assert "memory_quarantine_list" in REGISTERED_TOOL_NAMES
