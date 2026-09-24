"""PRD-CORE-294 FR03: correct or retire a learning by id through one implementation.

Drives trw-memory's ``memory_update`` surface (``memory_update_impl``) against a
real SQLite store; trw-mcp's ``trw_learn(learning_id=...)`` side of the same
contract lives in ``trw-mcp/tests/test_learn_update_by_id.py``.
"""

from __future__ import annotations

import tempfile
from collections.abc import Iterator
from typing import cast
from unittest.mock import patch

import pytest

from trw_memory.integrations._backend import create_backend_from_config
from trw_memory.lifecycle.correction import LearningPatch, parse_patch
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry, MemoryStatus
from trw_memory.storage.interface import StorageBackend
from trw_memory.tools.recall import memory_recall_impl
from trw_memory.tools.update import memory_update_impl

NAMESPACE = "project:default"


@pytest.fixture()
def store() -> Iterator[tuple[StorageBackend, MemoryConfig]]:
    with tempfile.TemporaryDirectory() as td:
        cfg = MemoryConfig(storage_backend="sqlite", storage_path=td)
        with create_backend_from_config(cfg, NAMESPACE) as backend:
            backend.store(
                MemoryEntry(
                    id="L-1",
                    content="retry the flaky network call with backoff",
                    detail="original detail",
                    tags=["net", "retry"],
                    importance=0.4,
                    namespace=NAMESPACE,
                )
            )
            yield backend, cfg


def _update(store: tuple[StorageBackend, MemoryConfig], entry_id: str, **patch_fields: object) -> dict[str, str]:
    """Drive the tool as the MCP wrapper does: parse the raw dict, then apply the typed patch."""
    backend, cfg = store
    parsed = parse_patch(patch_fields)
    if not isinstance(parsed, LearningPatch):
        return parsed
    return memory_update_impl(entry_id, parsed, NAMESPACE, backend=backend, config=cfg)


def _get(store: tuple[StorageBackend, MemoryConfig], entry_id: str) -> MemoryEntry:
    entry = store[0].get(entry_id, namespace=NAMESPACE)
    assert entry is not None
    return entry


def test_named_fields_change_and_the_rest_do_not(store: tuple[StorageBackend, MemoryConfig]) -> None:
    result = _update(store, "L-1", impact=0.9, detail="corrected detail")

    assert result == {"learning_id": "L-1", "status": "updated", "changes": "detail updated, impact→0.9"}
    entry = _get(store, "L-1")
    assert (entry.id, entry.importance, entry.detail) == ("L-1", 0.9, "corrected detail")
    assert entry.content == "retry the flaky network call with backoff"
    assert entry.tags == ["net", "retry"]


def test_tags_replace_but_tags_add_appends_without_duplicates(store: tuple[StorageBackend, MemoryConfig]) -> None:
    _update(store, "L-1", tags_add=["retry", "backoff", "backoff"])
    assert _get(store, "L-1").tags == ["net", "retry", "backoff"]

    _update(store, "L-1", tags=["only"])
    assert _get(store, "L-1").tags == ["only"]


def test_unknown_id_fails_loudly(store: tuple[StorageBackend, MemoryConfig]) -> None:
    result = _update(store, "L-missing", impact=0.5)

    assert result["status"] == "not_found"
    assert result["error_type"] == "learning_not_found"


@pytest.mark.parametrize(
    ("patch_fields", "field"),
    [({"impact": 1.5}, "impact"), ({"status": "archived"}, "status"), ({"q_value": 0.3}, "q_value")],
)
def test_invalid_patch_is_refused_and_nothing_is_written(
    store: tuple[StorageBackend, MemoryConfig], patch_fields: dict[str, object], field: str
) -> None:
    result = _update(store, "L-1", **patch_fields)

    assert result["status"] == "invalid"
    assert result["error"].startswith(f"Invalid {field}")
    assert _get(store, "L-1").importance == 0.4


def test_unsubstantiated_verified_promotion_is_refused(store: tuple[StorageBackend, MemoryConfig]) -> None:
    result = _update(store, "L-1", confidence="verified")

    assert result["status"] == "invalid"
    assert result["reason"] == "unsubstantiated_verified"
    assert _get(store, "L-1").confidence == "unverified"


def test_supersedes_closes_the_prior_validity_window(store: tuple[StorageBackend, MemoryConfig]) -> None:
    backend, _cfg = store
    backend.store(MemoryEntry(id="L-new", content="use the retry helper instead", namespace=NAMESPACE))

    result = _update(store, "L-new", supersedes="L-1")

    assert result["changes"] == "supersedes→L-1"
    prior = _get(store, "L-1")
    assert prior.invalid_from is not None
    assert prior.invalidated_by == "L-new"


def test_retired_learning_leaves_default_recall_but_stays_addressable(
    store: tuple[StorageBackend, MemoryConfig],
) -> None:
    backend, cfg = store

    def recalled_ids() -> list[str]:
        with patch("trw_memory.tools.recall.get_local_embedder", return_value=None):
            result = memory_recall_impl(
                "retry flaky network backoff", NAMESPACE, backend=backend, config=cfg, include_org_memories=False
            )
        return [str(row["id"]) for row in cast("list[dict[str, object]]", result["memories"])]

    assert recalled_ids() == ["L-1"]

    assert _update(store, "L-1", status="obsolete")["changes"] == "status→obsolete"

    assert recalled_ids() == []
    obsolete = backend.list_entries(namespace=NAMESPACE, status=MemoryStatus.OBSOLETE)
    assert [entry.id for entry in obsolete] == ["L-1"]


def test_the_mcp_surface_is_registered_over_the_same_function() -> None:
    from trw_memory.lifecycle import correction
    from trw_memory.server import REGISTERED_TOOL_NAMES
    from trw_memory.tools import update

    assert update.apply_correction is correction.apply_correction
    assert "memory_update" in REGISTERED_TOOL_NAMES


def test_a_correction_is_audited_with_what_changed(
    store: tuple[StorageBackend, MemoryConfig], monkeypatch: pytest.MonkeyPatch
) -> None:
    from trw_memory.tools import update

    events: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(update, "append_audit_event", lambda _cfg, op, **kw: events.append((op, kw["data"])))

    _update(store, "L-1", impact=0.7)
    _update(store, "L-1")  # nothing named -> no_changes -> no audit event

    assert events == [("update", {"entry_id": "L-1", "changes": "impact→0.7"})]


def test_tags_add_applies_to_the_stored_row_not_a_stale_read(store: tuple[StorageBackend, MemoryConfig]) -> None:
    from trw_memory.lifecycle.correction import Store, apply_correction

    stale = _get(store, "L-1")
    assert _update(store, "L-1", tags_add=["first"])["status"] == "updated"

    result = apply_correction(Store(*store), stale, LearningPatch(tags_add=["second"]))

    assert result["status"] == "updated"
    assert _get(store, "L-1").tags == ["net", "retry", "first", "second"]


def test_concurrent_tags_add_from_two_connections_loses_nothing(store: tuple[StorageBackend, MemoryConfig]) -> None:
    import threading

    from trw_memory.lifecycle.correction import Store, apply_correction

    _backend, cfg = store
    start = threading.Barrier(2)

    def worker(prefix: str) -> None:
        with create_backend_from_config(cfg, NAMESPACE) as own:
            start.wait()
            for n in range(15):
                # Each caller reads, then corrects: the window a lost update needs.
                entry = own.get("L-1", namespace=NAMESPACE)
                assert entry is not None
                apply_correction(Store(own, cfg), entry, LearningPatch(tags_add=[f"{prefix}{n}"]))

    threads = [threading.Thread(target=worker, args=(p,)) for p in ("a", "b")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    tags = set(_get(store, "L-1").tags)
    assert {f"{p}{n}" for p in "ab" for n in range(15)} <= tags


def test_a_failed_correction_leaves_a_prior_in_another_store_open(store: tuple[StorageBackend, MemoryConfig]) -> None:
    """The prior closes only after the new learning's write commits; a failure before that leaves it open."""
    from trw_memory.lifecycle.correction import Store, apply_correction

    backend, cfg = store
    backend.store(MemoryEntry(id="L-new", content="use the retry helper instead", namespace=NAMESPACE))
    new_entry = _get(store, "L-new")
    with tempfile.TemporaryDirectory() as other_dir:
        other_cfg = MemoryConfig(storage_backend="sqlite", storage_path=other_dir)
        with create_backend_from_config(other_cfg, "user") as other:
            other.store(MemoryEntry(id="L-prior", content="the old way", namespace="user"))
            prior = other.get("L-prior", namespace="user")

            with (
                patch.object(backend, "update", side_effect=RuntimeError("disk full")),
                pytest.raises(RuntimeError, match="disk full"),
            ):
                apply_correction(
                    Store(backend, cfg),
                    new_entry,
                    LearningPatch(detail="corrected", supersedes="L-prior"),
                    prior=(Store(other, other_cfg), prior),
                )

            reread = other.get("L-prior", namespace="user")
            assert reread is not None
            assert reread.invalid_from is None
