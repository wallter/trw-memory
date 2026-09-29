"""PRD-CORE-309 FR01/FR02 (B71-116, B71-115): one pinned-identity canary predicate on every import path.

A source store's system canary (a pinned id and content) is skipped and counted; a row with a pinned
canary's identity that also carries user data is rejected, never silently skipped; a row that only
carries the ``system_canary`` flag is an ordinary row. Namespace rename is not an import: it moves
every row, canaries included.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tests.conftest import make_entry
from tests.test_tools_checkout_import import _NS, _project_store, user_store  # noqa: F401 -- the fixture
from trw_memory.cli import main
from trw_memory.models.memory import MemoryEntry
from trw_memory.namespaces.curate import NamespaceStores, rename_namespace
from trw_memory.security.canary import _CANARY_FIXTURES, PINNED_HASHES
from trw_memory.storage.sqlite_backend import SQLiteBackend
from trw_memory.tools.checkout_import import memory_import_checkout_impl

from ._test_cli_support import _CLI, _real_import_target, _reopen_import_target

(_ID, _CONTENT), (_ID2, _CONTENT2) = _CANARY_FIXTURES[:2]
_FLAG = {"system_canary": "true"}
_AT = datetime(2026, 9, 1, tzinfo=timezone.utc)


def _canary(entry_id: str = _ID, content: str = _CONTENT, **fields: object) -> MemoryEntry:
    """A pinned canary exactly as ``_runtime_canary`` seeds it; *fields* change it."""
    fields = {"metadata": {**_FLAG, "provenance_content_hash": PINNED_HASHES[entry_id]}, **fields}
    return MemoryEntry(id=entry_id, content=content, namespace="default", **fields)


#: A pinned identity plus anything the seeder does not write is user data (fail-closed): each is rejected.
_USER_DATA: dict[str, dict[str, object]] = {
    "metadata": {"metadata": {**_FLAG, "provenance_content_hash": PINNED_HASHES["canary-003"], "note": "mine"}},
    "assertions": {"assertions": [{"type": "glob_exists", "target": "src/*.py"}]},
    "anchors": {"anchors": [{"file": "src/a.py", "symbol_name": "run", "symbol_type": "function"}]},
    "supersedes": {"invalid_from": datetime(2030, 1, 1, tzinfo=timezone.utc), "invalidated_by": "L-9"},
    "importance": {"importance": 0.9},
    "type": {"type": "incident"},
    "stripped flag, custom key": {"metadata": {"provenance_content_hash": PINNED_HASHES["canary-009"], "note": "x"}},
}
#: A recalled canary: only the bookkeeping counters the store bumps differ from the seeded row.
_RECALLED = {"access_count": 3, "recall_count": 2, "session_count": 1, "last_accessed_at": _AT}


def _flag_stripped(entry_id: str = _ID) -> MemoryEntry:
    """A seeded canary as 4.0.0's intake stored it: the reserved ``system_canary`` key stripped, nothing else."""
    return _canary(
        entry_id, dict(_CANARY_FIXTURES)[entry_id], metadata={"provenance_content_hash": PINNED_HASHES[entry_id]}
    )


def _near_matches() -> list[MemoryEntry]:
    return [
        _canary(cid, content, **fields)
        for (cid, content), fields in zip(_CANARY_FIXTURES[2:], _USER_DATA.values(), strict=False)
    ]


def _exported(entry: MemoryEntry) -> dict[str, object]:
    return json.loads(json.dumps(entry.to_dict(), default=str))


# --- CLI import -----------------------------------------------------------------------------------


@patch(f"{_CLI}._create_local_backend")
@patch(f"{_CLI}.MemoryConfig")
def test_near_match_canary_goes_to_rejection_sidecar(
    config_cls: MagicMock, backend_fn: MagicMock, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    near = _exported(_canary(_ID2, _CONTENT2, detail="my own note", tags=["mine"]))
    rows = [_exported(_canary()), near, {"id": "F-1", "content": "an ordinary flagged row", "metadata": _FLAG}]
    config, backend = _real_import_target(tmp_path)
    config_cls.return_value, backend_fn.return_value = config, backend
    (path := tmp_path / "rows.json").write_text(json.dumps(rows), encoding="utf-8")

    assert main(["import", str(path)]) == 1

    assert "Imported 1 entries, skipped 0, system canaries skipped: 1" in capsys.readouterr().out
    sidecar = [json.loads(line) for line in (tmp_path / "rows.json.rejected.jsonl").read_text().splitlines()]
    assert [(r["index"], r["id"], r["row"]) for r in sidecar] == [(1, _ID2, near)]
    with _reopen_import_target(tmp_path) as store:
        assert [e.content for e in store.list_entries(namespace="default", limit=10)] == ["an ordinary flagged row"]


@patch(f"{_CLI}._create_local_backend")
@patch(f"{_CLI}.MemoryConfig")
def test_a_malformed_canary_is_rejected_and_the_import_goes_on(
    config_cls: MagicMock, backend_fn: MagicMock, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A pinned identity MemoryEntry cannot hold is user data in the sidecar, not a crash (sol r2 P2)."""
    malformed = {**_exported(_canary()), "metadata": {}, "nudge_line": None}
    rows = [malformed, {"id": "F-1", "content": "an ordinary row"}]
    config, backend = _real_import_target(tmp_path)
    config_cls.return_value, backend_fn.return_value = config, backend
    (path := tmp_path / "rows.json").write_text(json.dumps(rows), encoding="utf-8")

    assert main(["import", str(path)]) == 1

    assert "Imported 1 entries" in capsys.readouterr().out
    sidecar = [json.loads(line) for line in (tmp_path / "rows.json.rejected.jsonl").read_text().splitlines()]
    assert [(r["index"], r["id"]) for r in sidecar] == [(0, _ID)]


# --- checkout import ------------------------------------------------------------------------------


def _with(path: Path, *entries: MemoryEntry) -> None:
    store = SQLiteBackend(path, dim=4)
    try:
        for entry in entries:
            store.store(entry)
    finally:
        store.close()


def test_checkout_import_skips_and_counts_a_pinned_canary_and_moves_a_flagged_row(
    tmp_path: Path,
    user_store: SQLiteBackend,  # noqa: F811
) -> None:
    work = tmp_path / "work.db"
    _project_store(work, ["L-1"])
    _with(work, _canary(), make_entry(entry_id="F-1", namespace="default", content="flagged", metadata=dict(_FLAG)))

    reply = memory_import_checkout_impl(_NS, str(work), ["L-1", "F-1"], backend=user_store)

    assert (reply["status"], reply["moved"], reply["canaries_skipped"]) == ("ok", 2, 1)
    assert reply["held"] == {"rows": 2, "vectors": 1, "edges": 0}
    assert user_store.get(_ID, namespace=_NS) is None, "the destination seeds its own canaries"
    assert SQLiteBackend(work, dim=4).count(namespace="default") == 0, "the copy is drained, canary included"


def test_checkout_import_refuses_a_near_match_canary_by_id(
    tmp_path: Path,
    user_store: SQLiteBackend,  # noqa: F811
) -> None:
    work = tmp_path / "work.db"
    _project_store(work, ["L-1"])
    _with(work, _canary(), _canary(_ID2, _CONTENT2, evidence=["commit abc"]))

    reply = memory_import_checkout_impl(_NS, str(work), ["L-1"], backend=user_store)

    assert (reply["status"], reply["conflicts"]) == ("conflict", [_ID2])
    assert user_store.count(namespace=_NS) == 0, "a refused import moves nothing"


@patch(f"{_CLI}._create_local_backend")
@patch(f"{_CLI}.MemoryConfig")
def test_cli_import_rejects_every_field_the_seeder_does_not_write_and_skips_a_recalled_canary(
    config_cls: MagicMock, backend_fn: MagicMock, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    near = [_exported(entry) for entry in _near_matches()]
    config, backend = _real_import_target(tmp_path)
    config_cls.return_value, backend_fn.return_value = config, backend
    (path := tmp_path / "rows.json").write_text(json.dumps([_exported(_canary(**_RECALLED)), *near]), encoding="utf-8")

    assert main(["import", str(path)]) == 1

    assert "Imported 0 entries, skipped 0, system canaries skipped: 1" in capsys.readouterr().out
    sidecar = [json.loads(line) for line in (tmp_path / "rows.json.rejected.jsonl").read_text().splitlines()]
    assert [r["row"] for r in sidecar] == near, "every near-match is kept for the operator, none dropped"


def test_checkout_import_refuses_every_field_the_seeder_does_not_write(
    tmp_path: Path,
    user_store: SQLiteBackend,  # noqa: F811
) -> None:
    work = tmp_path / "work.db"
    _project_store(work, ["L-1"])
    _with(work, *_near_matches())

    reply = memory_import_checkout_impl(_NS, str(work), ["L-1"], backend=user_store)

    assert (reply["status"], reply["conflicts"]) == ("conflict", sorted(e.id for e in _near_matches()))
    assert len(reply["conflicts"]) == len(_USER_DATA)


def test_checkout_import_skips_a_recalled_canary(
    tmp_path: Path,
    user_store: SQLiteBackend,  # noqa: F811
) -> None:
    work = tmp_path / "work.db"
    _project_store(work, ["L-1"])
    _with(work, _canary())
    store = SQLiteBackend(work, dim=4)
    try:
        store.update(_ID, namespace="default", **_RECALLED)
    finally:
        store.close()

    reply = memory_import_checkout_impl(_NS, str(work), ["L-1"], backend=user_store)

    assert (reply["status"], reply["moved"], reply["canaries_skipped"]) == ("ok", 1, 1)


# --- rename is not an import ----------------------------------------------------------------------


def test_rename_still_moves_a_canary_row(tmp_path: Path) -> None:
    store = SQLiteBackend(tmp_path / "memory.db")
    try:
        store.store(_canary().model_copy(update={"namespace": "project:old-11111111"}))
        result = rename_namespace(NamespaceStores.shared(store), "project:old-11111111", "project:new-22222222")
        assert (result.moved, store.get(_ID, namespace="project:new-22222222") is not None) == (1, True)
    finally:
        store.close()


@patch(f"{_CLI}._create_local_backend")
@patch(f"{_CLI}.MemoryConfig")
def test_cli_import_skips_a_flag_stripped_canary(
    config_cls: MagicMock, backend_fn: MagicMock, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config, backend = _real_import_target(tmp_path)
    config_cls.return_value, backend_fn.return_value = config, backend
    (path := tmp_path / "rows.json").write_text(json.dumps([_exported(_flag_stripped())]), encoding="utf-8")

    assert main(["import", str(path)]) == 0

    assert "Imported 0 entries, skipped 0, system canaries skipped: 1" in capsys.readouterr().out
    assert not (tmp_path / "rows.json.rejected.jsonl").exists()


def test_checkout_import_skips_a_flag_stripped_canary(
    tmp_path: Path,
    user_store: SQLiteBackend,  # noqa: F811
) -> None:
    work = tmp_path / "work.db"
    _project_store(work, ["L-1"])
    _with(work, _flag_stripped())

    reply = memory_import_checkout_impl(_NS, str(work), ["L-1"], backend=user_store)

    assert (reply["status"], reply["moved"], reply["canaries_skipped"]) == ("ok", 1, 1)


def _through_intake(tmp_path: Path, canary_id: str) -> MemoryEntry:
    """A seeded canary exported and imported through the real intake, as 4.0.0's CLI import did, read back."""
    from trw_memory.cli_storage import _rebuild_own_export
    from trw_memory.models.config import MemoryConfig
    from trw_memory.security._runtime_canary import _seeded_canary
    from trw_memory.security.write_gate import guarded_store

    config = MemoryConfig(storage_path=str(tmp_path / "intake"))
    store = SQLiteBackend(tmp_path / "intake" / "memory.db", dim=config.embedding_dim)
    try:
        assert guarded_store(store, _rebuild_own_export(_exported(_seeded_canary(canary_id)), "default"), config=config)
        entry = store.get(canary_id, namespace="default")
    finally:
        store.close()
    assert entry is not None and "trust_score" in entry.metadata and "system_canary" not in entry.metadata
    return entry


@patch(f"{_CLI}._create_local_backend")
@patch(f"{_CLI}.MemoryConfig")
def test_cli_import_skips_a_canary_that_went_through_an_import_and_rejects_one_a_user_edited(
    config_cls: MagicMock, backend_fn: MagicMock, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    imported = _through_intake(tmp_path, _ID)
    edited = _through_intake(tmp_path, _ID2)
    edited = edited.model_copy(update={"metadata": {**edited.metadata, "note": "mine"}})
    rows = [_exported(imported), _exported(edited)]
    config, backend = _real_import_target(tmp_path)
    config_cls.return_value, backend_fn.return_value = config, backend
    (path := tmp_path / "rows.json").write_text(json.dumps(rows), encoding="utf-8")

    assert main(["import", str(path)]) == 1

    assert "Imported 0 entries, skipped 0, system canaries skipped: 1" in capsys.readouterr().out
    sidecar = [json.loads(line) for line in (tmp_path / "rows.json.rejected.jsonl").read_text().splitlines()]
    assert [r["id"] for r in sidecar] == [_ID2]


def test_checkout_import_skips_a_canary_that_went_through_an_import(
    tmp_path: Path,
    user_store: SQLiteBackend,  # noqa: F811
) -> None:
    work = tmp_path / "work.db"
    _project_store(work, ["L-1"])
    _with(work, _through_intake(tmp_path, _ID))

    reply = memory_import_checkout_impl(_NS, str(work), ["L-1"], backend=user_store)

    assert (reply["status"], reply["moved"], reply["canaries_skipped"]) == ("ok", 1, 1)
