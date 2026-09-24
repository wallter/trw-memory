"""PRD-CORE-293 rule R1: a store that still carries the retired reward columns keeps working.

3.0.0 never drops or ALTERs the ``q_value`` / ``q_observations`` / ``helpful_count``
/ ``unhelpful_count`` columns of an existing store: it stops reading and writing
them, and a fresh store omits them. These tests open a store that still has the
columns (a store the released 2.0.0 wheel created: one row at the column
defaults, one with populated values) and run
create / amend / retire / close / reopen through the public backend.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from trw_memory.models.memory import MemoryEntry, MemoryStatus
from trw_memory.storage.sqlite_backend import SQLiteBackend

RETIRED = ("q_value", "q_observations", "helpful_count", "unhelpful_count")
NS = "project/legacy"


def _columns(path: Path) -> set[str]:
    with sqlite3.connect(path) as conn:
        return {row[1] for row in conn.execute("PRAGMA table_info(memories)")}


def _entry(entry_id: str, content: str) -> MemoryEntry:
    now = datetime.now(timezone.utc)
    return MemoryEntry(id=entry_id, content=content, namespace=NS, created_at=now, updated_at=now)


FIXTURE_2_0_0 = Path(__file__).parent / "fixtures" / "store_2_0_0" / "memory.db"
# Values the released 2.0.0 wrote (fixtures/store_2_0_0/make_store.py); 3.0.0 must leave them untouched.
LEGACY_VALUES = {"L-default": (0.5, 0, 0, 0), "L-populated": (0.9, 7, 4, 2)}


def _retired_values(path: Path) -> dict[str, tuple[object, ...]]:
    with sqlite3.connect(path) as conn:
        rows = conn.execute(f"SELECT id, {', '.join(RETIRED)} FROM memories").fetchall()
    return {str(row[0]): tuple(row[1:]) for row in rows}


@pytest.fixture
def legacy_store(tmp_path: Path) -> Path:
    """A copy of a store created by the released trw-memory 2.0.0 wheel, reward columns populated."""
    path = tmp_path / "memory.db"
    shutil.copyfile(FIXTURE_2_0_0, path)
    assert _retired_values(path) == LEGACY_VALUES
    return path


def test_fresh_store_omits_retired_columns(tmp_path: Path) -> None:
    path = tmp_path / "fresh.db"
    with SQLiteBackend(path) as backend:
        backend.store(_entry("F-1", "fresh"))
    assert not set(RETIRED) & _columns(path)


def test_legacy_store_lifecycle(legacy_store: Path) -> None:
    with SQLiteBackend(legacy_store) as backend:
        # read both legacy rows
        for entry_id in ("L-default", "L-populated"):
            entry = backend.get(entry_id, namespace=NS)
            assert entry is not None
            assert not set(RETIRED) & set(entry.model_dump())
        # create
        backend.store(_entry("L-new", "created against a legacy store"))
        # amend
        amended = backend.update("L-populated", namespace=NS, detail="amended detail", importance=0.8)
        assert amended is not None
        assert amended.detail == "amended detail"
        # retire
        retired = backend.update("L-default", namespace=NS, status=MemoryStatus.ARCHIVED.value)
        assert retired is not None
        active_ids = {e.id for e in backend.list_entries(namespace=NS, status=MemoryStatus.ACTIVE)}
        assert active_ids == {"L-populated", "L-new"}

    # close + reopen: the columns are still there (never dropped or altered) and
    # every row still deserializes.
    assert set(RETIRED) <= _columns(legacy_store)
    with SQLiteBackend(legacy_store) as backend:
        populated = backend.get("L-populated", namespace=NS)
        assert populated is not None
        assert populated.detail == "amended detail"
        assert populated.importance == 0.8
        default = backend.get("L-default", namespace=NS)
        assert default is not None
        assert default.status == MemoryStatus.ARCHIVED.value
        assert backend.get("L-new", namespace=NS) is not None
    after = _retired_values(legacy_store)
    # amend and retire leave the 2.0.0 values in place; a row written by 3.0.0
    # names none of the retired columns, so it takes the 2.0.0 column defaults
    assert after == {**LEGACY_VALUES, "L-new": (0.5, 0, 0, 0)}


@patch("trw_memory.cli._create_local_backend")
@patch("trw_memory.cli.MemoryConfig")
def test_2_0_0_export_imports_with_ids_and_every_kept_field(
    config_cls: MagicMock, backend_fn: MagicMock, tmp_path: Path
) -> None:
    """PRD-CORE-293-FR04: a 2.0.0 export (retired keys populated) rebuilds whole, ids included."""
    from trw_memory.cli import main
    from trw_memory.integrations._backend import make_entry

    from ._test_cli_support import _real_import_target, _reopen_import_target

    rows: list[dict[str, object]] = []
    for i, text in enumerate(("config loads before routes", "handlers never block the loop")):
        entry = make_entry(text, detail=f"detail {i}", tags=["core293"], importance=0.7)
        row = json.loads(json.dumps(entry.model_copy(update={"evidence": [f"commit-{i}"]}).to_dict(), default=str))
        row.update(q_value=0.9, q_observations=7, helpful_count=4, unhelpful_count=2)
        rows.append(row)
    config, backend = _real_import_target(tmp_path)
    config_cls.return_value, backend_fn.return_value = config, backend
    source = tmp_path / "export-2.0.0.json"
    source.write_text(json.dumps(rows), encoding="utf-8")

    assert main(["import", str(source)]) == 0

    with _reopen_import_target(tmp_path) as store:
        stored = {e.id: e.to_dict() for e in store.list_entries(namespace="default", limit=10)}
    assert set(stored) == {row["id"] for row in rows}
    # The write gate stamps metadata/tags and the store keeps its own sync bookkeeping.
    stamped = {"metadata", "tags", "sync_hash", "sync_seq", "last_synced_at"}
    for row in rows:
        after = json.loads(json.dumps(stored[str(row["id"])], default=str))
        assert not set(RETIRED) & set(after)
        kept = {k: v for k, v in row.items() if k not in stamped and k not in RETIRED}
        assert {k: v for k, v in after.items() if k not in stamped} == kept
