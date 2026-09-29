"""PRD-CORE-332 S1 (review core332-s1 r1): every accepted form of an ``anchors`` update re-posts the index.

``update(anchors=...)`` accepts Anchor models, plain dicts, and a JSON-serialized list (the
column's own storage form). The index must follow the value written to the column in every
form; a JSON string used to post nothing, silently emptying the row's postings.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.conftest import make_entry
from trw_memory.models.memory import Anchor
from trw_memory.storage.sqlite_backend import SQLiteBackend


def _postings(conn: sqlite3.Connection) -> set[tuple[str, str, str]]:
    rows = conn.execute("SELECT namespace, file, entry_id FROM anchor_postings").fetchall()
    return {(str(r[0]), str(r[1]), str(r[2])) for r in rows}


@pytest.fixture
def backend(tmp_path: Path) -> Iterator[SQLiteBackend]:
    store = SQLiteBackend(tmp_path / "anchors.db")
    yield store
    store.close()


@pytest.mark.parametrize(
    "value",
    [
        [Anchor(file="two.py", symbol_name="f")],
        [{"file": "two.py", "symbol_name": "f"}],
        json.dumps([{"file": "two.py", "symbol_name": "f"}]),
    ],
    ids=["model", "dict", "json-string"],
)
def test_every_anchors_update_form_reposts_the_index(backend: SQLiteBackend, value: object) -> None:
    entry = make_entry(entry_id="L-1")
    entry.anchors = [Anchor(file="one.py", symbol_name="g")]
    backend.store(entry)

    updated = backend.update("L-1", namespace="default", anchors=value)

    assert updated is not None
    assert [a.file for a in updated.anchors] == ["two.py"]
    assert _postings(backend._conn) == {("default", "two.py", "L-1")}


def test_an_empty_string_anchors_update_clears_the_row_and_its_postings(backend: SQLiteBackend) -> None:
    """CORE-332-F3 (review core332-s1 r2): ``anchors=""`` clears like ``[]`` instead of raising."""
    entry = make_entry(entry_id="L-1")
    entry.anchors = [Anchor(file="one.py", symbol_name="g")]
    backend.store(entry)

    updated = backend.update("L-1", namespace="default", anchors="")

    assert updated is not None
    assert updated.anchors == []
    assert _postings(backend._conn) == set()
    stored = backend._conn.execute("SELECT anchors FROM memories WHERE id = 'L-1'").fetchone()[0]
    assert json.loads(stored) == []
