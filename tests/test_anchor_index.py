"""PRD-CORE-332 S1: the ``anchor_postings`` index, kept exact by every writer of ``memories``.

FR01: one posting per distinct normalized anchor file per row, re-pointed inside the
row write's own lock and transaction by ``store``/``store_many``/``update``/``delete``/
``delete_many``/``delete_namespace``, and re-derived by the two recovery writers
(salvage restore and cold-tier rebuild) through ``rebuild_anchor_postings``.

The census at the bottom is FR01's soundness scope: every SQL statement in trw-memory's
source that inserts into or deletes from ``memories`` -- or UPDATEs it with a statement
naming ``anchors`` or a dynamic SET list -- must sit in a function that maintains the
index, or in the recorded exemption list. It cannot see SQL assembled from fragments
the patterns do not match, nor another process writing the same file.
"""

from __future__ import annotations

import ast
import re
import sqlite3
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

import pytest

from tests._test_cold_rebuild_support import _make_yaml
from tests.conftest import make_entry
from trw_memory.models.memory import Anchor, MemoryEntry, MemoryStatus
from trw_memory.storage.interface import StorageBackend
from trw_memory.storage.sqlite_backend import SQLiteBackend
from trw_memory.storage.yaml_backend import YAMLBackend

_SRC = Path(__file__).resolve().parents[1] / "src" / "trw_memory"


def _anchor(file: str, symbol: str = "sym") -> Anchor:
    return Anchor(file=file, symbol_name=symbol)


def _anchored(entry_id: str, *files: str, namespace: str = "default") -> MemoryEntry:
    entry = make_entry(entry_id=entry_id, namespace=namespace)
    entry.anchors = [_anchor(f, f"s{i}") for i, f in enumerate(files)]
    return entry


def _postings(conn: sqlite3.Connection) -> set[tuple[str, str, str]]:
    rows = conn.execute("SELECT namespace, file, entry_id FROM anchor_postings").fetchall()
    return {(str(r[0]), str(r[1]), str(r[2])) for r in rows}


def _tags(conn: sqlite3.Connection) -> list[str]:
    return [str(r[0]) for r in conn.execute("SELECT tag FROM memory_tags WHERE entry_id = 'L-1'").fetchall()]


@pytest.fixture
def backend(tmp_path: Path) -> Iterator[SQLiteBackend]:
    store = SQLiteBackend(tmp_path / "anchors.db")
    yield store
    store.close()


# ---------------------------------------------------------------------------
# normalize_anchor_file: the one key definition
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("a/b.py", "a/b.py"),
        ("./a//b.py", "a/b.py"),
        ("a/./b.py", "a/b.py"),
        ("./httpx/_client.py", "httpx/_client.py"),
        ("Pkg/Mod.py", "Pkg/Mod.py"),  # case preserved
        ("a/b/", "a/b"),
        ("", None),
        ("   ", None),
        (".", None),
        ("./", None),
        ("/abs/x.py", None),
        ("//net/x.py", None),
        ("../x.py", None),
        ("a/../b.py", None),
        ("a/..", None),
        (None, None),
        (42, None),
    ],
)
def test_normalize_anchor_file_forms(raw: object, expected: str | None) -> None:
    from trw_memory.storage._anchor_index import normalize_anchor_file

    assert normalize_anchor_file(raw) == expected


# ---------------------------------------------------------------------------
# Write-path maintenance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("writer", ["store", "store_many"])
def test_store_indexes_anchor_files(backend: SQLiteBackend, writer: str) -> None:
    """Each distinct normalized anchor file gets one posting; ``./a//b.py`` and ``a/b.py`` collapse."""
    entry = _anchored("L-1", "./httpx//_client.py", "httpx/_client.py", "httpx/_models.py")
    if writer == "store":
        backend.store(entry)
    else:
        assert backend.store_many([entry]) == 1

    assert _postings(backend._conn) == {
        ("default", "httpx/_client.py", "L-1"),
        ("default", "httpx/_models.py", "L-1"),
    }


def test_store_overwrite_replaces_anchor_postings(backend: SQLiteBackend) -> None:
    """INSERT OR REPLACE of the same (namespace, id) re-points; a row without anchors posts nothing."""
    backend.store(_anchored("L-1", "a.py", "b.py"))
    backend.store(_anchored("L-1", "c.py"))
    backend.store(make_entry(entry_id="L-2"))

    assert _postings(backend._conn) == {("default", "c.py", "L-1")}


def test_store_skips_unnormalizable_anchor_files(backend: SQLiteBackend) -> None:
    """``Anchor`` refuses these forms; an entry built around its validator still posts only the good one."""
    entry = make_entry(entry_id="L-1")
    entry.anchors = [Anchor.model_construct(file=f, symbol_name="s") for f in ("/abs/x.py", "../up.py", "ok.py")]
    backend.store(entry)

    assert _postings(backend._conn) == {("default", "ok.py", "L-1")}


def test_update_repoints_anchor_postings(backend: SQLiteBackend) -> None:
    backend.store(_anchored("L-1", "a.py", "b.py"))

    backend.update("L-1", namespace="default", anchors=[_anchor("./c.py")])
    assert _postings(backend._conn) == {("default", "c.py", "L-1")}

    # An update that does not name ``anchors`` leaves the postings alone, tags included.
    backend.update("L-1", namespace="default", content="new text", tags=["t1"])
    assert _postings(backend._conn) == {("default", "c.py", "L-1")}
    assert _tags(backend._conn) == ["t1"]

    backend.update("L-1", namespace="default", anchors=[])
    assert _postings(backend._conn) == set()


def test_update_anchors_leaves_tag_postings_alone(backend: SQLiteBackend) -> None:
    entry = _anchored("L-1", "a.py")
    entry.tags = ["keep"]
    backend.store(entry)

    backend.update("L-1", namespace="default", anchors=[_anchor("b.py")])

    assert _tags(backend._conn) == ["keep"]
    assert _postings(backend._conn) == {("default", "b.py", "L-1")}


@pytest.mark.parametrize("deleter", ["delete", "delete_many", "delete_by_namespace"])
def test_delete_purges_anchor_postings(backend: SQLiteBackend, deleter: str) -> None:
    """Deleting a row drops its postings; the same id and file in another namespace survive."""
    backend.store(_anchored("L-1", "a.py", "b.py", namespace="project:doomed"))
    backend.store(_anchored("L-1", "a.py", namespace="project:kept"))

    if deleter == "delete":
        assert backend.delete("L-1", namespace="project:doomed") is True
    elif deleter == "delete_many":
        assert backend.delete_many(["L-1", "L-missing"], namespace="project:doomed") == 1
    else:
        assert backend.delete_by_namespace("project:doomed") == 1

    assert _postings(backend._conn) == {("project:kept", "a.py", "L-1")}


# ---------------------------------------------------------------------------
# Recovery writers re-derive the index from the column
# ---------------------------------------------------------------------------


def test_salvage_restore_reposts_anchor_postings(tmp_path: Path) -> None:
    """``_restore_rows`` inserts salvaged rows with raw SQL; the postings must follow them."""
    from trw_memory.storage._recovery import _restore_rows

    source = SQLiteBackend(tmp_path / "old.db")
    source.store(_anchored("L-1", "./pkg/a.py", "pkg/b.py"))
    source.close()
    old = sqlite3.connect(tmp_path / "old.db")
    old.row_factory = sqlite3.Row
    rows = old.execute("SELECT * FROM memories").fetchall()
    old.close()

    fresh = SQLiteBackend(tmp_path / "new.db")
    try:
        _restore_rows(fresh._conn, rows, db_path=tmp_path / "new.db")
        assert _postings(fresh._conn) == {("default", "pkg/a.py", "L-1"), ("default", "pkg/b.py", "L-1")}
    finally:
        fresh.close()


def test_cold_rebuild_reposts_anchor_postings(tmp_path: Path) -> None:
    from trw_memory.storage._cold_rebuild import rebuild_from_cold

    _make_yaml(tmp_path, "L-COLD", anchors=[{"file": "./pkg/cold.py", "symbol_name": "f"}])
    fresh = SQLiteBackend(tmp_path / "rebuilt.db")
    try:
        assert rebuild_from_cold(tmp_path, fresh._conn) == 1
        assert _postings(fresh._conn) == {("default", "pkg/cold.py", "L-COLD")}
    finally:
        fresh.close()


def _tag_postings(conn: sqlite3.Connection) -> set[tuple[str, str, str]]:
    rows = conn.execute("SELECT namespace, tag, entry_id FROM memory_tags").fetchall()
    return {(str(r[0]), str(r[1]), str(r[2])) for r in rows}


def test_salvage_restore_reposts_tag_postings(tmp_path: Path) -> None:
    """``_restore_rows`` inserts salvaged rows with raw SQL; ``memory_tags`` must follow them (PRD-CORE-332 F2)."""
    from trw_memory.storage._recovery import _restore_rows

    entry = _anchored("L-1", "a.py")
    entry.tags = ["alpha", "beta"]
    source = SQLiteBackend(tmp_path / "old.db")
    source.store(entry)
    source.close()
    old = sqlite3.connect(tmp_path / "old.db")
    old.row_factory = sqlite3.Row
    rows = old.execute("SELECT * FROM memories").fetchall()
    old.close()

    fresh = SQLiteBackend(tmp_path / "new.db")
    try:
        _restore_rows(fresh._conn, rows, db_path=tmp_path / "new.db")
        assert _tag_postings(fresh._conn) == {
            ("default", "alpha", "L-1"),
            ("default", "beta", "L-1"),
        }
    finally:
        fresh.close()


def test_cold_rebuild_reposts_tag_postings(tmp_path: Path) -> None:
    """Cold-tier rebuild raw-INSERTs rows too; ``memory_tags`` must be re-derived (PRD-CORE-332 F2)."""
    from trw_memory.storage._cold_rebuild import rebuild_from_cold

    _make_yaml(tmp_path, "L-COLD")  # default fixture tags: ["alpha", "beta"]
    fresh = SQLiteBackend(tmp_path / "rebuilt.db")
    try:
        assert rebuild_from_cold(tmp_path, fresh._conn) == 1
        assert _tag_postings(fresh._conn) == {
            ("default", "alpha", "L-COLD"),
            ("default", "beta", "L-COLD"),
        }
    finally:
        fresh.close()


def test_salvage_restore_and_cold_rebuild_tags_are_recall_findable(tmp_path: Path) -> None:
    """Tag-filtered recall (``derive_tag_neighbours``, which reads ``memory_tags`` directly rather than the
    ``memories.tags`` JSON column) finds a restored row across both recovery paths, and a listing over the
    ``memory_tags`` postings table itself finds them too (PRD-CORE-332 F2)."""
    from trw_memory.models.config import MemoryConfig
    from trw_memory.retrieval.tag_derivation import derive_tag_neighbours
    from trw_memory.storage._cold_rebuild import rebuild_from_cold
    from trw_memory.storage._recovery import _restore_rows

    shared_tags = ["alpha", "beta"]  # >= config.graph_tag_min_shared_tags (default 2)

    salvaged_entry = _anchored("L-SALVAGE", "a.py")
    salvaged_entry.tags = shared_tags
    source = SQLiteBackend(tmp_path / "old.db")
    source.store(salvaged_entry)
    source.close()
    old = sqlite3.connect(tmp_path / "old.db")
    old.row_factory = sqlite3.Row
    rows = old.execute("SELECT * FROM memories").fetchall()
    old.close()

    _make_yaml(tmp_path, "L-COLD", tags=shared_tags)

    fresh = SQLiteBackend(tmp_path / "new.db")
    try:
        _restore_rows(fresh._conn, rows, db_path=tmp_path / "new.db")
        assert rebuild_from_cold(tmp_path, fresh._conn) == 1

        neighbours = derive_tag_neighbours(fresh._conn, "L-SALVAGE", namespace="default", config=MemoryConfig())
        listed = fresh.list_entries(namespace="default", tags=shared_tags, status=None)
    finally:
        fresh.close()

    assert [n.entry_id for n in neighbours] == ["L-COLD"]
    assert {e.id for e in listed} == {"L-SALVAGE", "L-COLD"}


# ---------------------------------------------------------------------------
# Census: every writer of ``memories`` maintains the index (FR01 soundness scope)
# ---------------------------------------------------------------------------

_MAINTAINERS = frozenset({"_replace_postings", "purge_postings_for", "rebuild_anchor_postings"})
_WRITE_SQL = re.compile(
    r"INSERT\s+(?:OR\s+(?:REPLACE|IGNORE)\s+)?INTO\s+memories(?!\w)"
    r"|DELETE\s+FROM\s+memories(?!\w)"
    r"|UPDATE\s+memories\s+SET\s+(?:\{|[^;]*\banchors\b)",
    re.IGNORECASE,
)
#: ``path::function`` -> why it may write ``memories`` without maintaining the index itself.
_EXEMPT = {
    "storage/_query_ops.py::delete_by_namespace": (
        "its only caller, _namespace_purge.delete_namespace, purges the snapshot ids' postings "
        "in the same transaction (asserted below)"
    ),
    "storage/probe_fixtures.py::generated_column_bomb": "builds a hostile probe fixture file, never a live store",
}
#: Writers the census MUST find, so a pattern that stops matching cannot pass vacuously.
_KNOWN_WRITERS = {
    "storage/_crud_ops.py::store",
    "storage/_crud_ops.py::store_many",
    "storage/_crud_ops.py::update",
    "storage/_crud_ops.py::delete",
    "storage/_crud_ops.py::delete_many",
    "storage/_query_ops.py::delete_by_namespace",
    "storage/_recovery.py::_restore_rows",
    "storage/_cold_rebuild.py::rebuild_from_cold",
}


def _text(node: ast.AST) -> str | None:
    """The SQL-ish text of a string literal; an f-string's holes read as ``{}``."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(v.value if isinstance(v, ast.Constant) else "{}" for v in node.values)  # type: ignore[misc]
    return None


def _writes_memories(node: ast.AST) -> bool:
    text = _text(node)
    if text is not None:
        return bool(_WRITE_SQL.search(text))
    # ``_delete_keyed(backend, "memories", ...)`` assembles its DELETE from a table argument.
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_delete_keyed"
        and len(node.args) > 1
        and _text(node.args[1]) == "memories"
    )


def _calls(func: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(func):
        if isinstance(node, ast.Call):
            target = node.func
            names.add(target.id if isinstance(target, ast.Name) else getattr(target, "attr", ""))
    return names


def _memories_writers() -> dict[str, set[str]]:
    """``path::function`` -> the helpers it calls, for every function holding a ``memories`` write."""
    found: dict[str, set[str]] = {}
    for path in sorted(_SRC.rglob("*.py")):
        rel = path.relative_to(_SRC).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        functions = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        inside: set[int] = set()
        for func in functions:
            if any(_writes_memories(n) for n in ast.walk(func)):
                inner = [f for f in ast.walk(func) if f is not func and isinstance(f, type(func))]
                if not any(any(_writes_memories(n) for n in ast.walk(f)) for f in inner):
                    found[f"{rel}::{func.name}"] = _calls(func)
            inside.update(id(n) for n in ast.walk(func))
        for node in ast.walk(tree):
            if id(node) not in inside and _writes_memories(node):
                found[f"{rel}::<module>"] = set()
    return found


def test_every_memories_writer_maintains_anchor_index() -> None:
    writers = _memories_writers()

    missing = _KNOWN_WRITERS - writers.keys()
    assert not missing, f"census patterns no longer find known writers: {sorted(missing)}"
    unmaintained = {
        key: sorted(calls & _MAINTAINERS)
        for key, calls in writers.items()
        if not calls & _MAINTAINERS and key not in _EXEMPT
    }
    assert not unmaintained, f"memories writers that do not maintain anchor_postings: {sorted(unmaintained)}"
    stale = set(_EXEMPT) - writers.keys()
    assert not stale, f"exemptions naming no current writer: {sorted(stale)}"


def test_namespace_purge_maintains_what_delete_by_namespace_exempts() -> None:
    """The ``delete_by_namespace`` exemption holds only while ``delete_namespace`` purges postings."""
    tree = ast.parse((_SRC / "storage" / "_namespace_purge.py").read_text(encoding="utf-8"))
    purge = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "delete_namespace")

    assert {"_delete_rows", "purge_postings_for"} <= _calls(purge)


# ---------------------------------------------------------------------------
# FR03 (S2): anchored_to, the read over the index
# ---------------------------------------------------------------------------


def _yaml_backend(tmp_path: Path) -> YAMLBackend:
    return YAMLBackend(tmp_path / "yaml-entries")


@pytest.fixture(params=["sqlite", "yaml"])
def any_backend(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[StorageBackend]:
    """The SQLite override and the interface default must answer the same way."""
    store: StorageBackend = (
        SQLiteBackend(tmp_path / "read.db") if request.param == "sqlite" else _yaml_backend(tmp_path)
    )
    yield store
    store.close()


def _put(store: StorageBackend, entry_id: str, *files: str, namespace: str = "default", **fields: object) -> None:
    entry = _anchored(entry_id, *files, namespace=namespace)
    store.store(entry.model_copy(update=fields))


def test_anchored_to_filters_namespace_and_file(any_backend: StorageBackend) -> None:
    _put(any_backend, "L-high", "httpx/_client.py", importance=0.9)
    _put(any_backend, "L-dotted", "./httpx/_client.py", "httpx/other.py", importance=0.5)
    _put(any_backend, "L-other-file", "httpx/other.py", importance=1.0)
    _put(any_backend, "L-other-ns", "httpx/_client.py", namespace="project:elsewhere", importance=1.0)
    any_backend.store(make_entry(entry_id="L-unanchored", content="httpx/_client.py", importance=1.0))

    found = any_backend.anchored_to("default", "httpx//_client.py", status=None, limit=10)

    assert [entry.id for entry in found] == ["L-high", "L-dotted"]
    assert [entry.id for entry in any_backend.anchored_to("default", "httpx/_client.py", status=None, limit=1)] == [
        "L-high"
    ]


def test_anchored_to_orders_by_importance_then_recency_then_id(any_backend: StorageBackend) -> None:
    older, newer = datetime(2026, 1, 1, tzinfo=timezone.utc), datetime(2026, 2, 1, tzinfo=timezone.utc)
    _put(any_backend, "L-b", "a.py", importance=0.5, updated_at=newer)
    _put(any_backend, "L-a", "a.py", importance=0.5, updated_at=newer)
    _put(any_backend, "L-old", "a.py", importance=0.5, updated_at=older)
    _put(any_backend, "L-top", "a.py", importance=0.8, updated_at=older)

    found = any_backend.anchored_to("default", "a.py", status=None, limit=10)

    assert [entry.id for entry in found] == ["L-top", "L-a", "L-b", "L-old"]


def test_anchored_to_status_filter(any_backend: StorageBackend) -> None:
    _put(any_backend, "L-live", "a.py")
    _put(any_backend, "L-gone", "a.py", status=MemoryStatus.OBSOLETE, importance=0.9)

    active = any_backend.anchored_to("default", "a.py", status=MemoryStatus.ACTIVE, limit=10)
    every = any_backend.anchored_to("default", "a.py", status=None, limit=10)

    assert [entry.id for entry in active] == ["L-live"]
    assert [entry.id for entry in every] == ["L-gone", "L-live"]


@pytest.mark.parametrize("file", ["", ".", "/abs/a.py", "../a.py", "a/../a.py"])
def test_anchored_to_rejects_unnormalizable_file(any_backend: StorageBackend, file: str) -> None:
    _put(any_backend, "L-1", "a.py")

    assert any_backend.anchored_to("default", file, status=None, limit=10) == []


def test_anchored_to_non_positive_limit_is_empty(any_backend: StorageBackend) -> None:
    _put(any_backend, "L-1", "a.py")

    assert any_backend.anchored_to("default", "a.py", status=None, limit=0) == []


def test_anchored_to_uses_index(backend: SQLiteBackend) -> None:
    backend.store_many([_anchored(f"L-{i:03d}", f"pkg/m{i % 50}.py") for i in range(500)])
    backend.anchored_to("default", "pkg/m1.py", status=MemoryStatus.ACTIVE, limit=5)  # opens the connection
    statements: list[str] = []
    backend._conn.set_trace_callback(statements.append)
    try:
        found = backend.anchored_to("default", "pkg/m1.py", status=MemoryStatus.ACTIVE, limit=5)
    finally:
        backend._conn.set_trace_callback(None)
    (read,) = [sql for sql in statements if "anchor_postings" in sql]

    plan = [str(row[-1]) for row in backend._conn.execute(f"EXPLAIN QUERY PLAN {read}")]

    assert len(found) == 5
    # The postings drive the loop; each hit is one probe of the memories key, never a walk of the namespace.
    assert plan[0].startswith("SEARCH anchor_postings USING PRIMARY KEY (namespace=? AND file=?)"), plan
    assert [line for line in plan if "memories" in line] == [
        "SEARCH memories USING INDEX sqlite_autoindex_memories_1 (namespace=? AND id=?)"
    ], plan
