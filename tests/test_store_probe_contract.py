"""PRD-QUAL-147 FR01-FR05: ``probe_store``'s contract, over the public fixture set adopters build against.

The fixtures come from ``trw_memory.storage.probe_fixtures`` so trw-mcp's doctor, ``holds_rows`` and
migrate tests classify exactly these files. Each case here pins a state, a count or a bound.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from trw_memory.models.memory import MemoryEntry
from trw_memory.security._runtime_canary import _seeded_canary
from trw_memory.security.canary import PINNED_HASHES
from trw_memory.storage import _probe
from trw_memory.storage import probe_fixtures as fixtures
from trw_memory.storage._probe import StoreProbe, StoreState, probe_store
from trw_memory.storage._schema import SCHEMA_VERSION
from trw_memory.storage.sqlite_backend import SQLiteBackend

_ROW = "INSERT INTO memories (id, content, namespace, created_at, updated_at) VALUES (?, 'x', 'default', 't', 't')"


def trw_store(
    path: Path, rows: dict[str, int] | None = None, *, canaries: int = 2, decoys: int = 0, legacy: bool = False
) -> Path:
    """*rows* real rows per namespace (default 3 in ``default``), *canaries* seeded canaries, *decoys*
    rows flagged ``system_canary`` with no pinned identity (real rows), optionally a pre-migration schema."""
    counts = {"default": 3} if rows is None else rows
    real = [MemoryEntry(id=f"{ns}-{i}", content=f"row {i}", namespace=ns) for ns, n in counts.items() for i in range(n)]
    decoy = [MemoryEntry(id=f"decoy-{i}", content="a decoy", metadata={"system_canary": "true"}) for i in range(decoys)]
    backend = SQLiteBackend(path)
    try:
        for entry in [*real, *map(_seeded_canary, list(PINNED_HASHES)[:canaries]), *decoy]:
            backend.store(entry)
    finally:
        backend.close()
    return fixtures.legacy_schema(path) if legacy else path


def _zero_tables(path: Path) -> Path:
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA user_version = 7")
    conn.close()
    return path


def _sidecars(path: Path) -> list[str]:
    return [suffix for suffix in ("-wal", "-shm", "-journal") if Path(f"{path}{suffix}").exists()]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sql(path: Path, *statements: str) -> None:
    conn = sqlite3.connect(path)
    try:
        for statement in statements:
            conn.execute(statement)
        conn.commit()
    finally:
        conn.close()


@pytest.mark.parametrize(
    ("build", "expected"),
    [
        (lambda p: p, StoreProbe(StoreState.ABSENT)),
        (fixtures.empty_file, StoreProbe(StoreState.UNINITIALIZED)),
        (_zero_tables, StoreProbe(StoreState.UNINITIALIZED)),
        (fixtures.foreign_db, StoreProbe(StoreState.NOT_TRW, tables=("notes", "tags"), detail="no memories table")),
        (fixtures.not_a_database, StoreProbe(StoreState.UNREADABLE, detail="DatabaseError: file is not a database")),
    ],
    ids=["absent", "0-byte", "zero-tables", "foreign", "not-a-database"],
)
def test_probe_classifies_store_states(tmp_path: Path, build: Callable[[Path], Path], expected: StoreProbe) -> None:
    path = build(tmp_path / "memory.db")
    assert probe_store(path) == expected
    assert path.exists() is (expected.state is not StoreState.ABSENT)  # ABSENT creates no file


def test_probe_counts_ready_store_and_reads_legacy_columns_as_migration_defaults(tmp_path: Path) -> None:
    """FR01: a store missing ``verification_checked_at`` and ``protection_tier`` reads each as its
    ``MIGRATE_COLS`` default, so its seeded canaries still classify as canaries (NULL would not)."""
    ready = probe_store(trw_store(tmp_path / "ready.db"))
    # ``tables`` is sorted, so ``anchor_postings`` (PRD-CORE-332, schema 12) now precedes ``memories``.
    assert (ready.state, ready.real_rows, "memories" in ready.tables) == (StoreState.READY, 3, True)
    legacy = trw_store(tmp_path / "legacy.db", legacy=True)
    columns = {row[1] for row in sqlite3.connect(legacy).execute("PRAGMA table_info(memories)")}
    assert not {"verification_checked_at", "protection_tier"} & columns
    assert sqlite3.connect(legacy).execute("PRAGMA user_version").fetchone()[0] < SCHEMA_VERSION
    assert probe_store(legacy).real_rows == 3


_BOMB_CHILD = """
import json, resource, sys, time
from pathlib import Path
from trw_memory.storage import probe_fixtures, probe_store
path = probe_fixtures.generated_column_bomb(Path(sys.argv[1]))
before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
start = time.monotonic()
result = probe_store(path)
elapsed = time.monotonic() - start
growth = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - before
print(json.dumps({"state": result.state.value, "detail": result.detail, "elapsed": elapsed,
                  "growth_mb": growth / (1 << 20 if sys.platform == "darwin" else 1 << 10),
                  "size": path.stat().st_size}))
"""


def test_generated_column_bomb_is_refused_within_deadline_with_bounded_memory(tmp_path: Path) -> None:
    """FR02: the evidence's 8 KB file (6.0 GB through the old query) is refused, never evaluated.
    In a child process with a timeout, so a regression cannot take the test worker down."""
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
    child = subprocess.run(
        [sys.executable, "-c", _BOMB_CHILD, str(tmp_path / "memory.db")],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
        check=True,
    )
    report = json.loads(child.stdout.strip().splitlines()[-1])
    assert report["state"] == "refused" and "memories" in report["detail"]
    assert report["size"] <= 8192
    assert report["elapsed"] < 2 * _probe.PROBE_DEADLINE_S
    assert report["growth_mb"] < 64


@pytest.mark.parametrize(
    "hostile",
    [
        "CREATE VIEW memories AS WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT x AS id FROM c",
        "CREATE TRIGGER t AFTER INSERT ON notes BEGIN SELECT 1; END",
        "CREATE INDEX ix ON notes (lower(body))",
    ],
    ids=["recursive-view", "trigger", "expression-index"],
)
def test_probe_refuses_hostile_schema_within_deadline(tmp_path: Path, hostile: str) -> None:
    path = fixtures.foreign_db(tmp_path / "memory.db")
    _sql(path, hostile)
    start = time.monotonic()
    assert probe_store(path).state is StoreState.REFUSED
    assert time.monotonic() - start < 2 * _probe.PROBE_DEADLINE_S


_COLUMNS = "id TEXT, namespace TEXT, content TEXT"


@pytest.mark.parametrize(
    ("schema", "reason"),
    [
        ("CREATE TABLE memories (id TEXT, namespace TEXT COLLATE nocase, content TEXT)", "no such collation"),
        (f"CREATE TABLE memories ({_COLUMNS} DEFAULT '{'x' * 20_000}')", "too big"),  # SQL_LENGTH 16,384
        (f"CREATE TABLE memories ({_COLUMNS} DEFAULT ({'1 + ' * 300}1))", "maximum depth 100"),
        (
            f"CREATE TABLE memories ({_COLUMNS}, " + ", ".join(f"c{i} TEXT" for i in range(200)) + ")",
            "too many columns",
        ),
    ],
    ids=["unknown-collation", "sql-length", "expr-depth", "too-many-columns"],
)
def test_schema_past_the_parse_limits_is_unreadable_quickly(tmp_path: Path, schema: str, reason: str) -> None:
    """A stored schema the probe connection cannot parse or collate fails cleanly as UNREADABLE, never raises."""
    path = tmp_path / "memory.db"
    rename = "UPDATE sqlite_master SET sql = replace(sql, 'COLLATE nocase', 'COLLATE no_such') WHERE name = 'memories'"
    _sql(
        path,
        schema,
        "INSERT INTO memories (id, namespace) VALUES ('a', 'default')",
        "PRAGMA writable_schema = ON",
        rename,
    )
    start = time.monotonic()
    result = probe_store(path, "default")
    assert result.state is StoreState.UNREADABLE and reason in result.detail, result
    assert time.monotonic() - start < 2 * _probe.PROBE_DEADLINE_S


def test_probe_connection_is_hardened(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The probe's own connection: no schema functions, cell checks on, a 2 MiB cache, parse limits set."""
    seen: dict[str, object] = {}

    def record(conn: sqlite3.Connection, namespace: str | None) -> int:
        for pragma in ("trusted_schema", "cell_size_check", "cache_size"):
            seen[pragma] = conn.execute(f"PRAGMA {pragma}").fetchone()[0]
        seen["expr_depth"] = conn.getlimit(sqlite3.SQLITE_LIMIT_EXPR_DEPTH)  # type: ignore[attr-defined]
        return 0

    monkeypatch.setattr(_probe, "_count", record)
    probe_store(trw_store(tmp_path / "memory.db"))
    assert seen == {"trusted_schema": 0, "cell_size_check": 1, "cache_size": -2048, "expr_depth": 100}


def test_probe_past_its_deadline_is_unreadable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = trw_store(tmp_path / "memory.db", canaries=0)
    _sql(path, *(_ROW.replace("?", f"'bulk-{i}'") for i in range(20_000)))
    monkeypatch.setattr(_probe, "PROBE_DEADLINE_S", 0.0)
    result = probe_store(path, "default")
    assert result.state is StoreState.UNREADABLE and "interrupted" in result.detail


def test_probe_never_writes_and_reads_live_wal(tmp_path: Path) -> None:
    """FR03: at rest the file is byte-identical with no sidecar; a live writer's WAL-only rows count."""
    path = trw_store(tmp_path / "memory.db")
    Path(f"{path}.oplock").unlink(missing_ok=True)  # the builder's own lock file, not the probe's
    digest, header, listing = _sha(path), path.read_bytes()[60:64], sorted(os.listdir(tmp_path))  # 60:64 user_version
    assert probe_store(path).real_rows == 3
    assert (_sha(path), path.read_bytes()[60:64], sorted(os.listdir(tmp_path))) == (digest, header, listing)
    assert _sidecars(path) == [] and not Path(f"{path}.oplock").exists()

    writer = sqlite3.connect(path)
    try:
        writer.execute("PRAGMA wal_autocheckpoint = 0")
        writer.execute(_ROW, ("live-1",))
        writer.commit()
        assert Path(f"{path}-wal").exists()
        digest = _sha(path)
        assert probe_store(path).real_rows == 4
        assert _sha(path) == digest
    finally:
        writer.close()


def test_probe_reads_a_store_in_a_read_only_directory(tmp_path: Path) -> None:
    """FR03: the probe writes nothing, so a store in a directory it cannot write to is READY."""
    store_dir = tmp_path / "ro"
    store_dir.mkdir()
    path = trw_store(store_dir / "memory.db")
    Path(f"{path}.oplock").unlink()
    store_dir.chmod(0o555)
    try:
        assert probe_store(path).real_rows == 3
        assert sorted(os.listdir(store_dir)) == ["memory.db"]
    finally:
        store_dir.chmod(0o755)


def test_probe_opens_any_path_spelling(tmp_path: Path) -> None:
    """FR03: a space, ``?``, ``#`` and ``%`` in the path, and a symlink to the store, all probe READY."""
    source = trw_store(tmp_path / "memory.db")
    odd = tmp_path / "a b?c#d%20e" / "memory.db"
    odd.parent.mkdir()
    shutil.copyfile(source, odd)
    link = tmp_path / "link.db"
    link.symlink_to(odd)
    assert [probe_store(p).real_rows for p in (odd, link)] == [3, 3]
    fifo = tmp_path / "fifo.db"
    os.mkfifo(fifo)
    assert probe_store(fifo) == StoreProbe(StoreState.UNREADABLE, detail="not a regular file")  # never blocks


def test_probe_detects_writer_mid_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """FR04: a writer that commits into a new ``-wal`` during the immutable read makes the probe read
    again, and the count includes its rows; a store that changes on both reads is UNREADABLE."""
    path = trw_store(tmp_path / "memory.db")
    real_count = _probe._count
    writers: list[sqlite3.Connection] = []

    def count_then_write(conn: sqlite3.Connection, namespace: str | None) -> int:
        counted = real_count(conn, namespace)
        if not writers:
            writers.append(sqlite3.connect(path))
            writers[0].execute("PRAGMA wal_autocheckpoint = 0")
            writers[0].execute(_ROW, ("mid-read",))
            writers[0].commit()
        return counted

    monkeypatch.setattr(_probe, "_count", count_then_write)
    try:
        assert probe_store(path).real_rows == 4
    finally:
        writers[0].close()

    def count_and_always_write(conn: sqlite3.Connection, namespace: str | None) -> int:
        _sql(path, _ROW.replace("?", f"'w-{time.monotonic_ns()}'"))
        return 0

    monkeypatch.setattr(_probe, "_count", count_and_always_write)
    assert probe_store(path) == StoreProbe(StoreState.UNREADABLE, detail="store changed during read")


def test_real_row_predicate_is_total(tmp_path: Path) -> None:
    """FR05: canaries are pinned identities (classify_canary), never a flag; malformed metadata is a real row."""
    path = trw_store(tmp_path / "memory.db", {"default": 3, "other": 1}, canaries=2)
    assert (probe_store(path).real_rows, probe_store(path, "default").real_rows) == (4, 3)
    decoyed = trw_store(tmp_path / "decoy.db", {"default": 3, "other": 1}, canaries=2, decoys=1)
    assert probe_store(decoyed).real_rows == 5  # a system_canary flag with no pinned identity is a real row
    path = decoyed
    first_canary = next(iter(PINNED_HASHES))
    _sql(path, "UPDATE memories SET metadata = '{not json' WHERE id IN ('default-0', 'decoy-0')")
    assert probe_store(path).real_rows == 5
    _sql(path, f"UPDATE memories SET metadata = '{{not json' WHERE id = '{first_canary}'")
    assert probe_store(path).real_rows == 5  # classify_canary compares no system metadata key
    _sql(path, f"UPDATE memories SET importance = 0.9 WHERE id = '{first_canary}'")
    assert probe_store(path).real_rows == 6  # a pinned id carrying anything else is user data


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("vector_clock", '{"node": 1e999}'),
        ("metadata", "{not json"),
        ("metadata", "[" * 100_000),
        ("tags", "not a list"),
        ("importance", "not a number"),
        ("created_at", "not a date"),
        ("anchors", '[{"file": 1e999}]'),
        ("assertions", "{"),
        ("recurrence", "1e999"),
    ],
)
def test_undecodable_pinned_row_is_real_and_never_raises(tmp_path: Path, column: str, value: str) -> None:
    """FR05 total: a pinned canary id whose row no entry can hold counts as real data, whatever the field."""
    path = trw_store(tmp_path / "memory.db", canaries=2)
    first_canary = next(iter(PINNED_HASHES))
    conn = sqlite3.connect(path)
    conn.execute(f"UPDATE memories SET {column} = ? WHERE id = ?", (value, first_canary))
    conn.commit()
    conn.close()
    result = probe_store(path)
    assert result.state is StoreState.READY and result.real_rows in (3, 4)
    if column == "vector_clock":
        assert result.real_rows == 4  # the reported crash: OverflowError in row_to_entry


def test_probe_is_lazy() -> None:
    """NFR01: ``import trw_memory.storage`` does not load the probe."""
    code = "import sys, trw_memory.storage; print('trw_memory.storage._probe' in sys.modules)"
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True)
    assert out.stdout.strip() == "False"
