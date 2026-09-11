"""Real SQLite streaming and bytes-mode replay for temporal selection."""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from trw_memory.models.memory import MemoryEntry
from trw_memory.retrieval.temporal_selection import TemporalSelection
from trw_memory.storage._resilient_fetch import FetchQuery
from trw_memory.storage._shared import ENTRY_COLUMNS
from trw_memory.storage._temporal_fetch import fetch_temporal_selection
from trw_memory.storage.sqlite_backend import SQLiteBackend


@pytest.mark.parametrize("corrupt", [False, True])
def test_stream_selects_beyond_prefix_and_recovers_bytes(tmp_path: Path, corrupt: bool) -> None:
    path = tmp_path / "memory.db"
    backend = SQLiteBackend(path)
    try:
        for i in range(40):
            backend.store(
                MemoryEntry(
                    id=f"old{i:03}",
                    content="old",
                    importance=0.9,
                    valid_from=datetime(2020, 1, 1, tzinfo=timezone.utc),
                    invalid_from=datetime(2021, 1, 1, tzinfo=timezone.utc),
                    invalidated_by="current",
                )
            )
        backend.store(MemoryEntry(id="current", content="current", importance=0.4))
        with sqlite3.connect(path) as connection:
            if corrupt:
                connection.execute("UPDATE memories SET content=CAST(X'FF' AS TEXT) WHERE id='old020'")
                connection.commit()  # Recovery connection must observe the committed corrupt row.
            query = FetchQuery(select_columns_sql=", ".join(ENTRY_COLUMNS), order_by="importance DESC, id")
            hits, delta = fetch_temporal_selection(
                connection,
                db_path=path,
                dbapi=sqlite3,
                query=query,
                selection=TemporalSelection(),
                limit=3,
                batch_size=7,
            )
            assert [e.id for e in hits] == ["current"]
            assert delta == int(corrupt)
            assert connection.execute("SELECT 1").fetchone() == (1,)  # Primary remains owned by caller.
    finally:
        backend.close()


def test_rejects_prefilter_limit_before_execution(tmp_path: Path) -> None:
    with sqlite3.connect(":memory:") as connection:
        with pytest.raises(ValueError, match="pre-eligibility"):
            fetch_temporal_selection(
                connection,
                db_path=tmp_path / "unused",
                dbapi=sqlite3,
                query=FetchQuery(select_columns_sql="id", limit=3),
                selection=TemporalSelection(),
                limit=3,
            )


def test_secondary_failure_is_not_successful_empty_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from trw_memory.storage import _temporal_fetch

    def corrupt_stream(*args, **kwargs):
        raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid")

    class Unavailable:
        @staticmethod
        def connect(path):
            raise sqlite3.OperationalError("secondary unavailable")

    monkeypatch.setattr(_temporal_fetch, "_select_stream", corrupt_stream)
    with sqlite3.connect(":memory:") as connection:
        with pytest.raises(sqlite3.OperationalError, match="secondary unavailable"):
            fetch_temporal_selection(
                connection,
                db_path=tmp_path / "unused",
                dbapi=Unavailable,
                query=FetchQuery(select_columns_sql="id"),
                selection=TemporalSelection(),
                limit=3,
            )


@pytest.mark.parametrize("fail_fetch", [False, True])
def test_fetch_is_batched_and_cursor_closes_on_early_exit_or_error(tmp_path: Path, fail_fetch: bool) -> None:
    path = tmp_path / "bounded.db"
    backend = SQLiteBackend(path)
    requests: list[int] = []
    closed: list[bool] = []

    class TrackedCursor:
        def __init__(self, cursor):
            self.cursor = cursor
            self.description = cursor.description

        def fetchall(self):
            raise AssertionError("Temporal selection must not materialize all rows")

        def fetchmany(self, size):
            requests.append(size)
            if fail_fetch:
                raise RuntimeError("injected fetch failure")
            return self.cursor.fetchmany(size)

        def close(self):
            closed.append(True)
            self.cursor.close()

    class TrackedConnection:
        def __init__(self, connection):
            self.connection = connection

        def execute(self, sql, params=()):
            return TrackedCursor(self.connection.execute(sql, params))

    try:
        for i in range(40):
            backend.store(MemoryEntry(id=f"eligible{i:03}", content="eligible"))
        with sqlite3.connect(path) as connection:

            def run():
                return fetch_temporal_selection(
                    TrackedConnection(connection),
                    db_path=path,
                    dbapi=sqlite3,
                    query=FetchQuery(select_columns_sql=", ".join(ENTRY_COLUMNS), order_by="id"),
                    selection=TemporalSelection(),
                    limit=1,
                    batch_size=7,
                )

            if fail_fetch:
                with pytest.raises(RuntimeError, match="injected fetch failure"):
                    run()
            else:
                hits, _ = run()
                assert [row.id for row in hits] == ["eligible000"]
            assert requests == [7]
            assert closed == [True]
            assert connection.execute("SELECT 1").fetchone() == (1,)
    finally:
        backend.close()


@pytest.mark.parametrize("include", [False, True])
def test_only_useful_candidates_are_constructed_and_bad_rows_do_not_fill_quota(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, include: bool
) -> None:
    from trw_memory.storage import _resilient_fetch

    path = tmp_path / "quota.db"
    backend = SQLiteBackend(path)
    original = _resilient_fetch.row_to_entry
    constructed: list[str] = []

    def observed(row, **kwargs):
        constructed.append(str(row[0]))
        return original(row, **kwargs)

    try:
        for i in range(40):
            backend.store(
                MemoryEntry(
                    id=f"old{i:03}",
                    content="old",
                    importance=0.9,
                    valid_from=datetime(2020, 1, 1, tzinfo=timezone.utc),
                    invalid_from=datetime(2021, 1, 1, tzinfo=timezone.utc),
                    invalidated_by="current",
                )
            )
        backend.store(MemoryEntry(id="current", content="current", importance=0.4))
        with sqlite3.connect(path) as connection:
            connection.execute("UPDATE memories SET status='invalid-schema-status' WHERE id='old000'")
            connection.commit()
            monkeypatch.setattr(_resilient_fetch, "row_to_entry", observed)
            hits, _ = fetch_temporal_selection(
                connection,
                db_path=path,
                dbapi=sqlite3,
                query=FetchQuery(select_columns_sql=", ".join(ENTRY_COLUMNS), order_by="importance DESC, id"),
                selection=TemporalSelection(include_superseded=include),
                limit=3,
                batch_size=7,
            )
        assert [e.id for e in hits] == (["current", "old001", "old002"] if include else ["current"])
        assert constructed == (["old000", "old001", "old002", "old003", "current"] if include else ["current"])
    finally:
        backend.close()


def test_selector_programming_error_is_not_quarantined_as_bad_data(tmp_path: Path) -> None:
    from trw_memory.storage._resilient_fetch import _decode_bytes_rows

    def broken_selector(row):
        raise KeyError("selector programming error")

    with pytest.raises(KeyError, match="selector programming error"):
        _decode_bytes_rows(
            [()], column_names=(), db_path=tmp_path / "unused", table="memories", retain_row=broken_selector
        )
