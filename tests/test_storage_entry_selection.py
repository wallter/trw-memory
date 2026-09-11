"""Full-entry admission before storage quotas; no model, namespace or replay bypass."""

import sqlite3
from datetime import datetime, timezone

import pytest

from trw_memory.models.memory import MemoryEntry
from trw_memory.retrieval.temporal_selection import TemporalSelection
from trw_memory.storage.sqlite_backend import SQLiteBackend
from trw_memory.storage.yaml_backend import YAMLBackend


@pytest.fixture(params=[SQLiteBackend, YAMLBackend])
def backend(request, tmp_path):
    value = request.param(tmp_path / "store")
    yield value
    value.close()


def entry(id, *, namespace="default", closed=False, importance=0.9):
    return MemoryEntry(
        id=id,
        namespace=namespace,
        content="needle",
        importance=importance,
        valid_from=datetime(2020, 1, 1, tzinfo=timezone.utc),
        invalid_from=datetime(2021, 1, 1, tzinfo=timezone.utc) if closed else None,
        invalidated_by="replacement" if closed else None,
        metadata={"keep": "yes" if id == "keep" else "no"},
    )


@pytest.mark.parametrize("method", ["search", "list_entries"])
def test_filter_precedes_cap_and_deferred_quota(backend, method):
    for i in range(20):
        backend.store(entry(f"reject{i}", closed=True))
    backend.store(entry("keep", closed=True, importance=0.1))
    kwargs = {
        "namespace": "default",
        "entry_filter": lambda e: e.metadata.get("keep") == "yes",
        "temporal_selection": TemporalSelection(include_superseded=True),
    }
    rows = (
        backend.search("needle", top_k=1, **kwargs) if method == "search" else backend.list_entries(limit=1, **kwargs)
    )
    assert [e.id for e in rows] == ["keep"]


@pytest.mark.parametrize("method", ["search", "list_entries"])
def test_filter_only_preserves_closed_and_scopes_before_callback(backend, method):
    backend.store(entry("same", namespace="foreign"))
    backend.store(entry("same", closed=True))
    seen = []

    def predicate(e):
        seen.append(e.namespace)
        return True

    kwargs = {"namespace": "default", "entry_filter": predicate}
    rows = (
        backend.search("needle", top_k=1, **kwargs) if method == "search" else backend.list_entries(limit=1, **kwargs)
    )
    assert len(rows) == 1 and rows[0].invalid_from is not None
    assert seen == ["default"]
    assert len(backend.list_entries(namespace="default")) == 1


@pytest.mark.parametrize("error", [ValueError("predicate"), UnicodeDecodeError("utf8", b"x", 0, 1, "predicate")])
def test_predicate_error_propagates(backend, error):
    backend.store(entry("keep"))
    calls = []

    def predicate(e):
        calls.append(e.id)
        raise error

    with pytest.raises(type(error)) as raised:
        backend.search("needle", entry_filter=predicate, temporal_selection=TemporalSelection())
    assert raised.value is error
    assert calls == ["keep"]


@pytest.mark.parametrize("temporal", [False, True])
def test_fts_late_match_beyond_500_and_namespace_collision(tmp_path, temporal):
    backend = SQLiteBackend(tmp_path / "fts.db")
    try:
        if not backend.fts_available:
            pytest.skip("FTS5 unavailable")
        for i in range(510):
            backend.store(entry(f"reject{i}"))
        backend.store(entry("keep", namespace="foreign"))
        backend.store(entry("keep", importance=0.1, closed=not temporal))
        seen = []

        def predicate(e):
            seen.append(e.namespace)
            return e.id == "keep"

        rows = backend.search_fts(
            "needle",
            top_k=1,
            namespace="default",
            entry_filter=predicate,
            temporal_selection=TemporalSelection() if temporal else None,
        )
        assert [e.id for e in rows] == ["keep"]
        assert set(seen) == {"default"}
    finally:
        backend.close()


def test_filter_survives_real_bytes_replay(tmp_path):
    backend = SQLiteBackend(tmp_path / "replay.db")
    try:
        backend.store(entry("bad"))
        backend.store(entry("keep", importance=0.1))
        with sqlite3.connect(backend._db_path) as connection:
            connection.execute("UPDATE memories SET detail=CAST(X'FF' AS TEXT) WHERE id='bad'")
        rows = backend.search("needle", top_k=1, entry_filter=lambda e: e.id == "keep")
        assert [e.id for e in rows] == ["keep"]
        assert backend.quarantine_count_utf8 == 1
    finally:
        backend.close()


def test_callback_replayed_only_after_real_decode_failure(tmp_path):
    from trw_memory.storage._resilient_fetch import FetchQuery
    from trw_memory.storage._shared import ENTRY_COLUMNS
    from trw_memory.storage._temporal_fetch import fetch_temporal_selection

    path = tmp_path / "repeat.db"
    backend = SQLiteBackend(path)
    try:
        backend.store(entry("first", importance=1.0))
        backend.store(entry("middle", importance=0.95))
        backend.store(entry("bad"))
        backend.store(entry("keep", importance=0.1))
        with sqlite3.connect(path) as connection:
            connection.execute("UPDATE memories SET detail=CAST(X'FF' AS TEXT) WHERE id='bad'")
            connection.commit()
            seen = []

            def predicate(e):
                seen.append(e.id)
                return e.id == "keep"

            rows, quarantined = fetch_temporal_selection(
                connection,
                db_path=path,
                dbapi=sqlite3,
                query=FetchQuery(select_columns_sql=", ".join(ENTRY_COLUMNS), order_by="importance DESC"),
                selection=None,
                limit=1,
                batch_size=1,
                entry_filter=predicate,
            )
            assert [e.id for e in rows] == ["keep"]
            assert seen == ["first", "first", "middle", "keep"]
            assert quarantined == 1
    finally:
        backend.close()


def test_callback_sees_only_schema_valid_rows(tmp_path):
    backend = SQLiteBackend(tmp_path / "schema.db")
    try:
        backend.store(entry("bad"))
        backend.store(entry("keep", importance=0.1))
        backend._conn.execute("UPDATE memories SET status='not-a-status' WHERE id='bad'")
        backend._conn.commit()
        seen = []

        def predicate(e):
            seen.append(e.id)
            return True

        rows = backend.search("needle", top_k=1, entry_filter=predicate)
        assert [e.id for e in rows] == ["keep"]
        assert seen == ["keep"]
    finally:
        backend.close()


@pytest.mark.parametrize("method", ["search", "list_entries"])
@pytest.mark.parametrize("limit", [1, 3, 10])
def test_yaml_filter_only_matches_legacy_order_including_ties(tmp_path, method, limit):
    backend = YAMLBackend(tmp_path / "ties")
    try:
        for name in ["z", "a", "m", "b", "keep"]:
            candidate = entry(name, closed=True)
            candidate.updated_at = datetime(2022, 1, 1, tzinfo=timezone.utc)
            backend.store(candidate)
        if method == "search":
            raw = backend.search("needle", top_k=limit)
            selected = backend.search("needle", top_k=limit, entry_filter=lambda _: True)
        else:
            raw = backend.list_entries(limit=limit)
            selected = backend.list_entries(limit=limit, entry_filter=lambda _: True)
        assert [e.id for e in selected] == [e.id for e in raw]
        assert all(e.invalid_from is not None for e in selected)
    finally:
        backend.close()


def test_inherited_no_fts_accepts_optional_selection(tmp_path):
    backend = YAMLBackend(tmp_path / "no-fts")
    try:

        def must_not_call(entry):
            raise AssertionError("No FTS candidates should reach selection")

        assert backend.search_fts("needle", temporal_selection=TemporalSelection(), entry_filter=must_not_call) == []
    finally:
        backend.close()
