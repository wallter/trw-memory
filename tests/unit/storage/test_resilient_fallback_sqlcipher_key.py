"""memory-storage-2: bytes-mode UTF-8 fallback must key its secondary connection.

On an encrypted (SQLCipher) store the fallback's secondary connection was opened
without applying the namespace key, so SQLCipher treated the handle as a blank DB
and every SELECT returned zero rows — silently dropping all data instead of
quarantining only the truly-undecodable rows.

These tests use a fake DB-API so they run without a real SQLCipher driver; they
assert on the *behavior* (the key pragma is issued, in the right order, with the
right value) rather than on existence.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from trw_memory.storage._resilient_fetch import (
    FetchQuery,
    fetch_rows_via_bytes_fallback,
)

_VALID_KEY = "a" * 64


class _FakeCursor:
    description: tuple[tuple[object, ...], ...] | None = (("id",),)

    def fetchall(self) -> list[tuple[object, ...]]:
        return []


class _RecordingConnection:
    """Records every ``execute`` so we can assert the keying sequence."""

    def __init__(self) -> None:
        self.text_factory: object = str
        self.executed: list[str] = []
        self.closed = False

    def execute(self, sql: str, parameters: object = (), /) -> _FakeCursor:
        self.executed.append(sql)
        return _FakeCursor()

    def close(self) -> None:
        self.closed = True


class _FakeDBAPI:
    def __init__(self) -> None:
        self.conn = _RecordingConnection()

    def connect(self, database: str) -> _RecordingConnection:
        return self.conn


def _query(*, sqlcipher_key_hex: str | None) -> FetchQuery:
    return FetchQuery(
        select_columns_sql="id",
        where_sql="1",
        sqlcipher_key_hex=sqlcipher_key_hex,
    )


def test_fallback_applies_key_pragma_before_select_on_encrypted_store() -> None:
    dbapi = _FakeDBAPI()
    fetch_rows_via_bytes_fallback(
        db_path=Path("/tmp/encrypted.db"),
        dbapi=dbapi,  # type: ignore[arg-type]
        query=_query(sqlcipher_key_hex=_VALID_KEY),
    )
    executed = dbapi.conn.executed
    key_stmts = [s for s in executed if s.startswith("PRAGMA key")]
    assert key_stmts, "the secondary connection was never keyed"
    assert f"x'{_VALID_KEY}'" in key_stmts[0]
    # The key pragma must precede the SELECT, or the SELECT runs against an
    # unkeyed (blank) handle and returns nothing.
    key_idx = next(i for i, s in enumerate(executed) if s.startswith("PRAGMA key"))
    select_idx = next(i for i, s in enumerate(executed) if s.lstrip().upper().startswith("SELECT"))
    assert key_idx < select_idx


def test_fallback_skips_key_pragma_on_plaintext_store() -> None:
    dbapi = _FakeDBAPI()
    fetch_rows_via_bytes_fallback(
        db_path=Path("/tmp/plain.db"),
        dbapi=dbapi,  # type: ignore[arg-type]
        query=_query(sqlcipher_key_hex=None),
    )
    assert not any(s.startswith("PRAGMA key") for s in dbapi.conn.executed)


def test_fallback_rejects_malformed_key_loudly() -> None:
    dbapi = _FakeDBAPI()
    # A malformed key must raise (not be swallowed as a silent zero-row drop).
    with pytest.raises(ValueError, match="64-character lowercase hex"):
        fetch_rows_via_bytes_fallback(
            db_path=Path("/tmp/encrypted.db"),
            dbapi=dbapi,  # type: ignore[arg-type]
            query=_query(sqlcipher_key_hex="not-hex"),
        )
