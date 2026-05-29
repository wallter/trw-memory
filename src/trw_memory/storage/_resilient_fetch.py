"""Resilient row materialisation — bad-UTF-8 row quarantine.

Belongs to the ``sqlite_backend.py`` facade. Re-exported there for
back-compat — ``SQLiteBackend._fetch_rows_resilient`` /
``_fetch_rows_via_bytes_fallback`` retain identical signatures via
1-line delegators that pass instance state.

The sqlite3 C extension raises ``sqlite3.OperationalError("Could not
decode to UTF-8 column ...")`` during row fetch. On SQLite >= 3.51
(pysqlite3) the same error surfaces during ``execute()`` because TEXT
columns are decoded eagerly. Either way the fallback opens a secondary
connection with ``text_factory=bytes`` to read raw rows, then decodes
each column individually, quarantining rows that can't decode at all.

The fallback re-executes the *same* query — including the caller's
WHERE clause, ORDER BY, and LIMIT — so the degraded path preserves
query semantics (status/namespace filters and row caps) rather than
returning every row in the table.

Returns ``(results, quarantine_delta)`` so callers can update their
own ``quarantine_count_utf8`` counter.

Extracted as PRD-DIST-245 Phase 1 batch 84.
"""

from __future__ import annotations

import contextlib
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import structlog

from trw_memory.models.memory import MemoryEntry
from trw_memory.storage._row_mapper import row_to_entry

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Typing protocols — the resilient path runs against either stdlib ``sqlite3``
# or the optional SQLCipher driver, so we describe the minimal surface we use
# rather than binding to a concrete class.
# ---------------------------------------------------------------------------


class _CursorLike(Protocol):
    """Minimal cursor surface used by the resilient fetch path."""

    description: tuple[tuple[object, ...], ...] | None

    def fetchall(self) -> list[tuple[object, ...]]: ...


class _ConnectionLike(Protocol):
    """Minimal connection surface used by the bytes-mode fallback."""

    text_factory: object

    def execute(self, sql: str, parameters: object = ..., /) -> _CursorLike: ...

    def close(self) -> None: ...


class _DBAPILike(Protocol):
    """Minimal DB-API module surface (``sqlite3`` or SQLCipher driver)."""

    def connect(self, database: str) -> _ConnectionLike: ...


@dataclass(frozen=True, slots=True)
class FetchQuery:
    """Reconstructable query the fallback re-executes in bytes mode.

    Captures the caller's filter clause, ORDER BY, and LIMIT so the
    degraded path honours the same row selection as the primary query.
    ``where_sql`` is the bare predicate (``"1"`` when unfiltered); the
    SELECT/FROM and trailing clauses are assembled by :meth:`build`.
    """

    select_columns_sql: str
    table: str = "memories"
    where_sql: str = "1"
    params: tuple[object, ...] = ()
    order_by: str = "updated_at DESC"
    limit: int | None = None

    def build(self) -> tuple[str, tuple[object, ...]]:
        """Return the ``(sql, params)`` pair to re-execute in bytes mode."""
        sql = (
            f"SELECT {self.select_columns_sql} FROM {self.table} "  # noqa: S608
            f"WHERE {self.where_sql} ORDER BY {self.order_by}"
        )
        params = self.params
        if self.limit is not None:
            sql += " LIMIT ?"
            params = (*params, self.limit)
        return sql, params


def is_utf8_decode_error(exc: BaseException) -> bool:
    """Return True if *exc* is a UTF-8 decode failure from the SQLite driver.

    On SQLite >= 3.51 the driver raises ``OperationalError("Could not
    decode to UTF-8 column ...")`` either during ``execute()`` or
    ``fetchall()``; older drivers raise ``UnicodeDecodeError``.
    """
    if isinstance(exc, UnicodeDecodeError):
        return True
    msg = str(exc)
    return "UTF-8" in msg or "decode" in msg.lower()


def _quarantine_log(
    *,
    db_path: Path,
    table: str,
    row_id: str | None,
    column: str,
    row_index: int | None,
    error: str | None = None,
) -> None:
    """Emit a structured quarantine warning with an ``outcome`` field."""
    logger.warning(
        "db_bad_utf8_row_quarantined",
        action="memory_row_utf8_quarantined",
        outcome="quarantined",
        row_id=row_id,
        column=column,
        db_path=str(db_path),
        table=table,
        row_index=row_index,
        error=error,
    )


def fetch_rows_resilient(
    cursor: _CursorLike,
    *,
    db_path: Path,
    dbapi: _DBAPILike,
    query: FetchQuery,
) -> tuple[list[MemoryEntry], int]:
    """Iterate cursor row-by-row, quarantining bad-UTF-8 rows.

    Returns ``(results, quarantine_delta)`` — caller adds the delta to
    its own ``quarantine_count_utf8`` counter. If ``fetchall()`` raises a
    UTF-8 decode error (older drivers), control routes to
    :func:`fetch_rows_via_bytes_fallback`, which re-executes *query* so
    the caller's filters and limit are preserved.
    """
    try:
        raw_rows = cursor.fetchall()
    except (sqlite3.OperationalError, UnicodeDecodeError) as exc:
        if not is_utf8_decode_error(exc):
            raise
        return fetch_rows_via_bytes_fallback(
            db_path=db_path,
            dbapi=dbapi,
            query=query,
        )

    results: list[MemoryEntry] = []
    quarantine_delta = 0
    for idx, raw_row in enumerate(raw_rows):
        try:
            entry = row_to_entry(tuple(raw_row))
            results.append(entry)
        except (UnicodeDecodeError, UnicodeEncodeError) as exc:
            quarantine_delta += 1
            row_id: str | None = None
            with contextlib.suppress(IndexError, ValueError, TypeError):
                row_id = str(raw_row[0])
            _quarantine_log(
                db_path=db_path,
                table=query.table,
                row_id=row_id,
                column="detail",
                row_index=idx,
                error=str(exc),
            )
    return results, quarantine_delta


def fetch_rows_via_bytes_fallback(
    *,
    db_path: Path,
    dbapi: _DBAPILike,
    query: FetchQuery,
) -> tuple[list[MemoryEntry], int]:
    """Bytes-mode connection re-execute to isolate bad-UTF-8 rows.

    Slow path — invoked only when the primary cursor's ``execute()`` or
    ``fetchall()`` raises a UTF-8 decode error. Re-executes *query*
    (including WHERE / ORDER BY / LIMIT) against a ``text_factory=bytes``
    connection so each column can be decoded individually and only the
    truly-undecodable rows are quarantined, while filters and the row cap
    are preserved. Column names for quarantine logs come from the
    bytes-mode cursor's ``description``.
    """
    sql, params = query.build()
    try:
        raw_conn = dbapi.connect(str(db_path))
        raw_conn.text_factory = bytes
        try:
            byte_cursor = raw_conn.execute(sql, params)
            raw_rows = byte_cursor.fetchall()
            column_names = _column_names(byte_cursor)
        finally:
            raw_conn.close()
    except sqlite3.Error as exc:
        # The secondary connection itself failed (locked, missing file,
        # cipher mismatch, ...). We cannot recover rows here; surface the
        # failure via the log and return empty rather than masking it as a
        # partial result — the caller's quarantine counter stays accurate.
        logger.warning(
            "db_utf8_fallback_failed",
            action="memory_row_utf8_quarantined",
            outcome="fallback_failed",
            db_path=str(db_path),
            table=query.table,
            error=str(exc),
        )
        return [], 0

    return _decode_bytes_rows(raw_rows, column_names=column_names, db_path=db_path, table=query.table)


def _decode_bytes_rows(
    raw_rows: list[tuple[object, ...]],
    *,
    column_names: tuple[str, ...],
    db_path: Path,
    table: str,
) -> tuple[list[MemoryEntry], int]:
    """Decode bytes-mode rows column-by-column, quarantining bad ones."""
    results: list[MemoryEntry] = []
    quarantine_delta = 0
    for idx, raw_row in enumerate(raw_rows):
        decoded, bad_col = _decode_row_columns(raw_row, column_names=column_names)
        if bad_col is not None:
            quarantine_delta += 1
            _quarantine_log(
                db_path=db_path,
                table=table,
                row_id=_row_id_from_bytes(raw_row),
                column=bad_col,
                row_index=idx,
            )
            continue

        try:
            entry = row_to_entry(tuple(decoded))
            results.append(entry)
        except (ValueError, TypeError, KeyError) as exc:
            # Columns decoded cleanly but model construction failed (bad
            # enum value, malformed JSON, schema drift). Quarantine the row
            # rather than failing the whole listing.
            quarantine_delta += 1
            _quarantine_log(
                db_path=db_path,
                table=table,
                row_id=_row_id_from_bytes(raw_row),
                column="row_to_entry",
                row_index=idx,
                error=str(exc),
            )
    return results, quarantine_delta


def _decode_row_columns(
    raw_row: tuple[object, ...],
    *,
    column_names: tuple[str, ...],
) -> tuple[list[object], str | None]:
    """Strict-decode each bytes column.

    Returns ``(decoded_values, bad_column_name)``. ``bad_column_name`` is
    ``None`` when every column decoded; otherwise it names the first
    undecodable column (best effort) and decoding stops there.
    """
    decoded: list[object] = []
    for col_idx, val in enumerate(raw_row):
        if isinstance(val, bytes):
            try:
                decoded.append(val.decode("utf-8", errors="strict"))
            except UnicodeDecodeError:
                name = column_names[col_idx] if col_idx < len(column_names) else "unknown"
                return decoded, name
        else:
            decoded.append(val)
    return decoded, None


def _column_names(cursor: _CursorLike) -> tuple[str, ...]:
    """Column names from a cursor's ``description`` (empty if unavailable)."""
    description = cursor.description
    if not description:
        return ()
    return tuple(str(col[0]) for col in description)


def _row_id_from_bytes(raw_row: tuple[object, ...]) -> str | None:
    """Best-effort id extraction from a bytes-mode row (column 0)."""
    if not raw_row:
        return None
    first = raw_row[0]
    if isinstance(first, bytes):
        return first.decode("utf-8", errors="replace")
    return str(first)
