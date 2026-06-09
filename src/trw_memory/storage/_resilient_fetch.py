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
# Fallback-failure observability counter
# ---------------------------------------------------------------------------
#
# The bytes-mode fallback fails open: when the *secondary* connection itself
# raises (locked, missing file, cipher mismatch, ...) it returns ([], 0) so the
# caller's per-row ``quarantine_count_utf8`` cannot reflect the drop — the rows
# were never readable, so they are not "quarantined". That made the silent drop
# invisible to counter-based monitoring (only a warning log fired). This
# process-wide counter makes the fail-open event *countable* without changing
# the fail-open behaviour or the ``(results, delta)`` return contract.


@dataclass(slots=True)
class _FallbackMetrics:
    """Process-wide counter for bytes-mode fallback hard failures."""

    bytes_fallback_failures: int = 0


_fallback_metrics = _FallbackMetrics()


def get_bytes_fallback_failures() -> int:
    """Return the number of bytes-mode fallback connections that failed open.

    Each increment corresponds to one ``fetch_rows_via_bytes_fallback`` call
    whose secondary connection raised ``sqlite3.Error`` — i.e. rows that were
    silently dropped (not row-level quarantined). Monitoring can poll this to
    detect a degraded backend that the per-row quarantine counter cannot see.
    """
    return _fallback_metrics.bytes_fallback_failures


def reset_bytes_fallback_failures() -> None:
    """Reset the fallback-failure counter (test isolation / monitoring window)."""
    _fallback_metrics.bytes_fallback_failures = 0


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
    #: SQLCipher key (64-char lowercase hex) for the namespace's encrypted DB.
    #: When set, the bytes-mode fallback keys its secondary connection before
    #: reading, so encrypted stores don't silently return zero rows. ``None``
    #: for plaintext stores.
    sqlcipher_key_hex: str | None = None

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
            _quarantine_log(
                db_path=db_path,
                table=query.table,
                row_id=_safe_row_id(raw_row),
                column="detail",
                row_index=idx,
                error=str(exc),
            )
        except (ValueError, TypeError, KeyError) as exc:
            # Columns decoded cleanly but model construction failed (bad enum
            # value, malformed JSON, schema drift). The slow bytes-mode path
            # already quarantines these; the fast path must too, or a single
            # forward-version / corrupt row collapses the whole listing. The
            # narrow exception set mirrors row_to_entry's failure modes —
            # enum/int/float coercion raise ValueError, shape errors raise
            # TypeError, and dict-keyed lookups raise KeyError.
            quarantine_delta += 1
            _quarantine_log(
                db_path=db_path,
                table=query.table,
                row_id=_safe_row_id(raw_row),
                column="row_to_entry",
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
            # On an encrypted store the secondary connection MUST be keyed before
            # any read, or every SELECT returns zero rows (SQLCipher treats the
            # unkeyed handle as a blank DB) — silently dropping every row instead
            # of quarantining only the bad-UTF-8 ones. Apply the key + cipher
            # pragmas here so the fallback decodes real data rather than nothing.
            if query.sqlcipher_key_hex is not None:
                _apply_fallback_sqlcipher_key(raw_conn, query.sqlcipher_key_hex)
            byte_cursor = raw_conn.execute(sql, params)
            raw_rows = byte_cursor.fetchall()
            column_names = _column_names(byte_cursor)
        finally:
            raw_conn.close()
    except sqlite3.Error as exc:
        # The secondary connection itself failed (locked, missing file,
        # cipher mismatch, ...). We cannot recover rows here; surface the
        # failure via the log AND a distinct process-wide counter, then return
        # empty rather than masking it as a partial result. The caller's per-row
        # quarantine counter stays accurate (these rows were never read, so they
        # are not row-level quarantined); the dedicated counter makes the silent
        # drop countable for monitoring (F1).
        _fallback_metrics.bytes_fallback_failures += 1
        logger.warning(
            "db_utf8_fallback_failed",
            action="memory_row_utf8_quarantined",
            outcome="fallback_failed",
            db_path=str(db_path),
            table=query.table,
            error=str(exc),
            bytes_fallback_failures=_fallback_metrics.bytes_fallback_failures,
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


def _apply_fallback_sqlcipher_key(conn: _ConnectionLike, sqlcipher_key_hex: str) -> None:
    """Key + configure a bytes-mode SQLCipher connection for the fallback read.

    Mirrors ``storage._connection.connect``'s keying path: validate the hex key,
    apply ``PRAGMA key`` and the shared KDF/cipher pragmas. Raises ``ValueError``
    on a malformed key — the caller's ``except sqlite3.Error`` does NOT catch
    that, so a misconfigured key surfaces loudly rather than silently dropping
    rows (the encrypted-store failure mode this fix exists to prevent).
    """
    from trw_memory.storage._connection import _apply_sqlcipher_pragmas_safe

    if len(sqlcipher_key_hex) != 64 or any(ch not in "0123456789abcdef" for ch in sqlcipher_key_hex):
        raise ValueError("sqlcipher_key_hex must be a 64-character lowercase hex string")
    conn.execute(f"PRAGMA key = \"x'{sqlcipher_key_hex}'\"")
    _apply_sqlcipher_pragmas_safe(conn)


def _column_names(cursor: _CursorLike) -> tuple[str, ...]:
    """Column names from a cursor's ``description`` (empty if unavailable)."""
    description = cursor.description
    if not description:
        return ()
    return tuple(str(col[0]) for col in description)


def _safe_row_id(raw_row: tuple[object, ...]) -> str | None:
    """Best-effort id extraction from a fast-path (already-decoded) row.

    Column 0 holds the entry id; coercion is wrapped so quarantine logging
    never raises while reporting a row that already failed to map.
    """
    row_id: str | None = None
    with contextlib.suppress(IndexError, ValueError, TypeError):
        row_id = str(raw_row[0])
    return row_id


def _row_id_from_bytes(raw_row: tuple[object, ...]) -> str | None:
    """Best-effort id extraction from a bytes-mode row (column 0)."""
    if not raw_row:
        return None
    first = raw_row[0]
    if isinstance(first, bytes):
        return first.decode("utf-8", errors="replace")
    return str(first)
