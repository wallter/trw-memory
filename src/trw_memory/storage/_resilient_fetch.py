"""Resilient row materialisation — bad-UTF-8 row quarantine.

Belongs to the ``sqlite_backend.py`` facade. Re-exported there for
back-compat — ``SQLiteBackend._fetch_rows_resilient`` /
``_fetch_rows_via_bytes_fallback`` retain identical signatures via
1-line delegators that pass instance state.

The sqlite3 C extension raises ``sqlite3.OperationalError("Could not
decode to UTF-8 column ...")`` during row fetch. The fallback opens a
secondary connection with ``text_factory=bytes`` to read raw rows,
then decodes each column individually, quarantining rows that can't
decode at all.

Returns ``(results, quarantine_delta)`` so callers can update their
own ``quarantine_count_utf8`` counter.

Extracted as PRD-DIST-245 Phase 1 batch 84.
"""

from __future__ import annotations

import contextlib
import sqlite3
from pathlib import Path
from typing import Any

import structlog

from trw_memory.models.memory import MemoryEntry
from trw_memory.storage._row_mapper import row_to_entry

logger = structlog.get_logger(__name__)


def fetch_rows_resilient(
    cursor: Any,
    *,
    db_path: Path,
    dbapi: Any,
    select_columns_sql: str,
    table: str = "memories",
) -> tuple[list[MemoryEntry], int]:
    """Iterate cursor row-by-row, quarantining bad-UTF-8 rows.

    Returns ``(results, quarantine_delta)`` — caller adds the delta to
    its own ``quarantine_count_utf8`` counter.
    """
    try:
        raw_rows = cursor.fetchall()
    except (sqlite3.OperationalError, UnicodeDecodeError) as exc:
        err_str = str(exc)
        if "UTF-8" not in err_str and "decode" not in err_str.lower() and not isinstance(exc, UnicodeDecodeError):
            raise
        return fetch_rows_via_bytes_fallback(
            cursor,
            db_path=db_path,
            dbapi=dbapi,
            select_columns_sql=select_columns_sql,
            table=table,
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
            with contextlib.suppress(Exception):
                row_id = str(raw_row[0])
            logger.warning(
                "db_bad_utf8_row_quarantined",
                action="memory_row_utf8_quarantined",
                row_id=row_id,
                column="detail",
                db_path=str(db_path),
                table=table,
                row_index=idx,
                error=str(exc),
            )
    return results, quarantine_delta


def fetch_rows_via_bytes_fallback(
    cursor: Any,
    *,
    db_path: Path,
    dbapi: Any,
    select_columns_sql: str,
    table: str = "memories",
) -> tuple[list[MemoryEntry], int]:
    """Bytes-mode connection re-execute to isolate bad-UTF-8 rows.

    Slow path — invoked only when the primary cursor's fetchall raises
    a UTF-8 decode error.
    """
    results: list[MemoryEntry] = []
    quarantine_delta = 0
    try:
        raw_conn = dbapi.connect(str(db_path))
        raw_conn.text_factory = bytes
        raw_rows = raw_conn.execute(
            f"SELECT {select_columns_sql} FROM {table} ORDER BY updated_at DESC"  # noqa: S608
        ).fetchall()
        raw_conn.close()
    except Exception:
        logger.warning(
            "db_utf8_fallback_failed",
            action="memory_row_utf8_quarantined",
            db_path=str(db_path),
            table=table,
        )
        return [], 0

    for idx, raw_row in enumerate(raw_rows):
        decoded: list[object] = []
        row_bad = False
        bad_col: str | None = None
        for col_idx, val in enumerate(raw_row):
            if isinstance(val, bytes):
                try:
                    decoded.append(val.decode("utf-8", errors="strict"))
                except (UnicodeDecodeError, ValueError):
                    row_bad = True
                    with contextlib.suppress(Exception):
                        bad_col = cursor.description[col_idx][0] if cursor.description else None
                    break
            else:
                decoded.append(val)

        if row_bad:
            quarantine_delta += 1
            row_id_str: str | None = None
            with contextlib.suppress(Exception):
                first = raw_row[0]
                row_id_str = first.decode("utf-8", errors="replace") if isinstance(first, bytes) else str(first)
            logger.warning(
                "db_bad_utf8_row_quarantined",
                action="memory_row_utf8_quarantined",
                row_id=row_id_str,
                column=bad_col or "unknown",
                db_path=str(db_path),
                table=table,
                row_index=idx,
            )
            continue

        try:
            entry = row_to_entry(tuple(decoded))
            results.append(entry)
        except Exception:
            quarantine_delta += 1
            logger.warning(
                "db_bad_utf8_row_quarantined",
                action="memory_row_utf8_quarantined",
                row_id=None,
                column="row_to_entry",
                db_path=str(db_path),
                table=table,
                row_index=idx,
            )

    return results, quarantine_delta
