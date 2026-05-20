"""SQLite persistence helpers for wiki page references."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from trw_memory.exceptions import StorageError
from trw_memory.models.memory import MemoryEntry
from trw_memory.wiki.models import WikiPage, validate_wiki_slug

if TYPE_CHECKING:
    from trw_memory.storage.sqlite_backend import SQLiteBackend

__all__ = [
    "StoredWikiReference",
    "purge_wiki_refs_for_entry",
    "query_wiki_inbound_refs",
    "query_wiki_outbound_refs",
    "replace_wiki_refs_for_entry",
]


class StoredWikiReference(BaseModel):
    """A persisted wiki reference row."""

    model_config = ConfigDict(extra="forbid", strict=True, validate_default=True)

    source_entry_id: str = Field(min_length=1)
    source_slug: str
    target_slug: str
    ref_type: str = Field(min_length=1, max_length=64)
    label: str = Field(default="", max_length=160)
    bidirectional: bool = True
    namespace: str = Field(default="default", min_length=1)


def replace_wiki_refs_for_entry(backend: SQLiteBackend, entry: MemoryEntry) -> int:
    """Replace persisted outbound wiki refs for ``entry`` from its metadata."""

    page = WikiPage.from_memory_metadata(entry.metadata)
    rows: list[tuple[str, str, str, str, str, int, str, str]] = []
    if page is not None:
        updated_at = _timestamp()
        rows = [
            (
                entry.id,
                page.slug,
                ref.target_slug,
                ref.ref_type,
                ref.label,
                int(ref.bidirectional),
                entry.namespace,
                updated_at,
            )
            for ref in page.outbound_refs
        ]

    try:
        with backend._lock:
            backend._conn.execute("DELETE FROM wiki_refs WHERE source_entry_id = ?", (entry.id,))
            if rows:
                backend._conn.executemany(
                    """
                    INSERT INTO wiki_refs (
                        source_entry_id, source_slug, target_slug, ref_type,
                        label, bidirectional, namespace, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )
            if backend._skip_commit_depth == 0:
                backend._conn.commit()
    except (sqlite3.Error, ValueError) as exc:
        raise StorageError(
            f"Failed to replace wiki refs for entry {entry.id}: {exc}", path=str(backend._db_path)
        ) from exc
    return len(rows)


def purge_wiki_refs_for_entry(backend: SQLiteBackend, entry_id: str) -> int:
    """Remove all persisted wiki refs whose source is ``entry_id``."""

    try:
        with backend._lock:
            cursor = backend._conn.execute("DELETE FROM wiki_refs WHERE source_entry_id = ?", (entry_id,))
            if backend._skip_commit_depth == 0:
                backend._conn.commit()
            return int(cursor.rowcount)
    except sqlite3.Error as exc:
        raise StorageError(
            f"Failed to purge wiki refs for entry {entry_id}: {exc}", path=str(backend._db_path)
        ) from exc


def query_wiki_outbound_refs(
    backend: SQLiteBackend,
    source_slug: str,
    *,
    namespace: str | None = None,
) -> list[StoredWikiReference]:
    """Return deterministic outbound refs from ``source_slug``."""

    return _query_refs(backend, "source_slug", validate_wiki_slug(source_slug), namespace=namespace)


def query_wiki_inbound_refs(
    backend: SQLiteBackend,
    target_slug: str,
    *,
    namespace: str | None = None,
) -> list[StoredWikiReference]:
    """Return deterministic inbound refs pointing at ``target_slug``."""

    return _query_refs(backend, "target_slug", validate_wiki_slug(target_slug), namespace=namespace)


def _query_refs(
    backend: SQLiteBackend,
    column: str,
    slug: str,
    *,
    namespace: str | None,
) -> list[StoredWikiReference]:
    clauses = [f"{column} = ?"]
    params: list[str] = [slug]
    if namespace is not None:
        clauses.append("namespace = ?")
        params.append(namespace)
    sql = f"""
        SELECT source_entry_id, source_slug, target_slug, ref_type, label, bidirectional, namespace
        FROM wiki_refs
        WHERE {" AND ".join(clauses)}
        ORDER BY source_slug, target_slug, ref_type, source_entry_id
    """  # noqa: S608 - column is an internal allow-listed argument.
    try:
        with backend._lock:
            rows = backend._conn.execute(sql, params).fetchall()
    except sqlite3.Error as exc:
        raise StorageError(f"Failed to query wiki refs for {slug}: {exc}", path=str(backend._db_path)) from exc
    return [
        StoredWikiReference(
            source_entry_id=str(row["source_entry_id"]),
            source_slug=str(row["source_slug"]),
            target_slug=str(row["target_slug"]),
            ref_type=str(row["ref_type"]),
            label=str(row["label"] or ""),
            bidirectional=bool(row["bidirectional"]),
            namespace=str(row["namespace"]),
        )
        for row in rows
    ]


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()
