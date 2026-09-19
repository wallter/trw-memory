"""Warm tier operations for tiered memory lifecycle.

Manages the sqlite-vec backed persistent index with JSONL sidecar fallback
for keyword search.
"""

from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path
from typing import TYPE_CHECKING, cast

import structlog

from trw_memory.lifecycle.tiers._warm_sidecar_cache import (
    ParsedSidecar,
    SidecarCache,
    SidecarRows,
    access_only_change,
    parse_sidecar,
    sidecar_key,
)
from trw_memory.lifecycle.tiers._warm_space import admit_warm_hits, admit_warm_vectors
from trw_memory.namespaces.validation import DEFAULT_NAMESPACE
from trw_memory.storage.persistence import lock_for_rmw

if TYPE_CHECKING:
    from trw_memory.embeddings.provenance import EmbeddingSpace, VectorProvenance
    from trw_memory.storage.sqlite_backend import SQLiteBackend

logger = structlog.get_logger(__name__)

#: The warm tier is a single-tenant sidecar store: it holds demoted vectors and
#: a JSONL keyword sidecar for ONE project's hot set, and its rows are never
#: partitioned by namespace. Schema 5 keys ``vec_index`` on
#: ``(namespace, entry_id)`` (PRD-CORE-245 FR02), so every warm read and write
#: uses this one constant — the value is arbitrary but must be identical on
#: both sides or a demoted vector becomes unreachable.
WARM_TIER_NAMESPACE = DEFAULT_NAMESPACE


class WarmTierStore:
    """Warm tier: sqlite-vec backed persistent index with JSONL sidecar.

    Args:
        base_dir: Base directory for memory storage.
    """

    def __init__(self, base_dir: Path) -> None:
        self._base_dir = base_dir
        # Cached SQLiteBackend to avoid open/close per operation
        self._warm_backend: SQLiteBackend | None = None
        self._warm_backend_dim: int | None = None
        # Parsed-sidecar cache keyed on (mtime_ns, size, inode); writers re-seed it
        # under the RMW lock so a recall's access-time mirror does not force
        # the next read to re-parse the file (see _warm_sidecar_cache).
        self._sidecar_cache = SidecarCache()

    def _get_warm_backend(self, dim: int | None = None) -> SQLiteBackend | None:
        """Lazy-init and cache a SQLiteBackend for warm tier operations.

        Args:
            dim: Embedding dimension (required for vector ops, None for metadata-only).

        Returns:
            Cached SQLiteBackend instance, or None if import fails.
        """
        try:
            from trw_memory.storage.sqlite_backend import SQLiteBackend as _SQLiteBackend
        except ImportError:
            return None

        # Normalise dim so that a call with dim=None and a subsequent call with
        # dim=384 (SQLiteBackend's internal default) do not needlessly recreate
        # the backend -- both resolve to the same effective dimension.
        effective_dim = dim if dim is not None else 384  # SQLiteBackend default

        # If dim changed, close old backend and recreate
        if self._warm_backend is not None and self._warm_backend_dim != effective_dim:
            self._warm_backend.close()
            self._warm_backend = None
            self._warm_backend_dim = None

        if self._warm_backend is None:
            db_path = self._warm_db_path()
            self._warm_backend = _SQLiteBackend(db_path, dim=effective_dim)
            self._warm_backend_dim = effective_dim

        return self._warm_backend

    def _warm_db_path(self) -> Path:
        """Resolve path to warm.db."""
        mem_dir = self._base_dir / "memory"
        mem_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        return mem_dir / "warm.db"

    def _warm_sidecar_path(self) -> Path:
        """Path to the warm tier keyword-search sidecar (JSONL)."""
        return self._warm_db_path().with_suffix(".jsonl")

    def _iter_sidecar_records(self, sidecar: Path) -> Iterator[tuple[int, dict[str, object]]]:
        """Yield ``(line_number, record)`` for each well-formed row in the sidecar.

        This is the single parsing Seam for the warm JSONL sidecar: every
        ``warm_add`` / ``warm_search`` / ``warm_entries`` / ``warm_remove`` path
        reads rows through here instead of re-deriving how corruption is handled.

        Fail-open is preserved -- blank lines are ignored, and a row that is not
        valid JSON, is not a JSON object, or is not valid UTF-8 is skipped rather
        than raised. Each skip emits a structured
        ``warm_tier_sidecar_corrupt_record_skipped`` event carrying the sidecar
        path, the 1-based line number, and the error class so operators get
        locality when warm recall misses because rows were dropped. The raw line
        and any memory payload are deliberately never logged.

        Rows are read as raw bytes and decoded per line so a single non-UTF-8
        row cannot abort the whole read: adjacent valid records (and their
        line-number locality) survive a corrupt byte sequence anywhere in the
        file. ``\r`` from CRLF-terminated rows is stripped, matching the prior
        ``str.splitlines`` behaviour.

        Only live rows are yielded: a row superseded by a later row with the
        same id (the append-log layout, see ``_warm_sidecar_cache``) is not.
        Parsed rows are cached against the file's ``(mtime_ns, size, inode)``; a
        cache hit yields shallow copies so callers may annotate records
        without poisoning the cache. Corrupt-row warnings therefore fire once
        per file version, not once per read. Writers re-seed the cache with
        the rows they wrote, so the first read after a write is a hit too.
        """
        for line_number, rec in self._current_parse(sidecar).rows:
            yield line_number, dict(rec)

    def _current_parse(self, sidecar: Path) -> ParsedSidecar:
        """Return the parse of the sidecar's current bytes, from the cache when it is still valid."""
        key = sidecar_key(sidecar)
        parsed = self._sidecar_cache.get(key)
        if parsed is None:
            parsed = self._parse_sidecar_records(sidecar)
            self._sidecar_cache.put(key, parsed)
        return parsed

    def _parse_sidecar_records(self, sidecar: Path) -> ParsedSidecar:
        """Parse the sidecar from disk (see ``_iter_sidecar_records`` for the contract)."""
        return parse_sidecar(sidecar)

    def get_embedding(self, entry_id: str) -> list[float] | None:
        """Return the stored warm-tier embedding for *entry_id*, if present."""
        backend = self._get_warm_backend()
        if backend is None:
            return None
        return backend.get_stored_embeddings([entry_id]).get(entry_id)

    def warm_add(
        self,
        entry_id: str,
        entry_data: dict[str, object],
        embedding: list[float] | None,
        *,
        provenance: VectorProvenance | None = None,
    ) -> None:
        """Insert or replace an entry in the warm store.

        When embedding is provided and sqlite-vec is available, stores the
        vector with its *provenance* (without it the vector is never
        dense-scored). Always writes to the JSONL sidecar for keyword search.

        Args:
            entry_id: Memory entry identifier.
            entry_data: Dict of entry fields.
            embedding: Optional dense embedding vector.
        """
        if embedding is not None:
            try:
                backend = self._get_warm_backend(dim=len(embedding))
                if backend is not None:
                    proof = {"provenance": provenance} if provenance is not None else {}
                    backend.upsert_vector(entry_id, embedding, namespace=WARM_TIER_NAMESPACE, **proof)
            except (OSError, ValueError):
                logger.debug("warm_tier_vec_upsert_failed", entry_id=entry_id, exc_info=True)

        # Always update sidecar for keyword search
        self._warm_sidecar_upsert(entry_id, entry_data)
        logger.debug("warm_tier_add", entry_id=entry_id, has_embedding=embedding is not None)

    def warm_add_many(
        self,
        items: list[tuple[str, dict[str, object], list[float] | None]],
    ) -> None:
        """Insert or replace several entries with ONE sidecar read-modify-write.

        ``warm_add`` costs a full sidecar parse (and, for an entry already
        present, a full rewrite) per call. Recall mirrors every returned entry
        back into the warm tier to refresh ``last_accessed_at``, so a 50-result
        recall over a 400-row sidecar was 50 parses + 50 rewrites -- measured at
        ~80% of recall latency (2.1 s of 2.6 s) on the LOCOMO benchmark. This
        path writes once for the whole batch (an append for new ids and
        access-only refreshes); the resulting live rows are identical to
        applying ``warm_add`` in order.
        """
        if not items:
            return
        for entry_id, _entry_data, embedding in items:
            if embedding is None:
                continue
            try:
                backend = self._get_warm_backend(dim=len(embedding))
                if backend is not None:
                    backend.upsert_vector(entry_id, embedding, namespace=WARM_TIER_NAMESPACE)
            except (OSError, ValueError):
                logger.debug("warm_tier_vec_upsert_failed", entry_id=entry_id, exc_info=True)
        # Last write wins for a duplicated id, matching sequential warm_add.
        records: dict[str, dict[str, object]] = {}
        for entry_id, entry_data, _embedding in items:
            records[entry_id] = self._sidecar_record(entry_id, entry_data)
        self._warm_sidecar_upsert_many(records)
        logger.debug("warm_tier_add_many", count=len(records))

    @staticmethod
    def _sidecar_record(entry_id: str, entry_data: dict[str, object]) -> dict[str, object]:
        # Use 'content' (MemoryEntry) or fall back to 'summary' (legacy)
        summary = str(entry_data.get("content", entry_data.get("summary", "")))
        return {
            "id": entry_id,
            "summary": summary,
            "tags": entry_data.get("tags", []),
            "entry": dict(entry_data),
        }

    def _warm_sidecar_upsert(self, entry_id: str, entry_data: dict[str, object]) -> None:
        """Write one entry's metadata to the warm sidecar JSONL for keyword search."""
        self._warm_sidecar_upsert_many({entry_id: self._sidecar_record(entry_id, entry_data)})

    def _warm_sidecar_upsert_many(self, records: dict[str, dict[str, object]]) -> None:
        """Upsert *records* (keyed by entry id) into the sidecar in one pass.

        New ids and access-only refreshes (recall's ``last_accessed_at`` mirror,
        see ``ACCESS_FIELDS``) are APPENDED: O(len(records)) bytes and encoding,
        with the superseded rows left as compaction debt. Any other change to an
        existing row -- or debt past the compaction threshold -- rewrites the
        live rows once, atomically, so stale content never outlives an update.
        """
        sidecar = self._warm_sidecar_path()
        sidecar.parent.mkdir(mode=0o700, parents=True, exist_ok=True)

        # The entire read-modify-write must be serialized: two concurrent
        # upserts (or an upsert racing purge_sidecar_entry) would otherwise
        # read the same snapshot and clobber each other's rows on rewrite, or
        # interleave the supersede decision with another writer's append. The
        # advisory lock is both in-process and cross-process (fcntl), matching
        # yaml_backend's RMW discipline.
        with lock_for_rmw(sidecar):
            dumped = [json.dumps(r) for r in records.values()]
            # What a parse of the new lines yields; the cache is re-seeded with
            # it below (still under the lock) so the next read is not a re-parse.
            new_rows = [cast("dict[str, object]", json.loads(line)) for line in dumped]
            if not sidecar.exists():
                self._replace_sidecar(sidecar, new_rows)
                return
            parsed = self._current_parse(sidecar)
            content_changed = False
            for rec in new_rows:
                old = parsed.live(str(rec.get("id", "")))
                if old is not None and not access_only_change(old, rec):
                    content_changed = True
                    break
            if content_changed or parsed.compaction_due_after(new_rows):
                # Survivors keep their order; updated rows move to the end.
                self._replace_sidecar(sidecar, parsed.merged(new_rows))
                return
            # A torn tail (crash mid-append) is terminated first so the new
            # rows do not fuse with it; the fragment stays a skipped line.
            with sidecar.open("a", encoding="utf-8") as fh:
                fh.write(("\n" if parsed.torn_tail else "") + "".join(line + "\n" for line in dumped))
            self._sidecar_cache.record_append(sidecar, parsed, new_rows)

    def _replace_sidecar(self, sidecar: Path, rows: list[dict[str, object]]) -> None:
        """Atomically replace the sidecar with *rows* (caller holds the RMW lock).

        Written to a sibling and renamed so a concurrent reader (and the parse
        cache) never sees a half-written file. Corrupt and superseded lines are
        dropped: this is also the compaction path.
        """
        tmp = sidecar.with_name(sidecar.name + ".tmp")
        tmp.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        os.replace(tmp, sidecar)
        self._sidecar_cache.record_rewrite(sidecar, rows)

    def warm_remove(self, entry_id: str) -> bool:
        """Delete an entry from the warm store and sidecar.

        Args:
            entry_id: Memory entry identifier to remove.
        """
        backend_present = False
        vector_removed = False
        try:
            backend = self._get_warm_backend()
            if backend is not None:
                backend_present = True
                backend.delete(entry_id, namespace=WARM_TIER_NAMESPACE)
                vector_deleted = backend.delete_vector(entry_id, namespace=WARM_TIER_NAMESPACE)
                vector_removed = vector_deleted or vector_removed
        except (OSError, ValueError):
            logger.debug("warm_tier_db_remove_failed", entry_id=entry_id, exc_info=True)
            vector_removed = False

        self.purge_sidecar_entry(entry_id)
        vector_still_present = False
        if backend_present:
            try:
                backend = self._get_warm_backend()
                if backend is not None:
                    vector_exists = getattr(backend, "vector_exists", None)
                    vector_still_present = (
                        bool(vector_exists(entry_id, namespace=WARM_TIER_NAMESPACE))
                        if callable(vector_exists)
                        else False
                    )
                else:
                    vector_still_present = False
            except (OSError, ValueError):
                vector_still_present = True

        logger.debug("warm_tier_remove", entry_id=entry_id)
        return not vector_still_present

    def purge_sidecar_entry(self, entry_id: str) -> bool:
        """Remove an entry from the warm sidecar without touching vector state."""
        sidecar = self._warm_sidecar_path()
        sidecar_removed = False
        # Same advisory lock as _warm_sidecar_upsert: purge is also a
        # read-modify-write on the sidecar, so it must serialize against
        # concurrent upserts or one side's write is lost.
        with lock_for_rmw(sidecar):
            if sidecar.exists():
                parsed = self._current_parse(sidecar)
                if parsed.live(entry_id) is not None:
                    # A rewrite, not an appended tombstone: erasure must physically
                    # remove every copy, superseded ones included.
                    rows = [rec for _line, rec in parsed.rows if str(rec.get("id", "")) != entry_id]
                    self._replace_sidecar(sidecar, rows)
                    sidecar_removed = True
        return sidecar_removed

    def discovery_entries(
        self,
        query_embedding: list[float] | None,
        *,
        covered_ids: frozenset[str] = frozenset(),
        namespace: str | None = None,
        query_space: EmbeddingSpace | None = None,
    ) -> list[dict[str, object]]:
        """Read full sidecars and uncapped vectors without a writable backend.

        SQLite mode=ro may create WAL coordination sidefiles; records, schema,
        archive contents and lifecycle access metadata are never written here.

        Every sidecar row is returned except those named in *covered_ids*: rows
        the caller already ranked from the primary store, which tier discovery
        would drop anyway. They are neither copied nor vector-scored, so a recall
        whose hybrid pool held the whole namespace pays per UNCOVERED row, not
        per sidecar row (a full copy was ~12 ms at 5,000 rows, ~60 ms at
        20,000). A covered row is still checked to belong to *namespace* when
        one is given (``NamespaceScopeError`` otherwise), as discovery checks
        every row it receives. Only vectors recorded in *query_space* are
        scored (``_space_gate``); ``None`` scores none.
        """
        sidecar = self._base_dir / "memory" / "warm.jsonl"
        if not sidecar.exists():
            return []
        parsed = self._current_parse(sidecar)
        if namespace is not None and parsed.names_other_namespace(namespace):
            from trw_memory.security.namespace_scope import NamespaceScopeError

            raise NamespaceScopeError("tier snapshot outside authorized namespace")
        entries = self._entries_by_id(parsed.rows_except(covered_ids))
        db_path = sidecar.with_suffix(".db")
        scored_ids = list(entries)
        if query_embedding is None or not scored_ids or not db_path.exists():
            return list(entries.values())
        try:
            import sqlite_vec

            with closing(sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)) as conn:
                conn.enable_load_extension(True)
                sqlite_vec.load(conn)
                conn.enable_load_extension(False)
                # Warm vectors have one fixed namespace; reject corrupted foreign rows
                # before the existing ID-based bulk decoder sees them.
                foreign = conn.execute(
                    "SELECT 1 FROM vec_index WHERE namespace != ? LIMIT 1", (WARM_TIER_NAMESPACE,)
                ).fetchone()
                if foreign:
                    from trw_memory.security.namespace_scope import NamespaceScopeError

                    raise NamespaceScopeError("warm vector index contains foreign namespace")
                vectors = admit_warm_vectors(conn, scored_ids, query_space)
            for entry_id, vector in vectors.items():
                if len(vector) != len(query_embedding):
                    continue
                distance_squared = sum((a - b) ** 2 for a, b in zip(vector, query_embedding, strict=True))
                entries[entry_id]["_tier_relevance"] = 1.0 - distance_squared / 2.0
        except (ImportError, sqlite3.Error, OSError, AttributeError):
            logger.debug("warm_tier_discovery_vectors_unavailable", exc_info=True)
        return list(entries.values())

    def warm_search(
        self,
        query_tokens: list[str],
        query_embedding: list[float] | None,
        top_k: int = 25,
        *,
        query_space: EmbeddingSpace | None = None,
    ) -> list[dict[str, object]]:
        """Search the warm tier for relevant entries.

        Performs dense vector search when embedding is available, keeping only
        hits whose stored vector is in *query_space* (``None`` keeps none);
        falls back to JSONL keyword search when no vector hit survives.

        Args:
            query_tokens: Tokenized query for keyword fallback.
            query_embedding: Optional dense query vector.
            top_k: Maximum results to return.

        Returns:
            List of dicts containing at minimum ``{"id": ..., "score": ...}``
            plus the serialized MemoryEntry payload when the sidecar has it.
        """
        if not query_tokens and query_embedding is None:
            return []

        sidecar_entries = self._warm_sidecar_entries_by_id()
        if query_embedding is not None:
            try:
                backend = self._get_warm_backend(dim=len(query_embedding))
                if backend is not None:
                    raw = admit_warm_hits(backend, backend.search_vectors(query_embedding, top_k=top_k), query_space)
                    if raw:
                        results: list[dict[str, object]] = []
                        for eid, dist in raw:
                            sidecar_entry = sidecar_entries.get(eid)
                            if sidecar_entry is None:
                                logger.debug("warm_tier_skip_orphaned_vector_hit", entry_id=eid)
                                continue
                            item = dict(sidecar_entry)
                            item["id"] = eid
                            # sqlite-vec vec0 float[] returns L2 (Euclidean)
                            # distance, not cosine. For the unit-normalized
                            # embeddings this engine stores
                            # (normalize_embeddings=True),
                            # cosine_similarity = 1 - distance**2 / 2 (since
                            # distance**2 = 2 * (1 - cos)). The prior
                            # `1 - distance` under-scored every moderately-similar
                            # hit (cos=0.5 -> 0.0) and could even go negative,
                            # distorting cross-tier importance scoring. Order is
                            # unchanged (both are monotonic in distance); only
                            # the magnitude is corrected.
                            similarity = 1.0 - (dist * dist) / 2.0
                            item["_tier_relevance"] = float(similarity)
                            item["score"] = float(similarity)
                            results.append(item)
                        return results
            except (OSError, ValueError):
                logger.debug("warm_tier_vec_search_failed", exc_info=True)

        return self._warm_keyword_search(query_tokens, top_k)

    def _warm_keyword_search(self, query_tokens: list[str], top_k: int) -> list[dict[str, object]]:
        """Search the warm sidecar JSONL for keyword matches."""
        sidecar = self._warm_sidecar_path()
        if not sidecar.exists() or not query_tokens:
            return []

        results: list[dict[str, object]] = []
        entry_map = self._warm_sidecar_entries_by_id()
        lower_tokens = {t.lower() for t in query_tokens}
        for _line_number, rec in self._iter_sidecar_records(sidecar):
            entry_id = str(rec.get("id", ""))
            entry_payload = entry_map.get(entry_id, {})
            text = str(rec.get("summary", "")).lower()
            text += " " + str(entry_payload.get("detail", "")).lower()
            tags = [str(t).lower() for t in cast("list[object]", rec.get("tags") or [])]
            text += " " + " ".join(tags)
            matched = sum(1 for tok in lower_tokens if tok in text)
            if matched > 0:
                score = matched / len(lower_tokens)
                item = dict(entry_map.get(entry_id, {"id": entry_id}))
                item["id"] = entry_id
                item["score"] = score
                results.append(item)

        results.sort(key=lambda r: float(str(r.get("score", 0))), reverse=True)
        return results[:top_k]

    def warm_entries(self, limit: int | None = None) -> list[dict[str, object]]:
        """Return the serialized warm-tier entries persisted in the sidecar."""
        entries = list(self._warm_sidecar_entries_by_id().values())
        return entries[:limit] if limit is not None else entries

    def _warm_sidecar_entries_by_id(self) -> dict[str, dict[str, object]]:
        """Hydrate the full entry payloads stored alongside the warm index."""
        sidecar = self._warm_sidecar_path()
        if not sidecar.exists():
            return {}
        return self._entries_by_id(self._current_parse(sidecar).rows)

    @staticmethod
    def _entries_by_id(rows: SidecarRows) -> dict[str, dict[str, object]]:
        """Entry payloads (fresh copies, safe to annotate) of *rows*, keyed by id."""
        entries: dict[str, dict[str, object]] = {}
        for _line_number, rec in rows:
            entry_id = str(rec.get("id", ""))
            if not entry_id:
                continue

            payload = rec.get("entry")
            if isinstance(payload, dict):
                item = dict(payload)
            else:
                raw_tags = rec.get("tags", [])
                tags = [str(tag) for tag in raw_tags] if isinstance(raw_tags, list) else []
                item = {
                    "id": entry_id,
                    "content": str(rec.get("summary", "")),
                    "tags": tags,
                }

            item.setdefault("id", entry_id)
            entries[entry_id] = item
        return entries

    def close(self) -> None:
        """Release warm tier resources."""
        if self._warm_backend is not None:
            self._warm_backend.close()
            self._warm_backend = None
            self._warm_backend_dim = None
