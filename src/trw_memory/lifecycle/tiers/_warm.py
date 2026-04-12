"""Warm tier operations for tiered memory lifecycle.

Manages the sqlite-vec backed persistent index with JSONL sidecar fallback
for keyword search.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, cast

import structlog

if TYPE_CHECKING:
    from trw_memory.storage.sqlite_backend import SQLiteBackend

logger = structlog.get_logger(__name__)


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
        mem_dir.mkdir(parents=True, exist_ok=True)
        return mem_dir / "warm.db"

    def _warm_sidecar_path(self) -> Path:
        """Path to the warm tier keyword-search sidecar (JSONL)."""
        return self._warm_db_path().with_suffix(".jsonl")

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
    ) -> None:
        """Insert or replace an entry in the warm store.

        When embedding is provided and sqlite-vec is available, stores the
        vector. Always writes to the JSONL sidecar for keyword search fallback.

        Args:
            entry_id: Memory entry identifier.
            entry_data: Dict of entry fields.
            embedding: Optional dense embedding vector.
        """
        if embedding is not None:
            try:
                backend = self._get_warm_backend(dim=len(embedding))
                if backend is not None:
                    backend.upsert_vector(entry_id, embedding)
            except (OSError, ValueError):
                logger.debug("warm_tier_vec_upsert_failed", entry_id=entry_id, exc_info=True)

        # Always update sidecar for keyword search
        self._warm_sidecar_upsert(entry_id, entry_data)
        logger.debug("warm_tier_add", entry_id=entry_id, has_embedding=embedding is not None)

    def _warm_sidecar_upsert(self, entry_id: str, entry_data: dict[str, object]) -> None:
        """Write entry metadata to the warm sidecar JSONL for keyword search.

        Optimized: for new entries (not already in sidecar), appends directly
        without reading/rewriting the entire file. For updates to existing
        entries, falls back to full read-filter-write.
        """
        sidecar = self._warm_sidecar_path()
        sidecar.parent.mkdir(parents=True, exist_ok=True)

        # Use 'content' (MemoryEntry) or fall back to 'summary' (legacy)
        summary = str(entry_data.get("content", entry_data.get("summary", "")))
        record: dict[str, object] = {
            "id": entry_id,
            "summary": summary,
            "tags": entry_data.get("tags", []),
            "entry": dict(entry_data),
        }

        # Fast path: if sidecar doesn't exist or entry is new, just append
        if not sidecar.exists():
            sidecar.write_text(json.dumps(record) + "\n", encoding="utf-8")
            return

        # Check if entry already exists (scan once)
        existing_lines = sidecar.read_text(encoding="utf-8").splitlines()
        entry_exists = False
        for line in existing_lines:
            line_s = line.strip()
            if not line_s:
                continue
            try:
                rec = json.loads(line_s)
                if str(rec.get("id", "")) == entry_id:
                    entry_exists = True
                    break
            except json.JSONDecodeError:
                continue

        if not entry_exists:
            # Append-only: O(1) write for new entries
            with sidecar.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
            return

        # Update path: filter out old record and rewrite (O(N) — infrequent)
        records: list[dict[str, object]] = []
        for line in existing_lines:
            line_s = line.strip()
            if not line_s:
                continue
            try:
                rec = json.loads(line_s)
                if str(rec.get("id", "")) != entry_id:
                    records.append(rec)
            except json.JSONDecodeError:
                continue
        records.append(record)
        sidecar.write_text(
            "\n".join(json.dumps(r) for r in records) + "\n",
            encoding="utf-8",
        )

    def warm_remove(self, entry_id: str) -> bool:
        """Delete an entry from the warm store and sidecar.

        Args:
            entry_id: Memory entry identifier to remove.
        """
        backend_present = False
        memory_removed = False
        vector_removed = False
        try:
            backend = self._get_warm_backend()
            if backend is not None:
                backend_present = True
                memory_removed = backend.delete(entry_id)
                vector_deleted = backend.delete_vector(entry_id)
                vector_removed = vector_deleted or vector_removed
        except (OSError, ValueError):
            logger.debug("warm_tier_db_remove_failed", entry_id=entry_id, exc_info=True)
            vector_removed = False
            memory_removed = False

        sidecar_removed = self.purge_sidecar_entry(entry_id)
        vector_still_present = False
        if backend_present:
            try:
                backend = self._get_warm_backend()
                if backend is not None:
                    vector_exists = getattr(backend, "vector_exists", None)
                    vector_still_present = bool(vector_exists(entry_id)) if callable(vector_exists) else False
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
        if sidecar.exists():
            lines = []
            for line in sidecar.read_text(encoding="utf-8").splitlines():
                line_s = line.strip()
                if not line_s:
                    continue
                try:
                    rec = json.loads(line_s)
                    if str(rec.get("id", "")) != entry_id:
                        lines.append(line_s)
                    else:
                        sidecar_removed = True
                except json.JSONDecodeError:
                    continue
            sidecar.write_text(
                "\n".join(lines) + "\n" if lines else "",
                encoding="utf-8",
            )
        return sidecar_removed

    def warm_search(
        self,
        query_tokens: list[str],
        query_embedding: list[float] | None,
        top_k: int = 25,
    ) -> list[dict[str, object]]:
        """Search the warm tier for relevant entries.

        Performs dense vector search when embedding is available; falls back
        to JSONL keyword search when embedding is None.

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
                    raw = backend.search_vectors(query_embedding, top_k=top_k)
                    if raw:
                        results: list[dict[str, object]] = []
                        for eid, dist in raw:
                            sidecar_entry = sidecar_entries.get(eid)
                            if sidecar_entry is None:
                                logger.debug("warm_tier_skip_orphaned_vector_hit", entry_id=eid)
                                continue
                            item = dict(sidecar_entry)
                            item["id"] = eid
                            item["_tier_relevance"] = float(1.0 - dist)
                            item["score"] = float(1.0 - dist)
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
        for line in sidecar.read_text(encoding="utf-8").splitlines():
            line_s = line.strip()
            if not line_s:
                continue
            try:
                rec = json.loads(line_s)
            except json.JSONDecodeError:
                continue
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
        entries: dict[str, dict[str, object]] = {}
        sidecar = self._warm_sidecar_path()
        if not sidecar.exists():
            return entries

        for line in sidecar.read_text(encoding="utf-8").splitlines():
            line_s = line.strip()
            if not line_s:
                continue
            try:
                rec = json.loads(line_s)
            except json.JSONDecodeError:
                continue
            if not isinstance(rec, dict):
                continue

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
