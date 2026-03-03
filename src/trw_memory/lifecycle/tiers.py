"""Tiered memory storage: Hot (LRU) / Warm (sqlite-vec sidecar) / Cold (YAML archive).

Implements lifecycle management for memory entries with automatic tier
transitions based on recency and importance scores.

Tier definitions:
- Hot: in-memory LRU cache (OrderedDict, O(1) ops)
- Warm: sqlite-vec backed persistent index or JSONL keyword sidecar
- Cold: YAML archive partitioned by {YYYY}/{MM}/
"""

from __future__ import annotations

import contextlib
import json
import math
from collections import OrderedDict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import NamedTuple, cast

import structlog

from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.retrieval.dense import cosine_similarity
from trw_memory.storage.persistence import read_yaml, write_yaml

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


class TierSweepResult(NamedTuple):
    """Outcome of a single sweep() pass across all tiers.

    Attributes:
        promoted: Entries moved up a tier (Cold→Warm).
        demoted: Entries moved down a tier (Hot→Warm, Warm→Cold).
        purged: Entries deleted from Cold tier (retention expired).
        errors: Per-entry failures that were logged and skipped.
    """

    promoted: int
    demoted: int
    purged: int
    errors: int

    @property
    def total(self) -> int:
        """Total number of entries affected by this sweep."""
        return self.promoted + self.demoted + self.purged + self.errors


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _days_since_access(
    entry: dict[str, object],
    today: date,
    fallback_days: int = 30,
) -> int:
    """Compute days since last access from an entry dict.

    Resolution order: last_accessed_at -> created_at -> fallback_days.

    Handles both datetime objects (from model_dump) and ISO strings (from YAML).

    Args:
        entry: Entry data dict (from YAML or model_dump).
        today: Reference date for computing delta.
        fallback_days: Days to return when no date is parseable.

    Returns:
        Number of days since the entry was last accessed.
    """
    for field in ("last_accessed_at", "created_at", "created"):
        val = entry.get(field)
        if val is None:
            continue
        # datetime object (from model_dump)
        if isinstance(val, datetime):
            return max(0, (today - val.date()).days)
        if isinstance(val, date) and not isinstance(val, datetime):
            return max(0, (today - val).days)
        # String (from YAML)
        raw = str(val)
        if not raw or raw == "None":
            continue
        try:
            if "T" in raw or " " in raw:
                dt = datetime.fromisoformat(raw)
                return max(0, (today - dt.date()).days)
            return max(0, (today - date.fromisoformat(raw)).days)
        except (ValueError, TypeError):
            continue
    return fallback_days


# ---------------------------------------------------------------------------
# Importance scoring
# ---------------------------------------------------------------------------


def compute_importance_score(
    entry: dict[str, object],
    query_tokens: list[str],
    query_embedding: list[float] | None = None,
    entry_embedding: list[float] | None = None,
    *,
    config: MemoryConfig | None = None,
) -> float:
    """Compute a composite importance score for a memory entry.

    Formula: score = w1*relevance + w2*recency + w3*importance

    Weights are normalized if they don't sum to 1.0.

    Args:
        entry: Memory entry as a dict (from YAML or model_dump).
        query_tokens: Tokenized query for token-overlap fallback.
        query_embedding: Optional dense query vector for cosine similarity.
        entry_embedding: Optional dense entry vector for cosine similarity.
        config: MemoryConfig for weights and decay settings.

    Returns:
        Composite importance score in [0.0, 1.0].
    """
    cfg = config or MemoryConfig()

    w1 = cfg.score_relevance_weight
    w2 = cfg.score_recency_weight
    w3 = cfg.score_importance_weight

    # Normalize weights if they don't sum to 1.0
    total_w = w1 + w2 + w3
    if total_w > 0 and abs(total_w - 1.0) > 1e-9:
        w1 /= total_w
        w2 /= total_w
        w3 /= total_w

    # Relevance: cosine similarity when both embeddings present, else token overlap
    if query_embedding is not None and entry_embedding is not None:
        relevance = max(0.0, cosine_similarity(query_embedding, entry_embedding))
    else:
        # Token overlap ratio fallback
        entry_text = (
            str(entry.get("content", "")).lower()
            + " "
            + str(entry.get("detail", "")).lower()
        )
        entry_tokens = set(entry_text.split())
        query_set = {t.lower() for t in query_tokens}
        if query_set:
            relevance = len(query_set & entry_tokens) / len(query_set)
        else:
            relevance = 0.0

    # Recency: exponential decay based on days since access
    today = date.today()
    days = _days_since_access(entry, today)
    half_life = cfg.decay_half_life_days
    decay_rate = math.log(2) / half_life if half_life > 0 else 0.0
    recency = math.exp(-decay_rate * days)

    # Importance: the entry's importance field (was 'impact' in LearningEntry)
    # Support both 'importance' (MemoryEntry) and 'impact' (legacy LearningEntry)
    raw_importance = entry.get("importance", entry.get("impact", 0.5))
    importance = float(str(raw_importance))
    importance = max(0.0, min(1.0, importance))

    score = w1 * relevance + w2 * recency + w3 * importance
    return max(0.0, min(1.0, score))


# ---------------------------------------------------------------------------
# TierManager
# ---------------------------------------------------------------------------


class TierManager:
    """Hot/Warm/Cold tier manager for memory entry lifecycle.

    Hot tier: in-memory LRU cache (OrderedDict, O(1) ops).
    Warm tier: sqlite-vec backed persistent index with JSONL sidecar fallback.
    Cold tier: YAML archive partitioned by {YYYY}/{MM}/.

    Usage::

        mgr = TierManager(base_dir=Path(".memory"))
        entry = mgr.hot_get("some-id")
        mgr.hot_put("some-id", memory_entry)
        result = mgr.sweep()
    """

    def __init__(
        self,
        base_dir: Path,
        config: MemoryConfig | None = None,
        entries_dir: Path | None = None,
    ) -> None:
        """Initialise TierManager.

        Args:
            base_dir: Base directory for memory storage.
            config: MemoryConfig for capacity/TTL settings.
            entries_dir: Optional explicit entries directory for sweep.
                         Defaults to base_dir / "entries".
        """
        self._base_dir = base_dir
        self._config = config or MemoryConfig()
        self._entries_dir: Path = entries_dir or (base_dir / "entries")

        # Hot tier: OrderedDict used as LRU cache
        # LRU invariant: MRU at the end (rightmost), LRU at the front (leftmost)
        self._hot: OrderedDict[str, MemoryEntry] = OrderedDict()

    # -----------------------------------------------------------------------
    # Hot Tier
    # -----------------------------------------------------------------------

    def hot_get(self, entry_id: str) -> MemoryEntry | None:
        """Return a cached entry, moving it to MRU position on hit.

        Args:
            entry_id: Memory entry identifier.

        Returns:
            MemoryEntry if in cache, None otherwise.
        """
        if entry_id not in self._hot:
            return None
        self._hot.move_to_end(entry_id)
        return self._hot[entry_id]

    def hot_put(self, entry_id: str, entry: MemoryEntry) -> None:
        """Add or refresh an entry in the hot cache.

        Evicts the LRU entry when capacity is exceeded.

        Args:
            entry_id: Memory entry identifier.
            entry: MemoryEntry to cache.
        """
        cfg = self._config

        if entry_id in self._hot:
            self._hot.move_to_end(entry_id)
            self._hot[entry_id] = entry
            return

        self._hot[entry_id] = entry
        self._hot.move_to_end(entry_id)

        # Evict LRU if over capacity
        if len(self._hot) > cfg.hot_max_entries:
            evicted_id, _ = self._hot.popitem(last=False)
            logger.debug(
                "hot_tier_evict",
                evicted_id=evicted_id,
                capacity=cfg.hot_max_entries,
            )

    def hot_clear(self) -> None:
        """Evict all entries from the hot cache (for testing / shutdown)."""
        self._hot.clear()

    @property
    def hot_size(self) -> int:
        """Number of entries currently in the hot cache."""
        return len(self._hot)

    # -----------------------------------------------------------------------
    # Warm Tier
    # -----------------------------------------------------------------------

    def _warm_db_path(self) -> Path:
        """Resolve path to warm.db."""
        mem_dir = self._base_dir / "memory"
        mem_dir.mkdir(parents=True, exist_ok=True)
        return mem_dir / "warm.db"

    def _warm_sidecar_path(self) -> Path:
        """Path to the warm tier keyword-search sidecar (JSONL)."""
        return self._warm_db_path().with_suffix(".jsonl")

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
        try:
            from trw_memory.storage.sqlite_backend import SQLiteBackend

            _SQLITE_AVAILABLE = True
        except ImportError:
            _SQLITE_AVAILABLE = False

        if _SQLITE_AVAILABLE and embedding is not None:
            try:
                db_path = self._warm_db_path()
                backend = SQLiteBackend(db_path, dim=len(embedding))
                try:
                    backend.upsert_vector(entry_id, embedding)
                finally:
                    backend.close()
            except Exception:
                logger.debug("warm_tier_vec_upsert_failed", entry_id=entry_id, exc_info=True)

        # Always update sidecar for keyword search
        self._warm_sidecar_upsert(entry_id, entry_data)
        logger.debug("warm_tier_add", entry_id=entry_id, has_embedding=embedding is not None)

    def _warm_sidecar_upsert(
        self, entry_id: str, entry_data: dict[str, object]
    ) -> None:
        """Write entry metadata to the warm sidecar JSONL for keyword search."""
        sidecar = self._warm_sidecar_path()
        records: list[dict[str, object]] = []
        if sidecar.exists():
            for line in sidecar.read_text(encoding="utf-8").splitlines():
                line_s = line.strip()
                if not line_s:
                    continue
                try:
                    rec = json.loads(line_s)
                    if str(rec.get("id", "")) != entry_id:
                        records.append(rec)
                except json.JSONDecodeError:
                    continue

        # Use 'content' (MemoryEntry) or fall back to 'summary' (legacy)
        summary = str(
            entry_data.get("content", entry_data.get("summary", ""))
        )
        record: dict[str, object] = {
            "id": entry_id,
            "summary": summary,
            "tags": entry_data.get("tags", []),
        }
        records.append(record)
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_text(
            "\n".join(json.dumps(r) for r in records) + "\n",
            encoding="utf-8",
        )

    def warm_remove(self, entry_id: str) -> None:
        """Delete an entry from the warm store and sidecar.

        Args:
            entry_id: Memory entry identifier to remove.
        """
        try:
            from trw_memory.storage.sqlite_backend import SQLiteBackend

            db_path = self._warm_db_path()
            if db_path.exists():
                backend = SQLiteBackend(db_path)
                try:
                    backend.delete(entry_id)
                finally:
                    backend.close()
        except Exception:
            logger.debug("warm_tier_db_remove_failed", entry_id=entry_id, exc_info=True)

        # Purge from sidecar
        sidecar = self._warm_sidecar_path()
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
                except json.JSONDecodeError:
                    continue
            sidecar.write_text(
                "\n".join(lines) + "\n" if lines else "",
                encoding="utf-8",
            )

        logger.debug("warm_tier_remove", entry_id=entry_id)

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
            List of dicts with at minimum ``{"id": ..., "score": ...}``.
        """
        if not query_tokens and query_embedding is None:
            return []

        if query_embedding is not None:
            try:
                from trw_memory.storage.sqlite_backend import SQLiteBackend

                db_path = self._warm_db_path()
                if db_path.exists():
                    backend = SQLiteBackend(db_path, dim=len(query_embedding))
                    try:
                        raw = backend.search_vectors(query_embedding, top_k=top_k)
                    finally:
                        backend.close()
                    if raw:
                        return [
                            {"id": eid, "score": float(1.0 - dist)}
                            for eid, dist in raw
                        ]
            except Exception:
                logger.debug("warm_tier_vec_search_failed", exc_info=True)

        return self._warm_keyword_search(query_tokens, top_k)

    def _warm_keyword_search(
        self, query_tokens: list[str], top_k: int
    ) -> list[dict[str, object]]:
        """Search the warm sidecar JSONL for keyword matches."""
        sidecar = self._warm_sidecar_path()
        if not sidecar.exists() or not query_tokens:
            return []

        results: list[dict[str, object]] = []
        lower_tokens = {t.lower() for t in query_tokens}
        for line in sidecar.read_text(encoding="utf-8").splitlines():
            line_s = line.strip()
            if not line_s:
                continue
            try:
                rec = json.loads(line_s)
            except json.JSONDecodeError:
                continue
            text = str(rec.get("summary", "")).lower()
            tags = [str(t).lower() for t in cast("list[object]", rec.get("tags") or [])]
            text += " " + " ".join(tags)
            matched = sum(1 for tok in lower_tokens if tok in text)
            if matched > 0:
                score = matched / len(lower_tokens)
                results.append({"id": str(rec.get("id", "")), "score": score})

        results.sort(key=lambda r: float(str(r.get("score", 0))), reverse=True)
        return results[:top_k]

    # -----------------------------------------------------------------------
    # Cold Tier
    # -----------------------------------------------------------------------

    def _cold_dir(self) -> Path:
        """Base cold archive directory (base_dir/memory/cold/)."""
        return self._base_dir / "memory" / "cold"

    def _cold_partition(self, ts: datetime | None = None) -> Path:
        """Return cold partition directory for a given datetime.

        Args:
            ts: Datetime to use for partitioning. Defaults to now (UTC).

        Returns:
            Path like base_dir/memory/cold/2026/02/
        """
        if ts is None:
            ts = datetime.now(timezone.utc)
        return self._cold_dir() / str(ts.year) / f"{ts.month:02d}"

    def _assert_within_cold_dir(self, path: Path) -> None:
        """Guard against path traversal attacks on cold archive operations.

        Args:
            path: Path to validate.

        Raises:
            ValueError: If path is outside the cold archive directory.
        """
        cold_base = self._cold_dir().resolve()
        resolved = path.resolve()
        if not resolved.is_relative_to(cold_base):
            raise ValueError(
                f"Path traversal guard: {path} is not under cold dir {cold_base}"
            )

    def _assert_within_base_dir(self, path: Path) -> None:
        """Guard against path traversal attacks on any base dir operations.

        Source entry_path for cold_archive must be inside base_dir.

        Args:
            path: Path to validate.

        Raises:
            ValueError: If path is outside the base directory.
        """
        base = self._base_dir.resolve()
        resolved = path.resolve()
        if not resolved.is_relative_to(base):
            raise ValueError(
                f"Path traversal guard: {path} is not under base_dir {base}"
            )

    def cold_archive(self, entry_id: str, entry_path: Path) -> None:
        """Move a warm-tier YAML entry to the cold archive partition.

        Writes the entry to base_dir/memory/cold/{YYYY}/{MM}/{filename} atomically,
        then removes the original file.

        Args:
            entry_id: Memory entry identifier.
            entry_path: Absolute path to the source YAML file.

        Raises:
            ValueError: If entry_path is outside base_dir (path traversal guard).
            Exception: Re-raises any read/write failure.
        """
        # Path traversal guard
        self._assert_within_base_dir(entry_path)

        partition = self._cold_partition()
        partition.mkdir(parents=True, exist_ok=True)
        dest = partition / entry_path.name

        try:
            data = read_yaml(entry_path)
            write_yaml(dest, data)
            # Remove from warm sidecar (best-effort)
            with contextlib.suppress(Exception):
                self.warm_remove(entry_id)
            # Delete original
            entry_path.unlink(missing_ok=True)
            logger.debug("cold_archive", entry_id=entry_id, dest=str(dest))
        except Exception:
            logger.warning(
                "cold_archive_failed",
                entry_id=entry_id,
                src=str(entry_path),
                dest=str(dest),
                exc_info=True,
            )
            raise

    def cold_promote(self, entry_id: str) -> dict[str, object] | None:
        """Move a cold-tier entry back to warm tier on access.

        Locates the YAML in the cold archive by scanning for a file containing
        the entry_id, updates last_accessed_at, adds to warm tier, and removes
        the cold file.

        Args:
            entry_id: Memory entry identifier to promote.

        Returns:
            Entry data dict if found and promoted, None otherwise.
        """
        cold_base = self._cold_dir()
        if not cold_base.exists():
            return None

        for yaml_file in cold_base.rglob("*.yaml"):
            try:
                data = read_yaml(yaml_file)
            except Exception:
                continue
            if str(data.get("id", "")) != entry_id:
                continue

            # Found — update last_accessed_at and move to warm
            data["last_accessed_at"] = datetime.now(timezone.utc).isoformat()
            try:
                write_yaml(yaml_file, data)
                self.warm_add(entry_id, data, None)
                yaml_file.unlink(missing_ok=True)
                logger.debug("cold_promote", entry_id=entry_id, src=str(yaml_file))
                return data
            except Exception:
                logger.warning(
                    "cold_promote_failed",
                    entry_id=entry_id,
                    path=str(yaml_file),
                    exc_info=True,
                )
                return None

        return None

    def cold_search(self, query_tokens: list[str]) -> list[dict[str, object]]:
        """Linear scan of the cold archive for keyword matches.

        Args:
            query_tokens: Tokens to match (case-insensitive).

        Returns:
            List of matching entry dicts (includes all YAML fields).
        """
        cold_base = self._cold_dir()
        if not cold_base.exists() or not query_tokens:
            return []

        lower_tokens = {t.lower() for t in query_tokens}
        results: list[dict[str, object]] = []

        for yaml_file in sorted(cold_base.rglob("*.yaml")):
            try:
                data = read_yaml(yaml_file)
            except Exception:
                continue

            text = str(data.get("content", data.get("summary", ""))).lower()
            tags = [str(t).lower() for t in cast("list[object]", data.get("tags") or [])]
            text += " " + " ".join(tags)

            if any(tok in text for tok in lower_tokens):
                results.append(data)

        return results

    # -----------------------------------------------------------------------
    # Sweep
    # -----------------------------------------------------------------------

    def sweep(self) -> TierSweepResult:
        """Execute lifecycle sweep across all tiers.

        Performs three transition checks in order:
        1. Hot → Warm: entries whose last_accessed_at exceeds hot_ttl_days.
        2. Warm → Cold: entries idle > cold_threshold_days with importance < 0.22.
        3. Cold → Purge: entries idle > retention_days with importance < 0.1.

        All thresholds are read from config at call time.
        Per-entry failures are logged and counted in ``errors``.

        Returns:
            TierSweepResult with counts of promoted, demoted, purged, and errors.
        """
        cfg = self._config
        today = date.today()
        promoted = 0
        demoted = 0
        purged = 0
        errors = 0

        entries_dir = self._entries_dir
        purge_audit_path = self._base_dir / "memory" / "purge_audit.jsonl"

        # 1. Hot → Warm: evict stale hot entries
        stale_hot_ids: list[str] = []
        for entry_id, entry in list(self._hot.items()):
            entry_dict = entry.model_dump()
            days = _days_since_access(entry_dict, today)
            if days > cfg.hot_ttl_days:
                stale_hot_ids.append(entry_id)

        for entry_id in stale_hot_ids:
            try:
                evicted = self._hot.pop(entry_id)
                self.warm_add(entry_id, evicted.model_dump(), None)
                demoted += 1
                logger.debug("sweep_hot_to_warm", entry_id=entry_id)
            except Exception:
                logger.warning("sweep_hot_to_warm_failed", entry_id=entry_id, exc_info=True)
                errors += 1

        # 2. Warm → Cold: scan entries directory for idle low-importance entries
        if entries_dir.exists():
            for yaml_file in sorted(entries_dir.glob("*.yaml")):
                if yaml_file.name in ("index.yaml",):
                    continue
                try:
                    data = read_yaml(yaml_file)
                    entry_id = str(data.get("id", ""))
                    if not entry_id:
                        continue
                    # Skip non-active entries
                    status_val = str(data.get("status", "active"))
                    if status_val != "active":
                        continue
                    days = _days_since_access(data, today)
                    importance = compute_importance_score(data, [], config=cfg)
                    if days > cfg.cold_threshold_days and importance < 0.22:
                        self.cold_archive(entry_id, yaml_file)
                        demoted += 1
                        logger.debug(
                            "sweep_warm_to_cold",
                            entry_id=entry_id,
                            days=days,
                            importance_score=importance,
                        )
                except Exception:
                    logger.warning(
                        "sweep_warm_to_cold_failed",
                        path=str(yaml_file),
                        exc_info=True,
                    )
                    errors += 1

        # 3. Cold → Purge: scan cold archive for expired entries
        cold_base = self._cold_dir()
        if cold_base.exists():
            for yaml_file in sorted(cold_base.rglob("*.yaml")):
                try:
                    data = read_yaml(yaml_file)
                    entry_id = str(data.get("id", ""))
                    days = _days_since_access(data, today)
                    importance = compute_importance_score(data, [], config=cfg)
                    if days > cfg.retention_days and importance < 0.1:
                        # Append to purge audit log before deleting
                        audit_record: dict[str, object] = {
                            "entry_id": entry_id,
                            "purged_at": datetime.now(timezone.utc).isoformat(),
                            "days_idle": days,
                            "importance_score": importance,
                            "importance": float(
                                str(data.get("importance", data.get("impact", 0.5)))
                            ),
                            "content": str(data.get("content", data.get("summary", ""))),
                        }
                        purge_audit_path.parent.mkdir(parents=True, exist_ok=True)
                        with purge_audit_path.open("a", encoding="utf-8") as fh:
                            fh.write(json.dumps(audit_record) + "\n")
                        yaml_file.unlink(missing_ok=True)
                        purged += 1
                        logger.debug(
                            "sweep_cold_purge",
                            entry_id=entry_id,
                            days=days,
                            importance_score=importance,
                        )
                except Exception:
                    logger.warning(
                        "sweep_cold_purge_failed",
                        path=str(yaml_file),
                        exc_info=True,
                    )
                    errors += 1

        logger.info(
            "tier_sweep_complete",
            promoted=promoted,
            demoted=demoted,
            purged=purged,
            errors=errors,
        )
        return TierSweepResult(
            promoted=promoted,
            demoted=demoted,
            purged=purged,
            errors=errors,
        )
