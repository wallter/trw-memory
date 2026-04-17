"""E2E tests for trw-memory package.

Covers the full memory lifecycle: CRUD, decay scoring, hybrid retrieval,
three-tier lifecycle, security (PII, encryption, audit), validation edge
cases, and SQLite backend resilience.

Based on: docs/testing/E2E-MEMORY-MANAGEMENT.md
"""

from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Re-use the canonical make_entry factory from conftest.
from conftest import make_entry, make_entry_dict

from trw_memory.client import MemoryClient
from trw_memory.exceptions import (
    DimensionMismatchError,
    MemoryNotFoundError,
)
from trw_memory.models.memory import MemoryEntry
from trw_memory.storage.sqlite_backend import SQLiteBackend

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> MemoryClient:
    """Isolated MemoryClient backed by SQLite in tmp_path."""
    monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "e2e_storage"))
    monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "sqlite")
    return MemoryClient(namespace="default", mode="local")


@pytest.fixture()
def client_ns_a(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> MemoryClient:
    """MemoryClient in namespace 'project:ns-a'."""
    monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "e2e_storage"))
    monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "sqlite")
    return MemoryClient(namespace="project:ns-a", mode="local")


@pytest.fixture()
def client_ns_b(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> MemoryClient:
    """MemoryClient in namespace 'project:ns-b'."""
    monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "e2e_storage"))
    monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "sqlite")
    return MemoryClient(namespace="project:ns-b", mode="local")


# ===================================================================
# 1. Store/Recall CRUD (5 tests)
# ===================================================================


class TestStoreCRUD:
    """Section 1 of E2E plan: core CRUD operations."""

    async def test_store_basic_entry(self, client: MemoryClient) -> None:
        """1.1 — Store a basic entry and verify the result dict shape."""
        result = await client.store(
            content="Always validate JWT tokens before accessing protected routes",
            tags=["security", "auth"],
            importance=0.8,
        )
        assert result["memory_id"].startswith("M-")
        assert result["status"] == "stored"
        assert result["namespace"] == "default"
        assert result["timestamp"]  # non-empty ISO string

    async def test_recall_basic_search(self, client: MemoryClient) -> None:
        """1.5 — Recall by keyword query returns relevant results."""
        await client.store(
            content="JWT tokens expire after 30 minutes",
            tags=["auth"],
            importance=0.8,
        )
        await client.store(
            content="Database indices improve query speed",
            tags=["perf"],
            importance=0.6,
        )
        results = await client.recall(query="authentication tokens")
        # At least one result should be returned
        assert len(results) >= 1
        # Results are ordered by score descending
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    async def test_recall_tag_filtering(self, client: MemoryClient) -> None:
        """1.6 — Recall with tag filter returns only matching entries."""
        await client.store(content="Auth entry about JWT", tags=["auth"], importance=0.7)
        await client.store(content="Perf entry about caching", tags=["perf"], importance=0.6)
        results = await client.recall(query="entry", tags=["auth"])
        assert len(results) >= 1
        for r in results:
            assert "auth" in r["tags"]

    async def test_forget_valid_entry(self, client: MemoryClient) -> None:
        """1.9 — Forget an existing entry successfully."""
        r = await client.store(content="temporary note", importance=0.3)
        memory_id = r["memory_id"]
        result = await client.forget(memory_id=memory_id)
        assert result["status"] == "deleted"
        assert result["memory_id"] == memory_id
        # Subsequent recall should not return this entry
        results = await client.recall(query="temporary note")
        found_ids = [res["memory_id"] for res in results]
        assert memory_id not in found_ids

    async def test_forget_wrong_namespace_raises(
        self,
        client_ns_a: MemoryClient,
        client_ns_b: MemoryClient,
    ) -> None:
        """1.11 — Forgetting an entry from the wrong namespace raises."""
        r = await client_ns_a.store(content="namespace-a data", importance=0.5)
        with pytest.raises(MemoryNotFoundError):
            await client_ns_b.forget(memory_id=r["memory_id"])


# ===================================================================
# 2. Decay Scoring (3 tests)
# ===================================================================


class TestDecayScoring:
    """Section 2 of E2E plan: time decay, Q-learning, composite utility."""

    def test_time_decay_floor_at_03(self) -> None:
        """2.1 — apply_time_decay never goes below 0.3 floor.

        Formula: decay_factor = max(0.3, 1.0 - (days/365) * 0.3)
        Floor of 0.3 kicks in at ~852 days.
        """
        from trw_memory.lifecycle.scoring import apply_time_decay

        # Fresh entry (just created) — full impact
        now = datetime.now(timezone.utc)
        decay_fresh = apply_time_decay(1.0, now)
        assert decay_fresh >= 0.95  # essentially no decay

        # 365 days old — decay_factor = max(0.3, 1.0 - 0.3) = 0.7
        year_old = now - timedelta(days=365)
        decay_year = apply_time_decay(1.0, year_old)
        assert decay_year >= 0.3
        assert 0.65 <= decay_year <= 0.75

        # Very old entry (1000 days) — hits the 0.3 floor
        very_old = now - timedelta(days=1000)
        decay_old = apply_time_decay(1.0, very_old)
        assert decay_old >= 0.3
        # Floor clamps: 1.0 - (1000/365)*0.3 = negative, so floor = 0.3
        assert decay_old == pytest.approx(0.3, abs=0.01)

    def test_q_learning_convergence(self) -> None:
        """2.2 — Q-learning updates converge toward target reward."""
        from trw_memory.lifecycle.scoring import update_q_value

        # Cold start, positive reward
        q_new = update_q_value(q_old=0.5, reward=1.0, alpha=0.15)
        assert q_new == pytest.approx(0.575, abs=0.01)

        # Negative reward
        q_neg = update_q_value(q_old=0.5, reward=0.0, alpha=0.15)
        assert q_neg == pytest.approx(0.425, abs=0.01)

        # Convergence over many iterations
        q = 0.5
        for _ in range(50):
            q = update_q_value(q, reward=0.8, alpha=0.15)
        assert abs(q - 0.8) < 0.05

    def test_composite_utility_score_ordering(self) -> None:
        """2.3 — High-impact recent entries score higher than low-impact old ones."""
        from trw_memory.lifecycle.scoring import compute_utility_score

        score_high = compute_utility_score(
            q_value=0.85,
            days_since_last_access=1,
            recurrence_count=10,
            base_impact=0.9,
            q_observations=10,
            access_count=10,
            source_type="human",
            half_life_days=14.0,
        )

        score_low = compute_utility_score(
            q_value=0.1,
            days_since_last_access=300,
            recurrence_count=1,
            base_impact=0.2,
            q_observations=2,
            access_count=0,
            source_type="agent",
            half_life_days=14.0,
        )

        assert score_high > score_low
        assert score_high > 0.5
        assert score_low < 0.3


# ===================================================================
# 3. Hybrid Retrieval (3 tests)
# ===================================================================


class TestHybridRetrieval:
    """Section 3 of E2E plan: BM25, dense, hybrid pipeline."""

    def test_bm25_keyword_search(self) -> None:
        """3.1 — BM25 ranks entries with query keyword overlap highest."""
        pytest.importorskip("rank_bm25")
        from trw_memory.retrieval.bm25 import bm25_search

        entries = [
            make_entry(
                entry_id="e1",
                content="pydantic v2 model validation with ConfigDict",
                tags=["pydantic"],
            ),
            make_entry(
                entry_id="e2",
                content="SQLAlchemy ORM session management",
                tags=["sqlalchemy"],
            ),
            make_entry(
                entry_id="e3",
                content="pydantic field validators and custom types",
                tags=["pydantic"],
            ),
        ]
        results = bm25_search("pydantic validation", entries, top_k=3)
        # Both pydantic entries should rank above SQLAlchemy
        result_ids = [eid for eid, _ in results]
        assert "e1" in result_ids
        assert "e3" in result_ids
        # SQLAlchemy entry should not appear or be ranked last
        if "e2" in result_ids:
            e2_pos = result_ids.index("e2")
            assert e2_pos == len(result_ids) - 1

    def test_dense_cosine_similarity(self) -> None:
        """3.4 — Dense search returns results ordered by cosine similarity."""
        from trw_memory.retrieval.dense import cosine_similarity

        # Test cosine similarity directly
        a = [1.0, 0.0, 0.0]
        b = [1.0, 0.0, 0.0]
        assert cosine_similarity(a, b) == pytest.approx(1.0)

        # Orthogonal vectors
        c = [0.0, 1.0, 0.0]
        assert cosine_similarity(a, c) == pytest.approx(0.0)

        # Dimension mismatch raises
        with pytest.raises(DimensionMismatchError, match="3 vs 2"):
            cosine_similarity([1.0, 0.0, 0.0], [1.0, 0.0])

    def test_hybrid_pipeline_bm25_only(self) -> None:
        """3.9 — Hybrid pipeline degrades to BM25-only when embedder is None."""
        pytest.importorskip("rank_bm25")
        from trw_memory.retrieval.pipeline import hybrid_search

        entries = [
            make_entry(entry_id="e1", content="pydantic validation patterns"),
            make_entry(entry_id="e2", content="react component lifecycle hooks"),
        ]
        results = hybrid_search(
            query="pydantic validation",
            entries=entries,
            embedder=None,
            top_k=5,
        )
        # Should return results via BM25 fallback
        assert len(results) >= 1
        # Pydantic entry should be ranked first
        assert results[0].id == "e1"


# ===================================================================
# 4. Three-Tier Lifecycle (2 tests)
# ===================================================================


class TestThreeTierLifecycle:
    """Section 5 of E2E plan: hot tier LRU, sweep transitions."""

    def test_prune_candidates_tier_classification(self) -> None:
        """2.6 — utility_based_prune_candidates classifies entries into tiers."""
        from trw_memory.lifecycle.scoring import utility_based_prune_candidates

        now = datetime.now(timezone.utc)

        entries = [
            # Tier 1: already resolved
            make_entry_dict(
                entry_id="resolved-1",
                content="resolved entry",
                importance=0.5,
                status="resolved",
                created_at=now - timedelta(days=10),
            ),
            # Tier 2: very low utility, old
            make_entry_dict(
                entry_id="low-util-1",
                content="very low utility",
                importance=0.01,
                status="active",
                created_at=now - timedelta(days=200),
                last_accessed_at=now - timedelta(days=200),
            ),
            # Should NOT be pruned: high importance, recent
            make_entry_dict(
                entry_id="high-imp-1",
                content="high importance recent",
                importance=0.9,
                status="active",
                created_at=now - timedelta(days=2),
            ),
        ]

        candidates = utility_based_prune_candidates(entries)
        candidate_ids = {str(c["id"]) for c in candidates}

        # Resolved entry should be a candidate
        assert "resolved-1" in candidate_ids
        # Very low utility should be a candidate
        assert "low-util-1" in candidate_ids
        # High importance entry should NOT be a candidate
        assert "high-imp-1" not in candidate_ids

    def test_sweep_resolved_entries_are_candidates(self) -> None:
        """2.7 — Resolved entries are always prune candidates regardless of age."""
        from trw_memory.lifecycle.scoring import utility_based_prune_candidates

        now = datetime.now(timezone.utc)
        entries = [
            make_entry_dict(
                entry_id="resolved-recent",
                content="just resolved",
                importance=0.9,
                status="resolved",
                created_at=now - timedelta(days=1),
            ),
        ]
        candidates = utility_based_prune_candidates(entries)
        assert len(candidates) == 1
        assert candidates[0]["id"] == "resolved-recent"


# ===================================================================
# 5. Security (3 tests)
# ===================================================================


class TestSecurity:
    """Section 7 of E2E plan: PII detection, encryption, audit."""

    def test_pii_detection_block_mode(self) -> None:
        """7.4 — PII detector in block mode raises on email address."""
        from trw_memory.exceptions import MemoryError as TrwMemoryError
        from trw_memory.security.pii import PIIAction, check_entry_pii

        entry = make_entry(
            content="Contact john@example.com for details",
            entry_id="pii-test-1",
        )
        with pytest.raises(TrwMemoryError, match="PII detected"):
            check_entry_pii(entry, action=PIIAction.BLOCK)

    def test_field_encryption_roundtrip(self) -> None:
        """7.9 — Encrypt then decrypt entry fields preserves content."""
        from trw_memory.security.encryption import (
            decrypt_entry_fields,
            derive_namespace_key,
            derive_namespace_key_bytes,
            encrypt_entry_fields,
            generate_master_key,
        )

        master_key = generate_master_key()
        assert len(derive_namespace_key(master_key, "test-ns")) == 64
        ns_key = derive_namespace_key_bytes(master_key, "test-ns")

        entry = make_entry(
            entry_id="enc-test-1",
            content="sensitive data",
            detail="very secret details",
        )

        encrypted = encrypt_entry_fields(entry, ns_key)
        # Encrypted content should differ from plaintext
        assert encrypted.content != "sensitive data"
        assert encrypted.detail != "very secret details"

        decrypted = decrypt_entry_fields(encrypted, ns_key)
        # Decrypted content should match original
        assert decrypted.content == "sensitive data"
        assert decrypted.detail == "very secret details"

    def test_audit_logging_records_operations(self, tmp_path: Path) -> None:
        """7.11 — Audit log records store/recall/delete events with hash chain."""
        from trw_memory.security.audit import AuditLog

        log_path = tmp_path / "audit.jsonl"
        audit = AuditLog(log_path)

        # Record three operations
        r1 = audit.append(action="store", target_id="M-001", namespace="test")
        r2 = audit.append(action="recall", target_id="", namespace="test")
        r3 = audit.append(action="delete", target_id="M-001", namespace="test")

        # Read all records
        records = audit.read_all()
        assert len(records) == 3
        # AuditRecord stores the action in the `op` field (short for operation).
        assert records[0].op == "store"
        assert records[1].op == "recall"
        assert records[2].op == "delete"

        # Verify hash chain integrity — verify_chain() returns a dict
        # with `valid`, `entries_checked`, `first_broken_at`, and
        # `broken_hash` keys (see security/audit.py).
        result = audit.verify_chain()
        assert result["valid"] is True
        assert result["entries_checked"] == 3
        assert result["first_broken_at"] is None

        # Second record should chain from first — the hash-chain field is `hash`.
        assert records[1].prev_hash == records[0].hash
        assert records[2].prev_hash == records[1].hash


# ===================================================================
# 6. Validation Edge Cases (3 tests)
# ===================================================================


class TestValidationEdgeCases:
    """Section 1.4 + 10 of E2E plan: input validation."""

    async def test_empty_content_raises_value_error(self, client: MemoryClient) -> None:
        """1.4 — Empty content string is rejected by schema validation."""
        from trw_memory.exceptions import SchemaValidationError

        with pytest.raises((ValueError, SchemaValidationError)):
            await client.store(content="", importance=0.5)

    async def test_importance_out_of_range_raises(self, client: MemoryClient) -> None:
        """1.4 — Importance outside [0,1] is rejected by schema validation."""
        from trw_memory.exceptions import SchemaValidationError

        with pytest.raises((ValueError, SchemaValidationError)):
            await client.store(content="test", importance=1.5)

        with pytest.raises((ValueError, SchemaValidationError)):
            await client.store(content="test", importance=-0.1)

    async def test_recall_limit_below_one_raises(self, client: MemoryClient) -> None:
        """Recall with limit < 1 raises ValueError."""
        with pytest.raises(ValueError, match="limit must be >= 1"):
            await client.recall(query="test", limit=0)


# ===================================================================
# 7. SQLite Backend (2 tests)
# ===================================================================


class TestSQLiteBackend:
    """Section 8 of E2E plan: concurrent writes, graceful degradation."""

    def test_concurrent_writes_with_wal(self, tmp_path: Path) -> None:
        """8.1 — Multiple threads writing concurrently succeed under WAL mode."""
        db_path = tmp_path / "concurrent.db"
        backend = SQLiteBackend(db_path)
        errors: list[Exception] = []
        barrier = threading.Barrier(4)

        def writer(prefix: str) -> None:
            try:
                barrier.wait(timeout=5)
                for i in range(25):
                    now = datetime.now(timezone.utc)
                    entry = MemoryEntry(
                        id=f"M-{prefix}-{i:03d}",
                        content=f"{prefix} entry {i}",
                        namespace="default",
                        created_at=now,
                        updated_at=now,
                    )
                    backend.store(entry)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=writer, args=(f"w{i}",)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        backend.close()
        assert errors == [], f"Concurrent write errors: {errors}"

        # Verify all 100 entries were stored
        verify_backend = SQLiteBackend(db_path)
        all_entries = verify_backend.list_entries(namespace="default", limit=200)
        verify_backend.close()
        assert len(all_entries) == 100

    def test_graceful_degradation_without_sqlite_vec(self, tmp_path: Path) -> None:
        """8.3 — SQLiteBackend works without sqlite-vec for metadata operations."""
        db_path = tmp_path / "no_vec.db"
        backend = SQLiteBackend(db_path)

        # Store and retrieve without vectors
        now = datetime.now(timezone.utc)
        entry = make_entry(
            entry_id="no-vec-1",
            content="test without vectors",
        )
        backend.store(entry)

        # Get by ID should work
        retrieved = backend.get("no-vec-1")
        assert retrieved is not None
        assert retrieved.content == "test without vectors"

        # Search should work (LIKE-based, no vector search)
        results = backend.search("test without", top_k=10, namespace="default")
        assert len(results) >= 1
        assert results[0].id == "no-vec-1"

        backend.close()


# ===================================================================
# Additional E2E scenarios
# ===================================================================


class TestRecallEmptyResults:
    """Verify recall handles queries with no matches gracefully."""

    async def test_recall_no_matches_returns_empty(self, client: MemoryClient) -> None:
        """1.8 — Recall with no matching entries returns empty list."""
        results = await client.recall(query="quantum_computing_patterns_xyz_nonexistent")
        assert results == []
        # No error raised


class TestSyncE2E:
    """Verify MemoryClient wires the package sync surface end-to-end."""

    @staticmethod
    def _mock_httpx_client(
        mock_client_cls: MagicMock,
        *,
        status_code: int,
        json_data: object | None = None,
    ) -> MagicMock:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = status_code
        mock_response.json.return_value = json_data if json_data is not None else []
        mock_client.post.return_value = mock_response
        mock_client.__enter__.return_value = mock_client
        mock_client.__exit__.return_value = False
        mock_client_cls.return_value = mock_client
        return mock_client

    async def test_store_sync_success_marks_entry_published(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "e2e_sync"))
        monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "sqlite")
        monkeypatch.setenv("MEMORY_SYNC_ENABLED", "true")
        monkeypatch.setenv("MEMORY_LOCAL_ONLY", "false")
        monkeypatch.setenv("MEMORY_PLATFORM_URL", "https://api.test.com")

        client = MemoryClient(namespace="default", mode="local")
        with patch("trw_memory.sync.remote.httpx.Client") as mock_client_cls:
            self._mock_httpx_client(mock_client_cls, status_code=200)
            stored = await client.store("syncable entry", importance=0.9)
            await client.close()

        reopened = MemoryClient(namespace="default", mode="local")
        entry = reopened._get_backend().get(stored["memory_id"])
        assert entry is not None
        assert entry.published_to_platform is True
        await reopened.close()

    async def test_recall_include_shared_merges_remote_results(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("MEMORY_STORAGE_PATH", str(tmp_path / "e2e_sync"))
        monkeypatch.setenv("MEMORY_STORAGE_BACKEND", "sqlite")
        monkeypatch.setenv("MEMORY_SYNC_ENABLED", "true")
        monkeypatch.setenv("MEMORY_LOCAL_ONLY", "false")
        monkeypatch.setenv("MEMORY_PLATFORM_URL", "https://api.test.com")

        client = MemoryClient(namespace="default", mode="local")
        await client.store("local entry", importance=0.8)

        with patch("trw_memory.sync.remote.httpx.Client") as mock_client_cls:
            self._mock_httpx_client(
                mock_client_cls,
                status_code=200,
                json_data=[{"summary": "remote shared entry", "impact": 0.6}],
            )
            results = await client.recall("entry", include_shared=True)

        assert any(result["source"] == "shared" for result in results)
        assert results[0]["source"] == "local"
        await client.close()


class TestForgetNonExistent:
    """Verify forget raises MemoryNotFoundError for missing entries."""

    async def test_forget_nonexistent_raises(self, client: MemoryClient) -> None:
        """1.10 — Forgetting a non-existent entry raises MemoryNotFoundError."""
        with pytest.raises(MemoryNotFoundError):
            await client.forget(memory_id="nonexistent-id-xyz")


class TestRRFFusion:
    """Verify the RRF fusion function directly."""

    def test_rrf_fuse_combines_rankings(self) -> None:
        """3.7 — RRF fusion combines BM25 and dense rankings."""
        from trw_memory.retrieval.fusion import rrf_fuse

        bm25_ranking = [("e1", 5.0), ("e2", 3.0), ("e3", 1.0)]
        dense_ranking = [("e2", 0.95), ("e1", 0.80), ("e4", 0.70)]

        fused = rrf_fuse([bm25_ranking, dense_ranking], k=60)
        fused_ids = [eid for eid, _ in fused]

        # e1 and e2 appear in both rankings, should be near the top
        assert "e1" in fused_ids[:3]
        assert "e2" in fused_ids[:3]
        # All entries from both rankings should appear
        assert set(fused_ids) == {"e1", "e2", "e3", "e4"}

    def test_rrf_fuse_empty_input(self) -> None:
        """3.10 — RRF fusion with empty rankings returns empty."""
        from trw_memory.retrieval.fusion import rrf_fuse

        assert rrf_fuse([]) == []
        assert rrf_fuse([[]]) == []


class TestConfigValidation:
    """Config edge cases from section 10 of the E2E plan."""

    def test_score_weights_must_sum_to_one(self) -> None:
        """10.1 — Score weights summing to != 1.0 raises validation error."""
        from pydantic import ValidationError

        from trw_memory.models.config import MemoryConfig

        with pytest.raises(ValidationError, match="sum to 1.0"):
            MemoryConfig(
                score_relevance_weight=0.5,
                score_recency_weight=0.5,
                score_importance_weight=0.5,
            )

    def test_decay_half_life_must_be_positive(self) -> None:
        """10.2 — Negative or zero decay_half_life_days raises validation error."""
        from pydantic import ValidationError

        from trw_memory.models.config import MemoryConfig

        with pytest.raises(ValidationError):
            MemoryConfig(
                decay_half_life_days=-1.0,
            )


class TestPIIRedactMode:
    """Additional PII scenarios from section 7.5."""

    def test_pii_redact_mode_masks_content(self) -> None:
        """7.5 — PII detector in redact mode masks email in content."""
        from trw_memory.security.pii import PIIAction, check_entry_pii

        entry = make_entry(
            content="Contact john@example.com for support",
            entry_id="pii-redact-1",
        )
        updated, matches = check_entry_pii(entry, action=PIIAction.REDACT)
        assert len(matches) > 0
        assert "john@example.com" not in updated.content
        assert "[REDACTED:" in updated.content


class TestHumanSourceBoost:
    """Decay scoring: human source entries get utility boost."""

    def test_human_source_scores_higher(self) -> None:
        """2.5 — Human-sourced entries get +0.1 utility boost."""
        from trw_memory.lifecycle.scoring import compute_utility_score

        score_human = compute_utility_score(
            q_value=0.5,
            days_since_last_access=30,
            recurrence_count=1,
            base_impact=0.5,
            q_observations=3,
            access_count=1,
            source_type="human",
            half_life_days=14.0,
        )

        score_agent = compute_utility_score(
            q_value=0.5,
            days_since_last_access=30,
            recurrence_count=1,
            base_impact=0.5,
            q_observations=3,
            access_count=1,
            source_type="agent",
            half_life_days=14.0,
        )

        assert score_human > score_agent
