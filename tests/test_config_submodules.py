"""Behavior tests for the MemoryConfig field-group mixin modules.

These modules define the field groups composed by ``MemoryConfig``. They carry
Pydantic ``Field`` constraints and ``Literal`` enumerations. The focused models
below exercise each group independently; integration tests also prove that the
public settings model inherits every field instead of redeclaring a stale copy.

The tests verify *behavior* — that valid values are accepted, that invalid
values (negative limits, out-of-range fractions, unknown enum members) are
rejected, and that defaults have sensible types — not merely that the classes
exist.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from trw_memory.models._config_daemon import _DaemonConfigMixin
from trw_memory.models._config_lifecycle import _LifecycleConfigMixin
from trw_memory.models._config_retrieval import _RetrievalConfigMixin
from trw_memory.models._config_security import _SecurityConfigMixin
from trw_memory.models._config_storage import _StorageConfigMixin
from trw_memory.models.config import MemoryConfig


# ---------------------------------------------------------------------------
# Concrete models composing each mixin so Pydantic field validation runs.
# Each mixin only declares fields (no methods), so a plain BaseModel subclass
# picks up every annotated Field with its constraints.
# ---------------------------------------------------------------------------
class _LifecycleModel(_LifecycleConfigMixin, BaseModel):
    pass


class _RetrievalModel(_RetrievalConfigMixin, BaseModel):
    pass


class _SecurityModel(_SecurityConfigMixin, BaseModel):
    pass


class _StorageModel(_StorageConfigMixin, BaseModel):
    pass


class _DaemonModel(_DaemonConfigMixin, BaseModel):
    """PRD-CORE-253 FR03/FR01: the loopback daemon's port, idle window, startup
    deadline and single-store path."""


def test_memory_config_composes_every_mixin_field_once() -> None:
    mixin_fields: set[str] = set()
    for mixin in (_DaemonModel, _LifecycleModel, _RetrievalModel, _SecurityModel, _StorageModel):
        mixin_fields.update(mixin.model_fields)

    assert mixin_fields == set(MemoryConfig.model_fields)
    # Read the class's OWN annotations, not the attribute. ``MemoryConfig``
    # declares no fields in its body, and ``cls.__annotations__`` is both
    # inherited-through and writable — ``unittest.mock.patch("...MemoryConfig")``
    # (test_cli_maintenance.py, and this package's own server tests) leaves the
    # security mixin's annotations visible there, which made this assertion fail
    # depending on which OTHER test file ran first. ``__dict__`` asks the
    # question the assertion actually means: does MemoryConfig declare a field of
    # its own instead of composing one from a mixin?
    own_annotations = set(MemoryConfig.__dict__.get("__annotations__", {}))
    assert not (own_annotations & mixin_fields)


def test_recent_fields_live_in_their_owning_mixins() -> None:
    cfg = MemoryConfig()
    assert cfg.hype_enabled is False
    assert cfg.hype_questions_per_entry == 3
    assert cfg.hype_min_question_chars == 8
    assert cfg.cold_search_cache_max == 1000
    assert cfg.lifecycle_use_fsrs is False


# ===========================================================================
# Lifecycle config mixin
# ===========================================================================
class TestLifecycleConfig:
    def test_defaults_are_sensible(self) -> None:
        cfg = _LifecycleModel()
        # ints are ints, not None
        assert isinstance(cfg.hot_max_entries, int)
        assert cfg.hot_max_entries == 50
        assert isinstance(cfg.hot_ttl_days, int)
        assert cfg.hot_ttl_days == 7
        assert cfg.cold_threshold_days == 90
        assert cfg.retention_days == 365
        # bools default sensibly
        assert cfg.dedup_enabled is True
        assert cfg.consolidation_enabled is True
        assert cfg.audit_enabled is True
        # PII action default literal
        assert cfg.pii_action == "warn"
        # float thresholds within [0, 1]
        assert 0.0 <= cfg.dedup_skip_threshold <= 1.0
        assert 0.0 <= cfg.dedup_merge_threshold <= 1.0

    def test_valid_values_accepted(self) -> None:
        cfg = _LifecycleModel(
            dedup_skip_threshold=0.99,
            dedup_merge_threshold=0.5,
            hot_max_entries=200,
            consolidation_min_cluster=4,
            pii_action="redact",
        )
        assert cfg.dedup_skip_threshold == 0.99
        assert cfg.hot_max_entries == 200
        assert cfg.consolidation_min_cluster == 4
        assert cfg.pii_action == "redact"

    def test_similarity_threshold_above_one_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _LifecycleModel(dedup_skip_threshold=1.5)

    def test_similarity_threshold_below_zero_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _LifecycleModel(dedup_merge_threshold=-0.1)

    def test_hot_max_entries_must_be_positive(self) -> None:
        # gt=0 — zero and negatives rejected
        with pytest.raises(ValidationError):
            _LifecycleModel(hot_max_entries=0)
        with pytest.raises(ValidationError):
            _LifecycleModel(hot_max_entries=-5)

    def test_ttl_days_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            _LifecycleModel(hot_ttl_days=0)
        with pytest.raises(ValidationError):
            _LifecycleModel(retention_days=-1)

    def test_decay_half_life_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            _LifecycleModel(decay_half_life_days=0.0)

    def test_consolidation_min_cluster_floor(self) -> None:
        # ge=2 — a cluster of 1 makes no sense
        with pytest.raises(ValidationError):
            _LifecycleModel(consolidation_min_cluster=1)
        # exactly 2 is allowed (boundary)
        assert _LifecycleModel(consolidation_min_cluster=2).consolidation_min_cluster == 2

    def test_unknown_pii_action_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _LifecycleModel(pii_action="delete")

    def test_impact_tier_caps_are_fractions(self) -> None:
        assert _LifecycleModel(impact_tier_critical_cap=0.0).impact_tier_critical_cap == 0.0
        assert _LifecycleModel(impact_tier_high_cap=1.0).impact_tier_high_cap == 1.0
        with pytest.raises(ValidationError):
            _LifecycleModel(impact_tier_critical_cap=1.01)

    def test_score_weights_are_fractions(self) -> None:
        with pytest.raises(ValidationError):
            _LifecycleModel(score_relevance_weight=2.0)
        with pytest.raises(ValidationError):
            _LifecycleModel(score_recency_weight=-0.5)

    def test_pii_custom_patterns_defaults_to_empty_list(self) -> None:
        cfg = _LifecycleModel()
        assert cfg.pii_custom_patterns == []
        # mutating one instance must not bleed into another (default_factory)
        cfg.pii_custom_patterns.append("foo")
        assert _LifecycleModel().pii_custom_patterns == []


# ===========================================================================
# Retrieval config mixin
# ===========================================================================
class TestRetrievalConfig:
    def test_defaults_are_sensible(self) -> None:
        cfg = _RetrievalModel()
        assert cfg.bm25_candidates == 50
        assert cfg.vector_candidates == 50
        assert cfg.rrf_k == 5  # meta-harness promoted default
        assert isinstance(cfg.rrf_k, int)
        assert cfg.recall_fusion_mode == "rrf"
        assert cfg.recall_top_k_multiplier == 3
        # opt-in confidence filter is OFF by default (None, not a number)
        assert cfg.recall_confidence_filter is None
        assert cfg.recall_preserve_hybrid_order is True

    def test_candidate_counts_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            _RetrievalModel(bm25_candidates=0)
        with pytest.raises(ValidationError):
            _RetrievalModel(vector_candidates=-1)
        with pytest.raises(ValidationError):
            _RetrievalModel(rrf_k=0)

    def test_importance_alpha_is_fraction(self) -> None:
        assert _RetrievalModel(rrf_importance_alpha=0.0).rrf_importance_alpha == 0.0
        assert _RetrievalModel(rrf_importance_alpha=1.0).rrf_importance_alpha == 1.0
        with pytest.raises(ValidationError):
            _RetrievalModel(rrf_importance_alpha=1.5)

    def test_importance_alpha_accepts_legacy_env_alias(self) -> None:
        # validation_alias AliasChoices accepts the memory_-prefixed key
        cfg = _RetrievalModel.model_validate({"memory_rrf_importance_alpha": 0.25})
        assert cfg.rrf_importance_alpha == 0.25

    def test_confidence_filter_when_set_is_clamped_fraction(self) -> None:
        assert _RetrievalModel(recall_confidence_filter=0.5).recall_confidence_filter == 0.5
        with pytest.raises(ValidationError):
            _RetrievalModel(recall_confidence_filter=1.2)
        with pytest.raises(ValidationError):
            _RetrievalModel(recall_confidence_filter=-0.01)

    def test_candidate_pool_size_floor(self) -> None:
        # ge=10 — a pool smaller than 10 is rejected
        with pytest.raises(ValidationError):
            _RetrievalModel(hybrid_search_candidate_pool_size=9)
        assert _RetrievalModel(hybrid_search_candidate_pool_size=10).hybrid_search_candidate_pool_size == 10

    def test_top_k_multiplier_bounds(self) -> None:
        # ge=1, le=50
        with pytest.raises(ValidationError):
            _RetrievalModel(recall_top_k_multiplier=0)
        with pytest.raises(ValidationError):
            _RetrievalModel(recall_top_k_multiplier=51)
        assert _RetrievalModel(recall_top_k_multiplier=50).recall_top_k_multiplier == 50

    def test_recency_weight_is_fraction(self) -> None:
        assert _RetrievalModel(recall_recency_weight=0.0).recall_recency_weight == 0.0
        with pytest.raises(ValidationError):
            _RetrievalModel(recall_recency_weight=1.5)

    def test_recency_halflife_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            _RetrievalModel(recall_recency_halflife_days=0.0)
        assert _RetrievalModel(recall_recency_halflife_days=30.0).recall_recency_halflife_days == 30.0

    def test_rerank_candidates_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            _RetrievalModel(recall_rerank_candidates=0)


# ===========================================================================
# Security config mixin
# ===========================================================================
class TestSecurityConfig:
    def test_defaults_are_sensible(self) -> None:
        cfg = _SecurityModel()
        assert cfg.poisoning_detection_enabled is True
        assert cfg.poisoning_detection_mode == "observe"
        assert cfg.trust_scoring_mode == "observe"
        assert cfg.recall_filter_mode == "redact"
        assert cfg.canary_fail_mode == "halt"
        assert cfg.memory_recovery_policy == "strict"
        # int budgets are ints, not None
        assert isinstance(cfg.max_entry_chars, int)
        assert cfg.max_entry_chars == 10_240
        assert isinstance(cfg.quarantine_ttl_seconds, int)
        # sync is opt-in OFF
        assert cfg.sync_enabled is False

    def test_retired_anomaly_bypass_field_is_gone(self) -> None:
        """``anomaly_bypass_source_prefixes`` was removed on 2026-07-30.

        PRD-DIST-2045 shipped it as a per-source anomaly-quarantine carve-out;
        ``209a47853`` then removed the carve-out from the runtime because
        ``metadata['source']`` is caller-supplied and any caller could spoof it.
        The FIELD was left behind "for compatibility", gating nothing — a settable
        security-shaped knob that silently does nothing is worse than an absent
        one, because an operator who sets it believes they have control they do
        not have (wiring-defect P12, superseded-authority shape).

        This test replaces two that asserted the field's default value and
        default_factory isolation. Both were true of a field nothing consumed,
        which is exactly why they stayed green while the control was dead.
        """
        assert "anomaly_bypass_source_prefixes" not in _SecurityModel.model_fields

    def test_z_threshold_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            _SecurityModel(poisoning_z_threshold=0.0)
        with pytest.raises(ValidationError):
            _SecurityModel(poisoning_z_threshold=-1.0)

    def test_trust_score_threshold_is_fraction(self) -> None:
        assert _SecurityModel(trust_score_threshold=0.0).trust_score_threshold == 0.0
        assert _SecurityModel(trust_score_threshold=1.0).trust_score_threshold == 1.0
        with pytest.raises(ValidationError):
            _SecurityModel(trust_score_threshold=1.1)

    def test_canary_injection_rate_bounds(self) -> None:
        # ge=3, le=5
        with pytest.raises(ValidationError):
            _SecurityModel(canary_injection_rate=2)
        with pytest.raises(ValidationError):
            _SecurityModel(canary_injection_rate=6)
        assert _SecurityModel(canary_injection_rate=3).canary_injection_rate == 3
        assert _SecurityModel(canary_injection_rate=5).canary_injection_rate == 5

    def test_canary_probe_interval_floor(self) -> None:
        with pytest.raises(ValidationError):
            _SecurityModel(canary_probe_interval=0)

    def test_unknown_enum_modes_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _SecurityModel(poisoning_detection_mode="quarantine_everything")
        with pytest.raises(ValidationError):
            _SecurityModel(trust_scoring_mode="permissive")
        with pytest.raises(ValidationError):
            _SecurityModel(recall_filter_mode="block")
        with pytest.raises(ValidationError):
            _SecurityModel(canary_fail_mode="ignore")
        with pytest.raises(ValidationError):
            _SecurityModel(memory_recovery_policy="best_effort")

    def test_max_entry_chars_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            _SecurityModel(max_entry_chars=0)

    def test_quarantine_ttl_allows_zero_but_not_negative(self) -> None:
        # ge=0 — zero TTL is a valid "no retention" setting
        assert _SecurityModel(quarantine_ttl_seconds=0).quarantine_ttl_seconds == 0
        with pytest.raises(ValidationError):
            _SecurityModel(quarantine_ttl_seconds=-1)

    def test_write_rate_limit_allows_zero(self) -> None:
        # ge=0 — 0 means "no writes" / fully throttled, still valid
        assert _SecurityModel(max_memory_writes_per_minute=0).max_memory_writes_per_minute == 0
        with pytest.raises(ValidationError):
            _SecurityModel(max_memory_writes_per_minute=-1)

    def test_corrupt_backup_keep_bounds(self) -> None:
        # ge=1, le=50
        with pytest.raises(ValidationError):
            _SecurityModel(memory_corrupt_backup_keep=0)
        with pytest.raises(ValidationError):
            _SecurityModel(memory_corrupt_backup_keep=51)

    def test_integrity_check_interval_bounds(self) -> None:
        # ge=0 (0 disables), le=1440
        assert _SecurityModel(memory_integrity_check_interval_minutes=0).memory_integrity_check_interval_minutes == 0
        with pytest.raises(ValidationError):
            _SecurityModel(memory_integrity_check_interval_minutes=1441)
        with pytest.raises(ValidationError):
            _SecurityModel(memory_integrity_check_interval_minutes=-1)

    def test_concurrent_writer_threshold_bounds(self) -> None:
        with pytest.raises(ValidationError):
            _SecurityModel(memory_concurrent_writer_warn_threshold=0)
        with pytest.raises(ValidationError):
            _SecurityModel(memory_concurrent_writer_warn_threshold=101)

    def test_snapshot_keep_bounds(self) -> None:
        with pytest.raises(ValidationError):
            _SecurityModel(memory_snapshot_daily_keep=0)
        with pytest.raises(ValidationError):
            _SecurityModel(memory_snapshot_daily_keep=366)
        with pytest.raises(ValidationError):
            _SecurityModel(memory_snapshot_weekly_keep=53)

    def test_sync_min_importance_is_fraction(self) -> None:
        assert _SecurityModel(sync_min_importance=0.7).sync_min_importance == 0.7
        with pytest.raises(ValidationError):
            _SecurityModel(sync_min_importance=1.5)

    def test_recovery_policy_accepts_legacy_alias(self) -> None:
        cfg = _SecurityModel.model_validate({"recovery_policy": "empty_ok"})
        assert cfg.memory_recovery_policy == "empty_ok"


# ===========================================================================
# Storage config mixin
# ===========================================================================
class TestStorageConfig:
    def test_defaults_are_sensible(self) -> None:
        cfg = _StorageModel()
        assert cfg.storage_backend == "sqlite"
        assert cfg.storage_path == ".memory"
        assert cfg.sqlite_db_name == "memory.db"
        assert isinstance(cfg.embedding_dim, int)
        assert cfg.embedding_dim == 384
        assert cfg.key_source == "env"
        assert cfg.rbac_mode == "local"
        assert cfg.default_role == "admin"
        assert cfg.encryption_enabled is False
        assert cfg.local_only is False

    def test_embedding_dim_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            _StorageModel(embedding_dim=0)
        with pytest.raises(ValidationError):
            _StorageModel(embedding_dim=-768)

    def test_unknown_storage_backend_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _StorageModel(storage_backend="postgres")

    def test_unknown_key_source_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _StorageModel(key_source="vault")

    def test_unknown_rbac_mode_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _StorageModel(rbac_mode="cluster")

    def test_unknown_default_role_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _StorageModel(default_role="superuser")

    def test_valid_enum_members_accepted(self) -> None:
        cfg = _StorageModel(
            storage_backend="yaml",
            key_source="file",
            rbac_mode="remote",
            default_role="reader",
        )
        assert cfg.storage_backend == "yaml"
        assert cfg.key_source == "file"
        assert cfg.rbac_mode == "remote"
        assert cfg.default_role == "reader"

    def test_namespace_roles_default_isolation(self) -> None:
        cfg = _StorageModel()
        assert cfg.namespace_roles == {}
        cfg.namespace_roles["project:x"] = "writer"
        assert _StorageModel().namespace_roles == {}

    def test_encryption_alias_accepted(self) -> None:
        cfg = _StorageModel.model_validate({"memory_encryption_enabled": True})
        assert cfg.encryption_enabled is True
