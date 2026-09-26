"""Standalone memory configuration.

The public :class:`MemoryConfig` composes field-group mixins so each setting is
declared once while callers retain one stable settings model. ``MEMORY_*``
environment variables and the legacy ``.trw/config.yaml`` source remain
backward compatible.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic_settings.sources import PydanticBaseSettingsSource

from trw_memory.exceptions import refuse_encryption_at_rest
from trw_memory.models._config_daemon import _DaemonConfigMixin
from trw_memory.models._config_lifecycle import _LifecycleConfigMixin
from trw_memory.models._config_retrieval import _RetrievalConfigMixin
from trw_memory.models._config_security import _SecurityConfigMixin
from trw_memory.models._config_sources import (
    _check_retired_hype_environment,
    _check_retired_hype_settings,
    _check_retired_local_only_environment,
    _check_retired_local_only_settings,
    _TRWConfigYamlSource,
    _warn_retired_environment,
    _warn_retired_settings,
)
from trw_memory.models._config_storage import _StorageConfigMixin

__all__ = ["DAEMON_WIDE_SECURITY_KEYS", "MemoryConfig", "daemon_wide_security"]


class MemoryConfig(
    _SecurityConfigMixin,
    _DaemonConfigMixin,
    _LifecycleConfigMixin,
    _RetrievalConfigMixin,
    _StorageConfigMixin,
    BaseSettings,
):
    """Configuration for the trw-memory package.

    All settings can be overridden via ``MEMORY_*`` environment variables.
    Example: ``MEMORY_STORAGE_BACKEND=yaml`` selects YAML-only storage.
    """

    model_config = SettingsConfigDict(
        env_prefix="MEMORY_",
        case_sensitive=False,
        extra="ignore",
    )

    def __repr__(self) -> str:
        """Return a concise view without secrets or remote credentials."""
        return (
            f"MemoryConfig("
            f"backend={self.storage_backend!r}, "
            f"path={self.storage_path!r}, "
            f"encryption={self.encryption_enabled}, "
            f"rbac={self.rbac_enabled})"
        )

    @model_validator(mode="after")
    def _check_weight_sum(self) -> MemoryConfig:
        total = self.score_relevance_weight + self.score_recency_weight + self.score_importance_weight
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Score weights must sum to 1.0, got {total:.3f}")
        return self

    @model_validator(mode="after")
    def _apply_security_switches(self) -> MemoryConfig:
        """4.0 refuses ``encryption_enabled`` at config load, as backend creation does too (C12), and
        ``platform_contact_enabled: false`` turns sync off here, so every publish and subscribe path inherits it (rc11)."""
        refuse_encryption_at_rest(self)
        self.sync_enabled = self.sync_enabled and self.platform_contact_enabled
        return self

    @model_validator(mode="after")
    def _derive_security_paths(self) -> MemoryConfig:
        """Keep audit, quarantine, provenance, and rate-limit state together.

        PRD-CORE-253 FR01: security state is the ``security`` sibling of the
        resolved store directory under EVERY supported base. The previous
        branch recognised only the home-fallback layout (a directory literally
        named ``memory`` inside one literally named ``.trw``), so an
        XDG_DATA_HOME base derived ``<xdg>/trw/.trw/security/quarantine.db`` --
        a nested ``.trw`` inside an XDG data directory, detached from the store
        it describes. One rule, no layout sniffing.
        """
        security_root = Path(self.storage_path).parent / "security"
        if not self.audit_log_path:
            self.audit_log_path = str(security_root / "audit.jsonl")
        if not self.quarantine_path:
            self.quarantine_path = str(security_root / "quarantine")
        if not self.quarantine_db_path:
            self.quarantine_db_path = str(security_root / "quarantine.db")
        if not self.rate_limit_state_path:
            self.rate_limit_state_path = str(security_root / "rate_limits.yaml")
        if not self.provenance_signing_key_path:
            self.provenance_signing_key_path = str(security_root / "ed25519_signing_key.bin")
        if not self.canary_fixtures_path:
            self.canary_fixtures_path = "package:canary"
        return self

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Load ``.trw/config.yaml`` after environment variables."""
        _check_retired_hype_settings(init_settings(), source="constructor")
        _check_retired_local_only_settings(init_settings(), source="constructor")
        # Inspect source data before Pydantic drops aliases of removed fields.
        _check_retired_hype_environment(dotenv_settings)
        _check_retired_local_only_environment(dotenv_settings)
        _warn_retired_settings(init_settings(), source="constructor")
        _warn_retired_environment(dotenv_settings)
        return (
            init_settings,
            env_settings,
            _TRWConfigYamlSource(settings_cls),
            dotenv_settings,
            file_secret_settings,
        )


#: Settings one daemon enforces for every client it serves (PRD-CORE-298 FR07).
#: The daemon resolves them from its own environment, so a client resolving a
#: different value would lose that policy without a sign; the client refuses.
DAEMON_WIDE_SECURITY_KEYS = (
    "rbac_enabled",
    "default_role",
    "namespace_roles",
    "enable_recall_filter",
    "recall_filter_mode",
    "canary_fail_mode",
    "poisoning_detection_mode",
    "enable_trust_scoring",
    "trust_scoring_mode",
    "provenance_required",
)


def daemon_wide_security(config: MemoryConfig) -> dict[str, object]:
    """The daemon-wide security settings *config* resolves, by field name."""
    return {key: getattr(config, key) for key in DAEMON_WIDE_SECURITY_KEYS}
