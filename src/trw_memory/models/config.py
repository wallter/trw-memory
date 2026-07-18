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

from trw_memory.models._config_lifecycle import _LifecycleConfigMixin
from trw_memory.models._config_retrieval import _RetrievalConfigMixin
from trw_memory.models._config_security import _SecurityConfigMixin
from trw_memory.models._config_sources import _TRWConfigYamlSource
from trw_memory.models._config_storage import _StorageConfigMixin

__all__ = ["MemoryConfig"]


class MemoryConfig(
    _SecurityConfigMixin,
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
    def _apply_local_only(self) -> MemoryConfig:
        """Disable every remote-capable setting when local-only mode is active."""
        if self.local_only:
            self.rbac_mode = "local"
            self.sync_enabled = False
            self.sync_namespace = ""
            self.platform_url = ""
            self.platform_api_key = ""
        return self

    @model_validator(mode="after")
    def _derive_security_paths(self) -> MemoryConfig:
        """Keep audit, quarantine, provenance, and rate-limit state together."""
        storage_root = Path(self.storage_path)
        if storage_root.name == "memory" and storage_root.parent.name == ".trw":
            security_root = storage_root.parent / "security"
        else:
            security_root = storage_root.parent / ".trw" / "security"
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
        return (
            init_settings,
            env_settings,
            _TRWConfigYamlSource(settings_cls),
            dotenv_settings,
            file_secret_settings,
        )
