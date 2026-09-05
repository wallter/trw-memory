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

from trw_memory.exceptions import ConfigError
from trw_memory.models._config_daemon import _DaemonConfigMixin
from trw_memory.models._config_lifecycle import _LifecycleConfigMixin
from trw_memory.models._config_retrieval import _RetrievalConfigMixin
from trw_memory.models._config_security import _SecurityConfigMixin
from trw_memory.models._config_sources import _TRWConfigYamlSource
from trw_memory.models._config_storage import _StorageConfigMixin

__all__ = ["MemoryConfig"]


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
    def _refuse_encrypted_single_store(self) -> MemoryConfig:
        """Reject encryption + a single store, because the key model is per-namespace.

        SQLCipher keys a whole FILE, but
        ``security.encryption.derive_namespace_key`` derives a DIFFERENT key per
        namespace. Under ``memory_single_store_path`` those two facts collide:
        the first namespace to open the shared file sets ``PRAGMA key`` to its
        own derived key, and every other namespace then cannot decrypt the file
        it is supposed to share. Silent-at-config, fatal-at-second-namespace.

        Refusing the combination outright is the honest position while the
        per-file key redesign is unwritten (PRD-CORE-253 FR09, Slice B). The
        alternative -- deriving one key for the file -- changes the key
        derivation for every existing encrypted store and needs its own
        migration, which is exactly why it is not a line of code here.
        """
        if self.encryption_enabled and self.memory_single_store_path:
            raise ConfigError(
                "encryption_enabled and memory_single_store_path cannot both be set: SQLCipher keys a "
                "whole file, but this package derives a per-NAMESPACE key, so only the first namespace "
                "to open the shared store could decrypt it. Single-file encryption keys are PRD-CORE-253 "
                "FR09 (Slice B). Until then, use one encrypted store per namespace (leave "
                "memory_single_store_path empty) or run the daemon unencrypted."
            )
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
        return (
            init_settings,
            env_settings,
            _TRWConfigYamlSource(settings_cls),
            dotenv_settings,
            file_secret_settings,
        )
