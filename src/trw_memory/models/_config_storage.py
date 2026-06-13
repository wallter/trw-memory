"""MemoryConfig field group mixins.

Split from ``models.config`` to keep the public settings surface deep while
keeping each module under the effective-LOC gate.
"""

from __future__ import annotations

from typing import Literal

from pydantic import AliasChoices, Field

__all__ = ["_StorageConfigMixin"]


class _StorageConfigMixin:
    # Storage
    storage_backend: Literal["sqlite", "yaml"] = Field(default="sqlite", description="Storage backend type")
    storage_path: str = Field(default=".memory", description="Root directory for memory storage files")
    sqlite_db_name: str = Field(default="memory.db", description="SQLite database filename within namespace dir")
    embedding_dim: int = Field(default=384, gt=0, description="Dimensionality of dense embedding vectors")
    embedding_model: str = Field(default="all-MiniLM-L6-v2", description="Sentence-transformer model for embeddings")

    # Encryption
    encryption_enabled: bool = Field(
        default=False,
        validation_alias=AliasChoices("encryption_enabled", "memory_encryption_enabled"),
        description="Enable field-level encryption",
    )
    encryption_algorithm: str = Field(
        default="AES-256-GCM", description="Encryption algorithm for field-level encryption"
    )
    key_source: Literal["keyring", "env", "file"] = Field(default="env", description="Source for encryption master key")
    key_file_path: str = Field(
        default="~/.trw-memory/master.key", description="Path to master key file when key_source='file'"
    )
    auto_generate_key: bool = Field(
        default=True,
        validation_alias=AliasChoices("auto_generate_key", "memory_auto_generate_key"),
        description="Generate and persist a master key if none exists",
    )
    key_rotation_backup: bool = Field(
        default=True,
        validation_alias=AliasChoices("key_rotation_backup", "memory_key_rotation_backup"),
        description="Create a backup before key rotation work",
    )

    # Local-only mode
    local_only: bool = Field(
        default=False,
        validation_alias=AliasChoices("local_only", "memory_local_only"),
        description="Restrict to local storage only (no remote sync)",
    )

    # RBAC
    rbac_enabled: bool = Field(
        default=False,
        validation_alias=AliasChoices("rbac_enabled", "memory_rbac_enabled"),
        description="Enable role-based access control",
    )
    rbac_mode: Literal["local", "remote"] = Field(
        default="local",
        validation_alias=AliasChoices("rbac_mode", "memory_rbac_mode"),
        description="RBAC enforcement layer",
    )
    default_role: Literal["admin", "reader", "writer", "none"] = "admin"
    namespace_roles: dict[str, str] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("namespace_roles", "memory_namespace_roles"),
        description="Per-namespace role overrides used when RBAC is enabled",
    )
