"""MemoryConfig field group mixins.

Split from ``models.config`` to keep the public settings surface deep while
keeping each module under the effective-LOC gate.
"""

from __future__ import annotations

from typing import Literal

from pydantic import AliasChoices, BaseModel, Field

from trw_memory._model_pin import DEFAULT_EMBEDDING_MODEL

__all__ = ["_StorageConfigMixin"]


class _StorageConfigMixin(BaseModel):
    # Storage
    storage_backend: Literal["sqlite", "yaml"] = Field(default="sqlite", description="Storage backend type")
    storage_path: str = Field(default=".memory", description="Root directory for memory storage files")
    sqlite_db_name: str = Field(default="memory.db", description="SQLite database filename within namespace dir")
    embedding_dim: int = Field(default=384, gt=0, description="Dimensionality of dense embedding vectors")
    embedding_model: str = Field(
        default=DEFAULT_EMBEDDING_MODEL,
        description=(
            "Sentence-transformer model for embeddings. Changing it leaves stored vectors in the old "
            "space: dense recall ignores them until `trw-memory reembed` re-encodes them"
        ),
    )

    # Encryption at rest is not supported: true is refused (EncryptionAtRestUnsupportedError).
    encryption_enabled: bool = Field(
        default=False,
        validation_alias=AliasChoices("encryption_enabled", "memory_encryption_enabled"),
        description="Refused when true: trw-memory does not encrypt its store; use full-disk encryption",
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
