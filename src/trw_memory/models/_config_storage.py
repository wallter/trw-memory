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
    storage_path: str = Field(
        default=".memory",
        description=(
            "Root directory for memory storage files. The default resolves beside the project's .trw "
            "(never the cwd) and is refused without one; an explicit value is used as given"
        ),
    )
    sqlite_db_name: str = Field(default="memory.db", description="SQLite database filename within namespace dir")
    embeddings_enabled: bool = Field(
        default=True,
        description=(
            "Load the local embedding model for dense recall. False (MEMORY_EMBEDDINGS_ENABLED=false) never "
            "loads it: stores write no vectors and recall is keyword-only"
        ),
    )
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
        validation_alias=AliasChoices("memory_encryption_enabled"),
        description="Refused when true: trw-memory does not encrypt its store; use full-disk encryption",
    )

    # RBAC
    rbac_enabled: bool = Field(
        default=False,
        validation_alias=AliasChoices("memory_rbac_enabled"),
        description="Enable role-based access control",
    )
    # `rbac_mode` was REMOVED on 2026-09-26 (PRD-QUAL-145 wave 3, DEFECT-LEDGER
    # UF-030): no enforcement path ever read it -- `require_namespace_permission`
    # (security/rbac.py) gates only on `rbac_enabled`. A settable "local"/"remote"
    # selector with no reader misrepresented RBAC as having a remote enforcement
    # mode it never had.
    default_role: Literal["admin", "reader", "writer", "none"] = "admin"
    namespace_roles: dict[str, str] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("memory_namespace_roles"),
        description="Per-namespace role overrides used when RBAC is enabled",
    )
