"""Security — access control, signing keys, audit, PII, poisoning.

Public API re-exported from submodules:
- ``keys`` — Ed25519 provenance-signing keys
- ``rbac`` — role-based access control
- ``audit`` — immutable SHA-256 hash chain audit log
- ``pii`` — PII detection and redaction
- ``poisoning`` — memory poisoning anomaly detection
"""

from typing import TYPE_CHECKING, Any

from trw_memory.exceptions import ModelNotCachedError
from trw_memory.security.audit import (
    AuditLog,
    AuditRecord,
)
from trw_memory.security.canary import (
    CanaryLearning,
    CanaryStore,
    CanaryVerificationResult,
)
from trw_memory.security.pii import (
    PIIAction,
    PIIMatch,
    PIIType,
    detect_pii,
    redact_text,
    shannon_entropy,
)
from trw_memory.security.provenance import (
    ProvenanceEntry,
)
from trw_memory.security.provenance import (
    append as provenance_append,
)
from trw_memory.security.provenance import (
    append_signed as provenance_append_signed,
)
from trw_memory.security.provenance import (
    verify as provenance_verify,
)
from trw_memory.security.provenance import (
    verify_signed as provenance_verify_signed,
)
from trw_memory.security.rbac import (
    ROLE_PERMISSIONS,
    Permission,
    Role,
    check_permission,
    require_namespace_permission,
)
from trw_memory.security.recall_filter import (
    RecallFilterResult,
    filter_recall_window,
)

if TYPE_CHECKING:
    from trw_memory.security.keys import (
        generate_ed25519_signing_key,
        get_or_create_ed25519_key,
        load_ed25519_signing_key,
    )

#: Re-exported on first access, not at import: ``keys`` imports nacl and cryptography,
#: which every importer of any ``trw_memory.security`` submodule (the edit hook's recall
#: admission among them) paid for at ~0.1-0.2 s without ever touching a key.
_LAZY_KEYS = frozenset({"generate_ed25519_signing_key", "get_or_create_ed25519_key", "load_ed25519_signing_key"})


def __getattr__(name: str) -> Any:
    if name in _LAZY_KEYS:
        from trw_memory.security import keys

        return getattr(keys, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ROLE_PERMISSIONS",
    "AuditLog",
    "AuditRecord",
    "CanaryLearning",
    "CanaryStore",
    "CanaryVerificationResult",
    "ModelNotCachedError",
    "PIIAction",
    "PIIMatch",
    "PIIType",
    "Permission",
    "ProvenanceEntry",
    "RecallFilterResult",
    "Role",
    "check_permission",
    "detect_pii",
    "filter_recall_window",
    "generate_ed25519_signing_key",
    "get_or_create_ed25519_key",
    "load_ed25519_signing_key",
    "provenance_append",
    "provenance_append_signed",
    "provenance_verify",
    "provenance_verify_signed",
    "redact_text",
    "require_namespace_permission",
    "shannon_entropy",
]
