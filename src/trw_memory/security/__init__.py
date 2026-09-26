"""Security — access control, signing keys, audit, PII, poisoning.

Public API re-exported from submodules:
- ``keys`` — Ed25519 provenance-signing keys
- ``rbac`` — role-based access control
- ``audit`` — immutable SHA-256 hash chain audit log
- ``pii`` — PII detection and redaction
- ``poisoning`` — memory poisoning anomaly detection
"""

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
from trw_memory.security.keys import (
    generate_ed25519_signing_key,
    get_or_create_ed25519_key,
    load_ed25519_signing_key,
)
from trw_memory.security.observe_clock import (
    ObserveClockState,
    read_observe_clock,
    start_observe_clock,
)
from trw_memory.security.pii import (
    PIIAction,
    PIIMatch,
    PIIType,
    detect_pii,
    redact_text,
    shannon_entropy,
)
from trw_memory.security.poisoning import (
    quarantine_entry,
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
from trw_memory.security.trust_scorer import (
    TrustScore,
    score_intake,
)

__all__ = [
    "ROLE_PERMISSIONS",
    "AuditLog",
    "AuditRecord",
    "CanaryLearning",
    "CanaryStore",
    "CanaryVerificationResult",
    "ModelNotCachedError",
    "ObserveClockState",
    "PIIAction",
    "PIIMatch",
    "PIIType",
    "Permission",
    "ProvenanceEntry",
    "RecallFilterResult",
    "Role",
    "TrustScore",
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
    "quarantine_entry",
    "read_observe_clock",
    "redact_text",
    "require_namespace_permission",
    "score_intake",
    "shannon_entropy",
    "start_observe_clock",
]
