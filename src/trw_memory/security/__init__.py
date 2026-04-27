"""Security — encryption, access control, key management, audit, PII, poisoning.

Public API re-exported from submodules:
- ``encryption`` — AES-256-GCM field-level encrypt/decrypt
- ``keys`` — master key retrieval, storage, and rotation
- ``rbac`` — role-based access control
- ``audit`` — immutable SHA-256 hash chain audit log
- ``pii`` — PII detection and redaction
- ``poisoning`` — memory poisoning anomaly detection
"""

from trw_memory.exceptions import (
    EncryptionUnavailableError,
    KeyRotationError,
    LocalOnlyViolationError,
    MasterKeyNotFoundError,
)
from trw_memory.security.audit import (
    AuditLog,
    AuditRecord,
)
from trw_memory.security.canary import (
    CanaryLearning,
    CanaryStore,
    CanaryVerificationResult,
    seed_canaries,
    verify_canaries,
)
from trw_memory.security.encryption import (
    decrypt_entry_fields,
    decrypt_field,
    derive_namespace_key,
    derive_namespace_key_bytes,
    encrypt_entry_fields,
    encrypt_field,
    generate_master_key,
    rotate_key,
)
from trw_memory.security.keys import (
    generate_ed25519_signing_key,
    get_master_key,
    get_or_create_ed25519_key,
    load_ed25519_signing_key,
    rotate_master_key,
    store_master_key,
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
    check_entry_pii,
    detect_pii,
    redact_text,
    shannon_entropy,
)
from trw_memory.security.poisoning import (
    AnomalyResult,
    AnomalyType,
    PoisoningDetector,
    quarantine_entry,
)
from trw_memory.security.provenance import (
    ProvenanceChain,
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
    require_permission,
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
    "AnomalyResult",
    "AnomalyType",
    "AuditLog",
    "AuditRecord",
    "CanaryLearning",
    "CanaryStore",
    "CanaryVerificationResult",
    "EncryptionUnavailableError",
    "KeyRotationError",
    "LocalOnlyViolationError",
    "MasterKeyNotFoundError",
    "ObserveClockState",
    "PIIAction",
    "PIIMatch",
    "PIIType",
    "Permission",
    "PoisoningDetector",
    "ProvenanceChain",
    "ProvenanceEntry",
    "RecallFilterResult",
    "Role",
    "TrustScore",
    "check_entry_pii",
    "check_permission",
    "decrypt_entry_fields",
    "decrypt_field",
    "derive_namespace_key",
    "derive_namespace_key_bytes",
    "detect_pii",
    "encrypt_entry_fields",
    "encrypt_field",
    "filter_recall_window",
    "generate_ed25519_signing_key",
    "generate_master_key",
    "get_master_key",
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
    "require_permission",
    "rotate_key",
    "rotate_master_key",
    "score_intake",
    "seed_canaries",
    "shannon_entropy",
    "start_observe_clock",
    "store_master_key",
    "verify_canaries",
]
