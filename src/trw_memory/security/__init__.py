"""Security — encryption, access control, key management, audit, PII, poisoning.

Public API re-exported from submodules:
- ``encryption`` — AES-256-GCM field-level encrypt/decrypt
- ``keys`` — master key retrieval, storage, and rotation
- ``rbac`` — role-based access control
- ``audit`` — immutable SHA-256 hash chain audit log
- ``pii`` — PII detection and redaction
- ``poisoning`` — memory poisoning anomaly detection
"""

from trw_memory.exceptions import LocalOnlyViolationError
from trw_memory.security.audit import (
    AuditLog,
    AuditRecord,
)
from trw_memory.security.encryption import (
    decrypt_entry_fields,
    decrypt_field,
    derive_namespace_key,
    encrypt_entry_fields,
    encrypt_field,
    generate_master_key,
)
from trw_memory.security.keys import (
    get_master_key,
    rotate_master_key,
    store_master_key,
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
from trw_memory.security.rbac import (
    ROLE_PERMISSIONS,
    Permission,
    Role,
    check_permission,
    require_namespace_permission,
    require_permission,
)

__all__ = [
    "ROLE_PERMISSIONS",
    "AnomalyResult",
    "AnomalyType",
    "AuditLog",
    "AuditRecord",
    "LocalOnlyViolationError",
    "PIIAction",
    "PIIMatch",
    "PIIType",
    "Permission",
    "PoisoningDetector",
    "Role",
    "check_entry_pii",
    "check_permission",
    "decrypt_entry_fields",
    "decrypt_field",
    "derive_namespace_key",
    "detect_pii",
    "encrypt_entry_fields",
    "encrypt_field",
    "generate_master_key",
    "get_master_key",
    "quarantine_entry",
    "redact_text",
    "require_namespace_permission",
    "require_permission",
    "rotate_master_key",
    "shannon_entropy",
    "store_master_key",
]
