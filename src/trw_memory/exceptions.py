"""Exception hierarchy for trw-memory."""

from __future__ import annotations


class MemoryError(Exception):
    """Base exception for all trw-memory errors.

    Args:
        message: Human-readable error description.
        path: Optional file path involved in the error.
    """

    def __init__(self, message: str, *, path: str = "") -> None:
        super().__init__(message)
        self.path = path


class StorageError(MemoryError):
    """Raised when a storage operation fails (read, write, lock)."""


class ConfigError(MemoryError):
    """Raised when configuration is invalid or cannot be loaded."""


class MemoryConnectionError(MemoryError):
    """Raised when no connection mode is available."""


class MemoryNotFoundError(MemoryError):
    """Raised when a memory entry is not found."""


class ToolAlreadyRegisteredError(MemoryError):
    """Raised when register_tools() is called twice."""


class AuthorizationError(MemoryError, PermissionError):
    """Raised when a permission check fails."""


class DimensionMismatchError(MemoryError):
    """Raised when vectors have incompatible dimensions."""


class LocalOnlyViolationError(MemoryError):
    """Raised when a network-capable operation is attempted in local-only mode."""


class RemoteCodeNotPermittedError(MemoryError):
    """Raised when loading a model would execute repo-supplied code without consent.

    PRD-SEC-014-FR02: ``trust_remote_code`` is reachable only through the typed
    ``embedding_trust_remote_code`` config field (default ``False``). Before this
    error existed, a model-name substring test opted a deployment into executing
    Hub-fetched code, so a ``.trw/config.yaml`` edit was sufficient to enable
    arbitrary code execution. The message names the field so the fail-closed path
    is actionable rather than merely refusing.
    """


class MasterKeyNotFoundError(MemoryError):
    """Raised when no usable master key exists in any configured source."""


class EncryptionUnavailableError(MemoryError):
    """Raised when an encryption-required runtime dependency is unavailable."""


class KeyRotationError(MemoryError):
    """Raised when key rotation fails or cannot be safely completed."""


class SchemaValidationError(MemoryError):
    """Raised when a memory entry fails the write-time schema/policy contract."""

    def __init__(
        self,
        message: str,
        *,
        path: str = "",
        failed_fields: list[str] | None = None,
        reason: str = "",
    ) -> None:
        super().__init__(message, path=path)
        self.failed_fields = failed_fields or []
        #: Machine-readable rejection code (PRD-CORE-244 FR02 uses
        #: ``"unsubstantiated_verified"``). Empty for the historical
        #: field-shape rejections, which are fully described by
        #: ``failed_fields``.
        self.reason = reason


class PIIBlockError(MemoryError):
    """Raised when a memory entry is blocked by the configured PII policy."""

    def __init__(self, message: str, *, path: str = "", detected_type: str = "") -> None:
        super().__init__(message, path=path)
        self.detected_type = detected_type


class PoisoningError(MemoryError):
    """Raised when content matches write-time poisoning or injection defenses."""

    def __init__(self, message: str, *, path: str = "", reason: str = "") -> None:
        super().__init__(message, path=path)
        self.reason = reason


class MemoryQuarantinedError(MemoryError):
    """Raised when a write was HELD for review rather than stored.

    A quarantine is deliberately not a ``PoisoningError``: the entry is durable in
    the review store and may be approved later, so reporting it as a rejection
    would misstate what happened to the caller's data.

    It exists because ``guarded_store`` reports a quarantine in its *return value*
    (``stored=False, quarantined=True``) rather than by raising, and a caller
    whose only return channel is ``None`` cannot pass that on. Three chat adapters
    discarded that result while their docstrings promised the opposite — "a chat
    history that silently dropped a turn would leave the caller unable to tell a
    censored transcript from a complete one". They now raise this.
    """

    def __init__(
        self,
        message: str,
        *,
        path: str = "",
        entry_id: str = "",
        anomaly_dimension: str = "",
    ) -> None:
        super().__init__(message, path=path)
        self.entry_id = entry_id
        self.anomaly_dimension = anomaly_dimension


class RateLimitError(MemoryError):
    """Raised when a caller exceeds the configured memory write rate limit."""

    def __init__(self, message: str, *, path: str = "", retry_after: float = 0.0) -> None:
        super().__init__(message, path=path)
        self.retry_after = retry_after


class Utf8ValidationError(SchemaValidationError):
    """Raised when a string field fails write-time UTF-8 validation.

    Carries the list of offending field names via failed_fields. Use this
    when a write would persist bytes that can't round-trip through
    str.encode('utf-8', errors='strict') — lone surrogates, invalid
    continuation bytes, etc.
    """


class StaleConnectionError(StorageError):
    """Raised when a connection was detected stale and could not be reopened.

    Normally SQLiteBackend._ensure_connection_fresh transparently reopens.
    This exception surfaces only when the reopen attempt itself fails.
    """


class CorruptDatabaseUnsalvageableError(StorageError):
    """Raised when a corrupt DB cannot be salvaged and strict policy refuses the empty fallback.

    Carries ``backup_path`` as a user-actionable breadcrumb so the forensic evidence
    is visible at the call site. Raised by :meth:`SQLiteBackend.recover_db` only when
    the primary SELECT salvage AND the ``sqlite3 .recover`` CLI fallback both yield
    zero rows from a non-empty ``.corrupt.bak`` file under the ``strict`` recovery
    policy (PRD-CORE-138).
    """

    def __init__(self, message: str, *, backup_path: str = "") -> None:
        super().__init__(f"{message} (backup preserved at: {backup_path})", path=backup_path)
        self.backup_path = backup_path


class SecurityDependencyError(MemoryError):
    """Base class for fail-loud SEC-001 dependency failures."""


class ScorerUnavailableError(SecurityDependencyError):
    """Raised when trust-scoring cannot be loaded or executed."""


class QuarantineUnreachableError(SecurityDependencyError):
    """Raised when the quarantine database cannot be reached."""


class SecurityDefaultUnresolvableError(SecurityDependencyError):
    """Raised when a SEC-001 default path or fixture cannot resolve at boot."""


class CanaryFixturesMissingError(SecurityDefaultUnresolvableError):
    """Raised when canary fixtures cannot be resolved at startup/runtime."""


class ProvenanceKeyUnavailableError(SecurityDependencyError):
    """Raised when provenance signing is required but no signing key is available."""


class SecurityTelemetryUnavailableError(SecurityDependencyError):
    """Raised when required SEC-001 telemetry cannot be appended."""


class CanaryTamperError(SecurityDependencyError):
    """Raised when a pinned canary hash drifts from the stored row."""


class DaemonError(MemoryError):
    """Base class for loopback memory-daemon failures (PRD-CORE-253 FR03)."""


class DaemonAlreadyRunningError(DaemonError):
    """Raised when a start is attempted while a live daemon holds the claim.

    Not a fault: it is the expected outcome of the second start, and the reason
    that start exits WITHOUT binding a port or rewriting the discovery file.
    """


class DaemonUnreachableError(DaemonError):
    """Raised when the store cannot be reached (PRD-CORE-253 FR08).

    Both reads and writes fail closed with this rather than degrading to a
    local snapshot: an agent that recalls from a stale view then writes a
    conclusion derived from it is split-brain with extra steps. The message
    carries the discovery-file path and the start command so the failure names
    its own remedy.
    """


class DaemonAuthError(DaemonError):
    """Raised when the daemon rejects the client's token (FR08 clause 3).

    Deliberately distinct from :class:`DaemonUnreachableError`: a rejection is
    never retried and never triggers token regeneration, because automatic
    rotation would let any local process force one by corrupting the file.
    """


class DaemonSecretUnreadableError(DaemonError):
    """Raised when a 0600 daemon secret exists but cannot be read.

    Deliberately distinct from "absent". Absent is first run and is answered by
    creating the file; unreadable is a symlink someone planted at the path, a
    permission change, or a file that is not UTF-8 — and creating over it would
    destroy the secret a live daemon is still authenticating against.
    """


class TokenUnreadableError(DaemonSecretUnreadableError):
    """Raised when the bearer-token file exists but cannot be read.

    Minting a replacement here would ROTATE the token out from under a running
    daemon, so every subsequent client call is rejected: the automatic rotation
    FR08 clause 3 forbids, arrived at from the other direction. The message
    names the file and why it could not be read.
    """


class DaemonRecordInvalidError(DaemonError):
    """Raised when ``daemon.json`` exists but cannot be trusted.

    A record that will not parse, names an unknown schema, or fails field
    validation proves nothing about whether a daemon is serving. Reading it as
    "no daemon" is what lets a second daemon bind a port and overwrite the
    record while the first is still live — two writers on one store. The remedy
    is an operator inspecting or removing the named file.
    """
