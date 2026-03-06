"""Immutable audit log with SHA-256 hash chain.

Each audit record links to the previous record via ``prev_hash``, forming
a tamper-evident chain.  The genesis record has ``prev_hash = ""``.
Records are persisted as append-only JSONL, one JSON object per line.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from trw_memory.exceptions import StorageError


class AuditRecord(BaseModel):
    """Single audit log entry with hash chain link."""

    model_config = ConfigDict(use_enum_values=True)

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    actor: str = ""
    action: str
    target_id: str = ""
    namespace: str = "default"
    detail: dict[str, str] = Field(default_factory=dict)
    prev_hash: str = ""
    record_hash: str = ""


class AuditLog:
    """Append-only audit log backed by JSONL with SHA-256 hash chain.

    Every record contains ``prev_hash`` (the ``record_hash`` of the
    previous record) and ``record_hash`` (SHA-256 of the canonical JSON
    of all fields *except* ``record_hash`` itself).
    """

    def __init__(self, log_path: Path) -> None:
        self._path = log_path
        self._last_hash = self._read_last_hash()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def append(
        self,
        action: str,
        target_id: str = "",
        namespace: str = "default",
        actor: str = "",
        detail: dict[str, str] | None = None,
    ) -> AuditRecord:
        """Append an audit record, chaining from the previous hash.

        Args:
            action: The action performed (store, recall, update, delete, export).
            target_id: The memory entry ID affected.
            namespace: Isolation namespace.
            actor: Who performed the action.
            detail: Arbitrary key-value metadata.

        Returns:
            The newly created and persisted :class:`AuditRecord`.
        """
        record = AuditRecord(
            action=action,
            target_id=target_id,
            namespace=namespace,
            actor=actor,
            detail=detail or {},
            prev_hash=self._last_hash,
        )

        # Compute the hash over all fields except record_hash
        record_data = record.model_dump(mode="json")
        record_data.pop("record_hash", None)
        computed_hash = self._compute_hash(record_data)
        record = record.model_copy(update={"record_hash": computed_hash})

        # Persist to JSONL
        self._append_line(record.model_dump(mode="json"))
        self._last_hash = computed_hash
        return record

    def verify_chain(self) -> tuple[bool, int, str]:
        """Verify the full hash chain.

        Returns:
            A tuple of ``(valid, record_count, error_message)``.
            *error_message* is empty when the chain is valid.
        """
        records = self.read_all()
        if not records:
            return (True, 0, "")

        prev_hash = ""
        for idx, record in enumerate(records):
            # Check prev_hash linkage
            if record.prev_hash != prev_hash:
                return (
                    False,
                    len(records),
                    f"Record {idx}: prev_hash mismatch "
                    f"(expected {prev_hash!r}, got {record.prev_hash!r})",
                )

            # Recompute record_hash and compare
            record_data = record.model_dump(mode="json")
            record_data.pop("record_hash", None)
            expected_hash = self._compute_hash(record_data)
            if record.record_hash != expected_hash:
                return (
                    False,
                    len(records),
                    f"Record {idx}: record_hash mismatch "
                    f"(expected {expected_hash!r}, "
                    f"got {record.record_hash!r})",
                )

            prev_hash = record.record_hash

        return (True, len(records), "")

    def read_all(self) -> list[AuditRecord]:
        """Read all audit records from the log file.

        Returns:
            List of :class:`AuditRecord` in chronological order.
        """
        if not self._path.exists():
            return []

        records: list[AuditRecord] = []
        with self._path.open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    data = json.loads(stripped)
                    records.append(AuditRecord.model_validate(data))
                except (json.JSONDecodeError, ValueError, KeyError) as exc:
                    raise StorageError(
                        f"Corrupt audit record at line {line_no}: {exc}",
                        path=str(self._path),
                    ) from exc
        return records

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_hash(self, record_data: dict[str, object]) -> str:
        """SHA-256 of the canonical JSON of *record_data*.

        Canonical form uses ``sort_keys=True`` and compact separators
        to ensure deterministic hashing.
        """
        canonical = json.dumps(
            record_data,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _read_last_hash(self) -> str:
        """Read the ``record_hash`` of the last record in the log.

        Returns:
            The hash string, or ``""`` if the log is empty or absent.
        """
        if not self._path.exists():
            return ""

        last_line = ""
        with self._path.open("r", encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if stripped:
                    last_line = stripped

        if not last_line:
            return ""

        try:
            data = json.loads(last_line)
            return str(data.get("record_hash", ""))
        except (json.JSONDecodeError, AttributeError):
            return ""

    def _append_line(self, record_data: dict[str, object]) -> None:
        """Append a single JSON line to the log file."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record_data, sort_keys=True, default=str) + "\n"
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(line)
