"""Immutable audit log with a PRD-aligned SHA-256 hash chain."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from trw_memory.exceptions import StorageError
from trw_memory.storage.persistence import lock_for_rmw

_GENESIS_HASH = "0" * 64


class AuditRecord(BaseModel):
    """Single append-only audit record."""

    model_config = ConfigDict(use_enum_values=True)

    ts: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    op: str
    id: str = ""
    actor: str = ""
    namespace: str = "default"
    data: dict[str, object] = Field(default_factory=dict)
    prev_hash: str = _GENESIS_HASH
    hash: str = ""


class AuditLog:
    """Append-only JSONL audit log with tamper-evident hash chaining."""

    def __init__(self, log_path: Path, *, fsync: bool = False) -> None:
        self._path = log_path
        self._fsync = fsync

    def append(
        self,
        op: str | None = None,
        *,
        action: str | None = None,
        entry_id: str = "",
        target_id: str = "",
        actor: str = "",
        namespace: str = "default",
        data: dict[str, object] | None = None,
    ) -> AuditRecord:
        """Append one record to the audit log."""
        effective_op = op or action
        effective_entry_id = entry_id or target_id
        if not effective_op:
            raise ValueError("append requires op or action")
        with lock_for_rmw(self._path):
            prev_hash = self._read_last_hash_unlocked()
            record = AuditRecord(
                op=effective_op,
                id=effective_entry_id,
                actor=actor,
                namespace=namespace,
                data=data or {},
                prev_hash=prev_hash,
            )
            payload = record.model_dump(mode="json")
            payload["hash"] = self._compute_hash(prev_hash, payload)
            self._append_line_unlocked(payload)
        return record.model_copy(update={"hash": str(payload["hash"])})

    def verify_chain(self) -> dict[str, object]:
        """Verify the full audit log and return a structured result."""
        records = self.read_all()
        if not records:
            return {"valid": True, "entries_checked": 0, "first_broken_at": None, "broken_hash": None}

        expected_prev = _GENESIS_HASH
        for line_no, record in enumerate(records, start=1):
            if record.prev_hash != expected_prev:
                return {
                    "valid": False,
                    "entries_checked": len(records),
                    "first_broken_at": line_no,
                    "broken_hash": record.prev_hash,
                }
            payload = record.model_dump(mode="json")
            expected_hash = self._compute_hash(record.prev_hash, payload)
            if record.hash != expected_hash:
                return {
                    "valid": False,
                    "entries_checked": len(records),
                    "first_broken_at": line_no,
                    "broken_hash": record.hash,
                }
            expected_prev = record.hash
        return {"valid": True, "entries_checked": len(records), "first_broken_at": None, "broken_hash": None}

    def compact(self, retention_days: int) -> int:
        """Drop records older than *retention_days* and re-chain the retained suffix."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
        with lock_for_rmw(self._path):
            records = [record for record in self.read_all() if record.ts >= cutoff]
            if not records:
                self._path.unlink(missing_ok=True)
                return 0

            chained: list[dict[str, object]] = []
            prev_hash = _GENESIS_HASH
            for record in records:
                payload = record.model_dump(mode="json")
                payload["prev_hash"] = prev_hash
                payload["hash"] = self._compute_hash(prev_hash, payload)
                chained.append(payload)
                prev_hash = str(payload["hash"])

            self._path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_name = tempfile.mkstemp(dir=str(self._path.parent), suffix=".audit.tmp")
            tmp_path = Path(tmp_name)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    for payload in chained:
                        fh.write(
                            json.dumps(payload, sort_keys=True, separators=(",", ":"), default=self._json_default)
                            + "\n"
                        )
                    fh.flush()
                    if self._fsync:
                        os.fsync(fh.fileno())
                tmp_path.replace(self._path)
            except Exception:
                tmp_path.unlink(missing_ok=True)
                raise
        return len(chained)

    def read_all(self) -> list[AuditRecord]:
        """Read all audit records from disk."""
        if not self._path.exists():
            return []
        records: list[AuditRecord] = []
        with self._path.open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    records.append(AuditRecord.model_validate(json.loads(stripped)))
                except (ValueError, TypeError, json.JSONDecodeError) as exc:
                    raise StorageError(f"Corrupt audit record at line {line_no}: {exc}", path=str(self._path)) from exc
        return records

    @staticmethod
    def _compute_hash(prev_hash: str, record_data: dict[str, object]) -> str:
        material = prev_hash + AuditLog._canonical_json(record_data, exclude={"hash"})
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    @staticmethod
    def _canonical_json(record_data: dict[str, object], *, exclude: set[str] | None = None) -> str:
        filtered = {key: value for key, value in record_data.items() if exclude is None or key not in exclude}
        return json.dumps(filtered, sort_keys=True, separators=(",", ":"), default=AuditLog._json_default)

    @staticmethod
    def _json_default(value: object) -> str:
        if isinstance(value, datetime):
            return value.isoformat()
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

    def _read_last_hash_unlocked(self) -> str:
        if not self._path.exists():
            return _GENESIS_HASH
        last_hash = _GENESIS_HASH
        with self._path.open("r", encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    data = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise StorageError(f"Corrupt audit record while loading tail: {exc}", path=str(self._path)) from exc
                last_hash = str(data.get("hash", _GENESIS_HASH))
        return last_hash

    def _append_line_unlocked(self, record_data: dict[str, object]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record_data, sort_keys=True, separators=(",", ":"), default=self._json_default) + "\n"
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(line)
            fh.flush()
            if self._fsync:
                os.fsync(fh.fileno())


def audit_verify(log_path: Path) -> dict[str, object]:
    """Verify an audit log and return the PRD-aligned result shape."""
    return AuditLog(log_path).verify_chain()
