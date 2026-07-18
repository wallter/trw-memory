"""Immutable audit log with a PRD-aligned SHA-256 hash chain."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from trw_memory.exceptions import StorageError
from trw_memory.storage.persistence import lock_for_rmw

_GENESIS_HASH = "0" * 64
_COMPACT_MANIFEST_NAME = "audit_compact_manifest.jsonl"


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
        if log_path.name == _COMPACT_MANIFEST_NAME:
            raise ValueError(f"audit log path uses reserved compact manifest name: {_COMPACT_MANIFEST_NAME}")
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
            if self._path.is_symlink():
                raise StorageError("Refusing symlink audit log path", path=str(self._path))
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
        """Drop records older than *retention_days* and re-chain the retained suffix.

        Before any records are dropped, the *current* chain-head hash and
        the pre-compact event count are appended to an immutable
        compact-manifest sidecar (``audit_compact_manifest.jsonl``). The
        manifest is the only durable record of the chain head that existed
        prior to re-chaining: ``compact`` re-roots the retained suffix at
        genesis, so without the manifest an insider could mutate old events
        and re-compact to produce a self-consistent — but fraudulent —
        chain that :meth:`verify_chain` would still report as valid. The
        append-only manifest lets post-hoc forensics detect a hash
        discontinuity between successive compacted ranges.
        """
        if self._path.is_symlink():
            raise StorageError("Refusing symlink audit log path", path=str(self._path))
        cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
        # Fast path: the log is append-only and chronological, so the OLDEST
        # record is line 1. If even the oldest record is within retention there
        # is nothing to drop — skip the lock + full read+rewrite entirely. This
        # bounds the common-case cost (called inline at every operation
        # boundary when security_maintenance_inline=True) to a single-line read
        # instead of holding lock_for_rmw across read_all() + the rewrite,
        # which would otherwise block all concurrent append() callers (they
        # share the same RMW lock) for the duration of a large-log rewrite.
        oldest_ts = self._read_oldest_ts()
        if oldest_ts is not None and oldest_ts >= cutoff:
            return self._count_records()
        with lock_for_rmw(self._path):
            if self._path.is_symlink():
                raise StorageError("Refusing symlink audit log path", path=str(self._path))
            all_records = self.read_all()
            # Capture the pre-compact chain head + count and pin it to the
            # manifest BEFORE any record is dropped or re-chained.
            if all_records:
                self._write_compact_manifest(
                    self._path,
                    head_hash=all_records[-1].hash,
                    event_count=len(all_records),
                )

            records = [record for record in all_records if record.ts >= cutoff]
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

    def _write_compact_manifest(self, audit_log_path: Path, *, head_hash: str, event_count: int) -> None:
        """Append the pre-compact chain head to the immutable compact manifest.

        Writes one JSONL line to ``audit_compact_manifest.jsonl`` alongside
        the audit log, recording the chain-head hash that existed *before*
        this compaction re-rooted the retained suffix at genesis. The
        manifest is append-only: each compaction adds exactly one line, so
        the full history of chain heads survives any number of compactions.
        Post-hoc verification can then detect a hash discontinuity between
        compacted ranges that an in-place re-chain would otherwise hide.
        """
        manifest_path = audit_log_path.parent / _COMPACT_MANIFEST_NAME
        entry: dict[str, object] = {
            "compact_at": datetime.now(timezone.utc).isoformat(),
            "chain_head_hash": head_hash,
            "event_count": event_count,
        }
        with lock_for_rmw(manifest_path):
            if manifest_path.is_symlink():
                raise StorageError("Refusing symlink compact manifest path", path=str(manifest_path))
            self._append_line_unlocked(entry, path=manifest_path)

    def read_all(self) -> list[AuditRecord]:
        """Read all audit records from disk."""
        records: list[AuditRecord] = []
        for line_no, data in self._iter_record_dicts():
            try:
                records.append(AuditRecord.model_validate(data))
            except (ValueError, TypeError) as exc:
                # Content-free: a pydantic ValidationError embeds the raw input
                # payload, so surface only the line number and exception type.
                raise StorageError(
                    f"Corrupt audit record at line {line_no}: {type(exc).__name__}",
                    path=str(self._path),
                ) from None
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

    def _iter_record_dicts(self) -> Iterator[tuple[int, dict[str, object]]]:
        """Yield ``(line_no, record)`` for each non-blank JSONL line.

        Single Seam owning the audit log's read-path corruption policy:
        it fails closed with a typed, content-free :class:`StorageError`
        on a non-UTF-8 byte stream, an undecodable line, or a line that
        is valid JSON but not an object. Callers (:meth:`read_all`,
        :meth:`_read_last_hash_unlocked`) layer their own per-record
        semantics on top, so the two read paths cannot drift into
        asymmetric corruption handling — a tamper-evident hash chain
        must treat a bad tail identically on every read. Diagnostics
        never embed the raw line, so a corrupted record's payload cannot
        leak into logs or exception messages.
        """
        if not self._path.exists():
            return
        try:
            with self._path.open("r", encoding="utf-8") as fh:
                for line_no, line in enumerate(fh, start=1):
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        data = json.loads(stripped)
                    except json.JSONDecodeError as exc:
                        raise StorageError(
                            f"Corrupt audit record at line {line_no}: {type(exc).__name__}",
                            path=str(self._path),
                        ) from exc
                    if not isinstance(data, dict):
                        raise StorageError(
                            f"Corrupt audit record at line {line_no}: expected JSON object",
                            path=str(self._path),
                        ) from None
                    yield line_no, data
        except (OSError, UnicodeDecodeError) as exc:
            raise StorageError(
                f"Unable to read audit log: {type(exc).__name__}",
                path=str(self._path),
            ) from exc

    def _read_oldest_ts(self) -> datetime | None:
        """Return the timestamp of the OLDEST (first) audit record, or None.

        The log is append-only and chronological, so line 1 is the oldest
        record. Used by :meth:`compact` to cheaply decide whether anything is
        old enough to prune before taking the RMW lock + reading the whole
        file. Returns ``None`` when the log is absent/empty or the first line's
        timestamp is missing/unparseable (callers then fall through to the full
        compaction path, which fails closed on corruption).
        """
        for _line_no, data in self._iter_record_dicts():
            raw_ts = data.get("ts")
            if not isinstance(raw_ts, str):
                return None
            try:
                parsed = datetime.fromisoformat(raw_ts)
            except ValueError:
                return None
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed
        return None

    def _count_records(self) -> int:
        """Count non-blank JSONL records without validating each payload."""
        return sum(1 for _ in self._iter_record_dicts())

    def _read_last_hash_unlocked(self) -> str:
        """Return the chain-head hash for the next append, failing closed.

        The append write-path links each new record's ``prev_hash`` to
        this value, so a corrupt tail must never crash the writer with a
        raw decode/attribute error nor silently re-root the chain to
        genesis — a truncated or tampered file resetting the hash linkage
        would defeat the tamper-evidence. A genuinely empty/absent log
        returns the genesis hash; a non-empty log whose any record lacks
        a usable hash is treated as corruption.
        """
        last_hash: str | None = None
        for _line_no, data in self._iter_record_dicts():
            hash_value = data.get("hash")
            if not isinstance(hash_value, str) or not hash_value:
                raise StorageError(
                    "Corrupt audit record at chain tail: missing hash",
                    path=str(self._path),
                ) from None
            last_hash = hash_value
        return last_hash if last_hash is not None else _GENESIS_HASH

    def _append_line_unlocked(self, record_data: dict[str, object], *, path: Path | None = None) -> None:
        target = path or self._path
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = (
            json.dumps(record_data, sort_keys=True, separators=(",", ":"), default=self._json_default) + "\n"
        ).encode("utf-8")
        flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT
        flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
        try:
            fd = os.open(target, flags, stat.S_IRUSR | stat.S_IWUSR)
        except OSError as exc:
            raise StorageError(
                f"Unable to append audit log: {type(exc).__name__}",
                path=str(target),
            ) from exc
        original_size: int | None = None
        try:
            opened = os.fstat(fd)
            original_size = opened.st_size
            leaf = os.lstat(target)
            if not stat.S_ISREG(leaf.st_mode) or (opened.st_dev, opened.st_ino) != (leaf.st_dev, leaf.st_ino):
                raise OSError("audit log path changed during secure open")
            fchmod = getattr(os, "fchmod", None)
            if fchmod is not None:
                fchmod(fd, stat.S_IRUSR | stat.S_IWUSR)
            remaining = memoryview(payload)
            while remaining:
                written = os.write(fd, remaining)
                if written <= 0:
                    raise OSError("failed to append audit record")
                remaining = remaining[written:]
            if self._fsync:
                os.fsync(fd)
        except BaseException as exc:
            if original_size is not None:
                try:
                    os.ftruncate(fd, original_size)
                except OSError:
                    pass
            if isinstance(exc, OSError):
                raise StorageError(
                    f"Unable to append audit log: {type(exc).__name__}",
                    path=str(target),
                ) from exc
            raise
        finally:
            os.close(fd)


def audit_verify(log_path: Path) -> dict[str, object]:
    """Verify an audit log and return the PRD-aligned result shape."""
    return AuditLog(log_path).verify_chain()
