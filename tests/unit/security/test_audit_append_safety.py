"""Secure audit-log append leaf and rollback behavior."""

from __future__ import annotations

import json
import os
import stat
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from trw_memory.exceptions import StorageError
from trw_memory.security.audit import AuditLog


def test_audit_log_rejects_reserved_compact_manifest_name(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="reserved compact manifest name"):
        AuditLog(tmp_path / "audit_compact_manifest.jsonl")


def test_append_rejects_dangling_symlink(tmp_path: Path) -> None:
    chain = tmp_path / "audit.jsonl"
    target = tmp_path / "outside.jsonl"
    try:
        chain.symlink_to(target)
    except OSError as exc:  # pragma: no cover - platform privilege boundary
        pytest.skip(f"symlinks unavailable: {exc}")

    with pytest.raises(StorageError, match="Refusing symlink"):
        AuditLog(chain).append("store", entry_id="M-001")

    assert not target.exists()


def test_compact_rejects_audit_log_symlink_without_touching_target(tmp_path: Path) -> None:
    target = tmp_path / "outside.jsonl"
    AuditLog(target).append("store", entry_id="M-outside")
    target_bytes = target.read_bytes()
    chain = tmp_path / "audit.jsonl"
    manifest = tmp_path / "audit_compact_manifest.jsonl"
    try:
        chain.symlink_to(target)
    except OSError as exc:  # pragma: no cover - platform privilege boundary
        pytest.skip(f"symlinks unavailable: {exc}")

    with pytest.raises(StorageError, match="Refusing symlink audit log path"):
        AuditLog(chain).compact(retention_days=365)

    assert chain.is_symlink()
    assert target.read_bytes() == target_bytes
    assert not manifest.exists()


def test_append_rolls_back_partial_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    chain = tmp_path / "audit.jsonl"
    log = AuditLog(chain)
    log.append("store", entry_id="M-001")
    original = chain.read_bytes()
    real_write = os.write
    calls = 0

    def partial_then_fail(fd: int, data: bytes | bytearray | memoryview) -> int:
        nonlocal calls
        calls += 1
        if calls == 1:
            return real_write(fd, data[:5])
        raise OSError("disk full")

    monkeypatch.setattr("trw_memory.security.audit.os.write", partial_then_fail)
    with pytest.raises(StorageError, match="Unable to append audit log"):
        log.append("forget", entry_id="M-001")
    assert chain.read_bytes() == original

    monkeypatch.setattr("trw_memory.security.audit.os.write", real_write)
    log.append("forget", entry_id="M-001")
    assert log.verify_chain()["valid"] is True
    if os.name != "nt":
        assert stat.S_IMODE(chain.stat().st_mode) == 0o600


def test_append_closes_descriptor_when_fstat_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    chain = tmp_path / "audit.jsonl"
    log = AuditLog(chain)
    log.append("store", entry_id="M-001")
    original = chain.read_bytes()
    real_close = os.close
    real_fstat = os.fstat
    real_open = os.open
    closed: list[int] = []
    data_fd: int | None = None

    def open_spy(path: str | bytes | os.PathLike[str] | os.PathLike[bytes], flags: int, mode: int = 0o777) -> int:
        nonlocal data_fd
        fd = real_open(path, flags, mode)
        if Path(path) == chain:
            data_fd = fd
        return fd

    def close_spy(fd: int) -> None:
        closed.append(fd)
        real_close(fd)

    def fail_data_fstat(fd: int) -> os.stat_result:
        if fd == data_fd:
            raise OSError("fstat failed")
        return real_fstat(fd)

    monkeypatch.setattr("trw_memory.security.audit.os.open", open_spy)
    monkeypatch.setattr("trw_memory.security.audit.os.close", close_spy)
    monkeypatch.setattr("trw_memory.security.audit.os.fstat", fail_data_fstat)

    with pytest.raises(StorageError, match="Unable to append audit log"):
        log.append("forget", entry_id="M-001")

    assert data_fd is not None
    assert data_fd in closed
    assert chain.read_bytes() == original


def test_compact_rejects_dangling_manifest_symlink(tmp_path: Path) -> None:
    chain = tmp_path / "audit.jsonl"
    log = AuditLog(chain)
    log.append("store", entry_id="M-001")
    record = json.loads(chain.read_text(encoding="utf-8"))
    record["ts"] = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
    chain.write_text(json.dumps(record) + "\n", encoding="utf-8")
    original = chain.read_bytes()
    manifest = tmp_path / "audit_compact_manifest.jsonl"
    outside = tmp_path / "outside-control.jsonl"
    try:
        manifest.symlink_to(outside)
    except OSError as exc:  # pragma: no cover - platform privilege boundary
        pytest.skip(f"symlinks unavailable: {exc}")

    with pytest.raises(StorageError, match="Refusing symlink compact manifest"):
        log.compact(retention_days=1)

    assert not outside.exists()
    assert chain.read_bytes() == original
