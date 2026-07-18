"""Secure provenance append leaf and rollback behavior."""

from __future__ import annotations

import os
import stat
from collections.abc import Callable
from pathlib import Path

import pytest

from trw_memory.security import provenance
from trw_memory.security.provenance import ProvenanceEntry


def _entry(index: int) -> ProvenanceEntry:
    return ProvenanceEntry(
        learning_id=f"L-{index:03d}",
        content_hash=f"hash-{index}",
        source_identity="append-safety-test",
    )


@pytest.fixture(params=["unsigned", "signed_degraded"])
def writer(request: pytest.FixtureRequest) -> Callable[[Path, ProvenanceEntry], str]:
    if request.param == "unsigned":
        return provenance.append
    return lambda path, entry: provenance.append_signed(path, entry, None)


def test_append_rejects_dangling_symlink(
    tmp_path: Path,
    writer: Callable[[Path, ProvenanceEntry], str],
) -> None:
    chain = tmp_path / "chain.jsonl"
    target = tmp_path / "outside.jsonl"
    try:
        chain.symlink_to(target)
    except OSError as exc:  # pragma: no cover - platform privilege boundary
        pytest.skip(f"symlinks unavailable: {exc}")

    with pytest.raises(OSError):
        writer(chain, _entry(1))

    assert not target.exists()


def test_unsigned_append_preserves_caller_supplied_signature(tmp_path: Path) -> None:
    chain = tmp_path / "chain.jsonl"
    entry = _entry(1).model_copy(update={"signature": "caller-supplied"})

    provenance.append(chain, entry)

    persisted = ProvenanceEntry.model_validate_json(chain.read_text(encoding="utf-8"))
    assert persisted.signature == "caller-supplied"


def test_append_rolls_back_partial_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    writer: Callable[[Path, ProvenanceEntry], str],
) -> None:
    chain = tmp_path / "chain.jsonl"
    writer(chain, _entry(1))
    original = chain.read_bytes()
    real_write = os.write
    calls = 0

    def partial_then_fail(fd: int, data: bytes | bytearray | memoryview) -> int:
        nonlocal calls
        calls += 1
        if calls == 1:
            return real_write(fd, data[:5])
        raise OSError("disk full")

    monkeypatch.setattr("trw_memory.security.provenance.os.write", partial_then_fail)
    with pytest.raises(OSError):
        writer(chain, _entry(2))
    assert chain.read_bytes() == original

    monkeypatch.setattr("trw_memory.security.provenance.os.write", real_write)
    writer(chain, _entry(3))
    assert len(chain.read_text(encoding="utf-8").splitlines()) == 2
    assert provenance.verify(chain) is True
    if os.name != "nt":
        assert stat.S_IMODE(chain.stat().st_mode) == 0o600
