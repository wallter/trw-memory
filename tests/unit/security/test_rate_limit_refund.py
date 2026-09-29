"""B71-81: a conflicting store returns exactly the slot it charged, by receipt."""

from __future__ import annotations

from pathlib import Path

import pytest

from trw_memory.exceptions import RateLimitError
from trw_memory.models.config import MemoryConfig
from trw_memory.security import runtime
from trw_memory.storage.persistence import read_yaml


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    now = [1000.0]
    monkeypatch.setattr(runtime, "time", lambda: now[0])
    return now


def _charge(cfg: MemoryConfig, session: str = "s") -> float | None:
    return runtime.enforce_write_rate_limit(cfg, session_id=session, actor="a", namespace="default", entry_id="M")


def _cfg(tmp_path: Path) -> MemoryConfig:
    return MemoryConfig(rate_limit_state_path=str(tmp_path / "rl.yaml"), max_memory_writes_per_minute=10)


def test_an_expired_charge_refunds_nothing(tmp_path: Path, clock: list[float]) -> None:
    """A charges, stalls past the window, then conflicts: the refund must not free a newer request's slot."""
    cfg = _cfg(tmp_path)
    receipt_a = _charge(cfg)
    clock[0] += 61.0
    for _ in range(10):
        clock[0] += 0.1
        _charge(cfg)

    runtime.refund_write_slot(cfg, session_id="s", receipt=receipt_a)

    with pytest.raises(RateLimitError):
        _charge(cfg)


def test_a_refund_removes_its_own_slot_not_a_newer_one(tmp_path: Path, clock: list[float]) -> None:
    cfg = _cfg(tmp_path)
    receipt_b = _charge(cfg)
    clock[0] += 1.0
    receipt_a = _charge(cfg)

    runtime.refund_write_slot(cfg, session_id="s", receipt=receipt_b)  # B conflicts

    assert read_yaml(Path(cfg.rate_limit_state_path))["sessions"]["s"] == [receipt_a]


def test_a_shared_batch_slot_carries_no_receipt(tmp_path: Path, clock: list[float]) -> None:
    """Inside single_write_operation one slot covers the batch, so no row may refund it."""
    cfg = _cfg(tmp_path)
    with runtime.single_write_operation():
        assert _charge(cfg) is None
        assert _charge(cfg) is None
