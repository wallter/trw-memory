"""14-day observe-mode calibration clock tests (PRD-SEC-001 Item 5).

Sprint-96 carry-forward-b.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Barrier, Event, Thread

import pytest

from trw_memory.security import trust_scorer as trust_scorer_mod
from trw_memory.security.observe_clock import (
    read_observe_clock,
    start_observe_clock,
)
from trw_memory.security.trust_scorer import score_intake


@pytest.fixture(autouse=True)
def _reset_clock_flag() -> None:
    """Reset the module-level clock flag between tests."""
    trust_scorer_mod._clock_started_for.clear()


def test_start_writes_sidecar(tmp_path: Path) -> None:
    state = start_observe_clock(tmp_path)
    sidecar = tmp_path / "memory" / "security" / "observe_start.yaml"
    assert sidecar.exists()
    assert state.phase == "observe"
    assert state.prd == "PRD-SEC-001"
    # started_at + 14 days == promotion_review_at
    started = datetime.fromisoformat(state.started_at)
    review = datetime.fromisoformat(state.promotion_review_at)
    assert abs((review - started) - timedelta(days=14)) < timedelta(seconds=1)


def test_start_is_idempotent(tmp_path: Path) -> None:
    first = start_observe_clock(tmp_path)
    # Second call must return the SAME started_at, not overwrite
    second = start_observe_clock(tmp_path)
    assert first.started_at == second.started_at
    assert first.promotion_review_at == second.promotion_review_at


def test_concurrent_starts_share_one_persisted_clock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from trw_memory.security import observe_clock

    original_write = observe_clock.write_yaml
    write_started = Event()
    release_write = Event()
    start = Barrier(3)
    returned: list[str] = []

    def _paused_write(path: Path, data: dict[str, object]) -> None:
        write_started.set()
        assert release_write.wait(timeout=2)
        original_write(path, data)

    monkeypatch.setattr(observe_clock, "write_yaml", _paused_write)

    def _start() -> None:
        start.wait()
        returned.append(start_observe_clock(tmp_path).started_at)

    threads = [Thread(target=_start) for _ in range(2)]
    for thread in threads:
        thread.start()
    start.wait()
    assert write_started.wait(timeout=2)
    release_write.set()
    for thread in threads:
        thread.join(timeout=2)

    persisted = read_observe_clock(tmp_path)
    assert persisted is not None
    assert all(not thread.is_alive() for thread in threads)
    assert returned == [persisted.started_at, persisted.started_at]


def test_failed_atomic_write_does_not_publish_sidecar(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from trw_memory.exceptions import StorageError
    from trw_memory.security import observe_clock

    def _fail_write(_path: Path, _data: dict[str, object]) -> None:
        raise StorageError("simulated write failure")

    monkeypatch.setattr(observe_clock, "write_yaml", _fail_write)
    with pytest.raises(StorageError, match="simulated write failure"):
        start_observe_clock(tmp_path)

    assert read_observe_clock(tmp_path) is None


def test_read_returns_none_when_unset(tmp_path: Path) -> None:
    assert read_observe_clock(tmp_path) is None


def test_read_after_start_returns_state(tmp_path: Path) -> None:
    started = start_observe_clock(tmp_path, prd="PRD-TEST-001", window_days=7)
    read = read_observe_clock(tmp_path)
    assert read is not None
    assert read.prd == "PRD-TEST-001"
    assert read.started_at == started.started_at
    # window_days honored
    started_dt = datetime.fromisoformat(read.started_at)
    review_dt = datetime.fromisoformat(read.promotion_review_at)
    assert abs((review_dt - started_dt) - timedelta(days=7)) < timedelta(seconds=1)


def test_score_intake_starts_clock_on_first_call(tmp_path: Path) -> None:
    sidecar = tmp_path / "memory" / "security" / "observe_start.yaml"
    assert not sidecar.exists()
    score_intake("hello", {}, trw_dir=tmp_path)
    assert sidecar.exists()


def test_score_intake_does_not_restart_clock(tmp_path: Path) -> None:
    score_intake("first", {}, trw_dir=tmp_path)
    first_state = read_observe_clock(tmp_path)
    assert first_state is not None
    # Simulate process restart -- reset the flag but keep the sidecar
    trust_scorer_mod._clock_started_for.clear()
    score_intake("second", {}, trw_dir=tmp_path)
    second_state = read_observe_clock(tmp_path)
    assert second_state is not None
    assert second_state.started_at == first_state.started_at


def test_score_intake_without_trw_dir_does_not_start_clock(tmp_path: Path) -> None:
    score_intake("hello", {})  # no trw_dir
    # Nothing written anywhere under tmp_path
    assert not any(tmp_path.rglob("observe_start.yaml"))


def _write_sidecar(trw_dir: Path, data: bytes) -> Path:
    sidecar = trw_dir / "memory" / "security" / "observe_start.yaml"
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_bytes(data)
    return sidecar


def test_read_fails_open_on_non_utf8_sidecar(tmp_path: Path) -> None:
    # A torn/partial write can leave non-UTF-8 bytes on disk. read_text(utf-8)
    # raises UnicodeDecodeError (a ValueError, NOT an OSError), so the seam must
    # list it explicitly or the security intake path crashes.
    _write_sidecar(tmp_path, b"\xff\xfe started_at: not-utf8")
    assert read_observe_clock(tmp_path) is None


def test_read_fails_open_on_malformed_yaml_sidecar(tmp_path: Path) -> None:
    _write_sidecar(tmp_path, b"started_at: [unclosed\nphase: observe")
    assert read_observe_clock(tmp_path) is None


def test_read_returns_none_on_non_mapping_sidecar(tmp_path: Path) -> None:
    # A scalar/list root (not a mapping) must not be coerced into a state model.
    _write_sidecar(tmp_path, b"- just\n- a\n- list\n")
    assert read_observe_clock(tmp_path) is None


def test_start_recovers_from_corrupt_sidecar(tmp_path: Path) -> None:
    # start_observe_clock reads first; a non-UTF-8 sidecar must not crash it --
    # it fails open to None and writes a fresh, valid clock.
    _write_sidecar(tmp_path, b"\xff\xfe\x00 garbage")
    state = start_observe_clock(tmp_path)
    assert state.phase == "observe"
    # The freshly written sidecar is now readable.
    reread = read_observe_clock(tmp_path)
    assert reread is not None
    assert reread.started_at == state.started_at


def test_score_intake_survives_corrupt_sidecar(tmp_path: Path) -> None:
    # The intake path (score_intake -> start_observe_clock -> read_observe_clock)
    # must not raise when a corrupt sidecar is present on disk.
    _write_sidecar(tmp_path, b"\xff\xfe non-utf8 clock")
    # Should not raise.
    score_intake("hello", {}, trw_dir=tmp_path)
    assert read_observe_clock(tmp_path) is not None


def test_clock_started_at_is_utc(tmp_path: Path) -> None:
    state = start_observe_clock(tmp_path)
    dt = datetime.fromisoformat(state.started_at)
    # Must carry tzinfo and be close to now
    assert dt.tzinfo is not None
    assert abs(dt - datetime.now(timezone.utc)) < timedelta(seconds=10)
