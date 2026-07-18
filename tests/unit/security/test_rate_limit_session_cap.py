"""Rate-limiter session-key bounds (disk-growth DoS hardening).

A caller-controlled session_id is used as a key in the persisted YAML rate-limit
state. Long IDs are hashed and the number of live session buckets is bounded.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from trw_memory.exceptions import RateLimitError
from trw_memory.models.config import MemoryConfig
from trw_memory.security.runtime import enforce_write_rate_limit
from trw_memory.storage.persistence import read_yaml


@pytest.mark.integration
def test_long_session_id_is_capped_before_persisting(tmp_path: Path) -> None:
    state = tmp_path / "rate_limits.yaml"
    config = MemoryConfig(
        rate_limit_state_path=str(state),
        max_memory_writes_per_minute=100,
    )
    long_id = "x" * 5000

    enforce_write_rate_limit(
        config,
        session_id=long_id,
        actor="agent",
        namespace="default",
        entry_id="M-1",
    )

    raw = read_yaml(state)
    assert isinstance(raw, dict)
    sessions = raw["sessions"]
    assert isinstance(sessions, dict)
    keys = list(sessions.keys())
    assert keys, "rate-limit state should have recorded the session"
    assert all(len(k) <= 256 for k in keys), "session_id key must be capped at 256 chars"
    assert long_id not in keys  # the 5000-char id is never persisted verbatim


@pytest.mark.integration
def test_normal_session_id_is_preserved(tmp_path: Path) -> None:
    state = tmp_path / "rate_limits.yaml"
    config = MemoryConfig(
        rate_limit_state_path=str(state),
        max_memory_writes_per_minute=100,
    )
    sid = "session-" + ("a" * 36)  # well under the cap

    enforce_write_rate_limit(
        config,
        session_id=sid,
        actor="agent",
        namespace="default",
        entry_id="M-1",
    )

    raw = read_yaml(state)
    assert isinstance(raw, dict)
    sessions = raw["sessions"]
    assert isinstance(sessions, dict)
    assert sid in sessions  # ordinary ids are untouched


@pytest.mark.integration
def test_long_session_ids_with_shared_prefix_do_not_collide(tmp_path: Path) -> None:
    state = tmp_path / "rate_limits.yaml"
    config = MemoryConfig(rate_limit_state_path=str(state), max_memory_writes_per_minute=100)
    prefix = "x" * 300

    for suffix in ("a", "b"):
        enforce_write_rate_limit(
            config,
            session_id=prefix + suffix,
            actor="agent",
            namespace="default",
            entry_id=f"M-{suffix}",
        )

    sessions = read_yaml(state)["sessions"]
    assert isinstance(sessions, dict)
    assert len(sessions) == 2
    assert all(str(key).startswith("sha256:") for key in sessions)


@pytest.mark.integration
def test_live_session_bucket_count_is_bounded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state = tmp_path / "rate_limits.yaml"
    config = MemoryConfig(rate_limit_state_path=str(state), max_memory_writes_per_minute=100)
    monkeypatch.setattr("trw_memory.security.runtime._MAX_LIVE_RATE_LIMIT_SESSIONS", 2)

    for session_id in ("one", "two"):
        enforce_write_rate_limit(
            config,
            session_id=session_id,
            actor="agent",
            namespace="default",
            entry_id="M-existing",
        )

    with pytest.raises(RateLimitError, match="capacity"):
        enforce_write_rate_limit(
            config,
            session_id="three",
            actor="agent",
            namespace="default",
            entry_id="M-new",
        )

    sessions = read_yaml(state)["sessions"]
    assert isinstance(sessions, dict)
    assert set(sessions) == {"one", "two"}


@pytest.mark.integration
def test_future_session_timestamps_are_discarded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state = tmp_path / "rate_limits.yaml"
    state.write_text("sessions:\n  future: [200.0]\n", encoding="utf-8")
    config = MemoryConfig(rate_limit_state_path=str(state), max_memory_writes_per_minute=1)
    monkeypatch.setattr("trw_memory.security.runtime.time", lambda: 100.0)

    enforce_write_rate_limit(
        config,
        session_id="current",
        actor="agent",
        namespace="default",
        entry_id="M-current",
    )

    sessions = read_yaml(state)["sessions"]
    assert isinstance(sessions, dict)
    assert set(sessions) == {"current"}
