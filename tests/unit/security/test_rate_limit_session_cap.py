"""Rate-limiter session_id length-cap regression (disk-growth DoS hardening).

A caller-controlled session_id is used as a key in the persisted YAML rate-limit
state. Without a length cap, a pathologically long id would balloon the state
file on every write. enforce_write_rate_limit caps the id at 256 chars first.
"""

from __future__ import annotations

from pathlib import Path

import pytest

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
