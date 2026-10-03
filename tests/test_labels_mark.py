"""PRD-SEC-023 FR04: the session high-water mark, and the raise-only stamp that carries it onto a written row.

A session starts at ``team``. Every row a recall returns raises the mark; nothing lowers it. A row written while the mark is above the row's
own label is stamped with the mark, so the label follows the data into every later session. A session that never rose adds no key at all.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from trw_memory.labels import LabelPolicy, Level, SessionMark
from trw_memory.models.memory import MemoryEntry


@pytest.fixture
def user_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    base = tmp_path / "user-base"
    base.mkdir()
    monkeypatch.setenv("TRW_USER_DIR", str(base))
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    return base


def _row(namespace: str = "default", **metadata: str) -> MemoryEntry:
    return MemoryEntry(id="L-x", content="c", namespace=namespace, metadata=dict(metadata))


def test_a_new_mark_starts_at_team_and_reports_nothing() -> None:
    mark = SessionMark()
    assert mark.level is Level.TEAM
    assert mark.reported() is None, "a session that never rose is byte-identical to today's"


def test_the_mark_only_rises() -> None:
    mark = SessionMark()

    assert mark.raise_to(Level.PERSONAL) is Level.PERSONAL
    assert mark.raise_to(Level.TEAM) is Level.PERSONAL, "a lower level does not lower it"
    assert mark.raise_to(Level.PUBLIC) is Level.PERSONAL
    assert mark.raise_to(Level.SENSITIVE) is Level.SENSITIVE
    assert mark.level is Level.SENSITIVE
    assert mark.reported() == "sensitive"


def test_the_mark_is_the_maximum_under_concurrent_raises() -> None:
    mark = SessionMark()
    levels = [Level.TEAM, Level.PERSONAL, Level.SENSITIVE, Level.PUBLIC] * 200
    barrier = threading.Barrier(8)

    def hammer(chunk: list[Level]) -> None:
        barrier.wait()
        for level in chunk:
            mark.raise_to(level)

    threads = [threading.Thread(target=hammer, args=(levels[i::8],)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert mark.level is Level.SENSITIVE


def test_a_row_written_above_its_own_label_is_stamped_with_the_mark(user_dir: Path) -> None:
    policy = LabelPolicy.current()

    stamped = policy.stamped({"source": "agent"}, Level.PERSONAL, Level.TEAM)

    assert stamped == {"source": "agent", "trw_label": "personal"}
    assert policy.label_of(_row(**stamped)) is Level.PERSONAL


def test_a_team_session_adds_no_metadata_key(user_dir: Path) -> None:
    """NFR01: existing tests assert ``entry.metadata`` exactly, so a session that never rose must not touch it."""
    policy = LabelPolicy.current()

    assert policy.stamped({}, Level.TEAM, Level.TEAM) == {}
    assert policy.stamped({"a": "b"}, Level.TEAM, Level.TEAM) == {"a": "b"}
    assert policy.stamped({"a": "b"}, Level.PERSONAL, Level.PERSONAL) == {"a": "b"}, "the row is already that high"


def test_a_stamp_is_raise_only(user_dir: Path) -> None:
    policy = LabelPolicy.current()

    assert policy.stamped({"trw_label": "sensitive"}, Level.PERSONAL, Level.SENSITIVE) == {"trw_label": "sensitive"}
    assert policy.stamped({"trw_label": "personal"}, Level.SENSITIVE, Level.PERSONAL) == {"trw_label": "sensitive"}
    assert policy.stamped({}, Level.SENSITIVE, Level.PERSONAL) == {"trw_label": "sensitive"}


def test_stamping_never_mutates_the_callers_metadata(user_dir: Path) -> None:
    original = {"a": "b"}

    LabelPolicy.current().stamped(original, Level.PERSONAL, Level.TEAM)

    assert original == {"a": "b"}


def test_highest_is_the_maximum_label_of_the_rows_and_team_for_none(user_dir: Path) -> None:
    policy = LabelPolicy.current()

    assert policy.highest([]) is Level.TEAM
    assert policy.highest([_row(), _row(trw_label="personal"), _row(trw_label="sensitive")]) is Level.SENSITIVE
    assert policy.highest([_row("user:alice")]) is Level.PERSONAL


def test_joined_keeps_the_maximum_stamp_and_adds_none_when_neither_has_one(user_dir: Path) -> None:
    policy = LabelPolicy.current()

    assert policy.joined({"k": "v"}, {"x": "y"}) == {"k": "v"}, "no stamp on either side: unchanged"
    assert policy.joined({}, {"trw_label": "personal"}) == {"trw_label": "personal"}
    assert policy.joined({"trw_label": "team"}, {"trw_label": "personal"}) == {"trw_label": "personal"}
    assert policy.joined({"trw_label": "sensitive"}, {"trw_label": "personal"}) == {"trw_label": "sensitive"}
    assert policy.joined({"trw_label": "personal"}, {"trw_label": "bogus"}) == {"trw_label": "sensitive"}, (
        "an unknown stamp is sensitive"
    )


def test_joined_never_lowers_in_either_direction(user_dir: Path) -> None:
    policy = LabelPolicy.current()
    stamps = [None, "team", "personal", "sensitive", "bogus"]
    for left in stamps:
        for right in stamps:
            a = {} if left is None else {"trw_label": left}
            b = {} if right is None else {"trw_label": right}
            for first, second in ((a, b), (b, a)):
                joined = policy.joined(first, second)
                for source in (first, second):
                    assert policy.label_of(_row(**joined)) >= policy.label_of(_row(**source)), (first, second)
