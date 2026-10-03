"""UF-PRD-04 slice 2b (AIKIDO census, group C as re-verified): trw-memory writers refuse a planted symlink.

Each writer is driven through the function its production caller uses, with a symlink planted at the path it
writes (the leaf it appends to or rewrites, the predictable ``.tmp`` it staged through, or the sibling lock file).
The file behind the link must keep its bytes (or, for a dangling link, must not be created) and the link must stay
a link. ``append_beneath(lock=True)`` is pinned against a reader holding a shared flock.
"""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

OUTSIDE = b"outside the store -- must never change\n"


def _plant(link: Path, outside_dir: Path, content: bytes = OUTSIDE) -> Path:
    outside_dir.mkdir(parents=True, exist_ok=True)
    victim = outside_dir / link.name
    victim.write_bytes(content)
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(victim)
    return victim


def _untouched(link: Path, victim: Path, content: bytes = OUTSIDE) -> None:
    assert victim.read_bytes() == content, "the writer changed the file behind the link"
    assert link.is_symlink(), "the planted link must still be a link"


# --- safe_fs.append_beneath(lock=True) ------------------------------------------------------------------------


@pytest.mark.skipif(sys.platform == "win32", reason="flock is POSIX; Windows appends unlocked")
def test_locked_append_waits_for_a_shared_lock_reader_then_appends(tmp_path: Path) -> None:
    import fcntl

    from trw_memory.safe_fs import append_beneath

    target = tmp_path / "logs" / "x.jsonl"
    target.parent.mkdir(parents=True)
    target.write_text("first\n", encoding="utf-8")
    done = threading.Event()

    def _writer() -> None:
        append_beneath(tmp_path, "logs/x.jsonl", b"second\n", mode=0o644, lock=True)
        done.set()

    with target.open("r", encoding="utf-8") as reader:
        fcntl.flock(reader.fileno(), fcntl.LOCK_SH)
        thread = threading.Thread(target=_writer)
        thread.start()
        thread.join(timeout=0.5)
        assert not done.is_set(), "the append must wait while a reader holds a shared lock"
        fcntl.flock(reader.fileno(), fcntl.LOCK_UN)
    thread.join(timeout=10)

    assert done.is_set()
    assert target.read_text(encoding="utf-8") == "first\nsecond\n"


# --- storage.persistence -------------------------------------------------------------------------------------


def test_append_jsonl_refuses_a_symlinked_log(tmp_path: Path) -> None:
    from trw_memory.exceptions import StorageError
    from trw_memory.storage.persistence import append_jsonl

    link = tmp_path / "store" / "security-events.jsonl"
    victim = _plant(link, tmp_path / "outside")

    with pytest.raises(StorageError):
        append_jsonl(link, {"event": "x"})

    _untouched(link, victim)


def test_append_jsonl_still_appends_one_line_per_record(tmp_path: Path) -> None:
    from trw_memory.storage.persistence import append_jsonl

    path = tmp_path / "store" / "events.jsonl"
    append_jsonl(path, {"n": 1})
    append_jsonl(path, {"n": 2})

    assert [json.loads(line)["n"] for line in path.read_text(encoding="utf-8").splitlines()] == [1, 2]


def test_lock_for_rmw_refuses_a_dangling_symlinked_lock(tmp_path: Path) -> None:
    """``open("a+b")`` followed a dangling link and created the file it names."""
    from trw_memory.storage.persistence import lock_for_rmw

    path = tmp_path / "store" / "state.yaml"
    path.parent.mkdir(parents=True)
    named = tmp_path / "outside" / "created-through-the-link"
    named.parent.mkdir()
    (path.parent / "state.yaml.lock").symlink_to(named)

    with pytest.raises(OSError), lock_for_rmw(path):
        pass

    assert not named.exists(), "the lock open created the file a dangling link names"


# --- lifecycle.tiers._warm sidecar -------------------------------------------------------------------------------


def test_warm_sidecar_append_refuses_a_symlinked_sidecar(tmp_path: Path) -> None:
    from trw_memory.lifecycle.tiers._warm import WarmTierStore

    store = WarmTierStore(tmp_path)
    link = store._warm_sidecar_path()
    victim = _plant(link, tmp_path / "outside")

    with pytest.raises(Exception):  # the refusal surfaces; the sidecar is not written through
        store.warm_add("L-1", {"id": "L-1", "summary": "s", "detail": "d"}, None)

    _untouched(link, victim)


def test_warm_sidecar_rewrite_refuses_a_symlink_at_its_tmp_name(tmp_path: Path) -> None:
    from trw_memory.lifecycle.tiers._warm import WarmTierStore

    store = WarmTierStore(tmp_path)
    store.warm_add("L-1", {"id": "L-1", "summary": "s", "detail": "d"}, None)
    sidecar = store._warm_sidecar_path()
    link = sidecar.with_name(sidecar.name + ".tmp")
    victim = _plant(link, tmp_path / "outside")

    store.warm_add("L-1", {"id": "L-1", "summary": "changed", "detail": "d"}, None)  # a content change rewrites

    _untouched(link, victim)
    rows = [json.loads(line) for line in sidecar.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert [r.get("summary") for r in rows if r.get("id") == "L-1"] == ["changed"]


# --- tools.maintain stamp -------------------------------------------------------------------------------------


def test_maintenance_stamp_refuses_a_symlinked_state_file(tmp_path: Path) -> None:
    from trw_memory.tools import maintain

    db = tmp_path / "store" / "memory.db"
    db.parent.mkdir(parents=True)
    backend = SimpleNamespace(db_path=db)
    link = db.parent / maintain.MAINTENANCE_STATE_FILE
    victim = _plant(link, tmp_path / "outside", b"{}")

    with pytest.raises(Exception):
        maintain._record_stamp(backend, "ns", attempted_at="2026-10-02T00:00:00Z", succeeded=True)  # type: ignore[arg-type]

    _untouched(link, victim, b"{}")
