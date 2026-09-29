"""PRD-CORE-306 B71-03 (worker-1): the ``hard_fail`` recovery sidecar must be
bound to the db file's identity (``st_dev``/``st_ino``), not just its path,
so a replaced or restored store at the same path does not inherit a stale
verdict from the file that used to live there.
"""

from __future__ import annotations

from pathlib import Path

from trw_memory.storage._recovery_preflight import (
    classify_recovery_preflight,
    write_recovery_state,
)


def test_hard_fail_verdict_is_not_inherited_by_a_replaced_db_file(tmp_path: Path) -> None:
    """A swapped-in fresh file at the same path must not inherit the old hard_fail."""
    db_path = tmp_path / "memory.db"
    db_path.write_bytes(b"store A - the one that hard-failed recovery")

    write_recovery_state(
        db_path, status="hard_fail", reason="inline_recovery_failed", db_size_bytes=db_path.stat().st_size
    )
    assert classify_recovery_preflight(db_path, inline_max_bytes=1024).classification == "hard_fail"

    # Replace the file at the same path with a fresh one (store B) — new inode.
    db_path.unlink()
    db_path.write_bytes(b"store B - brand new, never recovered")

    preflight = classify_recovery_preflight(db_path, inline_max_bytes=1024)

    assert preflight.classification != "hard_fail", "store B inherited store A's stale hard_fail verdict"
    assert preflight.persisted_status == ""


def test_hard_fail_verdict_still_applies_to_the_same_unmodified_file(tmp_path: Path) -> None:
    """Non-vacuity: an unchanged file still honors its own recorded hard_fail."""
    db_path = tmp_path / "memory.db"
    db_path.write_bytes(b"0123456789")

    write_recovery_state(db_path, status="hard_fail", reason="inline_recovery_failed", db_size_bytes=10)

    assert classify_recovery_preflight(db_path, inline_max_bytes=1024).classification == "hard_fail"


def test_a_sidecar_with_no_recorded_identity_keeps_its_verdict(tmp_path: Path) -> None:
    """A legacy sidecar (no identity field) keeps its hard_fail, as before this change: never fail open."""
    db_path = tmp_path / "memory.db"
    db_path.write_bytes(b"0123456789")
    state_path = db_path.with_name(f"{db_path.name}.recovery.json")
    state_path.write_text('{"status": "hard_fail", "reason": "legacy", "db_size_bytes": 10}', encoding="utf-8")

    assert classify_recovery_preflight(db_path, inline_max_bytes=1024).classification == "hard_fail"


def test_a_failed_recovery_that_removed_the_db_is_remembered_on_the_next_open(tmp_path: Path) -> None:
    """sol r1: strict recovery moved the db away and failed; the hard_fail written with no file must hold."""
    db_path = tmp_path / "memory.db"
    write_recovery_state(db_path, status="hard_fail", reason="inline_recovery_failed", db_size_bytes=0)
    assert not db_path.exists()

    assert classify_recovery_preflight(db_path, inline_max_bytes=1024).classification == "hard_fail"
