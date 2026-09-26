"""Tests for compute_anchor_validity (PRD-CORE-111).

Covers:
- All anchors valid returns 1.0
- Partial validity returns correct fraction
- Empty anchor list returns 1.0
- All anchors missing returns 0.0
- File exists but symbol not in content returns 0.0
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from trw_memory.lifecycle.anchor_validation import compute_anchor_validity


def test_3_of_3_valid_returns_1(tmp_path: Path) -> None:
    """3 anchors, all files+symbols exist -> 1.0."""
    (tmp_path / "a.py").write_text("def foo(): pass")
    (tmp_path / "b.py").write_text("class Bar: pass")
    (tmp_path / "c.py").write_text("def baz(): pass")

    anchors = [
        {"file": "a.py", "symbol_name": "foo"},
        {"file": "b.py", "symbol_name": "Bar"},
        {"file": "c.py", "symbol_name": "baz"},
    ]
    assert compute_anchor_validity(anchors, tmp_path) == 1.0


def test_an_anchor_file_over_the_read_cap_is_not_valid_and_not_read_whole(tmp_path: Path) -> None:
    """C12 rc4: an unbounded read of a multi-GB anchor file could exhaust the shared daemon."""
    from trw_memory.lifecycle.verification import MAX_FILE_SIZE_BYTES

    (tmp_path / "big.py").write_bytes(b"def foo(): pass\n" + b"#" * MAX_FILE_SIZE_BYTES)
    (tmp_path / "small.py").write_text("def foo(): pass")

    assert compute_anchor_validity([{"file": "big.py", "symbol_name": "foo"}], tmp_path) == 0.0
    assert compute_anchor_validity([{"file": "small.py", "symbol_name": "foo"}], tmp_path) == 1.0


def test_2_of_3_returns_067(tmp_path: Path) -> None:
    """3 anchors, 2 valid, 1 missing file -> 0.67."""
    (tmp_path / "a.py").write_text("def foo(): pass")
    (tmp_path / "b.py").write_text("class Bar: pass")
    # c.py does not exist

    anchors = [
        {"file": "a.py", "symbol_name": "foo"},
        {"file": "b.py", "symbol_name": "Bar"},
        {"file": "c.py", "symbol_name": "baz"},
    ]
    result = compute_anchor_validity(anchors, tmp_path)
    assert result == pytest.approx(0.67, abs=0.01)


def test_0_anchors_returns_1(tmp_path: Path) -> None:
    """Empty anchor list returns 1.0 (no anchors = no staleness)."""
    assert compute_anchor_validity([], tmp_path) == 1.0


def test_all_missing_returns_0(tmp_path: Path) -> None:
    """3 anchors, all files missing -> 0.0."""
    anchors = [
        {"file": "missing_a.py", "symbol_name": "foo"},
        {"file": "missing_b.py", "symbol_name": "Bar"},
        {"file": "missing_c.py", "symbol_name": "baz"},
    ]
    assert compute_anchor_validity(anchors, tmp_path) == 0.0


def test_file_exists_but_symbol_missing(tmp_path: Path) -> None:
    """File exists but symbol name is not in content -> 0.0."""
    (tmp_path / "module.py").write_text("def other_function(): pass")

    anchors = [
        {"file": "module.py", "symbol_name": "missing_symbol"},
    ]
    assert compute_anchor_validity(anchors, tmp_path) == 0.0


def test_1_of_2_valid_returns_05(tmp_path: Path) -> None:
    """2 anchors, 1 valid -> 0.5."""
    (tmp_path / "good.py").write_text("def present_fn(): pass")

    anchors = [
        {"file": "good.py", "symbol_name": "present_fn"},
        {"file": "bad.py", "symbol_name": "absent_fn"},
    ]
    assert compute_anchor_validity(anchors, tmp_path) == 0.5


def test_symbol_in_comment_counts(tmp_path: Path) -> None:
    """Symbol name appearing in a comment also counts as found."""
    (tmp_path / "mod.py").write_text("# References MyClass\nclass Other: pass")
    anchors = [{"file": "mod.py", "symbol_name": "MyClass"}]
    assert compute_anchor_validity(anchors, tmp_path) == 1.0


def test_empty_symbol_name_not_counted(tmp_path: Path) -> None:
    """Anchor with empty symbol_name is not counted as valid."""
    (tmp_path / "mod.py").write_text("def foo(): pass")
    anchors = [{"file": "mod.py", "symbol_name": ""}]
    # empty symbol_name skipped -> 0/1 = 0.0
    assert compute_anchor_validity(anchors, tmp_path) == 0.0


def test_project_root_as_string(tmp_path: Path) -> None:
    """project_root can be passed as a str."""
    (tmp_path / "mod.py").write_text("MY_CONST = 42")
    anchors = [{"file": "mod.py", "symbol_name": "MY_CONST"}]
    assert compute_anchor_validity(anchors, str(tmp_path)) == 1.0


def test_anchor_model_instance_valid(tmp_path: Path) -> None:
    """Anchor model instances (not dicts) are handled via anchor.file / anchor.symbol_name."""
    from trw_memory.models.memory import Anchor

    (tmp_path / "mod.py").write_text("def my_func(): pass")
    anchor = Anchor(file="mod.py", symbol_name="my_func")
    result = compute_anchor_validity([anchor], tmp_path)
    assert result == 1.0


def test_anchor_model_instance_invalid_file(tmp_path: Path) -> None:
    """Anchor model instance with missing file returns 0.0 (file not found branch)."""
    from trw_memory.models.memory import Anchor

    anchor = Anchor(file="nonexistent.py", symbol_name="my_func")
    result = compute_anchor_validity([anchor], tmp_path)
    assert result == 0.0


def test_os_error_on_read_skips_anchor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """OSError during file read is caught and the anchor is skipped (not counted).

    Reads now go through ``open_checkout_file_fd`` (PRD-SEC-016 round-2
    finding 2), so the OSError is simulated at ``os.fdopen`` -- the read call
    ``_read_anchor_file`` actually makes -- rather than the retired
    ``Path.read_text``.
    """
    import trw_memory.lifecycle.anchor_validation as anchor_validation_module

    (tmp_path / "mod.py").write_text("def my_func(): pass")
    anchors = [{"file": "mod.py", "symbol_name": "my_func"}]

    def _raise_fdopen(*args: object, **kwargs: object) -> object:
        raise OSError("simulated read error")

    monkeypatch.setattr(anchor_validation_module.os, "fdopen", _raise_fdopen)
    result = compute_anchor_validity(anchors, tmp_path)
    assert result == 0.0


def test_compute_anchor_validity_is_pure(tmp_path: Path) -> None:
    """NFR02 (PRD :435): compute_anchor_validity performs no filesystem writes.

    Snapshots the directory tree (paths + contents + mtimes) before and after a
    call, and asserts the tree is unchanged and its inputs are not mutated.
    """
    (tmp_path / "good.py").write_text("def present_fn(): pass")

    def snapshot(root: Path) -> dict[str, tuple[bytes, float]]:
        state: dict[str, tuple[bytes, float]] = {}
        for p in sorted(root.rglob("*")):
            if p.is_file():
                st = p.stat()
                state[str(p.relative_to(root))] = (p.read_bytes(), st.st_mtime)
        return state

    anchors = [
        {"file": "good.py", "symbol_name": "present_fn"},
        {"file": "bad.py", "symbol_name": "absent_fn"},
    ]

    before = snapshot(tmp_path)
    result = compute_anchor_validity(anchors, tmp_path)
    after = snapshot(tmp_path)

    assert before == after, "compute_anchor_validity mutated the filesystem"
    # Inputs must not be mutated either.
    assert anchors == [
        {"file": "good.py", "symbol_name": "present_fn"},
        {"file": "bad.py", "symbol_name": "absent_fn"},
    ]
    # 1 valid of 2 -> 0.5.
    assert result == 0.5


def test_compute_anchor_validity_does_not_walk_project_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: the function must score anchors directly, never walk the tree.

    This is the regression test for the removed marker-scan bonus, which used
    to call ``Path.iterdir()`` on every directory under ``project_root`` (and
    ``read_text()`` every candidate file) looking for an inline
    ``mcp.trw.recall(id=...)`` comment. Two live trw-mcp servers were measured
    pegged at ~50% CPU for 4h40m because of this scan running on every
    ``trw_learn`` call and every journaled-learn replay across 82 nested git
    worktrees. ``compute_anchor_validity`` must resolve validity purely from
    the anchors' own declared files — it must never call ``iterdir()`` on the
    project root or any subdirectory.
    """
    (tmp_path / "good.py").write_text("def present_fn(): pass")
    subdir = tmp_path / "unrelated_subdir"
    subdir.mkdir()
    (subdir / "other.py").write_text("mcp.trw.recall(id=L-should-not-be-read)\n")

    def _forbidden_iterdir(self: Path) -> object:
        raise AssertionError(f"compute_anchor_validity must not walk the tree (iterdir on {self})")

    monkeypatch.setattr(Path, "iterdir", _forbidden_iterdir)

    anchors = [{"file": "good.py", "symbol_name": "present_fn"}]
    result = compute_anchor_validity(anchors, tmp_path)
    assert result == 1.0


def test_a_recall_marker_never_changes_the_score(tmp_path: Path) -> None:
    """A marker naming the learning, anywhere in the tree, leaves validity unchanged."""
    (tmp_path / "good.py").write_text("def present_fn(): pass  # mcp.trw.recall(id=L-abcd)\n")
    (tmp_path / "notes.md").write_text("mcp.trw.recall(id=L-abcd)\n")
    anchors = [{"file": "good.py", "symbol_name": "present_fn"}, {"file": "gone.py", "symbol_name": "x"}]

    assert compute_anchor_validity(anchors, tmp_path) == 0.5


# --- PRD-SEC-016 round-2 finding 2: a stored (raw-dict) anchor's `file` cannot escape project_root ---


def test_an_absolute_anchor_file_never_reads_outside_the_root(tmp_path: Path) -> None:
    """``root / "/etc/passwd"`` in the old implementation discards ``root`` entirely (pathlib semantics).

    Anchor.file's OWN pydantic validator would reject this if the data
    arrived as an ``Anchor`` instance, but ``_reverify_anchors`` reads raw
    stored dicts straight off a learning -- exactly the shape this test uses
    -- never through that validator.
    """
    outside = tmp_path.parent / f"outside-secret-{tmp_path.name}.txt"
    outside.write_text("root\n")  # the "symbol_name" this test searches for
    try:
        anchors = [{"file": str(outside), "symbol_name": "root"}]

        result = compute_anchor_validity(anchors, tmp_path)

        assert result == 0.0, "an absolute anchor path must never be read, let alone score as valid"
    finally:
        outside.unlink(missing_ok=True)


def test_a_traversal_anchor_file_never_reads_outside_the_root(tmp_path: Path) -> None:
    """A relative ``..`` escape must be refused, not merely 'not found'."""
    outside = tmp_path.parent / f"outside-secret-{tmp_path.name}-2.txt"
    outside.write_text("needle\n")
    try:
        traversal = os.path.relpath(outside, tmp_path)
        assert traversal.startswith("..")
        anchors = [{"file": traversal, "symbol_name": "needle"}]

        result = compute_anchor_validity(anchors, tmp_path)

        assert result == 0.0
    finally:
        outside.unlink(missing_ok=True)


@pytest.mark.skipif(sys.platform == "win32", reason="os.symlink requires elevated privileges on Windows")
def test_a_symlinked_anchor_file_never_reads_outside_the_root(tmp_path: Path) -> None:
    """A symlink INSIDE project_root pointing outside it must be refused, matching FR03's verify_assertions rule."""
    outside = tmp_path.parent / f"outside-secret-{tmp_path.name}-3.txt"
    outside.write_text("def leaked_symbol(): pass\n")
    try:
        (tmp_path / "link.py").symlink_to(outside)
        anchors = [{"file": "link.py", "symbol_name": "leaked_symbol"}]

        result = compute_anchor_validity(anchors, tmp_path)

        assert result == 0.0
    finally:
        outside.unlink(missing_ok=True)


def test_an_anchor_model_instance_with_a_legitimate_relative_file_still_works(tmp_path: Path) -> None:
    """Regression control: the fix must not break the ordinary, validated ``Anchor``-instance case."""
    from trw_memory.models.memory import Anchor

    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "mod.py").write_text("def my_func(): pass")
    anchor = Anchor(file="src/mod.py", symbol_name="my_func")

    assert compute_anchor_validity([anchor], tmp_path) == 1.0
