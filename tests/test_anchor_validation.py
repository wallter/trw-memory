"""Tests for compute_anchor_validity (PRD-CORE-111).

Covers:
- All anchors valid returns 1.0
- Partial validity returns correct fraction
- Empty anchor list returns 1.0
- All anchors missing returns 0.0
- File exists but symbol not in content returns 0.0
"""

from __future__ import annotations

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
    """OSError during file read is caught and the anchor is skipped (not counted)."""
    (tmp_path / "mod.py").write_text("def my_func(): pass")
    anchors = [{"file": "mod.py", "symbol_name": "my_func"}]

    original_read_text = Path.read_text

    def _raise_on_mod(self: Path, *args: object, **kwargs: object) -> str:
        if self.name == "mod.py":
            raise OSError("simulated read error")
        return original_read_text(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "read_text", _raise_on_mod)
    result = compute_anchor_validity(anchors, tmp_path)
    assert result == 0.0


class TestMarkerBonus:
    """FR03/FR05: inline comment marker adds 0.5 to valid_count, capped at 1.0."""

    def test_marker_bonus_lifts_partial_validity(self, tmp_path: Path) -> None:
        """A marker adds 0.5 to valid_count: 1 valid of 2 -> (1+0.5)/2 = 0.75."""
        (tmp_path / "good.py").write_text("def present_fn(): pass")
        # bad.py is missing -> 1 of 2 valid.
        (tmp_path / "notes.md").write_text("See mcp.trw.recall(id=L-a3Fq) for context\n")
        anchors = [
            {"file": "good.py", "symbol_name": "present_fn"},
            {"file": "bad.py", "symbol_name": "absent_fn"},
        ]
        # Without the id: raw 0.5.
        assert compute_anchor_validity(anchors, tmp_path) == 0.5
        # With the id and a marker present: (1.0 + 0.5) / 2 = 0.75.
        assert compute_anchor_validity(anchors, tmp_path, learning_id="L-a3Fq") == 0.75

    def test_marker_bonus_capped_at_one(self, tmp_path: Path) -> None:
        """The 0.5 bonus never pushes validity above 1.0 (min cap).

        2 valid of 2 -> 1.0 raw; +0.5 bonus -> 2.5/2 = 1.25 -> capped 1.0.
        """
        (tmp_path / "a.py").write_text("def foo(): pass")
        (tmp_path / "b.py").write_text("class Bar: pass")
        (tmp_path / "code.py").write_text("# mcp.trw.recall(id=L-a3Fq)\n")
        anchors = [
            {"file": "a.py", "symbol_name": "foo"},
            {"file": "b.py", "symbol_name": "Bar"},
        ]
        assert compute_anchor_validity(anchors, tmp_path, learning_id="L-a3Fq") == 1.0

    def test_marker_bonus_partial_boost(self, tmp_path: Path) -> None:
        """Bonus is a genuine +0.5, not a jump straight to 1.0.

        0 valid anchors of 3 -> 0.0 raw; a marker adds 0.5 -> 0.5/3 = 0.17.
        """
        (tmp_path / "notes.md").write_text("marker mcp.trw.recall(id=L-b2Xp) here\n")
        anchors = [
            {"file": "missing_a.py", "symbol_name": "foo"},
            {"file": "missing_b.py", "symbol_name": "bar"},
            {"file": "missing_c.py", "symbol_name": "baz"},
        ]
        assert compute_anchor_validity(anchors, tmp_path) == 0.0
        result = compute_anchor_validity(anchors, tmp_path, learning_id="L-b2Xp")
        assert result == pytest.approx(0.17, abs=0.01)

    def test_no_marker_no_bonus(self, tmp_path: Path) -> None:
        """learning_id supplied but no matching marker present -> no bonus."""
        (tmp_path / "good.py").write_text("def present_fn(): pass")
        (tmp_path / "notes.md").write_text("no marker here\n")
        anchors = [
            {"file": "good.py", "symbol_name": "present_fn"},
            {"file": "bad.py", "symbol_name": "absent_fn"},
        ]
        assert compute_anchor_validity(anchors, tmp_path, learning_id="L-a3Fq") == 0.5

    def test_marker_for_different_id_ignored(self, tmp_path: Path) -> None:
        """A marker for a DIFFERENT learning id does not grant the bonus."""
        (tmp_path / "good.py").write_text("def present_fn(): pass")
        (tmp_path / "notes.md").write_text("mcp.trw.recall(id=L-zzzz)\n")
        anchors = [
            {"file": "good.py", "symbol_name": "present_fn"},
            {"file": "bad.py", "symbol_name": "absent_fn"},
        ]
        assert compute_anchor_validity(anchors, tmp_path, learning_id="L-a3Fq") == 0.5

    def test_multi_id_marker_matches_member(self, tmp_path: Path) -> None:
        """A comma-separated multi-id marker grants the bonus to each member id."""
        (tmp_path / "good.py").write_text("def present_fn(): pass")
        (tmp_path / "code.py").write_text("# mcp.trw.recall(id=L-aaaa,L-a3Fq)\n")
        anchors = [
            {"file": "good.py", "symbol_name": "present_fn"},
            {"file": "bad.py", "symbol_name": "absent_fn"},
        ]
        # (1 valid + 0.5 bonus) / 2 = 0.75.
        assert compute_anchor_validity(anchors, tmp_path, learning_id="L-a3Fq") == 0.75


def test_compute_anchor_validity_is_pure(tmp_path: Path) -> None:
    """NFR02 (PRD :435): compute_anchor_validity performs no filesystem writes.

    Snapshots the directory tree (paths + contents + mtimes) before and after a
    call that exercises both anchor checks and the marker scan, and asserts the
    tree is unchanged and its inputs are not mutated.
    """
    (tmp_path / "good.py").write_text("def present_fn(): pass")
    (tmp_path / "notes.md").write_text("mcp.trw.recall(id=L-a3Fq)\n")

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
    result = compute_anchor_validity(anchors, tmp_path, learning_id="L-a3Fq")
    after = snapshot(tmp_path)

    assert before == after, "compute_anchor_validity mutated the filesystem"
    # Inputs must not be mutated either.
    assert anchors == [
        {"file": "good.py", "symbol_name": "present_fn"},
        {"file": "bad.py", "symbol_name": "absent_fn"},
    ]
    # (1 valid + 0.5 marker bonus) / 2 = 0.75.
    assert result == 0.75
