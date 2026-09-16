"""Tests for grep_absent assertion verification.

PRD-CORE-086 FR04: verify_assertions() for grep_absent type.
"""

from __future__ import annotations

from pathlib import Path

from trw_memory.lifecycle.verification import verify_assertions
from trw_memory.models.memory import Assertion, AssertionType


class TestGrepAbsent:
    """Test grep_absent assertion type."""

    def test_grep_absent_no_match(self, tmp_path: Path) -> None:
        """Pattern not found in any file -> passed=True."""
        (tmp_path / "safe.py").write_text("def safe_function(): pass")
        assertions = [
            Assertion(type=AssertionType.GREP_ABSENT, pattern="eval\\(", target="*.py"),
        ]
        results = verify_assertions(assertions, tmp_path)
        assert len(results) == 1
        assert results[0].passed is True
        assert "correctly absent" in results[0].evidence

    def test_grep_absent_found(self, tmp_path: Path) -> None:
        """Pattern found in a file -> passed=False."""
        (tmp_path / "bad.py").write_text("result = eval(user_input)")
        assertions = [
            Assertion(type=AssertionType.GREP_ABSENT, pattern="eval\\(", target="*.py"),
        ]
        results = verify_assertions(assertions, tmp_path)
        assert len(results) == 1
        assert results[0].passed is False
        assert "unexpectedly found" in results[0].evidence

    def test_grep_absent_empty_dir(self, tmp_path: Path) -> None:
        """No files matching target -> passed=True (nothing to find)."""
        assertions = [
            Assertion(type=AssertionType.GREP_ABSENT, pattern="danger", target="*.py"),
        ]
        results = verify_assertions(assertions, tmp_path)
        assert results[0].passed is True
        assert "correctly absent" in results[0].evidence

    def test_grep_absent_multiple_files_all_clean(self, tmp_path: Path) -> None:
        (tmp_path / "a.py").write_text("safe code")
        (tmp_path / "b.py").write_text("also safe")
        assertions = [
            Assertion(type=AssertionType.GREP_ABSENT, pattern="exec\\(", target="*.py"),
        ]
        results = verify_assertions(assertions, tmp_path)
        assert results[0].passed is True

    def test_grep_absent_one_file_has_match(self, tmp_path: Path) -> None:
        (tmp_path / "a.py").write_text("safe code")
        (tmp_path / "b.py").write_text("exec('danger')")
        assertions = [
            Assertion(type=AssertionType.GREP_ABSENT, pattern="exec\\(", target="*.py"),
        ]
        results = verify_assertions(assertions, tmp_path)
        assert results[0].passed is False


class TestAbsenceIsNotProvenByNotLooking:
    """A negative assertion over files that were never read is UNVERIFIED.

    ``grep_absent`` / ``glob_absent`` are how a security invariant or a
    forbidden-pattern rule is expressed, so their green is load-bearing. The
    verifier used to reach it by not looking: an ``OSError`` on read hit
    ``continue``, a failed directory walk returned ``[]``, and ``passed`` was
    computed as ``len(matching_files) == 0`` — so an unreadable tree reported
    "pattern correctly absent from 0 file(s)".

    That is this framework's own rule broken inside its own gate: absence of a
    measurement is not a measurement of absence. ``passed=None`` (the state an
    invalid regex already uses) is the honest answer.

    Reported by a cross-family audit 2026-09-12.
    """

    def test_an_unreadable_file_makes_absence_unverified(self, tmp_path: Path) -> None:
        (tmp_path / "readable.py").write_text("safe code")
        unreadable = tmp_path / "unreadable.py"
        unreadable.write_text("danger lives here")
        unreadable.chmod(0o000)
        try:
            assertions = [Assertion(type=AssertionType.GREP_ABSENT, pattern="danger", target="*.py")]
            results = verify_assertions(assertions, tmp_path)
            assert results[0].passed is None, "a file we could not read was counted as proof of absence"
            assert "unverified" in results[0].evidence
        finally:
            unreadable.chmod(0o644)

    def test_an_unwalkable_target_is_unverified_not_absent(self, tmp_path: Path) -> None:
        from unittest.mock import patch

        assertions = [Assertion(type=AssertionType.GREP_ABSENT, pattern="danger", target="*.py")]
        with patch.object(Path, "glob", side_effect=OSError("permission denied")):
            results = verify_assertions(assertions, tmp_path)
        assert results[0].passed is None
        # Asserting the EVIDENCE, not just the verdict. verify_assertions wraps
        # every assertion in a catch-all that also yields passed=None, so the
        # verdict alone cannot tell a handled walk failure from an unhandled
        # crash — the probe that revealed this showed the test staying green with
        # the guard disabled, because len(None) raised and the catch-all caught it.
        assert "could not enumerate" in results[0].evidence
        assert "verification error" not in results[0].evidence
        assert "could not enumerate" in results[0].evidence

    def test_glob_absent_is_also_unverified_when_the_walk_fails(self, tmp_path: Path) -> None:
        from unittest.mock import patch

        assertions = [Assertion(type=AssertionType.GLOB_ABSENT, pattern="", target="*.secret")]
        with patch.object(Path, "glob", side_effect=OSError("permission denied")):
            results = verify_assertions(assertions, tmp_path)
        assert results[0].passed is None
        # The evidence, not just the verdict — see the note on the grep case above.
        assert "could not enumerate" in results[0].evidence
        assert "verification error" not in results[0].evidence

    def test_a_genuinely_clean_readable_tree_still_passes(self, tmp_path: Path) -> None:
        """Non-vacuity partner. Absence must stay PROVABLE, or the fix would have
        turned every negative assertion permanently unverified and the gate would
        be as useless in the other direction."""
        (tmp_path / "a.py").write_text("safe code")
        (tmp_path / "b.py").write_text("also safe")
        assertions = [Assertion(type=AssertionType.GREP_ABSENT, pattern="danger", target="*.py")]
        results = verify_assertions(assertions, tmp_path)
        assert results[0].passed is True
        assert "correctly absent" in results[0].evidence

    def test_a_real_match_still_fails_even_with_an_unreadable_sibling(self, tmp_path: Path) -> None:
        """A found pattern outranks an unread file: the claim is already refuted,
        so downgrading to 'unverified' would hide a confirmed violation."""
        (tmp_path / "bad.py").write_text("danger is right here")
        unreadable = tmp_path / "unreadable.py"
        unreadable.write_text("whatever")
        unreadable.chmod(0o000)
        try:
            assertions = [Assertion(type=AssertionType.GREP_ABSENT, pattern="danger", target="*.py")]
            results = verify_assertions(assertions, tmp_path)
            assert results[0].passed is False
            assert "unexpectedly found" in results[0].evidence
        finally:
            unreadable.chmod(0o644)
