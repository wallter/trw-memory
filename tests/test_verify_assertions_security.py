"""Security tests for assertion verification engine.

PRD-CORE-086: Ensure no shell execution, path traversal, or ReDoS vectors.
PRD-SEC-016 FR03: a symlink inside the checkout is never followed to read a
file outside it.
"""

from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path

import pytest
from pydantic import ValidationError

from trw_memory.lifecycle.verification import MAX_PATTERN_LEN, verify_assertions
from trw_memory.models.memory import Assertion, AssertionType


class TestNoShellExecution:
    """Verify the verification engine has no shell execution paths."""

    def test_no_shell_execution_in_verification(self) -> None:
        """Grep verification.py for dangerous execution functions."""
        from pathlib import Path

        verification_path = Path(__file__).parent.parent / "src" / "trw_memory" / "lifecycle" / "verification.py"
        assert verification_path.exists(), f"verification.py not found at {verification_path}"

        content = verification_path.read_text()
        dangerous_patterns = [
            "subprocess",
            "os.system",
            "os.popen",
            "eval(",
            "exec(",
            "os.exec",
            "__import__",
        ]
        for pattern in dangerous_patterns:
            assert pattern not in content, (
                f"SECURITY: verification.py contains '{pattern}' — "
                f"shell execution is forbidden in assertion verification"
            )


class TestPathTraversal:
    """Verify that path traversal is rejected at the model level."""

    def test_absolute_path_rejected(self) -> None:
        with pytest.raises(ValidationError, match="absolute paths not allowed"):
            Assertion(type=AssertionType.GLOB_EXISTS, pattern="", target="/etc/passwd")

    def test_path_traversal_rejected_leading(self) -> None:
        with pytest.raises(ValidationError, match="path traversal"):
            Assertion(type=AssertionType.GLOB_EXISTS, pattern="", target="../secret.txt")

    def test_path_traversal_rejected_middle(self) -> None:
        with pytest.raises(ValidationError, match="path traversal"):
            Assertion(type=AssertionType.GREP_PRESENT, pattern="x", target="src/../../etc/passwd")

    def test_safe_path_with_dots_allowed(self) -> None:
        """Paths with dots that aren't traversal should be OK."""
        a = Assertion(type=AssertionType.GLOB_EXISTS, pattern="", target="src/.hidden/file.py")
        assert a.target == "src/.hidden/file.py"

    def test_dotdot_in_filename_allowed(self) -> None:
        """A filename containing '..' but not as a path component should be OK."""
        a = Assertion(type=AssertionType.GLOB_EXISTS, pattern="", target="file..name.py")
        assert a.target == "file..name.py"


class TestPatternLengthLimit:
    """Verify pattern length limit for ReDoS mitigation."""

    def test_pattern_length_limit_501(self) -> None:
        with pytest.raises(ValidationError, match="pattern exceeds 500 character limit"):
            Assertion(
                type=AssertionType.GREP_PRESENT,
                pattern="a" * 501,
                target="*.py",
            )

    def test_pattern_length_limit_500_ok(self) -> None:
        a = Assertion(
            type=AssertionType.GREP_PRESENT,
            pattern="a" * 500,
            target="*.py",
        )
        assert len(a.pattern) == 500

    def test_pattern_length_limit_1000(self) -> None:
        with pytest.raises(ValidationError, match="pattern exceeds 500 character limit"):
            Assertion(
                type=AssertionType.GREP_PRESENT,
                pattern="x" * 1000,
                target="*.py",
            )


class TestPatternCompileIsBounded:
    """C12 rc3: ``regex`` unrolls counted repeats while compiling, before any deadline; refuse, never a verdict."""

    @pytest.mark.parametrize(
        "pattern",
        [
            "a{1000000}",
            "(a{1000}){1000}",
            "a{1000000,}",
            "a{0000001000000}",
            "a{0001000000}",
            "(?x)a{1000 000}",
            "(?V1x)a{1000 000}",
            "(?V0x)a{1000 000}",
            "(?i-x:b)(?x:a{1000 000})",
        ],
    )
    @pytest.mark.parametrize("kind", [AssertionType.GREP_PRESENT, AssertionType.GREP_ABSENT])
    def test_an_expensive_pattern_is_unverified_without_compiling(
        self, tmp_path: Path, pattern: str, kind: AssertionType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from trw_memory.lifecycle import verification

        compiled: list[str] = []
        monkeypatch.setattr(verification.regex, "compile", lambda p, *a, **k: compiled.append(p))
        (tmp_path / "a.txt").write_text("a" * 10 + "\n")

        [result] = verify_assertions([Assertion(type=kind, pattern=pattern, target="a.txt")], tmp_path)

        assert (result.passed, compiled) == (None, []), result
        assert "MAX_PATTERN_COST" in result.evidence

    @pytest.mark.parametrize("pattern", ["a{2,}", r"\d{4}-\d{2}-\d{2}", "[0-9a-f]{40}|a{3}", "a{e<=1}"])
    def test_an_ordinary_counted_pattern_still_verifies(self, tmp_path: Path, pattern: str) -> None:
        (tmp_path / "a.txt").write_text("aaa 2026-09-25 " + "f" * 40 + "\n")

        [result] = verify_assertions(
            [Assertion(type=AssertionType.GREP_PRESENT, pattern=pattern, target="a.txt")], tmp_path
        )

        assert result.passed is True, result


@pytest.mark.skipif(sys.platform == "win32", reason="AF_UNIX sockets and FIFOs are POSIX")
class TestGlobCountsSpecialFiles:
    """C12 rc3: the walker dropped sockets, FIFOs and devices, so ``glob_absent`` certified an existing socket absent."""

    @pytest.fixture
    def checkout(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        import socket

        monkeypatch.chdir(tmp_path)  # a relative bind keeps the socket path under the AF_UNIX length limit
        server = socket.socket(socket.AF_UNIX)
        server.bind("control.sock")
        server.close()
        os.mkfifo(tmp_path / "events.fifo")
        return tmp_path

    @pytest.mark.parametrize("target", ["control.sock", "events.fifo", "*.sock"])
    def test_both_glob_modes_see_a_special_file(self, checkout: Path, target: str) -> None:
        exists, absent = verify_assertions(
            [
                Assertion(type=AssertionType.GLOB_EXISTS, pattern="", target=target),
                Assertion(type=AssertionType.GLOB_ABSENT, pattern="", target=target),
            ],
            checkout,
        )
        assert (exists.passed, absent.passed) == (True, False), (exists, absent)

    def test_a_trailing_slash_still_matches_directories_only(self, checkout: Path) -> None:
        [result] = verify_assertions(
            [Assertion(type=AssertionType.GLOB_ABSENT, pattern="", target="control.sock/")], checkout
        )
        assert result.passed is True, result

    def test_grep_never_opens_a_fifo(self, checkout: Path) -> None:
        [result] = verify_assertions(
            [Assertion(type=AssertionType.GREP_PRESENT, pattern="x", target="events.fifo")], checkout
        )
        assert result.passed is False, result


@pytest.mark.skipif(sys.platform == "win32", reason="os.symlink requires elevated privileges on Windows")
class TestCatastrophicBacktrackingIsBounded:
    """C12: a caller-supplied pattern cannot hang the shared daemon; it comes back unverified."""

    _EVIL = "(a|aa)+$"  # exponential under backtracking; stdlib re never returns on this input

    @pytest.fixture
    def checkout(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        from trw_memory.lifecycle import verification

        monkeypatch.setattr(verification, "GREP_MATCH_DEADLINE_SECONDS", 0.3)
        (tmp_path / "a.txt").write_text("a" * 5000 + "!\n")
        (tmp_path / "b.txt").write_text("needle\n")
        return tmp_path

    @pytest.mark.parametrize("kind", [AssertionType.GREP_PRESENT, AssertionType.GREP_ABSENT])
    def test_a_pathological_pattern_returns_unverified_within_the_budget(
        self, checkout: Path, kind: AssertionType
    ) -> None:
        import time

        started = time.monotonic()
        [result] = verify_assertions([Assertion(type=kind, pattern=self._EVIL, target="a.txt")], checkout)
        assert time.monotonic() - started < 3
        assert result.passed is None
        assert "budget" in result.evidence

    def test_the_budget_covers_the_whole_call_not_each_assertion(self, checkout: Path) -> None:
        import time

        evil = [Assertion(type=AssertionType.GREP_PRESENT, pattern=self._EVIL, target="a.txt")] * 20
        started = time.monotonic()
        results = verify_assertions(
            [*evil, Assertion(type=AssertionType.GREP_PRESENT, pattern="needle", target="b.txt")], checkout
        )
        assert time.monotonic() - started < 3, "20 assertions must share one budget, not take 20 of them"
        assert [r.passed for r in results] == [None] * 21

    def test_an_ordinary_pattern_still_verifies(self, checkout: Path) -> None:
        [present, absent] = verify_assertions(
            [
                Assertion(type=AssertionType.GREP_PRESENT, pattern=r"ne+dle", target="b.txt"),
                Assertion(type=AssertionType.GREP_ABSENT, pattern=r"ne+dle", target="a.txt"),
            ],
            checkout,
        )
        assert (present.passed, absent.passed) == (True, True)


class TestSymlinkEscape:
    """PRD-SEC-016 FR03: a symlink inside the checkout never proves a claim about a file outside it."""

    def test_a_symlink_out_of_the_root_is_never_read(self, tmp_path: Path) -> None:
        """Ad731f961 returns passed=True with evidence naming link.txt as a match -- the bug this closes."""
        checkout = tmp_path / "checkout"
        checkout.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        secret = outside / "secret.txt"
        secret.write_text("the pattern lives here, but outside the checkout")
        (checkout / "link.txt").symlink_to(secret)

        results = verify_assertions(
            [Assertion(type=AssertionType.GREP_PRESENT, pattern="the pattern", target="link.txt")], checkout
        )

        assert results[0].passed is None
        assert "outside the checkout" in results[0].evidence
        assert "link.txt" in results[0].evidence

    def test_a_symlinked_directory_is_never_read(self, tmp_path: Path) -> None:
        """grep_absent over a symlinked directory cannot claim absence of a file it declined to read.

        ``linkdir/*.txt`` is a glob whose directory component IS the symlink
        (Python 3.13+'s ``**`` no longer descends into a symlinked directory
        at all, so that spelling proves nothing here; naming the linked
        directory directly still resolves it the old way).
        """
        checkout = tmp_path / "checkout"
        checkout.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "note.txt").write_text("forbidden content")
        (checkout / "linkdir").symlink_to(outside)

        results = verify_assertions(
            [Assertion(type=AssertionType.GREP_ABSENT, pattern="forbidden", target="linkdir/*.txt")], checkout
        )

        assert results[0].passed is None

    def test_a_regular_in_root_file_is_unaffected(self, tmp_path: Path) -> None:
        """The containment check must not regress the ordinary, no-symlink case."""
        checkout = tmp_path / "checkout"
        checkout.mkdir()
        (checkout / "real.txt").write_text("hello_world")

        present = verify_assertions(
            [Assertion(type=AssertionType.GREP_PRESENT, pattern="hello_world", target="*.txt")], checkout
        )
        absent = verify_assertions(
            [Assertion(type=AssertionType.GREP_ABSENT, pattern="nope", target="*.txt")], checkout
        )

        assert present[0].passed is True
        assert absent[0].passed is True


@pytest.mark.skipif(sys.platform == "win32", reason="os.symlink requires elevated privileges on Windows")
class TestGlobSymlinkEscape:
    """PRD-SEC-016 round-4 finding 1: glob_exists/glob_absent must not count a match outside the checkout.

    ``_verify_glob`` never had the containment check ``_verify_grep`` gained
    in FR03 -- it counted whatever ``project_root.glob(target)`` returned,
    including a match reached only through a symlink pointing outside the
    checkout, and reported the count as a pass. This is an existence leak,
    not a content leak: even without reading the target's bytes, a
    ``glob_exists`` true/false answer discloses whether another tenant's file
    (or a specific path like ``../other-checkout/secret``) exists.
    """

    def test_glob_exists_does_not_pass_on_a_symlinked_match(self, tmp_path: Path) -> None:
        """Ad731f961-era bug: passed=True (with a match count) through a symlink to an outside file."""
        checkout = tmp_path / "checkout"
        checkout.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "secret.txt").write_text("another tenant's file")
        (checkout / "link.txt").symlink_to(outside / "secret.txt")

        results = verify_assertions(
            [Assertion(type=AssertionType.GLOB_EXISTS, pattern="", target="link.txt")], checkout
        )

        assert results[0].passed is None
        assert "outside the checkout" in results[0].evidence

    def test_glob_exists_does_not_pass_on_a_match_behind_a_symlinked_directory(self, tmp_path: Path) -> None:
        checkout = tmp_path / "checkout"
        checkout.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "note.txt").write_text("forbidden")
        (checkout / "linkdir").symlink_to(outside)

        results = verify_assertions(
            [Assertion(type=AssertionType.GLOB_EXISTS, pattern="", target="linkdir/*.txt")], checkout
        )

        assert results[0].passed is None

    def test_glob_absent_does_not_pass_when_the_only_match_is_outside_the_checkout(self, tmp_path: Path) -> None:
        """A symlinked match must not let glob_absent report a false 'correctly no files matching'."""
        checkout = tmp_path / "checkout"
        checkout.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "note.txt").write_text("forbidden")
        (checkout / "linkdir").symlink_to(outside)

        results = verify_assertions(
            [Assertion(type=AssertionType.GLOB_ABSENT, pattern="", target="linkdir/*.txt")], checkout
        )

        assert results[0].passed is None

    def test_glob_exists_and_glob_absent_are_unaffected_by_a_regular_in_root_file(self, tmp_path: Path) -> None:
        """Regression control: the containment check must not break the ordinary, no-symlink case."""
        checkout = tmp_path / "checkout"
        checkout.mkdir()
        (checkout / "real.txt").write_text("content")

        exists = verify_assertions([Assertion(type=AssertionType.GLOB_EXISTS, pattern="", target="*.txt")], checkout)
        absent = verify_assertions([Assertion(type=AssertionType.GLOB_ABSENT, pattern="", target="*.md")], checkout)

        assert exists[0].passed is True
        assert absent[0].passed is True


@pytest.mark.skipif(sys.platform == "win32", reason="os.symlink requires elevated privileges on Windows")
class TestRootAncestorSwapRefused:
    """PRD-SEC-016 round-7 finding 1: a swapped ANCESTOR of ``project_root`` must refuse, not silently enumerate.

    ``_iter_files`` enumerates with ``project_root.glob(target)`` and
    ``_check_containment`` re-resolves ``project_root`` -- both ordinary,
    symlink-following pathlib operations computed fresh at call time. A
    tenant who can write an ancestor of a DIFFERENT tenant's checkout root
    (the daemon can serve nested checkouts) can swap that ancestor for a
    symlink between grant time and a ``memory_verify`` call: both operations
    then silently agree on the redirected location, giving a file-existence
    oracle over an attacker-chosen directory with no race and no content
    read required. ``verify_assertions`` now refuses the WHOLE batch, up
    front, when any component of ``project_root`` itself resolves through a
    symlink.
    """

    def test_verify_assertions_refuses_when_an_ancestor_of_root_is_a_symlink(self, tmp_path: Path) -> None:
        real_parent = tmp_path / "real_parent"
        real_parent.mkdir()
        checkout = real_parent / "checkout"
        checkout.mkdir()
        (checkout / "present.txt").write_text("hello")

        swapped_ancestor = tmp_path / "swapped"
        swapped_ancestor.symlink_to(real_parent)
        redirected_root = swapped_ancestor / "checkout"

        results = verify_assertions(
            [Assertion(type=AssertionType.GLOB_EXISTS, pattern="", target="present.txt")], redirected_root
        )

        assert results[0].passed is None
        assert "could not be verified" in results[0].evidence
        assert "swapped for a symlink" in results[0].evidence

    def test_verify_assertions_is_unaffected_when_no_ancestor_is_a_symlink(self, tmp_path: Path) -> None:
        """Regression control: an ordinary, untampered root must not be refused by the new check."""
        checkout = tmp_path / "checkout"
        checkout.mkdir()
        (checkout / "present.txt").write_text("hello")

        results = verify_assertions(
            [Assertion(type=AssertionType.GLOB_EXISTS, pattern="", target="present.txt")], checkout
        )

        assert results[0].passed is True


@pytest.mark.skipif(sys.platform == "win32", reason="os.symlink requires elevated privileges on Windows")
class TestAncestorSwappedDuringEnumeration:
    """PRD-SEC-016 round-8 finding 1: a restore-then-redirect race across the ONE anchor open.

    The round-7 fix (``TestRootAncestorSwapRefused`` above) closed the case
    where an ancestor is ALREADY a symlink at call time. Round-8's finding is
    narrower and came AFTER that fix shipped: the old code's check
    (``open_anchored_walk`` + immediate ``os.close()``) and its enumeration
    (``project_root.glob(target)``, run moments later) were two INDEPENDENT
    path-based resolutions -- a tenant who can write the ancestor could
    restore it (real, non-symlink) for the check, let the check pass, then
    swap it back to a symlink before the glob ran, and the glob would
    silently follow it. This plants exactly that sequence -- the ancestor is
    swapped in the moment BETWEEN the anchor open returning and the
    enumeration that follows it -- and proves the fix: the already-open fd
    is immune, because nothing downstream of it re-resolves the ancestor's
    path string.
    """

    def test_swap_immediately_after_the_anchor_open_does_not_redirect_enumeration(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # `swappable` starts as a REAL directory (not a symlink) -- an
        # already-symlinked ancestor is round-7's separate, already-closed
        # case (it fails the anchor open outright, before any enumeration
        # question even arises). Round-8's finding is the restore-THEN-
        # redirect sequence: real at check time, swapped for a symlink the
        # instant after.
        swappable = tmp_path / "swappable"
        swappable.mkdir()
        checkout = swappable / "checkout"
        checkout.mkdir()
        (checkout / "present.txt").write_text("hello")

        decoy_parent = tmp_path / "decoy_parent"
        decoy_parent.mkdir()
        decoy_checkout = decoy_parent / "checkout"
        decoy_checkout.mkdir()
        # No present.txt here at all -- and a distinguishing marker file, so
        # a redirect would be visible in the evidence rather than merely
        # producing a coincidentally-identical count.
        (decoy_checkout / "decoy-only.txt").write_text("this must never be counted")

        root = swappable / "checkout"

        from trw_memory.lifecycle import verification as verification_module

        real_open_anchored_walk = verification_module.open_anchored_walk

        def redirecting_open(path: Path) -> int:
            # The anchor open runs against the REAL (untampered, non-symlink)
            # ancestor -- this is the "restore it for the check" half of the
            # attack.
            fd = real_open_anchored_walk(path)
            # The instant the anchor is open, redirect the ancestor -- the
            # "swap it back before the glob runs" half. Renaming (not
            # deleting) the original directory keeps its inode -- and this
            # fd, which was opened by inode, not by path -- fully intact and
            # readable; only a FRESH by-name lookup of "swappable/checkout"
            # is affected. The old code's subsequent ``project_root.glob
            # (target)`` was exactly such a fresh, by-name lookup, so it
            # would resolve THIS new destination; the fix's enumeration,
            # which never re-resolves a path string, must not.
            swappable.rename(tmp_path / "swappable-real")
            swappable.symlink_to(decoy_parent)
            return fd

        monkeypatch.setattr(verification_module, "open_anchored_walk", redirecting_open)

        results = verify_assertions([Assertion(type=AssertionType.GLOB_EXISTS, pattern="", target="present.txt")], root)

        assert results[0].passed is True
        assert "1 file(s) found" in results[0].evidence
        assert "decoy" not in results[0].evidence

    def test_a_second_call_after_the_swap_completes_does_see_the_redirect(self, tmp_path: Path) -> None:
        """Non-vacuity partner: the swap mechanism itself genuinely works, so the immunity above is the fix, not an inert test double."""
        swappable = tmp_path / "swappable"
        swappable.mkdir()
        checkout = swappable / "checkout"
        checkout.mkdir()
        (checkout / "present.txt").write_text("hello")

        decoy_parent = tmp_path / "decoy_parent"
        decoy_parent.mkdir()
        (decoy_parent / "checkout").mkdir()

        root = swappable / "checkout"

        first = verify_assertions([Assertion(type=AssertionType.GLOB_EXISTS, pattern="", target="present.txt")], root)
        assert first[0].passed is True

        shutil.rmtree(swappable)
        swappable.symlink_to(decoy_parent)

        second = verify_assertions([Assertion(type=AssertionType.GLOB_EXISTS, pattern="", target="present.txt")], root)
        assert second[0].passed is None
        assert "swapped for a symlink" in second[0].evidence


@pytest.mark.skipif(sys.platform == "win32", reason="os.symlink requires elevated privileges on Windows")
class TestSymlinkOracleParity:
    """PRD-SEC-016 round-8 finding 2: a live vs. a dangling outside target must answer IDENTICALLY.

    The old ``_check_containment``/``path.is_file()`` pairing classified a
    symlinked candidate only by resolving what it points at: a LIVE outside
    target hit the "outside the checkout" bucket, but a DANGLING one made
    ``path.is_file()`` return ``False`` and the code ``continue``d past it
    silently -- invisible to ``grep_absent``/``glob_absent``, which then
    reported a false "correctly absent"/"correctly no files matching." That
    is an existence oracle: the assertion's own verdict discloses whether
    the thing on the other side of the link exists. The fix classifies a
    symlinked candidate the instant ``os.DirEntry``/a fresh ``fstatat`` says
    so -- before ever looking at what it points to -- so the verdict cannot
    depend on that.
    """

    @pytest.fixture
    def live_target(self, tmp_path: Path) -> Path:
        checkout = tmp_path / "checkout-live"
        checkout.mkdir()
        outside = tmp_path / "outside-live"
        outside.mkdir()
        (outside / "secret.txt").write_text("forbidden")
        (checkout / "link.txt").symlink_to(outside / "secret.txt")
        return checkout

    @pytest.fixture
    def dangling_target(self, tmp_path: Path) -> Path:
        checkout = tmp_path / "checkout-dangling"
        checkout.mkdir()
        (checkout / "link.txt").symlink_to(tmp_path / "outside-dangling" / "never-created.txt")
        return checkout

    @pytest.mark.parametrize(
        ("assertion_type", "pattern"),
        [
            (AssertionType.GREP_PRESENT, "forbidden"),
            (AssertionType.GREP_ABSENT, "forbidden"),
            (AssertionType.GLOB_EXISTS, ""),
            (AssertionType.GLOB_ABSENT, ""),
        ],
    )
    def test_live_and_dangling_symlinked_targets_answer_identically(
        self,
        live_target: Path,
        dangling_target: Path,
        assertion_type: AssertionType,
        pattern: str,
    ) -> None:
        live_result = verify_assertions(
            [Assertion(type=assertion_type, pattern=pattern, target="link.txt")], live_target
        )[0]
        dangling_result = verify_assertions(
            [Assertion(type=assertion_type, pattern=pattern, target="link.txt")], dangling_target
        )[0]

        # Both must be UNVERIFIED -- neither confirmed present nor confirmed
        # absent -- and, critically, the SAME verdict regardless of whether
        # the thing on the other side of the symlink exists.
        assert live_result.passed is None, live_result
        assert dangling_result.passed is None, dangling_result
        assert "outside the checkout" in live_result.evidence
        assert "outside the checkout" in dangling_result.evidence

    def test_an_intermediate_symlinked_directory_also_answers_identically(self, tmp_path: Path) -> None:
        """The same parity for ``linkdir/*.txt`` -- an intermediate component, not the leaf."""
        live_checkout = tmp_path / "live-checkout"
        live_checkout.mkdir()
        live_outside = tmp_path / "live-outside"
        live_outside.mkdir()
        (live_outside / "note.txt").write_text("forbidden")
        (live_checkout / "linkdir").symlink_to(live_outside)

        dangling_checkout = tmp_path / "dangling-checkout"
        dangling_checkout.mkdir()
        (dangling_checkout / "linkdir").symlink_to(tmp_path / "dangling-outside")

        live = verify_assertions(
            [Assertion(type=AssertionType.GLOB_EXISTS, pattern="", target="linkdir/*.txt")], live_checkout
        )[0]
        dangling = verify_assertions(
            [Assertion(type=AssertionType.GLOB_EXISTS, pattern="", target="linkdir/*.txt")], dangling_checkout
        )[0]

        assert live.passed is None, live
        assert dangling.passed is None, dangling


class TestIncompleteClassificationIsUnverifiedNotAbsent:
    """PRD-SEC-016 round-9 review: a candidate the walk could not classify or enumerate is UNVERIFIED, never silently a passing absence.

    Round-8's fix classified every matched candidate with a fresh, dir_fd-
    anchored stat instead of cached scandir metadata (finding 2) -- but that
    stat, or the actual open behind it, can still fail on its own (the
    candidate vanished, or a nested subtree could not be scanned at all). A
    round-9 review found the walk was silently treating "could not tell" the
    same as "not a match": a `grep_absent`/`glob_absent` claim would then
    pass over ground it never actually inspected -- this framework's own
    "absence of a measurement is not a measurement of absence" rule, broken
    inside its own gate. Both cases below are failing-first against the
    pre-fix walk (which dropped these candidates with a bare ``continue``).
    """

    def test_a_matched_candidate_that_vanishes_before_classification_is_unverified(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "vanished.py").write_text("danger lives here")
        real_stat = os.stat

        def flaky_stat(path: object, *args: object, **kwargs: object) -> os.stat_result:
            if path == "vanished.py":
                raise FileNotFoundError(2, "No such file or directory", "vanished.py")
            return real_stat(path, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(os, "stat", flaky_stat)

        results = verify_assertions(
            [Assertion(type=AssertionType.GREP_ABSENT, pattern="danger", target="*.py")], tmp_path
        )

        assert results[0].passed is None, results[0]
        assert "could not classify" in results[0].evidence, results[0]

    def test_a_nested_scan_failure_reached_through_double_star_is_unverified(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "leaked.py").write_text("danger lives here too")

        real_open = os.open
        real_scandir = os.scandir
        sub_fds: set[int] = set()

        def spying_open(path: object, *args: object, **kwargs: object) -> int:
            fd = real_open(path, *args, **kwargs)  # type: ignore[arg-type]
            if path == "sub":
                sub_fds.add(fd)
            return fd

        def flaky_scandir(path: object) -> object:
            if isinstance(path, int) and path in sub_fds:
                raise PermissionError(13, "Permission denied")
            return real_scandir(path)  # type: ignore[arg-type]

        monkeypatch.setattr(os, "open", spying_open)
        monkeypatch.setattr(os, "scandir", flaky_scandir)

        results = verify_assertions(
            [Assertion(type=AssertionType.GREP_ABSENT, pattern="danger", target="**/*.py")], tmp_path
        )

        # The file that WOULD have proven the pattern present sits entirely
        # inside the subtree the walk could not scan -- so "correctly
        # absent" would have been a false claim; the fix must refuse it.
        assert results[0].passed is None, results[0]
        assert "could not classify" in results[0].evidence, results[0]

    def test_a_genuinely_scannable_double_star_tree_still_passes(self, tmp_path: Path) -> None:
        """Non-vacuity partner: an ordinary, fully-scannable nested tree is unaffected by the incomplete-tracking above."""
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "safe.py").write_text("nothing hazardous here")

        results = verify_assertions(
            [Assertion(type=AssertionType.GREP_ABSENT, pattern="danger", target="**/*.py")], tmp_path
        )

        assert results[0].passed is True, results[0]


class TestDuplicateEnumerationIsCollapsed:
    """PRD-SEC-016 round-9 review, P2: adjacent ``**`` components must not double-count one real file."""

    def test_adjacent_double_star_components_report_each_file_once(self, tmp_path: Path) -> None:
        nested = tmp_path / "a" / "b"
        nested.mkdir(parents=True)
        (nested / "target.py").write_text("pass")

        results = verify_assertions(
            [Assertion(type=AssertionType.GLOB_EXISTS, pattern="", target="**/**/*.py")], tmp_path
        )

        assert results[0].passed is True, results[0]
        assert "1 file(s) found" in results[0].evidence, results[0]

    def test_adjacent_double_star_visits_each_directory_once(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Round-10 review, item 2: the walk itself must not revisit a subtree, not merely dedup its output afterward.

        The prior test above only proves the REPORTED file count is right --
        it would pass even if the walk scanned ``a/b`` twice and merely
        deduplicated the two identical results. This spies on every
        directory the walk actually opens (identified by ``(st_dev,
        st_ino)``, not by name, so two different opens of the SAME directory
        are unambiguously the same sample) and asserts each one is visited
        exactly once.
        """
        nested = tmp_path / "a" / "b"
        nested.mkdir(parents=True)
        (nested / "target.py").write_text("pass")

        real_open = os.open
        visited_dir_identities: list[tuple[int, int]] = []

        def spying_open(path: object, *args: object, **kwargs: object) -> int:
            fd = real_open(path, *args, **kwargs)  # type: ignore[arg-type]
            st = os.fstat(fd)
            import stat as stat_module

            if stat_module.S_ISDIR(st.st_mode):
                visited_dir_identities.append((st.st_dev, st.st_ino))
            return fd

        monkeypatch.setattr(os, "open", spying_open)

        results = verify_assertions(
            [Assertion(type=AssertionType.GLOB_EXISTS, pattern="", target="**/**/*.py")], tmp_path
        )

        assert results[0].passed is True, results[0]
        assert len(visited_dir_identities) == len(set(visited_dir_identities)), visited_dir_identities


class TestGlobDirectoryMatching:
    """PRD-SEC-016 round-10 review, item 1: a glob target naming a directory must match.

    ``_walk_checkout`` only fed ``_WalkResult.files`` from regular-file
    leaves, so ``glob_exists``/``glob_absent`` never matched a directory --
    unlike the ``pathlib.Path.glob("src")`` it replaced, which does. This
    keeps every no-follow security property (descriptor-anchored walk, a
    symlinked directory is still refused rather than matched, and the same
    unverified response whether the thing behind an outside symlink exists
    or is dangling) while restoring the directory-match behavior.
    """

    def test_glob_exists_matches_a_bare_directory_target(self, tmp_path: Path) -> None:
        (tmp_path / "src").mkdir()

        results = verify_assertions([Assertion(type=AssertionType.GLOB_EXISTS, pattern="", target="src")], tmp_path)

        assert results[0].passed is True, results[0]
        assert "1 file(s) found" in results[0].evidence, results[0]

    def test_glob_absent_reports_false_when_a_directory_target_exists(self, tmp_path: Path) -> None:
        (tmp_path / "src").mkdir()

        results = verify_assertions([Assertion(type=AssertionType.GLOB_ABSENT, pattern="", target="src")], tmp_path)

        assert results[0].passed is False, results[0]

    def test_glob_exists_matches_a_wildcard_directory_target(self, tmp_path: Path) -> None:
        (tmp_path / "build-output").mkdir()

        results = verify_assertions([Assertion(type=AssertionType.GLOB_EXISTS, pattern="", target="build-*")], tmp_path)

        assert results[0].passed is True, results[0]

    @pytest.mark.skipif(sys.platform == "win32", reason="os.symlink requires elevated privileges on Windows")
    def test_a_symlinked_directory_target_is_refused_not_matched(self, tmp_path: Path) -> None:
        outside = tmp_path / "outside"
        outside.mkdir()

        checkout = tmp_path / "checkout"
        checkout.mkdir()
        (checkout / "linkdir").symlink_to(outside)

        results = verify_assertions([Assertion(type=AssertionType.GLOB_EXISTS, pattern="", target="linkdir")], checkout)

        # Never a silent False/True: an outside directory reached only via a
        # symlink is neither confirmed present INSIDE the checkout nor
        # confirmed absent.
        assert results[0].passed is None, results[0]
        assert "outside the checkout" in results[0].evidence, results[0]

    @pytest.mark.skipif(sys.platform == "win32", reason="os.symlink requires elevated privileges on Windows")
    def test_symlinked_directory_target_answers_identically_live_vs_dangling(self, tmp_path: Path) -> None:
        live_checkout = tmp_path / "live-checkout"
        live_checkout.mkdir()
        live_outside = tmp_path / "live-outside"
        live_outside.mkdir()
        (live_checkout / "linkdir").symlink_to(live_outside)

        dangling_checkout = tmp_path / "dangling-checkout"
        dangling_checkout.mkdir()
        (dangling_checkout / "linkdir").symlink_to(tmp_path / "never-created-outside")

        live = verify_assertions(
            [Assertion(type=AssertionType.GLOB_EXISTS, pattern="", target="linkdir")], live_checkout
        )[0]
        dangling = verify_assertions(
            [Assertion(type=AssertionType.GLOB_EXISTS, pattern="", target="linkdir")], dangling_checkout
        )[0]

        assert live.passed is None, live
        assert dangling.passed is None, dangling
        assert live.evidence == dangling.evidence, (live.evidence, dangling.evidence)


class TestGlobTraversalIsBounded:
    """C12-R: a caller-supplied glob cannot make the walk combinatorial; an exhausted bound is unverified."""

    _TARGET = "**/a/" * 10 + "**/missing"  # 21 components; unmemoized, >1,000,000 listings over 31 dirs

    @staticmethod
    def _chain(root: Path, depth: int = 31) -> None:
        (root / "/".join(["a"] * depth)).mkdir(parents=True)

    @staticmethod
    def _count_listings(monkeypatch: pytest.MonkeyPatch, cap: int) -> list[int]:
        """Count every directory listing; past *cap* abort, so the unbounded walk fails fast rather than hanging."""
        real_scandir = os.scandir
        calls = [0]

        def counting(path: object) -> object:
            calls[0] += 1
            if calls[0] > cap:
                raise RuntimeError(f"more than {cap} directory listings")
            return real_scandir(path)  # type: ignore[call-overload]

        monkeypatch.setattr(os, "scandir", counting)
        return calls

    def test_the_walk_lists_each_directory_once_per_pattern_component(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._chain(tmp_path)
        linear = 32 * 21  # (31 dirs + the root) x pattern components
        calls = self._count_listings(monkeypatch, cap=linear * 4)

        [result] = verify_assertions(
            [Assertion(type=AssertionType.GLOB_ABSENT, pattern="", target=self._TARGET)], tmp_path
        )

        assert calls[0] <= linear, calls
        assert result.passed is True, result

    @pytest.mark.parametrize("kind", [AssertionType.GLOB_ABSENT, AssertionType.GLOB_EXISTS, AssertionType.GREP_ABSENT])
    @pytest.mark.parametrize("bound", ["MAX_WALK_WORK", "GREP_MATCH_DEADLINE_SECONDS"])
    def test_an_exhausted_bound_is_unverified_never_verified(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: AssertionType, bound: str
    ) -> None:
        from trw_memory.lifecycle import verification

        self._chain(tmp_path, depth=5)
        (tmp_path / "a" / "a" / "a" / "x.py").write_text("needle\n")
        monkeypatch.setattr(verification, bound, 0)

        [result] = verify_assertions([Assertion(type=kind, pattern="nothing-here", target="**/*.py")], tmp_path)

        assert result.passed is None, result

    def test_every_assertion_after_the_walk_budget_runs_out_is_unverified_cleanly(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An earlier walk can leave the shared work budget negative; a later walk must not crash on it."""
        from trw_memory.lifecycle import verification

        self._chain(tmp_path, depth=5)
        monkeypatch.setattr(verification, "MAX_WALK_WORK", 3)
        kinds = [AssertionType.GLOB_EXISTS, AssertionType.GREP_PRESENT, AssertionType.GLOB_ABSENT]

        results = verify_assertions([Assertion(type=k, pattern="x", target="**/*.py") for k in kinds], tmp_path)

        assert [r.passed for r in results] == [None, None, None]
        assert not [r.evidence for r in results if "verification error" in (r.evidence or "")]

    def test_a_deadline_that_passes_while_classifying_entries_is_unverified(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Code review: the listing check ran before the per-entry stats, so a slow classification still passed."""
        import time
        from types import SimpleNamespace

        from trw_memory.lifecycle import _checkout_walk

        checkout = tmp_path / "checkout"  # one file, no subdirectory: no listing follows its stat
        checkout.mkdir()
        (checkout / "a.txt").write_text("x\n")
        clock = [time.monotonic()]

        class SlowClassification:
            """The walker's ``os``, except that each per-entry stat "takes" a minute on a fake clock."""

            def __getattr__(self, name: str) -> object:
                return getattr(os, name)

            @staticmethod
            def stat(*args: object, **kwargs: object) -> os.stat_result:
                clock[0] += 60
                return os.stat(*args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(_checkout_walk, "time", SimpleNamespace(monotonic=lambda: clock[0]))
        monkeypatch.setattr(_checkout_walk, "os", SlowClassification())

        [result] = verify_assertions(
            [Assertion(type=AssertionType.GLOB_ABSENT, pattern="", target="**/missing")], checkout
        )

        assert result.passed is None, result

    def test_the_deadline_stops_a_listing_part_way_through(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One huge or slow directory is read no further than the time budget, not the whole work budget."""
        import time
        from types import SimpleNamespace

        from trw_memory.lifecycle import _checkout_walk

        for index in range(50):
            (tmp_path / f"f{index}.txt").write_text("x\n")
        clock = [time.monotonic()]

        def tick() -> float:
            clock[0] += 1.0  # every clock read "takes" a second: the 5 s budget ends a few entries in
            return clock[0]

        monkeypatch.setattr(_checkout_walk, "time", SimpleNamespace(monotonic=tick))
        taken: list[str] = []
        real_scandir = os.scandir

        def counting(path: object) -> object:
            listing = real_scandir(path)  # type: ignore[call-overload]

            class Counted:
                def __enter__(self) -> Counted:
                    return self

                def __exit__(self, *exc: object) -> None:
                    listing.close()

                def __iter__(self) -> Counted:
                    return self

                def __next__(self) -> object:
                    entry = next(listing)
                    taken.append(entry.name)
                    return entry

            return Counted()

        monkeypatch.setattr(os, "scandir", counting)

        [result] = verify_assertions([Assertion(type=AssertionType.GLOB_ABSENT, pattern="", target="*.txt")], tmp_path)

        assert result.passed is None, result
        assert len(taken) < 10, f"read {len(taken)} entries after the deadline"

    def test_one_huge_directory_is_read_no_further_than_the_budget(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from trw_memory.lifecycle import verification

        for index in range(50):
            (tmp_path / f"f{index}.txt").write_text("x\n")
        monkeypatch.setattr(verification, "MAX_WALK_WORK", 10)
        taken: list[int] = []
        real_scandir = os.scandir

        def counting(path: object) -> object:
            listing = real_scandir(path)  # type: ignore[call-overload]

            class Counted:
                def __enter__(self) -> Counted:
                    return self

                def __exit__(self, *exc: object) -> None:
                    listing.close()

                def __iter__(self) -> Counted:
                    return self

                def __next__(self) -> object:
                    taken.append(1)
                    return next(listing)

            return Counted()

        monkeypatch.setattr(os, "scandir", counting)

        [result] = verify_assertions([Assertion(type=AssertionType.GLOB_ABSENT, pattern="", target="*.py")], tmp_path)

        assert result.passed is None, result
        assert len(taken) <= 10, len(taken)


class TestTrailingSlashMatchesDirectoriesOnly:
    """C12-R: ``Path.glob("AGENTS.md/")`` matches nothing; the walker must not return the regular file."""

    @pytest.mark.parametrize("target", ["AGENTS.md/", "src/", "*/", "AGENTS.md", "src"])
    @pytest.mark.parametrize("kind", [AssertionType.GLOB_EXISTS, AssertionType.GLOB_ABSENT])
    def test_both_polarities_agree_with_pathlib(self, tmp_path: Path, target: str, kind: AssertionType) -> None:
        (tmp_path / "AGENTS.md").write_text("x\n")
        (tmp_path / "src").mkdir()
        found = bool(list(tmp_path.glob(target)))

        [result] = verify_assertions([Assertion(type=kind, pattern="", target=target)], tmp_path)

        assert result.passed is (found if kind is AssertionType.GLOB_EXISTS else not found), (target, result)

    def test_grep_never_reads_a_file_named_with_a_trailing_slash(self, tmp_path: Path) -> None:
        (tmp_path / "AGENTS.md").write_text("needle\n")

        [result] = verify_assertions(
            [Assertion(type=AssertionType.GREP_PRESENT, pattern="needle", target="AGENTS.md/")], tmp_path
        )

        assert result.passed is False, result


class TestCallerTextIsCappedFirst:
    """rc6 C12: an assertion's pattern and target are length-checked before any regex or walk touches them."""

    LONG_TARGET = "a" * (MAX_PATTERN_LEN + 1)

    @pytest.mark.parametrize("pattern", ["{" * 120_000, "(?" * 60_000])
    def test_the_cost_scans_are_linear_even_without_the_cap(
        self, pattern: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from trw_memory.lifecycle import verification

        monkeypatch.setattr(verification, "MAX_PATTERN_LEN", len(pattern))  # let the scanners see all of it
        start = time.monotonic()
        verification._pattern_cost(pattern)
        assert time.monotonic() - start < 1.0

    @pytest.mark.parametrize("pattern", ["{" * 120_000, "(?" * 60_000])
    def test_an_overlong_stored_pattern_is_refused_before_any_regex(
        self, tmp_path: Path, pattern: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from trw_memory.lifecycle import verification

        touched: list[str] = []
        monkeypatch.setattr(verification, "_pattern_cost", lambda p: touched.append(p) or 0)
        (tmp_path / "a.txt").write_text("a\n")
        # A row written before the cap still loads; it is refused at verify time, never failed.
        stored = Assertion.model_construct(type="grep_present", pattern=pattern, target="a.txt")

        start = time.monotonic()
        [result] = verify_assertions([stored], tmp_path)

        assert (result.passed, touched) == (None, [])
        assert "refused" in result.evidence
        assert time.monotonic() - start < 1.0

    def test_an_overlong_target_is_refused_before_the_walk(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from trw_memory.lifecycle import verification

        walked: list[object] = []
        monkeypatch.setattr(verification, "_walk_checkout", lambda *args: walked.append(args))
        long = Assertion(type=AssertionType.GLOB_EXISTS, target=self.LONG_TARGET)

        [result] = verify_assertions([long], tmp_path)

        assert (result.passed, walked) == (None, [])

    def test_update_refuses_an_overlong_assertion(self) -> None:
        from trw_memory.lifecycle.correction import parse_patch

        refused = parse_patch({"assertions": [{"type": "glob_exists", "target": self.LONG_TARGET}]})

        assert isinstance(refused, dict)
        assert refused["status"] == "invalid"

    def test_store_refuses_an_overlong_assertion(self, tmp_path: Path) -> None:
        from trw_memory.models.config import MemoryConfig
        from trw_memory.storage.sqlite_backend import SQLiteBackend
        from trw_memory.tools.store import memory_store_impl

        with SQLiteBackend(tmp_path / "memory.db") as backend:
            result = memory_store_impl(
                "a learning",
                "project:probe-00000000",
                backend=backend,
                config=MemoryConfig(),
                assertions=[Assertion(type=AssertionType.GLOB_EXISTS, target=self.LONG_TARGET)],
            )
            assert result["status"] == "invalid"
            assert backend.count(namespace="project:probe-00000000") == 0

    def test_import_refuses_an_overlong_assertion(self) -> None:
        from tests.conftest import make_entry
        from trw_memory.cli_storage import _rebuild_own_export

        row = make_entry(entry_id="M-long").model_copy(
            update={"assertions": [Assertion(type=AssertionType.GLOB_EXISTS, target=self.LONG_TARGET)]}
        )
        with pytest.raises(ValueError, match="longer than"):
            _rebuild_own_export(row.model_dump(mode="json"), "project:probe-00000000")

    def test_a_long_list_past_the_budget_compiles_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """memory_update does not bound how many assertions an entry has; past the budget each costs O(1)."""
        from trw_memory.lifecycle import verification

        compiled: list[str] = []
        monkeypatch.setattr(verification.regex, "compile", lambda p, *a, **k: compiled.append(p))
        monkeypatch.setattr(verification, "GREP_MATCH_DEADLINE_SECONDS", 0.0)
        many = [Assertion(type=AssertionType.GREP_PRESENT, pattern="{" * 500, target="a.txt")] * 5_000

        start = time.monotonic()
        results = verify_assertions(many, tmp_path)

        assert {r.passed for r in results} == {None}
        assert compiled == []
        assert time.monotonic() - start < 2.0

    def test_sync_apply_refuses_an_overlong_assertion(self, tmp_path: Path) -> None:
        from trw_memory.models.config import MemoryConfig
        from trw_memory.models.memory import MemoryEntry
        from trw_memory.storage.sqlite_backend import SQLiteBackend
        from trw_memory.tools.sync import memory_sync_apply_impl

        pulled = MemoryEntry(
            id="T-long",
            content="a pulled row",
            namespace="project:probe-00000000",
            assertions=[Assertion(type=AssertionType.GLOB_EXISTS, target=self.LONG_TARGET)],
        ).model_dump(mode="json")
        with SQLiteBackend(tmp_path / "memory.db") as backend:
            result = memory_sync_apply_impl("project:probe-00000000", pulled, backend=backend, config=MemoryConfig())
            assert result["status"] == "invalid"
            assert backend.get("T-long", namespace="project:probe-00000000") is None

    @pytest.mark.parametrize("shape", ["model", "dict"])
    def test_store_input_validation_refuses_either_shape_per_item(self, shape: str) -> None:
        """A bulk caller sends raw dicts; an overlong one is that item's rejection, not a crash."""
        from trw_memory.exceptions import SchemaValidationError
        from trw_memory.security.poisoning import validate_store_inputs

        long = Assertion(type=AssertionType.GLOB_EXISTS, target=self.LONG_TARGET)
        short = Assertion(type=AssertionType.GLOB_EXISTS, target="a.txt")
        as_shape = (lambda a: a.model_dump(mode="json")) if shape == "dict" else (lambda a: a)
        common = {"content": "x", "detail": "", "tags": None, "metadata": None, "importance": 0.5}

        validate_store_inputs(**common, assertions=[as_shape(short)])  # type: ignore[arg-type]
        with pytest.raises(SchemaValidationError) as refused:
            validate_store_inputs(**common, assertions=[as_shape(long)])  # type: ignore[arg-type]
        assert refused.value.failed_fields == ["assertions"]
