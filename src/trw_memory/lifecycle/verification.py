"""Executable assertion verification engine.

Verifies grep/glob assertions against the codebase. Pure functions,
read-only, no shell commands — only a bounded no-follow dir_fd walk + a deadline-bounded regex search.

PRD-CORE-086 FR04: verify_assertions() engine.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import regex
import structlog

from trw_memory._dir_trust import open_anchored_walk
from trw_memory.exceptions import UntrustedDirectoryError
from trw_memory.lifecycle._checkout_walk import _Budget, _read_bytes_through_checkout, _walk_checkout
from trw_memory.models._assertion_cap import MAX_PATTERN_LEN, overlong
from trw_memory.models.memory import Assertion, AssertionResult, AssertionType

logger = structlog.get_logger(__name__)

# Default directories to exclude from file scanning
DEFAULT_EXCLUDES: frozenset[str] = frozenset(
    {
        ".git",
        "__pycache__",
        "node_modules",
        ".egg-info",
        "dist",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
    }
)

# Security limits
MAX_FILE_SIZE_BYTES: int = 1_048_576  # 1MB
BINARY_CHECK_BYTES: int = 512
#: Wall time one ``verify_assertions`` call may spend walking, reading and matching across all
#: its assertions (C12, C12-R). Patterns and targets are caller-supplied and the daemon is shared:
#: stdlib ``re`` cannot be interrupted, so a catastrophic-backtracking pattern would hang every
#: tenant; ``regex`` enforces the deadline inside the matcher and releases the GIL while it runs.
#: The budget is per call, not per assertion, because the assertion list itself is unbounded; an
#: exhausted one makes the assertion it interrupts, and every later one, unverified.
GREP_MATCH_DEADLINE_SECONDS: float = 5.0
#: Directory listings plus listed entries one call's walks may consume (see ``_Budget``).
MAX_WALK_WORK: int = 1_000_000
#: ``regex`` unrolls a counted repeat when it compiles (``a{1000000}`` is ~281 MiB, before any
#: deadline check), so a grep pattern is compiled only if its length times the product of every
#: brace group's largest number stays within this (C12 rc3). The product over-counts sequential
#: repeats and so also bounds nested ones; verbose mode is refused because its whitespace and
#: comments still count inside a quantifier. An over-budget pattern is unverified, never a verdict.
MAX_PATTERN_COST: int = 100_000
#: Any inline flag group naming ``x``, however spelled (``(?x)``, ``(?V1x)``, ``(?i-x:...)``); a comment ``(?#x)`` is
#: refused too, which only costs a verdict.
#: Both scanners stop at the next opener, so each runs in linear time (rc6 C12); the length cap comes first anyway.
_VERBOSE_FLAG = regex.compile(r"\(\?[^()<:'=!>]*x")
_BRACES = regex.compile(r"\{[^{}]*\}")


def _pattern_cost(pattern: str) -> int:
    if len(pattern) > MAX_PATTERN_LEN:
        return MAX_PATTERN_COST + 1
    cost = MAX_PATTERN_COST + 1 if _VERBOSE_FLAG.search(pattern) else len(pattern)
    for braces in _BRACES.findall(pattern):
        # a whole digit run is one count (leading zeroes included); 10 significant digits already exceed the budget
        cost *= max([1, *(int(run.lstrip("0")[:10] or 0) for run in regex.findall(r"\d+", braces))])
        if cost > MAX_PATTERN_COST:
            break
    return cost


def _result(assertion: Assertion, passed: bool | None, evidence: str) -> AssertionResult:
    return AssertionResult(
        type=assertion.type, pattern=assertion.pattern, target=assertion.target, passed=passed, evidence=evidence
    )


def verify_assertions(
    assertions: list[Assertion],
    project_root: Path | None,
    *,
    exclude_patterns: frozenset[str] | None = None,
) -> list[AssertionResult]:
    """Verify a list of assertions against the codebase.

    Args:
        assertions: Assertions to verify.
        project_root: Root directory for glob/grep operations.
        exclude_patterns: Directory names to skip. Defaults to DEFAULT_EXCLUDES.

    Returns:
        One AssertionResult per input assertion, in the same order.
    """
    if not assertions:
        return []

    if project_root is None:
        logger.debug("project_root unavailable, skipping assertion verification")
        return [_result(a, None, "project_root unavailable") for a in assertions]

    # PRD-SEC-016 round-8 finding 1 (round-8 review, item 1): the ONLY
    # existence/trust probe of `project_root` is this single no-follow open --
    # there is deliberately no separate `project_root.exists()`
    # (symlink-following) check beforehand. A pre-check like that is itself a
    # second, INDEPENDENT path-based resolution: a swapped ancestor pointing
    # at some OTHER destination would make `.exists()` answer for that
    # destination, not for the real checkout, before the anchored walk ever
    # ran -- a bare existence oracle over an attacker-chosen path, and exactly
    # the kind of "check via one path op, act via another" gap this PRD
    # closes everywhere else. `open_anchored_walk` decides both availability
    # and trust in the one call: `FileNotFoundError` means genuinely absent
    # (skip, matching the historic "no verification, not a failure" contract
    # for a project_root that legitimately doesn't exist yet), anything else
    # means a component could not be opened securely (refuse the batch). The
    # anchor fd this returns then stays open for EVERY assertion in the
    # batch, so every enumeration below walks THIS descriptor (or one opened
    # from it via `open_component_fd`), never a fresh path string -- there is
    # nothing left to re-swap between "the check" and "the use," because they
    # are the same open.
    try:
        anchor_fd = open_anchored_walk(project_root)
    except (OSError, UntrustedDirectoryError) as exc:
        cause = exc if isinstance(exc, OSError) else exc.__cause__
        if isinstance(cause, FileNotFoundError):
            logger.debug("project_root unavailable, skipping assertion verification")
            return [_result(a, None, "project_root unavailable") for a in assertions]
        refused = _log_root_swap_refused(project_root, exc)
        return [_result(a, None, refused) for a in assertions]

    excludes = exclude_patterns if exclude_patterns is not None else DEFAULT_EXCLUDES
    start_time = time.monotonic()

    results = []
    budget = _Budget(time.monotonic() + GREP_MATCH_DEADLINE_SECONDS, MAX_WALK_WORK)
    try:
        for assertion in assertions:
            if time.monotonic() >= budget.deadline:  # the list is unbounded: past the budget each one costs O(1)
                results.append(
                    _result(assertion, None, "the call's verification budget is spent; assertion unverified")
                )
                continue
            try:
                result = _verify_single(assertion, anchor_fd, excludes, budget)
            except Exception as exc:
                # WARNING, not debug. This handler is sound — an error becomes
                # UNVERIFIED rather than passing — but at debug the only record that
                # a security assertion never ran is dropped before any processor sees
                # it, so a corpus of "unverified" results has no attached cause.
                logger.warning("assertion_verification_error", assertion_type=str(assertion.type), exc_info=True)
                result = _result(assertion, None, f"verification error: {exc}")
            results.append(result)
    finally:
        os.close(anchor_fd)

    duration_ms = (time.monotonic() - start_time) * 1000
    logger.info(
        "assertion_verification_complete",
        assertion_count=len(assertions),
        duration_ms=round(duration_ms, 1),
        passing=sum(1 for r in results if r.passed is True),
        failing=sum(1 for r in results if r.passed is False),
        skipped=sum(1 for r in results if r.passed is None),
        project_root=str(project_root),
    )
    return results


def _verify_single(
    assertion: Assertion,
    anchor_fd: int,
    excludes: frozenset[str],
    budget: _Budget,
) -> AssertionResult:
    """Verify a single assertion."""
    if overlong(assertion):  # before any regex or walk touches caller text: refused, never a verdict
        return _result(
            assertion, None, f"pattern or target longer than {MAX_PATTERN_LEN} characters; refused unverified"
        )
    if assertion.type in (AssertionType.GREP_PRESENT, "grep_present"):
        return _verify_grep(assertion, anchor_fd, excludes, budget, expect_present=True)
    if assertion.type in (AssertionType.GREP_ABSENT, "grep_absent"):
        return _verify_grep(assertion, anchor_fd, excludes, budget, expect_present=False)
    if assertion.type in (AssertionType.GLOB_EXISTS, "glob_exists"):
        return _verify_glob(assertion, anchor_fd, excludes, budget, expect_exists=True)
    if assertion.type in (AssertionType.GLOB_ABSENT, "glob_absent"):
        return _verify_glob(assertion, anchor_fd, excludes, budget, expect_exists=False)
    return _result(assertion, None, f"unknown assertion type: {assertion.type}")


def _verify_grep(
    assertion: Assertion,
    anchor_fd: int,
    excludes: frozenset[str],
    budget: _Budget,
    *,
    expect_present: bool,
) -> AssertionResult:
    """Verify a grep_present or grep_absent assertion."""
    try:
        if _pattern_cost(assertion.pattern) > MAX_PATTERN_COST:
            raise regex.error(f"refused before compiling: its cost exceeds MAX_PATTERN_COST ({MAX_PATTERN_COST})")
        compiled = regex.compile(assertion.pattern)
    except regex.error as e:
        return _result(assertion, None, f"invalid regex: {e}")

    matching_files: list[str] = []
    oversized_files: list[str] = []
    #: Candidates that were NOT searched. For a NEGATIVE assertion these are the
    #: difference between "the pattern is absent" and "we did not look", and the
    #: old code spent them as the former.
    unsearched: list[str] = []

    walked = _walk_checkout(anchor_fd, assertion.target, excludes, budget)
    if walked is None:
        return _result(
            assertion, None, f"could not enumerate files matching '{assertion.target}'; assertion unverified"
        )
    #: Pattern-matching candidates whose path -- leaf or an intermediate
    #: directory component -- was a symlink (PRD-SEC-016 FR03/round-8 finding
    #: 2). Never opened at all: a `grep_present` cannot pass on their
    #: content, and a `grep_absent` cannot claim absence over a target it
    #: deliberately declined to read. The walk that produced this list never
    #: distinguished a dangling symlink from a live one -- both are refused
    #: the instant `os.DirEntry.is_symlink()` says so, before any stat of
    #: whatever they point at -- so the answer here is identical whichever
    #: state the outside target is in.
    outside_checkout: list[str] = list(walked.refused)
    #: Pattern-matching candidates (or a subtree ``**`` reached) the walk
    #: could not classify or enumerate at all (round-9 review) -- a vanished
    #: file, a symlink-classified-then-open-failed race, or a nested scan
    #: failure. Same rule as `outside_checkout`: neither proves presence NOR
    #: absence, so it can never be silently spent as either.
    incomplete_candidates: list[str] = list(walked.incomplete)
    scanned = 0

    try:
        for rel_path in walked.files:
            # C12-R: reads are charged to the call's deadline too, checked before every file.
            if time.monotonic() >= budget.deadline:
                raise TimeoutError
            # Read through the SAME anchor_fd the enumeration above used (round-8
            # review item 3): a component swapped for a symlink after this walk
            # enumerated it (and before this read) is refused here, not
            # followed, and -- unlike re-deriving a fresh anchor from
            # `project_root`'s string per file -- nothing about `project_root`'s
            # own ancestors is ever re-resolved for the read either.
            data = _read_bytes_through_checkout(anchor_fd, rel_path, max_bytes=MAX_FILE_SIZE_BYTES)
            if data is None:
                unsearched.append(f"unreadable: {rel_path.name}")
                continue
            if len(data) > MAX_FILE_SIZE_BYTES:
                logger.debug("file_exceeds_size_limit", path=str(rel_path), size=len(data))
                oversized_files.append(f"file exceeds 1MB limit: {rel_path}")
                continue
            if b"\x00" in data[:BINARY_CHECK_BYTES]:
                continue

            scanned += 1
            text = data.decode("utf-8", errors="replace")
            # Clamped at 0, which `regex` treats as already expired: a negative timeout means none at all.
            if compiled.search(text, timeout=max(budget.deadline - time.monotonic(), 0.0), concurrent=True):
                matching_files.append(str(rel_path))
    except TimeoutError:
        # Neither presence nor absence was measured, so the assertion is unverified either way.
        return _result(
            assertion, None, f"verification exceeded its {GREP_MATCH_DEADLINE_SECONDS:g}s budget; assertion unverified"
        )

    if expect_present:
        passed: bool | None = len(matching_files) > 0
        if passed:
            evidence = f"pattern found in {len(matching_files)} file(s): {', '.join(matching_files[:5])}"
        elif outside_checkout:
            passed = None
            evidence = f"target resolves outside the checkout, so it was never read: {', '.join(outside_checkout[:5])}"
        elif incomplete_candidates:
            passed = None
            evidence = (
                f"could not fully enumerate matches for '{assertion.target}': {', '.join(incomplete_candidates[:5])}"
            )
        else:
            evidence = f"pattern not found in {scanned} file(s) matching '{assertion.target}'"
            if oversized_files:
                evidence = f"{evidence}; skipped {', '.join(oversized_files[:5])}"
    else:
        unsearched = [
            *unsearched,
            *(f"outside the checkout: {name}" for name in outside_checkout),
            *(f"could not classify: {name}" for name in incomplete_candidates),
        ]
        # A NEGATIVE assertion is a claim that nothing contains the pattern, and
        # a file that was never searched cannot support it. Skipped candidates
        # therefore make the result UNVERIFIED (``passed=None``, the state an
        # invalid regex already uses), not passing.
        #
        # This is the framework's own rule applied to its own gate: absence of a
        # measurement is not a measurement of absence. The old code reported
        # "pattern correctly absent from 0 file(s)" — green — for a directory it
        # could not read, and oversized files were disclosed in the evidence
        # string while still counting as proof of absence.
        skipped = [*oversized_files, *unsearched]
        if matching_files:
            passed = False
            evidence = f"pattern unexpectedly found in {len(matching_files)} file(s): {', '.join(matching_files[:5])}"
        elif skipped:
            passed = None
            evidence = (
                f"pattern absent from the {scanned} file(s) searched, but "
                f"{len(skipped)} could not be read, so absence is unverified: {', '.join(skipped[:5])}"
            )
        else:
            passed = True
            evidence = f"pattern correctly absent from {scanned} file(s) matching '{assertion.target}'"

    return _result(assertion, passed, evidence)


def _verify_glob(
    assertion: Assertion,
    anchor_fd: int,
    excludes: frozenset[str],
    budget: _Budget,
    *,
    expect_exists: bool,
) -> AssertionResult:
    """Verify a glob_exists or glob_absent assertion.

    PRD-SEC-016 round-8 finding 2: a matching candidate's path -- its own leaf
    component OR any intermediate directory component -- is refused the
    moment ``_walk_checkout`` sees it is a symlink, before any enumeration
    through it and regardless of whether the thing it points at exists. That
    is stricter than (and replaces) round-4 finding 1's ``path.resolve(strict
    =True)``-after-the-fact containment check, which needed the target to be
    reachable at all to classify it, and so answered differently for a live
    vs. a dangling symlink.
    """
    walked = _walk_checkout(anchor_fd, assertion.target, excludes, budget)
    if walked is None:
        # Same rule as _verify_grep: a walk that failed proves nothing in EITHER
        # direction. glob_absent is the dangerous one — "correctly no files
        # matching" over a directory we could not read is a green gate.
        return _result(
            assertion, None, f"could not enumerate files matching '{assertion.target}'; assertion unverified"
        )

    # Round-10 review, item 1: a directory leaf (`walked.glob_only`) is just as
    # much "found matching the target" for a glob assertion as a file leaf
    # is -- `pathlib.Path.glob` matches directories too, and `_walk_checkout`
    # already refuses a symlinked directory candidate the same way it
    # refuses a symlinked file (see `walked.refused`), so nothing here
    # weakens the no-follow guarantee.
    inside = walked.files + walked.glob_only
    excluded = [f"outside the checkout: {name}" for name in walked.refused] + [
        f"could not classify: {name}" for name in walked.incomplete
    ]

    if expect_exists:
        passed: bool | None = len(inside) > 0
        if passed:
            evidence = f"{len(inside)} file(s) found matching '{assertion.target}'"
        elif excluded:
            # A match existed only outside the checkout -- this is neither
            # "found inside" nor "confirmed absent," so it is unverified
            # rather than a false negative.
            passed = None
            evidence = f"only match(es) outside the checkout or unreadable: {', '.join(excluded[:5])}"
        else:
            evidence = f"no files found matching '{assertion.target}'"
    else:
        if inside:
            passed = False
            evidence = f"{len(inside)} file(s) unexpectedly found matching '{assertion.target}'"
        elif excluded:
            # An outside-checkout match means absence inside the checkout is
            # not actually proven -- unverified, not a pass.
            passed = None
            evidence = (
                f"no matches confirmed inside the checkout, but {len(excluded)} could not be verified: "
                f"{', '.join(excluded[:5])}"
            )
        else:
            passed = True
            evidence = f"correctly no files matching '{assertion.target}'"

    return _result(assertion, passed, evidence)


def _log_root_swap_refused(project_root: Path, exc: Exception) -> str:
    """Log the refusal (outside any ``except`` block, matching ``_dir_trust.py::_refuse``'s pattern) and return its reason."""
    logger.error("verify_root_swap_refused", project_root=str(project_root), error=str(exc))
    return f"the checkout root {project_root} could not be verified (a component may have been swapped for a symlink): {exc}"
