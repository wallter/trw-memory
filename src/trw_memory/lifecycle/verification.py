"""Executable assertion verification engine.

Verifies grep/glob assertions against the codebase. Pure functions,
read-only, no shell commands — only pathlib.glob() + re.search().

PRD-CORE-086 FR04: verify_assertions() engine.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import structlog

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

    if project_root is None or not project_root.exists():
        logger.debug("project_root unavailable, skipping assertion verification")
        return [
            AssertionResult(
                type=a.type,
                pattern=a.pattern,
                target=a.target,
                passed=None,
                evidence="project_root unavailable",
            )
            for a in assertions
        ]

    excludes = exclude_patterns if exclude_patterns is not None else DEFAULT_EXCLUDES
    start_time = time.monotonic()

    results = []
    for assertion in assertions:
        try:
            result = _verify_single(assertion, project_root, excludes)
        except Exception as exc:
            # WARNING, not debug. This handler is sound — an error becomes
            # UNVERIFIED rather than passing — but at debug the only record that
            # a security assertion never ran is dropped before any processor sees
            # it, so a corpus of "unverified" results has no attached cause.
            logger.warning("assertion_verification_error", assertion_type=str(assertion.type), exc_info=True)
            result = AssertionResult(
                type=assertion.type,
                pattern=assertion.pattern,
                target=assertion.target,
                passed=None,
                evidence=f"verification error: {exc}",
            )
        results.append(result)

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
    project_root: Path,
    excludes: frozenset[str],
) -> AssertionResult:
    """Verify a single assertion."""
    if assertion.type in (AssertionType.GREP_PRESENT, "grep_present"):
        return _verify_grep(assertion, project_root, excludes, expect_present=True)
    if assertion.type in (AssertionType.GREP_ABSENT, "grep_absent"):
        return _verify_grep(assertion, project_root, excludes, expect_present=False)
    if assertion.type in (AssertionType.GLOB_EXISTS, "glob_exists"):
        return _verify_glob(assertion, project_root, excludes, expect_exists=True)
    if assertion.type in (AssertionType.GLOB_ABSENT, "glob_absent"):
        return _verify_glob(assertion, project_root, excludes, expect_exists=False)
    return AssertionResult(
        type=assertion.type,
        pattern=assertion.pattern,
        target=assertion.target,
        passed=None,
        evidence=f"unknown assertion type: {assertion.type}",
    )


def _verify_grep(
    assertion: Assertion,
    project_root: Path,
    excludes: frozenset[str],
    *,
    expect_present: bool,
) -> AssertionResult:
    """Verify a grep_present or grep_absent assertion."""
    try:
        compiled = re.compile(assertion.pattern)
    except re.error as e:
        return AssertionResult(
            type=assertion.type,
            pattern=assertion.pattern,
            target=assertion.target,
            passed=None,
            evidence=f"invalid regex: {e}",
        )

    matching_files: list[str] = []
    oversized_files: list[str] = []
    #: Candidates that were NOT searched. For a NEGATIVE assertion these are the
    #: difference between "the pattern is absent" and "we did not look", and the
    #: old code spent them as the former.
    unsearched: list[str] = []
    scanned = 0

    candidates = _iter_files(project_root, assertion.target, excludes)
    if candidates is None:
        return AssertionResult(
            type=assertion.type,
            pattern=assertion.pattern,
            target=assertion.target,
            passed=None,
            evidence=f"could not enumerate files matching '{assertion.target}'; assertion unverified",
        )

    for path in candidates:
        if not path.is_file():
            continue

        # Size check
        try:
            size = path.stat().st_size
        except OSError:
            unsearched.append(f"unreadable: {path.name}")
            continue
        if size > MAX_FILE_SIZE_BYTES:
            logger.debug("file_exceeds_size_limit", path=str(path), size=size)
            oversized_files.append(f"file exceeds 1MB limit: {path.relative_to(project_root)}")
            continue

        # Binary check
        binary = _is_binary(path)
        if binary is None:
            unsearched.append(f"unreadable: {path.name}")
            continue
        if binary:
            continue

        # Read and search
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            unsearched.append(f"unreadable: {path.name}")
            continue

        scanned += 1
        if compiled.search(content):
            matching_files.append(str(path.relative_to(project_root)))

    if expect_present:
        passed = len(matching_files) > 0
        if passed:
            evidence = f"pattern found in {len(matching_files)} file(s): {', '.join(matching_files[:5])}"
        else:
            evidence = f"pattern not found in {scanned} file(s) matching '{assertion.target}'"
            if oversized_files:
                evidence = f"{evidence}; skipped {', '.join(oversized_files[:5])}"
    else:
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

    return AssertionResult(
        type=assertion.type,
        pattern=assertion.pattern,
        target=assertion.target,
        passed=passed,
        evidence=evidence,
    )


def _verify_glob(
    assertion: Assertion,
    project_root: Path,
    excludes: frozenset[str],
    *,
    expect_exists: bool,
) -> AssertionResult:
    """Verify a glob_exists or glob_absent assertion."""
    matches = _iter_files(project_root, assertion.target, excludes)
    if matches is None:
        # Same rule as _verify_grep: a walk that failed proves nothing in EITHER
        # direction. glob_absent is the dangerous one — "correctly no files
        # matching" over a directory we could not read is a green gate.
        return AssertionResult(
            type=assertion.type,
            pattern=assertion.pattern,
            target=assertion.target,
            passed=None,
            evidence=f"could not enumerate files matching '{assertion.target}'; assertion unverified",
        )

    if expect_exists:
        passed = len(matches) > 0
        if passed:
            evidence = f"{len(matches)} file(s) found matching '{assertion.target}'"
        else:
            evidence = f"no files found matching '{assertion.target}'"
    else:
        passed = len(matches) == 0
        if passed:
            evidence = f"correctly no files matching '{assertion.target}'"
        else:
            evidence = f"{len(matches)} file(s) unexpectedly found matching '{assertion.target}'"

    return AssertionResult(
        type=assertion.type,
        pattern=assertion.pattern,
        target=assertion.target,
        passed=passed,
        evidence=evidence,
    )


def _iter_files(
    project_root: Path,
    target: str,
    excludes: frozenset[str],
) -> list[Path] | None:
    """Files matching *target*, or ``None`` when the glob could not be walked.

    ``None`` and ``[]`` are different answers and the caller must not conflate
    them: ``[]`` means the pattern matched nothing, ``None`` means we do not know
    what it would have matched. Returning ``[]`` for a failed walk made a
    ``grep_absent`` assertion report "pattern correctly absent from 0 file(s)" —
    a green security gate over a directory it never read.
    """
    try:
        candidates = list(project_root.glob(target))
    except (ValueError, OSError) as e:
        logger.warning("glob_error", target=target, error=str(e))
        return None

    return [
        p
        for p in candidates
        if not any(part in excludes or part.endswith(".egg-info") for part in p.relative_to(project_root).parts)
    ]


def _is_binary(path: Path) -> bool | None:
    """Binary? ``None`` when the file could not be read at all.

    "Cannot read" used to be reported as "binary", which is a reasonable-looking
    lie: both make the caller skip the file, but only one of them means the file
    holds no searchable text. For a NEGATIVE assertion that difference is the
    whole verdict — a skipped-because-unreadable file leaves absence unproven,
    and this was the FIRST place an unreadable file was silently dropped, before
    the read in ``_verify_grep`` ever ran.
    """
    try:
        chunk = path.read_bytes()[:BINARY_CHECK_BYTES]
    except OSError:
        return None
    return b"\x00" in chunk
