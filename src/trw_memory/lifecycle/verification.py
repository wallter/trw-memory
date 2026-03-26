"""Executable assertion verification engine.

Verifies grep/glob assertions against the codebase. Pure functions,
read-only, no shell commands — only pathlib.glob() + re.search().

PRD-CORE-086 FR04: verify_assertions() engine.
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path

from trw_memory.models.memory import Assertion, AssertionResult, AssertionType

logger = logging.getLogger(__name__)

# Default directories to exclude from file scanning
DEFAULT_EXCLUDES: frozenset[str] = frozenset({
    ".git",
    "__pycache__",
    "node_modules",
    ".egg-info",
    "dist",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
})

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
                type=a.type, pattern=a.pattern, target=a.target,
                passed=None, evidence="project_root unavailable",
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
            logger.debug("assertion verification error", exc_info=exc)
            result = AssertionResult(
                type=assertion.type, pattern=assertion.pattern,
                target=assertion.target, passed=None,
                evidence=f"verification error: {exc}",
            )
        results.append(result)

    duration_ms = (time.monotonic() - start_time) * 1000
    logger.debug(
        "assertion_verification_complete: count=%d duration_ms=%.1f passing=%d failing=%d",
        len(assertions),
        round(duration_ms, 1),
        sum(1 for r in results if r.passed is True),
        sum(1 for r in results if r.passed is False),
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
    elif assertion.type in (AssertionType.GREP_ABSENT, "grep_absent"):
        return _verify_grep(assertion, project_root, excludes, expect_present=False)
    elif assertion.type in (AssertionType.GLOB_EXISTS, "glob_exists"):
        return _verify_glob(assertion, project_root, excludes, expect_exists=True)
    elif assertion.type in (AssertionType.GLOB_ABSENT, "glob_absent"):
        return _verify_glob(assertion, project_root, excludes, expect_exists=False)
    else:
        return AssertionResult(
            type=assertion.type, pattern=assertion.pattern,
            target=assertion.target, passed=None,
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
            type=assertion.type, pattern=assertion.pattern,
            target=assertion.target, passed=None,
            evidence=f"invalid regex: {e}",
        )

    matching_files: list[str] = []
    scanned = 0

    for path in _iter_files(project_root, assertion.target, excludes):
        if not path.is_file():
            continue

        # Size check
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size > MAX_FILE_SIZE_BYTES:
            logger.debug("file exceeds size limit: path=%s size=%d", str(path), size)
            continue

        # Binary check
        if _is_binary(path):
            continue

        # Read and search
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
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
    else:
        passed = len(matching_files) == 0
        if passed:
            evidence = f"pattern correctly absent from {scanned} file(s) matching '{assertion.target}'"
        else:
            evidence = f"pattern unexpectedly found in {len(matching_files)} file(s): {', '.join(matching_files[:5])}"

    return AssertionResult(
        type=assertion.type, pattern=assertion.pattern,
        target=assertion.target, passed=passed, evidence=evidence,
    )


def _verify_glob(
    assertion: Assertion,
    project_root: Path,
    excludes: frozenset[str],
    *,
    expect_exists: bool,
) -> AssertionResult:
    """Verify a glob_exists or glob_absent assertion."""
    matches = list(_iter_files(project_root, assertion.target, excludes))

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
        type=assertion.type, pattern=assertion.pattern,
        target=assertion.target, passed=passed, evidence=evidence,
    )


def _iter_files(
    project_root: Path,
    target: str,
    excludes: frozenset[str],
) -> list[Path]:
    """Iterate files matching target glob, filtering excludes."""
    try:
        candidates = list(project_root.glob(target))
    except (ValueError, OSError) as e:
        logger.debug("glob error: target=%s error=%s", target, str(e))
        return []

    return [
        p for p in candidates
        if not any(part in excludes or part.endswith(".egg-info") for part in p.relative_to(project_root).parts)
    ]


def _is_binary(path: Path) -> bool:
    """Check if a file is binary by looking for null bytes in the first 512 bytes."""
    try:
        chunk = path.read_bytes()[:BINARY_CHECK_BYTES]
        return b"\x00" in chunk
    except OSError:
        return True  # Can't read -> treat as binary
