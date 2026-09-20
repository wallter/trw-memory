"""Fail the CI job unless the requested test suite actually ran.

A job can succeed without testing anything: a guarded pytest step that was
skipped, a run that collected nothing, or a report whose every test was skipped.
This check runs last and turns each of those into a failure.

The suite marker is written by the pytest step itself, after pytest returns
successfully, and holds the suite that step ran. It is never an echo of the
requested input, which would make the comparison below prove nothing.

Kept byte-identical in trw-mcp and trw-memory (a structure test compares them).
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

SUITES = ("unit", "full")


def _counts(junit: Path) -> tuple[int, int]:
    """(tests, skipped) summed over every <testsuite> in a pytest JUnit report."""
    root = ET.parse(junit).getroot()  # noqa: S314 -- same file as above
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    tests = sum(int(s.get("tests", "0")) for s in suites)
    skipped = sum(int(s.get("skipped", "0")) for s in suites)
    return tests, skipped


def verify(requested: str, junit: Path, marker: Path) -> str | None:
    """None when the requested suite ran and executed at least one test, else why not."""
    if requested not in SUITES:
        return f"requested suite {requested!r} is not one of {SUITES}"
    if not marker.is_file():
        return f"suite marker {marker} is missing: the {requested} pytest step did not complete"
    ran = marker.read_text(encoding="utf-8").strip()
    if ran != requested:
        return f"suite marker says {ran!r} but {requested!r} was requested"
    if not junit.is_file():
        return f"JUnit report {junit} is missing"
    try:
        tests, skipped = _counts(junit)
    except (ET.ParseError, ValueError) as exc:
        return f"JUnit report {junit} is unreadable: {exc}"
    if tests == 0:
        return "the JUnit report counts zero tests"
    if tests - skipped <= 0:
        return f"every one of the {tests} tests was skipped"
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requested", required=True)
    parser.add_argument("--junit", type=Path, required=True)
    parser.add_argument("--marker", type=Path, required=True)
    args = parser.parse_args(argv)
    problem = verify(args.requested, args.junit, args.marker)
    if problem is not None:
        print(f"suite verification FAILED: {problem}", file=sys.stderr)
        return 1
    tests, skipped = _counts(args.junit)
    print(f"suite verification ok: {args.requested} suite, {tests - skipped} executed, {skipped} skipped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
