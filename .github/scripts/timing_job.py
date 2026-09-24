"""Run the host-resource budget tests with CI unset and publish what they measured (PRD-QUAL-141).

Non-gating by construction: this always exits 0, and the outcome is a status in
``timing-results.json`` (``ok`` | ``budget_exceeded`` | ``no_tests_executed`` | ``setup_error``)
plus a job annotation. The ci.yml job that calls it is ``continue-on-error`` and nothing needs it,
so publish never waits on a runner's speed.

It measures only on a GitHub runner, or when asked with ``--local``. Everywhere else (notably the
monorepo's local replay of ci.yml before a release) it prints why and does nothing.

    python .github/scripts/timing_job.py --local                 # local smoke, whole marked set
    python .github/scripts/timing_job.py --local -- tests/x.py   # a subset
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

MARKER = "requires_local_timing"


def summarize(junit: Path, raw: Path, *, revision: str, returncode: int) -> dict[str, Any]:
    """The artifact: counts, one status, and every recorded measurement with its unit and limit."""
    result: dict[str, Any] = {
        "marker": MARKER,
        "revision": revision,
        "runner_os": os.environ.get("RUNNER_OS", platform.system()),
        "python": platform.python_version(),
        "pytest_returncode": returncode,
        "selected": 0,
        "executed": 0,
        "skipped": 0,
        "failed": 0,
        "errors": 0,
        "measurements": [],
        "outcomes": {},
    }
    if not junit.is_file():
        result["status"] = "setup_error"
        return result
    suites = ET.parse(junit).getroot()  # noqa: S314 - pytest's own junit file, written a line above
    suite = suites if suites.tag == "testsuite" else suites.find("testsuite")
    if suite is None:
        result["status"] = "setup_error"
        return result
    selected = int(suite.get("tests", 0))
    skipped = int(suite.get("skipped", 0))
    result.update(
        selected=selected,
        skipped=skipped,
        executed=selected - skipped,
        failed=int(suite.get("failures", 0)),
        errors=int(suite.get("errors", 0)),
    )
    if raw.is_file():
        report = json.loads(raw.read_text(encoding="utf-8"))
        result["measurements"] = report.get("measurements", [])
        result["outcomes"] = report.get("outcomes", {})
    if result["executed"] == 0:
        result["status"] = "no_tests_executed"
    elif result["errors"]:
        result["status"] = "setup_error"
    elif result["failed"]:
        result["status"] = "budget_exceeded"
    else:
        result["status"] = "ok"
    return result


def _summary_markdown(result: dict[str, Any]) -> str:
    lines = [
        f"### Host-resource budgets (non-gating): {result['status']}",
        "",
        f"selected {result['selected']} · executed {result['executed']} · skipped {result['skipped']} · "
        f"failed {result['failed']} · errors {result['errors']} · revision `{result['revision'][:12]}`",
        "",
        "| measurement | value | budget | ok |",
        "|---|---|---|---|",
    ]
    for m in result["measurements"]:
        op = ">=" if m.get("bound") == "min" else "<="
        lines.append(
            f"| {m['name']} | {m['value']:.4g} {m['unit']} | {op} {m['limit']:g} {m['unit']} | {'yes' if m['ok'] else '**no**'} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--local", action="store_true", help="measure on this machine (not a runner)")
    parser.add_argument("--out", default="timing-results.json")
    parser.add_argument("pytest_args", nargs="*", help="after --: extra pytest args or a test subset")
    args = parser.parse_args(argv)

    if not (args.local or os.environ.get("GITHUB_ACTIONS")):
        print("timing job: not on a GitHub runner and --local not given; nothing measured")
        return 0

    root = Path.cwd()
    junit, raw, out = root / "timing-junit.xml", root / "timing-raw.json", root / args.out
    for stale in (junit, raw):
        stale.unlink(missing_ok=True)
    env = {k: v for k, v in os.environ.items() if k not in ("CI", "GITHUB_ACTIONS")}
    env["TRW_TIMING_REPORT"] = str(raw)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(root / "src"), env.get("PYTHONPATH")]))
    command = [
        sys.executable,
        "-m",
        "pytest",
        *(args.pytest_args or ["tests"]),
        "-m",
        MARKER,
        "-p",
        "no:cacheprovider",
        "-q",
        f"--junitxml={junit}",
    ]
    if importlib.util.find_spec("xdist") is not None:
        command += ["-n", "0"]  # measure serially: parallel workers contend for the host being measured
    print("timing job:", " ".join(command), flush=True)
    returncode = subprocess.run(command, env=env, check=False).returncode  # noqa: S603 - fixed argv: this interpreter + pytest
    revision = (
        os.environ.get("GITHUB_SHA")
        or subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S607 - revision label only; empty if git is absent
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
    )

    result = summarize(junit, raw, revision=revision, returncode=returncode)
    out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as handle:
            handle.write(_summary_markdown(result))
    level = {"ok": "notice", "budget_exceeded": "warning"}.get(result["status"], "error")
    print(
        f"::{level}::host-resource budgets {result['status']}: executed {result['executed']}, failed {result['failed']}"
    )
    return 0  # trw:intentional non-gating by construction; the status lives in the artifact


if __name__ == "__main__":
    raise SystemExit(main())
