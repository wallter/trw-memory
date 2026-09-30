"""Host-resource budgets: one marker, one assertion helper, one report (PRD-QUAL-141).

A wall-clock, throughput or RSS budget measures the machine a test runs on, not the code.
Such an assertion lives in a test marked ``requires_local_timing`` and is written as
``assert_budget(...)``; everything deterministic stays in an unmarked (gating) test.

- Local runs (``CI`` and ``GITHUB_ACTIONS`` unset) execute the marked tests: budgets are
  enforced where the machine is known.
- A CI runner skips them in the gating suite, so a slow runner never blocks publish.
- The mirror's non-gating timing job runs ``-m requires_local_timing`` with both variables
  unset and ``TRW_TIMING_REPORT`` set; every measured value is written there as JSON.

The conftest imports the hook functions below; nothing here runs unless pytest calls it.
"""

from __future__ import annotations

import json
import os
import platform
from pathlib import Path
from typing import Any

import pytest

MARKER = "requires_local_timing"
REPORT_ENV = "TRW_TIMING_REPORT"
TRUST_LOAD_ENV = "TRW_TIMING_TRUST_LOAD"
#: What a budget missed above the trust load does: ``skip`` (default; a routine run on a busy host is not red) or ``fail``
#: (the release gate's dedicated timing stage sets this so it can report UNTRUSTED as its own verdict).
UNTRUSTED_MODE_ENV = "TRW_TIMING_UNTRUSTED"
#: 1-minute load average above which a MISSED budget is reported UNTRUSTED (see ``assert_budget``).
DEFAULT_TRUST_LOAD = 12.0
UNTRUSTED_PREFIX = "UNTRUSTED"

#: Budget measurements made in this process: test id, name, value, limit, unit, ok.
RECORDS: list[dict[str, Any]] = []
#: Outcome of every marked test in this process: test id -> passed/failed/skipped.
OUTCOMES: dict[str, str] = {}


def on_ci_runner() -> bool:
    return bool(os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"))


def trust_load_limit() -> float:
    """Load ceiling above which a budget miss is untrusted: ``TRW_TIMING_TRUST_LOAD`` or the default."""
    raw = os.environ.get(TRUST_LOAD_ENV)
    try:
        return float(raw) if raw else DEFAULT_TRUST_LOAD
    except ValueError:
        return DEFAULT_TRUST_LOAD


def assert_budget(name: str, value: float, limit: float, unit: str, *, at_least: bool = False) -> None:
    """Record a measured host-resource value, then assert it against its budget.

    ``unit`` is required and explicit (``"s"``, ``"ms"``, ``"MB"``, ``"ops/s"``). The budget is
    an upper bound unless ``at_least`` (throughput, rates).

    A budget that is met passes at any load. A budget that is MISSED while the 1-minute load
    average exceeds ``TRW_TIMING_TRUST_LOAD`` (default 12.0) was measured on a saturated host, so it
    is not evidence about the code: by default the test SKIPS with the visible reason
    ``timing not asserted: load X.X > N (...)`` (a routine run must not go red on load the lane tooling
    itself allows). With ``TRW_TIMING_UNTRUSTED=fail`` (the release gate's timing stage) it fails with a
    message that starts ``UNTRUSTED (load X.X > N)``, so the gate counts it apart from a trusted miss. Default 12 is twice the 6
    performance cores of the reference host: the budgets passed alone at load 8.5 and missed at
    load 14-17 (KNOWN-RED 2026-09-28), and 12 splits those two populations. Wall-clock is kept
    (not CPU time) because these budgets time I/O, locks and subprocesses.
    """
    ok = value >= limit if at_least else value <= limit
    RECORDS.append(
        {
            "test": os.environ.get("PYTEST_CURRENT_TEST", "").rsplit(" ", 1)[0],
            "name": name,
            "value": float(value),
            "limit": float(limit),
            "unit": unit,
            "bound": "min" if at_least else "max",
            "ok": ok,
        }
    )
    if ok:
        return
    miss = f"{name}: {value:.4g} {unit} {'below' if at_least else 'above'} budget {limit:g} {unit}"
    trust = trust_load_limit()
    try:
        load1 = os.getloadavg()[0]
    except OSError:
        load1 = 0.0
    if load1 > trust:
        if os.environ.get(UNTRUSTED_MODE_ENV, "skip") == "fail":
            raise AssertionError(f"{UNTRUSTED_PREFIX} (load {load1:.1f} > {trust:g}): {miss}")
        pytest.skip(f"timing not asserted: load {load1:.1f} > {trust:g} ({miss})")  # skip-category: opt-in
    raise AssertionError(miss)


def apply_timing_policy(items: list[pytest.Item]) -> None:
    """Skip marked tests on a CI runner (the gating suite); run them everywhere else."""
    if not on_ci_runner():
        return
    skip = pytest.mark.skip(reason="host-resource budget: measured by the non-gating timing job, not the gate")
    for item in items:
        if item.get_closest_marker(MARKER) is not None:
            item.add_marker(skip)


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    if MARKER not in report.keywords:
        return
    if report.when == "call" or report.outcome != "passed":
        OUTCOMES[report.nodeid] = report.outcome if report.when == "call" else f"{report.outcome}_in_{report.when}"


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    target = os.environ.get(REPORT_ENV)
    if not target:
        return
    worker = getattr(session.config, "workerinput", {}).get("workerid")
    path = Path(f"{target}.{worker}" if worker else target)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "outcomes": OUTCOMES,
                "measurements": RECORDS,
                "python": platform.python_version(),
                "platform": platform.platform(),
                "cpu_count": os.cpu_count(),
                "pytest_exitstatus": int(exitstatus),
            },
            indent=1,
        ),
        encoding="utf-8",
    )
