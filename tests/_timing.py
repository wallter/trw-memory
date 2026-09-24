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

#: Budget measurements made in this process: test id, name, value, limit, unit, ok.
RECORDS: list[dict[str, Any]] = []
#: Outcome of every marked test in this process: test id -> passed/failed/skipped.
OUTCOMES: dict[str, str] = {}


def on_ci_runner() -> bool:
    return bool(os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"))


def assert_budget(name: str, value: float, limit: float, unit: str, *, at_least: bool = False) -> None:
    """Record a measured host-resource value, then assert it against its budget.

    ``unit`` is required and explicit (``"s"``, ``"ms"``, ``"MB"``, ``"ops/s"``). The budget is
    an upper bound unless ``at_least`` (throughput, rates).
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
    assert ok, f"{name}: {value:.4g} {unit} {'below' if at_least else 'above'} budget {limit:g} {unit}"


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
