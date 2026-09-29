"""Per-test and per-suite duration budgets (PRD-INFRA-197-FR05).

Canonical source: ``trw-mcp/tests/_duration_budget.py``. ``trw-memory/tests/_duration_budget.py``
is a byte-identical copy; ``scripts/tests/test_duration_budget_parity.py`` enforces that,
following the ``_trw_home.py`` canonical-copy-plus-parity-check pattern used elsewhere in
this monorepo.

Marker
------
``@pytest.mark.duration_exempt(reason="...")`` exempts a test from the per-test threshold.
``reason`` is required and must be non-empty -- using the marker without one is a
collection-time error with a clear message, regardless of the mode below. This is
deliberately *not* the ``slow`` marker: trw-memory deselects ``slow`` by default via
``addopts``, which would silently remove exempted tests from every default run instead of
just flagging their duration.

Modes (env ``TRW_PYTEST_DURATION_MODE``)
-----------------------------------------
- unset / ``"off"`` (the default): no behaviour change beyond registering the marker and
  validating its usage at collection time. Lands safely everywhere.
- ``"report"``: prints a terminal-summary section listing the offenders (top 20 by
  duration: node id + seconds) and the suite wall-clock total. Never fails the run.
- ``"enforce"``: the same report, plus a non-zero exit status when any unexempt test is
  over its per-test threshold, or the suite exceeds its budget.

Thresholds
----------
Per-test threshold = ``2.0s * float(env TRW_PYTEST_DURATION_FACTOR, default 1.0)``.
Per-test duration = setup + call + teardown (every phase that produced a report; a
failure, skip or error still contributes whatever phases ran).

Suite budget = controller elapsed wall-clock from session start to session finish,
compared against env ``TRW_PYTEST_SUITE_BUDGET_S`` (unset = no suite budget).

Aggregation and xdist
----------------------
Duration/offender aggregation, printing and enforcement happen on the controller only.
Under xdist, each worker COLLECTS and runs its own assigned tests in its own process;
the controller never re-collects those items itself, it only computes the distribution
plan from the node ids workers report back. That means marker validation
(``pytest_collection_modifyitems``) has to run in every process -- workers included --
or a bare ``duration_exempt`` on a worker-collected test would pass silently under
``-n``. A validated exemption's reason is then carried to the controller the same way
every other per-test fact is: stamped onto the test's own report as a ``user_properties``
entry in ``pytest_runtest_makereport`` (run in whichever process ran the test), which
xdist serializes and forwards with the report to the controller's
``pytest_runtest_logreport``. ``hasattr(config, "workerinput")`` is true only inside a
worker process (xdist sets it there and only there); ``pytest_runtest_logreport`` uses it
to accumulate ``_DURATIONS``/``_EXEMPT`` and ``pytest_sessionfinish``/``pytest_terminal_summary``
use it to print and enforce, on the controller only -- exactly once. A worker that
crashes mid-run simply stops producing reports; the controller's aggregation only ever
adds up whatever reports actually arrived, so a crash cannot corrupt or fail this plugin.
"""

from __future__ import annotations

import math
import os
import time
from collections.abc import Generator

import pytest

MARKER = "duration_exempt"
MODE_ENV = "TRW_PYTEST_DURATION_MODE"
FACTOR_ENV = "TRW_PYTEST_DURATION_FACTOR"
SUITE_BUDGET_ENV = "TRW_PYTEST_SUITE_BUDGET_S"

DEFAULT_PER_TEST_THRESHOLD_S = 2.0
TOP_N_OFFENDERS = 20

_MODES = ("off", "report", "enforce")

#: nodeid -> accumulated setup+call+teardown seconds. Controller-only; reset per process.
_DURATIONS: dict[str, float] = {}
#: nodeid -> exemption reason, populated during collection on the controller.
_EXEMPT: dict[str, str] = {}
#: Wall-clock start of the controller's session, set in pytest_configure.
_SESSION_START: float | None = None
#: Whether *this* process is an xdist worker; set once in pytest_configure.
_IS_WORKER: bool = False
#: (mode, per-test threshold, suite budget), resolved ONCE at configure time: a test that
#: changes the env later (this plugin's own malformed-setting test) must not reach the
#: outer session's hooks.
_SETTINGS: tuple[str, float, float | None] = ("off", 2.0, None)


def _is_worker(config: pytest.Config) -> bool:
    """True only inside an xdist worker process."""
    return hasattr(config, "workerinput")


def _mode() -> str:
    raw = os.environ.get(MODE_ENV, "off").strip().lower()
    if raw not in _MODES:
        # A typo must not quietly switch enforcement off.
        raise pytest.UsageError(f"{MODE_ENV}={raw!r} is not one of {', '.join(_MODES)}")
    return raw


def _env_seconds(name: str) -> float | None:
    """A positive, finite float from env *name*, ``None`` when unset; anything else stops the run."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        raise pytest.UsageError(f"{name}={raw!r} is not a number") from None
    if not math.isfinite(value) or value <= 0:
        raise pytest.UsageError(f"{name}={raw!r} must be a positive, finite number")
    return value


def _threshold_s() -> float:
    return DEFAULT_PER_TEST_THRESHOLD_S * (_env_seconds(FACTOR_ENV) or 1.0)


def _suite_budget_s() -> float | None:
    return _env_seconds(SUITE_BUDGET_ENV)


def _exemption_reason(item: pytest.Item) -> str | None:
    """The test's exemption reason, or ``None`` if it carries no exemption marker."""
    marker = item.get_closest_marker(MARKER)
    if marker is None:
        return None
    if marker.args:
        return str(marker.args[0])
    reason = marker.kwargs.get("reason")
    return str(reason) if reason else None


def pytest_configure(config: pytest.Config) -> None:
    global _IS_WORKER, _SESSION_START, _DURATIONS, _EXEMPT, _SETTINGS
    config.addinivalue_line(
        "markers",
        f"{MARKER}(reason): exempt a test from the duration budget; reason is required and non-empty",
    )
    _IS_WORKER = _is_worker(config)
    _SETTINGS = (_mode(), _threshold_s(), _suite_budget_s())
    _DURATIONS = {}
    _EXEMPT = {}
    if not _IS_WORKER:
        _SESSION_START = time.monotonic()


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Fail collection with a clear message if ``duration_exempt`` is used without a reason.

    Runs in every process, xdist workers included: xdist COLLECTS inside each worker
    (the controller only computes the distribution plan from node ids reported back by
    workers; it does not run this hook over the real ``items`` itself), so validating on
    the controller alone would let a bare ``duration_exempt`` on a worker-collected test
    pass silently under ``-n``. A worker's own exemption reasons are not recorded into
    ``_EXEMPT`` here -- see ``pytest_runtest_makereport``/``pytest_runtest_logreport``
    below for how those reach the controller's aggregation.
    """
    bad: list[str] = []
    for item in items:
        marker = item.get_closest_marker(MARKER)
        if marker is None:
            continue
        reason = _exemption_reason(item)
        if not reason:
            bad.append(item.nodeid)
        elif not _is_worker(config):
            _EXEMPT[item.nodeid] = reason
    if bad:
        listed = "\n".join(f"  - {nodeid}" for nodeid in bad)
        pytest.exit(
            f"@pytest.mark.{MARKER} requires a non-empty reason=..., e.g. "
            f'@pytest.mark.{MARKER}(reason="why"). Missing/empty reason on:\n{listed}',
            returncode=2,
        )


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item, call: pytest.CallInfo[None]
) -> Generator[None, pytest.TestReport, pytest.TestReport]:
    """Stamp an exempt test's reason onto its own report, in whichever process ran it.

    ``user_properties`` are serialized and forwarded from an xdist worker to the
    controller with the report itself (see the module docstring), so this is how a
    worker-collected exemption reaches the controller's ``_EXEMPT`` when the controller
    never saw that item during its own (worker-side) collection pass.
    """
    report = yield
    reason = _exemption_reason(item)
    if reason:
        report.user_properties.append((MARKER, reason))
    return report


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    if _IS_WORKER:
        return
    for key, reason in report.user_properties:
        if key == MARKER:
            _EXEMPT[report.nodeid] = reason
    if _SETTINGS[0] == "off":
        return
    _DURATIONS[report.nodeid] = _DURATIONS.get(report.nodeid, 0.0) + report.duration


class _Results:
    def __init__(
        self,
        *,
        offenders: list[tuple[str, float]],
        offender_count: int,
        threshold_s: float,
        suite_elapsed_s: float,
        suite_budget_s: float | None,
        over_budget: bool,
    ) -> None:
        self.offenders = offenders
        self.offender_count = offender_count
        self.threshold_s = threshold_s
        self.suite_elapsed_s = suite_elapsed_s
        self.suite_budget_s = suite_budget_s
        self.over_budget = over_budget


_RESULTS_KEY = pytest.StashKey[_Results]()


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    config = session.config
    if _is_worker(config):
        return
    mode = _SETTINGS[0]
    if mode == "off":
        return

    threshold = _SETTINGS[1]
    offenders = sorted(
        (
            (nodeid, duration)
            for nodeid, duration in _DURATIONS.items()
            if nodeid not in _EXEMPT and duration > threshold
        ),
        key=lambda pair: pair[1],
        reverse=True,
    )
    suite_elapsed = (time.monotonic() - _SESSION_START) if _SESSION_START is not None else 0.0
    budget = _SETTINGS[2]
    over_budget = budget is not None and suite_elapsed > budget

    config.stash[_RESULTS_KEY] = _Results(
        offenders=offenders[:TOP_N_OFFENDERS],
        offender_count=len(offenders),
        threshold_s=threshold,
        suite_elapsed_s=suite_elapsed,
        suite_budget_s=budget,
        over_budget=over_budget,
    )

    upgradeable = {int(pytest.ExitCode.OK), int(pytest.ExitCode.TESTS_FAILED)}
    if mode == "enforce" and (offenders or over_budget) and int(exitstatus) in upgradeable:
        # Only upgrade a clean or already-failed status. An interrupted, internal-error
        # or usage-error run has a more specific and more important story to tell than
        # "some test ran long" -- overwriting it here would mask that story.
        session.exitstatus = pytest.ExitCode.TESTS_FAILED


def pytest_terminal_summary(terminalreporter: object, exitstatus: int, config: pytest.Config) -> None:
    del exitstatus
    if _is_worker(config):
        return
    if _SETTINGS[0] == "off":
        return
    results = config.stash.get(_RESULTS_KEY, None)
    if results is None:
        return

    write = terminalreporter.section  # type: ignore[attr-defined]
    write("duration budget (PRD-INFRA-197-FR05)")
    tr = terminalreporter
    if results.offenders:
        tr.write_line(f"{results.offender_count} test(s) over the {results.threshold_s:g}s per-test threshold:")  # type: ignore[attr-defined]
        for nodeid, duration in results.offenders:
            tr.write_line(f"  {duration:8.2f}s  {nodeid}")  # type: ignore[attr-defined]
    else:
        tr.write_line(f"no tests over the {results.threshold_s:g}s per-test threshold")  # type: ignore[attr-defined]
    if results.suite_budget_s is not None:
        state = "OVER" if results.over_budget else "within"
        tr.write_line(  # type: ignore[attr-defined]
            f"suite wall-clock: {results.suite_elapsed_s:.2f}s ({state} the {results.suite_budget_s:g}s budget)"
        )
    else:
        tr.write_line(f"suite wall-clock: {results.suite_elapsed_s:.2f}s (no suite budget set)")  # type: ignore[attr-defined]
