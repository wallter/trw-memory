"""Plugin tests for the duration-budget marker and modes (PRD-INFRA-197-FR05).

Every test below runs the plugin inside its own ``pytester``-managed pytest *subprocess*
against a freshly-copied plugin module (not the already-imported ``tests._duration_budget``
that governs this very outer suite). Subprocess isolation means each inner run gets its own
fresh module globals, so nothing here can leak state into -- or read state from -- the
outer suite's own duration tracking, and the behaviour under test (worker-vs-controller
detection, env-var reads) matches production exactly rather than an in-process approximation.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest_plugins = ["pytester"]

_PLUGIN_SOURCE = Path(__file__).with_name("_duration_budget.py").read_text(encoding="utf-8")


def _install_plugin(pytester: pytest.Pytester) -> None:
    """Plant a private copy of the real plugin as its own top-level module."""
    pytester.makepyfile(duration_budget_plugin=_PLUGIN_SOURCE)
    pytester.makeconftest("pytest_plugins = ['duration_budget_plugin']")


def _planted_tests(pytester: pytest.Pytester, body: str) -> None:
    pytester.makepyfile(test_planted=body)


def test_report_mode_lists_offender_and_suite_total(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_plugin(pytester)
    _planted_tests(
        pytester,
        """
        import time

        def test_slow():
            time.sleep(0.3)

        def test_fast():
            pass
        """,
    )
    monkeypatch.setenv("TRW_PYTEST_DURATION_MODE", "report")
    monkeypatch.setenv("TRW_PYTEST_DURATION_FACTOR", "0.1")  # threshold = 0.2s
    result = pytester.runpytest_subprocess()

    assert result.ret == 0, "report mode must never fail the run"
    result.stdout.fnmatch_lines(
        [
            "*duration budget (PRD-INFRA-197-FR05)*",
            "*1 test(s) over the 0.2s per-test threshold:*",
            "*test_planted.py::test_slow*",
        ]
    )


def test_enforce_mode_fails_on_offender(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_plugin(pytester)
    _planted_tests(
        pytester,
        """
        import time

        def test_slow():
            time.sleep(0.3)
        """,
    )
    monkeypatch.setenv("TRW_PYTEST_DURATION_MODE", "enforce")
    monkeypatch.setenv("TRW_PYTEST_DURATION_FACTOR", "0.1")
    result = pytester.runpytest_subprocess()

    assert result.ret != 0, "enforce mode must fail the run when a test is over threshold"
    result.stdout.fnmatch_lines(["*duration budget (PRD-INFRA-197-FR05)*"])


def test_exempt_marker_excludes_offender_from_enforcement(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_plugin(pytester)
    _planted_tests(
        pytester,
        """
        import time
        import pytest

        @pytest.mark.duration_exempt(reason="known-heavy fixture, tracked separately")
        def test_slow():
            time.sleep(0.3)
        """,
    )
    monkeypatch.setenv("TRW_PYTEST_DURATION_MODE", "enforce")
    monkeypatch.setenv("TRW_PYTEST_DURATION_FACTOR", "0.1")
    result = pytester.runpytest_subprocess()

    assert result.ret == 0, "an exempt test must not fail enforce mode even though it is over threshold"
    result.stdout.fnmatch_lines(["*no tests over the 0.2s per-test threshold*"])


def test_exempt_marker_without_reason_errors_at_collection(pytester: pytest.Pytester) -> None:
    _install_plugin(pytester)
    _planted_tests(
        pytester,
        """
        import pytest

        @pytest.mark.duration_exempt()
        def test_missing_reason():
            pass
        """,
    )
    result = pytester.runpytest_subprocess()

    assert result.ret == 2, "a bare duration_exempt() with no reason must error at collection"
    combined = "\n".join(result.stdout.lines + result.stderr.lines)
    assert "duration_exempt requires a non-empty reason" in combined
    assert "test_missing_reason" in combined


def test_off_mode_is_silent_even_for_a_slow_test(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_plugin(pytester)
    _planted_tests(
        pytester,
        """
        import time

        def test_slow():
            time.sleep(0.3)
        """,
    )
    monkeypatch.delenv("TRW_PYTEST_DURATION_MODE", raising=False)
    monkeypatch.setenv("TRW_PYTEST_DURATION_FACTOR", "0.1")  # would flag under report/enforce
    result = pytester.runpytest_subprocess()

    assert result.ret == 0
    combined = "\n".join(result.stdout.lines)
    assert "duration budget" not in combined, "off mode (the default) must print nothing"


def test_suite_budget_exceeded_enforces_even_with_no_slow_test(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_plugin(pytester)
    _planted_tests(
        pytester,
        """
        def test_fast_one():
            pass

        def test_fast_two():
            pass
        """,
    )
    monkeypatch.setenv("TRW_PYTEST_DURATION_MODE", "enforce")
    monkeypatch.setenv("TRW_PYTEST_SUITE_BUDGET_S", "0.0001")
    result = pytester.runpytest_subprocess()

    assert result.ret != 0, "a suite over its wall-clock budget must fail enforce mode"
    result.stdout.fnmatch_lines(["*OVER the 0.0001s budget*"])


def test_an_interrupted_session_keeps_its_own_status_instead_of_being_masked(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    """P2: enforce must upgrade only OK/TESTS_FAILED, never overwrite a more specific status."""
    _install_plugin(pytester)
    _planted_tests(
        pytester,
        """
        import time
        import pytest

        def test_slow():
            time.sleep(0.3)

        def test_interrupt():
            pytest.exit("simulated interruption", returncode=pytest.ExitCode.INTERRUPTED)
        """,
    )
    monkeypatch.setenv("TRW_PYTEST_DURATION_MODE", "enforce")
    monkeypatch.setenv("TRW_PYTEST_DURATION_FACTOR", "0.1")  # test_slow is over threshold
    result = pytester.runpytest_subprocess()

    assert result.ret == pytest.ExitCode.INTERRUPTED, (
        "an interrupted session must keep INTERRUPTED, not be overwritten with TESTS_FAILED "
        "just because a slow test also ran"
    )


def test_an_existing_test_failure_stays_tests_failed_under_enforce(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    """P2 companion: TESTS_FAILED is upgradeable (it's already the enforce target), so it stays TESTS_FAILED."""
    _install_plugin(pytester)
    _planted_tests(
        pytester,
        """
        import time

        def test_slow():
            time.sleep(0.3)

        def test_actually_fails():
            assert False
        """,
    )
    monkeypatch.setenv("TRW_PYTEST_DURATION_MODE", "enforce")
    monkeypatch.setenv("TRW_PYTEST_DURATION_FACTOR", "0.1")
    result = pytester.runpytest_subprocess()

    assert result.ret == pytest.ExitCode.TESTS_FAILED


def test_xdist_aggregates_offender_on_the_controller(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Under -n 2, the offender must be reported once by the controller, not lost to a worker."""
    pytest.importorskip("xdist", reason="pytest-xdist not installed; xdist aggregation cannot be exercised")
    _install_plugin(pytester)
    _planted_tests(
        pytester,
        """
        import time

        def test_slow():
            time.sleep(0.3)

        def test_fast_a():
            pass

        def test_fast_b():
            pass

        def test_fast_c():
            pass
        """,
    )
    monkeypatch.setenv("TRW_PYTEST_DURATION_MODE", "report")
    monkeypatch.setenv("TRW_PYTEST_DURATION_FACTOR", "0.1")
    result = pytester.runpytest_subprocess("-n", "2")

    if any("unrecognized arguments" in line or ("-n" in line and "not found" in line) for line in result.stderr.lines):
        pytest.skip("pytest-xdist -n option not honored in this environment; cannot exercise xdist aggregation")

    assert result.ret == 0
    result.stdout.fnmatch_lines(
        [
            "*duration budget (PRD-INFRA-197-FR05)*",
            "*1 test(s) over the 0.2s per-test threshold:*",
            "*test_planted.py::test_slow*",
        ]
    )


def test_xdist_honours_an_exempt_offender_under_enforce(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    """P1: exemption must reach the controller even when the marker was validated on a worker.

    Before the fix, ``_EXEMPT`` was only ever populated by the controller's own collection
    pass, which under ``-n`` never sees the real items (xdist collects inside each worker).
    An exempt slow test would therefore be flagged as an offender and fail enforce mode.
    """
    pytest.importorskip("xdist", reason="pytest-xdist not installed; xdist aggregation cannot be exercised")
    _install_plugin(pytester)
    _planted_tests(
        pytester,
        """
        import time
        import pytest

        @pytest.mark.duration_exempt(reason="known-heavy fixture, tracked separately")
        def test_slow():
            time.sleep(0.3)

        def test_fast_a():
            pass

        def test_fast_b():
            pass

        def test_fast_c():
            pass
        """,
    )
    monkeypatch.setenv("TRW_PYTEST_DURATION_MODE", "enforce")
    monkeypatch.setenv("TRW_PYTEST_DURATION_FACTOR", "0.1")
    result = pytester.runpytest_subprocess("-n", "2")

    if any("unrecognized arguments" in line or ("-n" in line and "not found" in line) for line in result.stderr.lines):
        pytest.skip("pytest-xdist -n option not honored in this environment; cannot exercise xdist aggregation")

    assert result.ret == 0, "an exempt test running on a worker must not fail enforce mode on the controller"
    result.stdout.fnmatch_lines(["*no tests over the 0.2s per-test threshold*"])


def test_xdist_rejects_a_bare_exempt_marker_at_collection(pytester: pytest.Pytester) -> None:
    """P1: marker validation must run in every process, not only the controller's own collection pass.

    xdist COLLECTS inside each worker, so the controller's own collection pass (single-
    process behaviour covered by ``test_exempt_marker_without_reason_errors_at_collection``)
    never even sees a worker-collected item; before the fix this bad marker passed silently
    under ``-n``. ``pytest.exit`` called from a worker's collection hook surfaces to the
    controller as a crashed-worker cascade rather than the single-process clean message --
    an existing xdist behaviour, not something this fix controls -- so this only asserts
    that the run fails and names the offending test, not the exact message shape.

    At host load ~30 the nested ``-n 2`` cascade once dropped the worker's message: the run
    still failed, but without the name (load flake, 2026-09-28). A failed run that lacks the
    name is therefore run ONCE more before the name is required; a run that passes is never
    retried, so the regression (a silent pass under ``-n``) still fails the first time.
    """
    pytest.importorskip("xdist", reason="pytest-xdist not installed; xdist aggregation cannot be exercised")
    _install_plugin(pytester)
    _planted_tests(
        pytester,
        """
        import pytest

        @pytest.mark.duration_exempt()
        def test_missing_reason():
            pass

        def test_fast():
            pass
        """,
    )
    result = pytester.runpytest_subprocess("-n", "2")

    if any("unrecognized arguments" in line or ("-n" in line and "not found" in line) for line in result.stderr.lines):
        pytest.skip("pytest-xdist -n option not honored in this environment; cannot exercise xdist aggregation")

    assert result.ret != 0, "a bare duration_exempt() must still error at collection under -n"
    if "test_missing_reason" not in "\n".join(result.stdout.lines + result.stderr.lines):
        result = pytester.runpytest_subprocess("-n", "2")
        assert result.ret != 0, "a bare duration_exempt() must still error at collection under -n"
    combined = "\n".join(result.stdout.lines + result.stderr.lines)
    assert "test_missing_reason" in combined


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("TRW_PYTEST_DURATION_MODE", "enfroce"),
        ("TRW_PYTEST_DURATION_FACTOR", "fast"),
        ("TRW_PYTEST_SUITE_BUDGET_S", "-5"),
        ("TRW_PYTEST_DURATION_FACTOR", "nan"),
        ("TRW_PYTEST_DURATION_FACTOR", "inf"),
        ("TRW_PYTEST_SUITE_BUDGET_S", "nan"),
        ("TRW_PYTEST_SUITE_BUDGET_S", "inf"),
    ],
)
def test_a_malformed_setting_stops_the_run_instead_of_disabling_the_budget(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch, name: str, value: str
) -> None:
    _install_plugin(pytester)
    _planted_tests(pytester, "def test_fast():\n    pass\n")
    if name != "TRW_PYTEST_DURATION_MODE":
        monkeypatch.setenv("TRW_PYTEST_DURATION_MODE", "enforce")
    monkeypatch.setenv(name, value)
    result = pytester.runpytest_subprocess()
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    result.stderr.fnmatch_lines([f"*{name}*"])
