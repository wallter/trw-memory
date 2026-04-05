"""Tests for Page-Hinkley change detection."""

from __future__ import annotations

import math
import random

from trw_memory.bandit.change_detection import PageHinkleyDetector


class TestStableSignal:
    """No alarm should fire on a stable distribution."""

    def test_stable_signal_no_alarm(self) -> None:
        """100 observations from N(0.6, 0.05) should not trigger alarm."""
        rng = random.Random(42)
        det = PageHinkleyDetector()
        for _ in range(100):
            value = rng.gauss(0.6, 0.05)
            fired = det.update(value)
            assert not fired, "Alarm should not fire on a stable signal"


class TestDistributionShift:
    """Alarm must fire when the distribution shifts significantly."""

    def test_detects_distribution_shift(self) -> None:
        """20 observations of 0.8 then sustained 0.2 -- alarm fires.

        With default threshold=20.0 and a 0.6-magnitude shift, the
        Page-Hinkley statistic accumulates gradually because the running
        mean adapts.  We feed enough post-shift observations to exceed
        the threshold.
        """
        det = PageHinkleyDetector()
        alarm_fired = False
        # 20 stable observations followed by up to 120 shift observations.
        for i in range(140):
            value = 0.8 if i < 20 else 0.2
            if det.update(value):
                alarm_fired = True
                break
        assert alarm_fired, "Alarm should fire when distribution shifts from 0.8 to 0.2"

    def test_detects_shift_with_sensitive_threshold(self) -> None:
        """With a lower threshold, shift is detected in fewer observations."""
        det = PageHinkleyDetector(alarm_threshold=3.0)
        alarm_fired = False
        for i in range(30):
            value = 0.8 if i < 20 else 0.2
            if det.update(value):
                alarm_fired = True
                break
        assert alarm_fired, "Lower threshold should detect shift quickly"


class TestFalseAlarmRate:
    """False alarm rate must be controlled under noisy conditions."""

    def test_false_alarm_rate_under_5_percent(self) -> None:
        """500 obs from N(0.6, 0.15) over 100 trials -- alarm rate < 5%."""
        alarms = 0
        for trial in range(100):
            rng = random.Random(trial + 1000)
            det = PageHinkleyDetector()
            for _ in range(500):
                value = rng.gauss(0.6, 0.15)
                if det.update(value):
                    alarms += 1
                    break  # count at most one alarm per trial
        assert alarms < 5, f"False alarm rate too high: {alarms}/100"


class TestReset:
    """Reset must clear all internal state."""

    def test_reset_clears_state(self) -> None:
        det = PageHinkleyDetector()
        # Feed some data
        for v in [0.5, 0.6, 0.7]:
            det.update(v)
        assert det._n > 0

        det.reset()
        assert det._n == 0
        assert det._sum == 0.0
        assert det._h == 0.0
        assert det._m == float("inf")


class TestSerialization:
    """Round-trip serialization must preserve state."""

    def test_to_dict_from_dict_round_trip(self) -> None:
        det = PageHinkleyDetector(delta=0.05, alarm_threshold=15.0)
        for v in [0.5, 0.6, 0.7, 0.4]:
            det.update(v)

        data = det.to_dict()
        restored = PageHinkleyDetector.from_dict(data)

        assert restored._delta == det._delta
        assert restored._alarm_threshold == det._alarm_threshold
        assert restored._n == det._n
        assert restored._sum == det._sum
        assert math.isclose(restored._h, det._h, rel_tol=1e-9)
        assert math.isclose(restored._h_down, det._h_down, rel_tol=1e-9)
        # _m and _m_down may be negative (not inf) after observations
        if det._m == float("inf"):
            assert restored._m == float("inf")
        else:
            assert math.isclose(restored._m, det._m, rel_tol=1e-9)
        if det._m_down == float("inf"):
            assert restored._m_down == float("inf")
        else:
            assert math.isclose(restored._m_down, det._m_down, rel_tol=1e-9)

    def test_from_dict_corrupt_returns_fresh(self) -> None:
        """Malformed dict produces a fresh detector with defaults."""
        corrupt: dict[str, object] = {"delta": "not_a_number", "n": "bad"}
        det = PageHinkleyDetector.from_dict(corrupt)  # type: ignore[arg-type]
        assert det._n == 0
        assert det._delta == 0.01
        assert det._alarm_threshold == 20.0


class TestCustomThresholds:
    """Custom delta and alarm_threshold must be stored correctly."""

    def test_custom_thresholds(self) -> None:
        det = PageHinkleyDetector(delta=0.1, alarm_threshold=50.0)
        assert det._delta == 0.1
        assert det._alarm_threshold == 50.0


class TestSingleObservation:
    """A single observation must never trigger alarm."""

    def test_single_observation_no_alarm(self) -> None:
        det = PageHinkleyDetector()
        assert not det.update(0.5), "First observation should never trigger alarm"

    def test_single_observation_extreme_no_alarm(self) -> None:
        """Even an extreme value as the very first observation cannot alarm."""
        det = PageHinkleyDetector()
        assert not det.update(100.0), "Single extreme observation should not trigger alarm"


class TestAlarmResetsState:
    """After alarm fires, detector state is reset for fresh monitoring."""

    def test_alarm_resets_state(self) -> None:
        det = PageHinkleyDetector(alarm_threshold=3.0)
        # Feed stable high values then sharp drop to trigger alarm
        for _ in range(20):
            det.update(0.8)

        alarm_fired = False
        for _ in range(20):
            if det.update(0.2):
                alarm_fired = True
                break

        assert alarm_fired, "Alarm should have fired"
        # After alarm, state should be reset
        assert det._n == 0
        assert det._sum == 0.0
        assert det._h == 0.0
        assert det._m == float("inf")
