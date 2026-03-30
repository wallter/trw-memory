"""Tests for PRD-QUAL-054 FR-01 (CLI error-boundary DRY), FR-03 (RRF k validation).

FR-02 (config weight validation) was already implemented and tested in
test_prd_qual_053.py — the ge=0.0/le=1.0 constraints are in place.
"""

from __future__ import annotations

import sys
from io import StringIO
from unittest.mock import patch

import pytest

from trw_memory.cli import _cli_error_boundary
from trw_memory.retrieval.fusion import rrf_fuse


# ---------------------------------------------------------------------------
# FR-01: _cli_error_boundary decorator
# ---------------------------------------------------------------------------


class TestCliErrorBoundary:
    """Verify the _cli_error_boundary decorator catches, logs, and exits."""

    def test_catches_exception_prints_stderr_and_exits(self) -> None:
        """Wrapped function that raises should print to stderr and exit(1)."""

        @_cli_error_boundary
        def boom() -> None:
            raise RuntimeError("kaboom")

        captured = StringIO()
        with patch("sys.stderr", captured):
            with pytest.raises(SystemExit) as exc_info:
                boom()

        assert exc_info.value.code == 1
        assert "kaboom" in captured.getvalue()

    def test_passes_system_exit_through(self) -> None:
        """SystemExit must not be intercepted — re-raise as-is."""

        @_cli_error_boundary
        def explicit_exit() -> None:
            raise SystemExit(42)

        with pytest.raises(SystemExit) as exc_info:
            explicit_exit()

        assert exc_info.value.code == 42

    def test_returns_result_on_success(self) -> None:
        """Normal return values must pass through unchanged."""

        @_cli_error_boundary
        def add(a: int, b: int) -> int:
            return a + b

        assert add(2, 3) == 5

    def test_preserves_function_name(self) -> None:
        """functools.wraps must preserve __name__ and __doc__."""

        @_cli_error_boundary
        def my_func() -> None:
            """My docstring."""

        assert my_func.__name__ == "my_func"
        assert my_func.__doc__ == "My docstring."

    def test_logs_error_with_structlog(self) -> None:
        """The decorator must log via structlog with command name and error."""

        @_cli_error_boundary
        def failing_cmd() -> None:
            raise ValueError("bad input")

        with patch("trw_memory.cli.logger") as mock_logger:
            captured = StringIO()
            with patch("sys.stderr", captured):
                with pytest.raises(SystemExit):
                    failing_cmd()

            mock_logger.error.assert_called_once()
            call_kwargs = mock_logger.error.call_args
            assert call_kwargs[0][0] == "cli_command_failed"
            assert call_kwargs[1]["command"] == "failing_cmd"
            assert "bad input" in call_kwargs[1]["error"]

    def test_chains_exception(self) -> None:
        """SystemExit should chain from the original exception."""

        @_cli_error_boundary
        def fail() -> None:
            raise TypeError("original")

        captured = StringIO()
        with patch("sys.stderr", captured):
            with pytest.raises(SystemExit) as exc_info:
                fail()

        assert exc_info.value.__cause__ is not None
        assert isinstance(exc_info.value.__cause__, TypeError)


# ---------------------------------------------------------------------------
# FR-03: RRF k parameter validation
# ---------------------------------------------------------------------------


class TestRRFKValidation:
    """Verify rrf_fuse guards against invalid k values."""

    def test_k_zero_uses_default(self) -> None:
        """k=0 is invalid (causes division by rank), must reset to 60."""
        ranking = [("a", 1.0)]
        result = rrf_fuse([ranking], k=0)
        # With k=60: score = 1/(60+0+1) = 1/61
        assert result[0][1] == pytest.approx(1.0 / 61.0)

    def test_k_negative_uses_default(self) -> None:
        """k=-1 is invalid, must reset to 60."""
        ranking = [("a", 1.0)]
        result = rrf_fuse([ranking], k=-1)
        # With k=60: score = 1/(60+0+1) = 1/61
        assert result[0][1] == pytest.approx(1.0 / 61.0)

    def test_k_valid_preserved(self) -> None:
        """k=30 is valid and must be used as-is."""
        ranking = [("a", 1.0)]
        result = rrf_fuse([ranking], k=30)
        # With k=30: score = 1/(30+0+1) = 1/31
        assert result[0][1] == pytest.approx(1.0 / 31.0)

    def test_k_one_is_valid(self) -> None:
        """k=1 is the minimum valid value."""
        ranking = [("a", 1.0)]
        result = rrf_fuse([ranking], k=1)
        # With k=1: score = 1/(1+0+1) = 1/2
        assert result[0][1] == pytest.approx(0.5)

    def test_k_invalid_logs_warning(self) -> None:
        """Invalid k should emit a structlog warning."""
        with patch("trw_memory.retrieval.fusion.logger") as mock_logger:
            rrf_fuse([[("a", 1.0)]], k=0)
            mock_logger.warning.assert_called_once()
            call_kwargs = mock_logger.warning.call_args
            assert call_kwargs[0][0] == "rrf_k_invalid"
            assert call_kwargs[1]["k"] == 0
            assert call_kwargs[1]["default"] == 60


# ---------------------------------------------------------------------------
# FR-02: Config weight validation (already tested in test_prd_qual_053.py)
# ---------------------------------------------------------------------------


class TestConfigWeightValidation:
    """Confirm ge=0.0 / le=1.0 constraints are enforced.

    These duplicate the tests in test_prd_qual_053.py to serve as
    FR-02 evidence in this PRD's test file.
    """

    def test_negative_weight_rejected(self) -> None:
        """Negative score weight must raise ValidationError."""
        from pydantic import ValidationError

        from trw_memory.models.config import MemoryConfig

        with pytest.raises(ValidationError, match="score_relevance_weight"):
            MemoryConfig(
                score_relevance_weight=-0.1,
                score_recency_weight=0.55,
                score_importance_weight=0.55,
            )

    def test_weight_above_one_rejected(self) -> None:
        """Score weight > 1.0 must raise ValidationError."""
        from pydantic import ValidationError

        from trw_memory.models.config import MemoryConfig

        with pytest.raises(ValidationError, match="score_recency_weight"):
            MemoryConfig(
                score_relevance_weight=0.0,
                score_recency_weight=1.5,
                score_importance_weight=0.0,
            )
