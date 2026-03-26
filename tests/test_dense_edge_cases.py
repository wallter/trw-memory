"""Edge-case tests for dense retrieval — DimensionMismatchError and cosine_similarity.

Verifies that dimension mismatches raise DimensionMismatchError (not generic
ValueError), orthogonal vectors return 0.0, and the exception types are
distinguishable.
"""

from __future__ import annotations

import math

import pytest

from trw_memory.exceptions import DimensionMismatchError, MemoryError
from trw_memory.retrieval.dense import cosine_similarity


class TestDimensionMismatchError:
    """Verify DimensionMismatchError is raised for incompatible vector dimensions."""

    def test_mismatched_dimensions_raises_dimension_mismatch(self) -> None:
        """cosine_similarity([1,2,3], [1,2]) raises DimensionMismatchError."""
        with pytest.raises(DimensionMismatchError, match="Dimension mismatch"):
            cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0])

    def test_dimension_mismatch_is_memory_error_subclass(self) -> None:
        """DimensionMismatchError is a MemoryError subclass."""
        with pytest.raises(MemoryError):
            cosine_similarity([1.0], [1.0, 2.0])

    def test_dimension_mismatch_distinguishable_from_generic_exception(self) -> None:
        """DimensionMismatchError can be caught separately from generic Exception."""
        try:
            cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0])
            pytest.fail("Should have raised DimensionMismatchError")
        except DimensionMismatchError:
            pass  # This is the expected path
        except Exception:
            pytest.fail("Caught generic Exception instead of DimensionMismatchError")


class TestCosineOrthogonality:
    """Verify cosine similarity of orthogonal vectors returns 0.0."""

    def test_orthogonal_vectors_return_zero(self) -> None:
        """Two perpendicular unit vectors have cosine similarity of 0.0."""
        result = cosine_similarity([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
        assert result == 0.0

    def test_identical_vectors_return_one(self) -> None:
        """Identical vectors have cosine similarity of 1.0."""
        result = cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        assert math.isclose(result, 1.0, rel_tol=1e-9)

    def test_opposite_vectors_return_negative_one(self) -> None:
        """Opposite vectors have cosine similarity of -1.0."""
        result = cosine_similarity([1.0, 0.0], [-1.0, 0.0])
        assert math.isclose(result, -1.0, rel_tol=1e-9)

    def test_zero_vector_returns_zero(self) -> None:
        """Zero vector against any vector returns 0.0 (not division error)."""
        result = cosine_similarity([0.0, 0.0, 0.0], [1.0, 2.0, 3.0])
        assert result == 0.0

    def test_empty_vectors_raise_mismatch_or_return_zero(self) -> None:
        """Empty vectors — same length, so cosine_similarity should return 0.0
        (both are zero-vectors)."""
        result = cosine_similarity([], [])
        assert result == 0.0

    def test_single_dimension(self) -> None:
        """Single-dimension positive vectors are perfectly aligned."""
        result = cosine_similarity([5.0], [3.0])
        assert math.isclose(result, 1.0, rel_tol=1e-9)

    def test_mismatched_by_one(self) -> None:
        """Off-by-one dimension mismatch is still caught."""
        with pytest.raises(DimensionMismatchError):
            cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0])
