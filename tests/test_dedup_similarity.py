"""Tests for cosine similarity behavior."""

from __future__ import annotations

import math

import pytest

from trw_memory.retrieval.dense import cosine_similarity


class TestCosineSimilarity:
    def test_identical_vectors(self) -> None:
        a = [1.0, 0.0, 0.0]
        assert cosine_similarity(a, a) == pytest.approx(1.0)

    def test_orthogonal_vectors(self) -> None:
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self) -> None:
        a = [1.0, 0.0, 0.0]
        b = [-1.0, 0.0, 0.0]
        assert cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_zero_vector_a(self) -> None:
        a = [0.0, 0.0, 0.0]
        b = [1.0, 0.0, 0.0]
        assert cosine_similarity(a, b) == 0.0

    def test_zero_vector_b(self) -> None:
        a = [1.0, 0.0, 0.0]
        b = [0.0, 0.0, 0.0]
        assert cosine_similarity(a, b) == 0.0

    def test_both_empty_lists(self) -> None:
        assert cosine_similarity([], []) == 0.0

    def test_partial_similarity(self) -> None:
        v = 1.0 / math.sqrt(2.0)
        a = [v, v, 0.0]
        b = [1.0, 0.0, 0.0]
        assert cosine_similarity(a, b) == pytest.approx(v, abs=1e-5)

    def test_unnormalized_vectors(self) -> None:
        a = [3.0, 0.0]
        b = [0.0, 4.0]
        assert cosine_similarity(a, b) == pytest.approx(0.0)
