"""Tests for trw_memory.utils.id_gen — compact base-62 ID generation."""

from __future__ import annotations

import re
from unittest.mock import patch

import pytest

from trw_memory.utils import generate_compact_id

# ---------------------------------------------------------------------------
# Pattern / charset tests
# ---------------------------------------------------------------------------


def test_compact_id_charset() -> None:
    """Generated ID matches pattern ^L-[a-zA-Z0-9]{4}$."""
    id_ = generate_compact_id()
    assert re.fullmatch(r"L-[a-zA-Z0-9]{4}", id_), f"ID {id_!r} does not match pattern"


def test_compact_id_length() -> None:
    """Default length is 4 random chars after the prefix dash."""
    id_ = generate_compact_id()
    suffix = id_.split("-", 1)[1]
    assert len(suffix) == 4


def test_compact_id_prefix_M() -> None:
    """prefix='M' produces IDs starting with 'M-'."""
    id_ = generate_compact_id(prefix="M")
    assert re.fullmatch(r"M-[a-zA-Z0-9]{4}", id_), f"ID {id_!r} does not match M-xxxx pattern"


def test_compact_id_custom_length() -> None:
    """Custom length parameter is respected."""
    id_ = generate_compact_id(length=8)
    suffix = id_.split("-", 1)[1]
    assert len(suffix) == 8
    assert re.fullmatch(r"[a-zA-Z0-9]{8}", suffix)


# ---------------------------------------------------------------------------
# Uniqueness / collision retry tests
# ---------------------------------------------------------------------------


def test_compact_id_uniqueness_1k_cumulative() -> None:
    """Generate 1000 IDs with cumulative existing_ids set; all must be unique."""
    seen: set[str] = set()
    for _ in range(1000):
        # Use length=6 to reduce collision probability in the loop
        id_ = generate_compact_id(prefix="L", length=6, existing_ids=seen)
        assert id_ not in seen, f"Duplicate ID generated: {id_!r}"
        seen.add(id_)
    assert len(seen) == 1000


def test_compact_id_collision_retry() -> None:
    """Collision retry: mock secrets.choice to force first N attempts to collide."""
    # Simulate the first two attempts producing "aaaa", then "bbbb"
    # so that when existing_ids={"L-aaaa"}, the first attempt collides
    # and the second succeeds.
    call_count = 0

    def mock_choice(seq: str) -> str:
        nonlocal call_count
        call_count += 1
        # First 4 calls (first ID attempt) → 'a'
        # Next 4 calls (second attempt) → 'b'
        if call_count <= 4:
            return "a"
        return "b"

    with patch("trw_memory.utils.id_gen.secrets.choice", side_effect=mock_choice):
        id_ = generate_compact_id(prefix="L", length=4, existing_ids={"L-aaaa"})

    assert id_ == "L-bbbb"


def test_compact_id_max_retries_raises() -> None:
    """RuntimeError raised when every attempt collides (max_retries exceeded)."""
    # Always return the same suffix
    with patch("trw_memory.utils.id_gen.secrets.choice", return_value="a"):
        existing = {"L-aaaa"}
        with pytest.raises(RuntimeError, match="exceeded"):
            generate_compact_id(prefix="L", length=4, existing_ids=existing, max_retries=3)


# ---------------------------------------------------------------------------
# Re-export from utils/__init__.py
# ---------------------------------------------------------------------------


def test_generate_compact_id_importable_from_utils_package() -> None:
    """generate_compact_id is importable from the utils package."""
    from trw_memory.utils import generate_compact_id as fn

    assert callable(fn)
