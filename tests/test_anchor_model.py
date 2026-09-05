"""Tests for Anchor model and MemoryEntry anchor fields (PRD-CORE-111).

Covers:
- Anchor model defaults and validation
- MemoryEntry.anchors max-3 constraint
- Anchor.file absolute path rejection
- Anchor.signature truncation at 200 chars
- Anchor.symbol_type literal validation
- MemoryEntry.anchor_validity default
- to_dict() includes anchors and anchor_validity
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from trw_memory.models.memory import Anchor, MemoryEntry

# ---------------------------------------------------------------------------
# Anchor model defaults
# ---------------------------------------------------------------------------


def test_anchor_defaults() -> None:
    """Anchor with only required fields has correct defaults."""
    anchor = Anchor(file="foo.py", symbol_name="bar")
    assert anchor.file == "foo.py"
    assert anchor.symbol_name == "bar"
    assert anchor.symbol_type == "function"
    assert anchor.signature == ""
    assert anchor.line_range is None


# ---------------------------------------------------------------------------
# Anchor.file validation
# ---------------------------------------------------------------------------


def test_reject_absolute_path() -> None:
    """Anchor.file starting with '/' raises ValidationError."""
    with pytest.raises(ValidationError, match="relative path"):
        Anchor(file="/abs/path.py", symbol_name="foo")


def test_relative_path_accepted() -> None:
    """Anchor.file with a relative path is accepted."""
    anchor = Anchor(file="src/module.py", symbol_name="my_func")
    assert anchor.file == "src/module.py"


# ---------------------------------------------------------------------------
# Anchor.symbol_type validation
# ---------------------------------------------------------------------------


def test_symbol_type_function_valid() -> None:
    """symbol_type='function' is accepted."""
    anchor = Anchor(file="a.py", symbol_name="fn", symbol_type="function")
    assert anchor.symbol_type == "function"


def test_symbol_type_all_valid() -> None:
    """All valid symbol_type values are accepted."""
    valid_types = ["function", "class", "method", "const", "type", "impl"]
    for st in valid_types:
        anchor = Anchor(file="a.py", symbol_name="s", symbol_type=st)  # type: ignore[arg-type]
        assert anchor.symbol_type == st


def test_symbol_type_bogus_raises() -> None:
    """Invalid symbol_type raises ValidationError."""
    with pytest.raises(ValidationError):
        Anchor(file="a.py", symbol_name="s", symbol_type="bogus")  # type: ignore[typeddict-item]


def test_symbol_type_literal_validation() -> None:
    """'function' is OK, 'bogus' raises."""
    anchor = Anchor(file="a.py", symbol_name="s", symbol_type="function")
    assert anchor.symbol_type == "function"

    with pytest.raises(ValidationError):
        Anchor(file="a.py", symbol_name="s", symbol_type="bogus")  # type: ignore[typeddict-item]


# ---------------------------------------------------------------------------
# Anchor.signature truncation
# ---------------------------------------------------------------------------


def test_signature_max_200_truncation() -> None:
    """A 250-char signature is truncated to 200 chars."""
    long_sig = "x" * 250
    anchor = Anchor(file="a.py", symbol_name="fn", signature=long_sig)
    assert len(anchor.signature) == 200
    assert anchor.signature == "x" * 200


def test_signature_at_200_unchanged() -> None:
    """A 200-char signature is unchanged."""
    sig_200 = "a" * 200
    anchor = Anchor(file="a.py", symbol_name="fn", signature=sig_200)
    assert len(anchor.signature) == 200


def test_signature_below_200_unchanged() -> None:
    """A signature under 200 chars is unchanged."""
    anchor = Anchor(file="a.py", symbol_name="fn", signature="def fn(x: int) -> str")
    assert anchor.signature == "def fn(x: int) -> str"


# ---------------------------------------------------------------------------
# MemoryEntry anchors max-3 constraint
# ---------------------------------------------------------------------------


def test_max_3_anchors() -> None:
    """MemoryEntry accepts up to 3 anchors."""
    anchors = [
        Anchor(file="a.py", symbol_name="fn_a"),
        Anchor(file="b.py", symbol_name="fn_b"),
        Anchor(file="c.py", symbol_name="fn_c"),
    ]
    entry = MemoryEntry(id="M-anc1", content="three anchors", anchors=anchors)
    assert len(entry.anchors) == 3


def test_4th_anchor_raises() -> None:
    """4 anchors raises ValidationError."""
    anchors = [
        Anchor(file="a.py", symbol_name="fn_a"),
        Anchor(file="b.py", symbol_name="fn_b"),
        Anchor(file="c.py", symbol_name="fn_c"),
        Anchor(file="d.py", symbol_name="fn_d"),
    ]
    with pytest.raises(ValidationError):
        MemoryEntry(id="M-anc2", content="four anchors", anchors=anchors)


def test_zero_anchors_ok() -> None:
    """MemoryEntry with no anchors is valid (default)."""
    entry = MemoryEntry(id="M-anc3", content="no anchors")
    assert entry.anchors == []


# ---------------------------------------------------------------------------
# MemoryEntry.anchor_validity
# ---------------------------------------------------------------------------


def test_anchor_validity_default() -> None:
    """MemoryEntry has anchor_validity=1.0 by default."""
    entry = MemoryEntry(id="M-av1", content="test")
    # PRD-CORE-244-FR01: unassessed is None, not a perfect score.
    assert entry.anchor_validity is None


def test_anchor_validity_custom() -> None:
    """anchor_validity can be set to values between 0.0 and 1.0."""
    entry = MemoryEntry(id="M-av2", content="test", anchor_validity=0.67)
    assert entry.anchor_validity == 0.67


def test_anchor_validity_zero() -> None:
    """anchor_validity=0.0 is valid (all anchors broken)."""
    entry = MemoryEntry(id="M-av3", content="test", anchor_validity=0.0)
    assert entry.anchor_validity == 0.0


def test_anchor_validity_out_of_range_raises() -> None:
    """anchor_validity > 1.0 raises ValidationError."""
    with pytest.raises(ValidationError):
        MemoryEntry(id="M-av4", content="test", anchor_validity=1.5)


# ---------------------------------------------------------------------------
# to_dict() includes anchors and anchor_validity
# ---------------------------------------------------------------------------


def test_anchors_in_to_dict() -> None:
    """to_dict() includes anchors and anchor_validity."""
    anchor = Anchor(file="src/mod.py", symbol_name="my_func", symbol_type="function")
    entry = MemoryEntry(
        id="M-dict1",
        content="with anchors",
        anchors=[anchor],
        anchor_validity=0.5,
    )
    d = entry.to_dict()

    assert "anchors" in d
    assert "anchor_validity" in d
    assert d["anchor_validity"] == 0.5
    assert isinstance(d["anchors"], list)
    assert len(d["anchors"]) == 1
    assert d["anchors"][0]["file"] == "src/mod.py"
    assert d["anchors"][0]["symbol_name"] == "my_func"
    assert d["anchors"][0]["symbol_type"] == "function"


def test_anchors_empty_in_to_dict() -> None:
    """to_dict() includes empty anchors list and anchor_validity=1.0 by default."""
    entry = MemoryEntry(id="M-dict2", content="no anchors")
    d = entry.to_dict()
    assert d["anchors"] == []
    assert d["anchor_validity"] is None
