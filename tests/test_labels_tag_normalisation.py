"""LABEL-TAG-STRIP: a tag matches a label rule however it is spelled on either side.

Rule tags and row tags were both only ``.lower()``ed, so a row tagged ``"finance "`` (trailing space), fullwidth
``FINANCE`` or ``"fin\\u200bance"`` (zero-width space) missed a ``finance`` rule and fell back to ``team``: a row the
operator labelled ``personal`` left the host. Both sides now go through ``labels.tag_key`` (NFKC, invisible
characters dropped, casefolded, whitespace trimmed and collapsed), the one normalisation every label match uses.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from trw_memory.labels import LabelPolicy, Level, Sink
from trw_memory.models.memory import MemoryEntry


@pytest.fixture
def user_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    base = tmp_path / "user-base"
    base.mkdir()
    monkeypatch.setenv("TRW_USER_DIR", str(base))
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    return base


def _write(base: Path, text: str) -> None:
    path = base / "labels.yaml"
    path.write_text(text, encoding="utf-8")
    path.chmod(0o600)


def _fullwidth(text: str) -> str:
    return "".join(chr(ord(ch) + 0xFEE0) for ch in text)


def _row(*tags: str, namespace: str = "default") -> MemoryEntry:
    return MemoryEntry(id="L-x", content="c", namespace=namespace, tags=list(tags))


@pytest.mark.parametrize(
    "tag",
    ["finance ", " finance", "Finance\t", _fullwidth("FINANCE"), "fin​ance", "FINANCE", "finance\u00a0"],
    ids=["trailing", "leading", "tab-case", "fullwidth", "zero-width", "upper", "nbsp"],
)
def test_a_row_tag_spelled_differently_still_matches_its_rule(user_dir: Path, tag: str) -> None:
    _write(user_dir, "version: 1\nrules:\n  - tags_any: [finance]\n    level: personal\n")
    policy = LabelPolicy.current()

    assert policy.label_of(_row(tag)) is Level.PERSONAL
    assert not policy.admit([_row(tag)], Sink.PLATFORM).admitted, "a personal row never reaches the platform"


def test_a_namespace_bound_rule_normalises_row_tags_the_same_way(user_dir: Path) -> None:
    _write(user_dir, 'version: 1\nrules:\n  - namespace: "user:local"\n    tags_any: [dots]\n    level: personal\n')

    assert LabelPolicy.current().label_of(_row("Dots ", namespace="user:local")) is Level.PERSONAL


def test_a_rule_tag_written_with_stray_spacing_or_case_matches_a_plain_row_tag(user_dir: Path) -> None:
    _write(
        user_dir, f'version: 1\nrules:\n  - tags_any: [" Health ", "{_fullwidth("TRAVEL")}"]\n    level: sensitive\n'
    )
    policy = LabelPolicy.current()

    assert policy.label_of(_row("health")) is Level.SENSITIVE
    assert policy.label_of(_row("travel")) is Level.SENSITIVE


def test_a_blank_rule_tag_never_discards_the_other_rules(user_dir: Path) -> None:
    """A rule tag that normalises to nothing is kept as the empty key, never a reason to load the file strict: strict
    drops every rule, so a ``finance: sensitive`` row would fall to ``team`` and egress (codex r1 block)."""
    _write(
        user_dir,
        'version: 1\nrules:\n  - tags_any: [finance]\n    level: sensitive\n  - tags_any: [" "]\n    level: personal\n',
    )
    policy = LabelPolicy.current()

    assert policy.source == "file"
    assert policy.label_of(_row("finance", namespace="project:alpha")) is Level.SENSITIVE
    assert not policy.admit([_row("finance", namespace="project:alpha")], Sink.PLATFORM).admitted
    assert policy.label_of(_row("\u200b")) is Level.PERSONAL, "a blank row tag still matches the blank rule, as before"


@pytest.mark.parametrize("invisible", ["\u034f", "\u3164", "\u115f", "\ufe0f", "\u180b", "\uffa0", "\u00ad"])
def test_a_default_ignorable_character_cannot_split_a_tag_from_its_rule(user_dir: Path, invisible: str) -> None:
    """Unicode default-ignorable characters that are not category Cf (codex r1 known issue)."""
    _write(user_dir, "version: 1\nrules:\n  - tags_any: [finance]\n    level: sensitive\n")

    assert LabelPolicy.current().label_of(_row(f"fin{invisible}ance")) is Level.SENSITIVE


def test_unrelated_tags_are_unchanged(user_dir: Path) -> None:
    _write(user_dir, "version: 1\nrules:\n  - tags_any: [finance]\n    level: personal\n")

    assert LabelPolicy.current().label_of(_row("financial-planning", "fin ance")) is Level.TEAM


# Unicode 16.0.0 Default_Ignorable_Code_Point, every range's endpoints (DerivedCoreProperties.txt), plus U+2065, which
# is DICP but category Cn, so a Cf-plus-hand-list check missed it (C1 red team on LABEL-TAG-STRIP-a).
_DICP_SAMPLES = [
    0x00AD,
    0x034F,
    0x061C,
    0x115F,
    0x1160,
    0x17B4,
    0x17B5,
    0x180B,
    0x180F,
    0x200B,
    0x200F,
    0x202A,
    0x202E,
    0x2060,
    0x2064,
    0x2065,
    0x2066,
    0x206F,
    0x3164,
    0xFE00,
    0xFE0F,
    0xFEFF,
    0xFFA0,
    0xFFF0,
    0xFFF8,
    0x1BCA0,
    0x1BCA3,
    0x1D173,
    0x1D17A,
    0xE0000,
    0xE0001,
    0xE001F,
    0xE007F,
    0xE00FF,
    0xE01EF,
    0xE01F0,
    0xE0FFF,
]


@pytest.mark.parametrize("point", _DICP_SAMPLES, ids=[f"U+{p:04X}" for p in _DICP_SAMPLES])
def test_no_default_ignorable_code_point_can_split_a_tag(user_dir: Path, point: int) -> None:
    _write(user_dir, "version: 1\nrules:\n  - tags_any: [finance]\n    level: personal\n")

    assert LabelPolicy.current().label_of(_row(f"fin{chr(point)}ance")) is Level.PERSONAL


def test_the_vendored_ignorable_table_is_at_least_the_running_unicode_version() -> None:
    """A floor, not an equality: an older Python (3.11+ ship Unicode 14 to 15.1) runs with a newer table, which is a
    superset. A Python whose Unicode is NEWER than the table fails here until ``labels/_default_ignorable.py`` is
    regenerated (scripts/gen_default_ignorable.py)."""
    import unicodedata
    from itertools import pairwise

    from trw_memory.labels._default_ignorable import DEFAULT_IGNORABLE_RANGES, UNICODE_VERSION

    def version(text: str) -> tuple[int, ...]:
        return tuple(int(part) for part in text.split("."))

    assert version(UNICODE_VERSION) >= version(unicodedata.unidata_version)
    assert all(a[1] < b[0] for a, b in pairwise(DEFAULT_IGNORABLE_RANGES))
    assert any(low <= 0x2065 <= high for low, high in DEFAULT_IGNORABLE_RANGES)
