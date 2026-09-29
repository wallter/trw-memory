"""PRD-QUAL-146 FR10: every skip site under trw-memory/tests carries a registered reason category."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from tests._skip_categories import CATEGORY_NAMES, scan_source, scan_tree

pytestmark = pytest.mark.unit

_TESTS = Path(__file__).resolve().parent


def test_every_skip_site_in_the_package_is_categorized() -> None:
    sites = scan_tree(_TESTS)
    uncategorized = [f"{s.path}:{s.line} {s.kind} reason={s.reason!r}" for s in sites if s.category is None]
    assert sites, "the census found no skip sites at all: the scanner is broken"
    assert uncategorized == [], "uncategorized skip sites (add a registered reason or a skip-category comment):\n" + (
        "\n".join(uncategorized)
    )
    counts = Counter(s.category for s in sites)
    assert set(counts) <= CATEGORY_NAMES


def test_a_reason_matching_no_category_is_uncategorized() -> None:
    [site] = scan_source('import pytest\npytest.skip("flaky, will look later")\n', "synthetic.py")
    assert (site.path, site.line, site.kind, site.category) == ("synthetic.py", 2, "skip", None)


def test_a_non_literal_reason_needs_a_registered_comment() -> None:
    bare = scan_source("import pytest\ndef f(r):\n    pytest.skip(r)\n", "s.py")
    commented = scan_source("import pytest\ndef f(r):\n    pytest.skip(r)  # skip-category: host-tool\n", "s.py")
    unregistered = scan_source("import pytest\ndef f(r):\n    pytest.skip(r)  # skip-category: someday\n", "s.py")
    assert [s.category for s in bare] == [None]
    assert [s.category for s in commented] == ["host-tool"]
    assert [s.category for s in unregistered] == [None]


def test_importorskip_is_an_optional_dependency_by_construction() -> None:
    [site] = scan_source('import pytest\nnp = pytest.importorskip("numpy")\n', "s.py")
    assert (site.kind, site.category) == ("importorskip", "optional-dependency")


def test_a_tracked_defect_id_counts_only_on_an_xfail() -> None:
    src = 'import pytest\npytest.xfail("PRD-FIX-1 open")\npytest.skip("PRD-FIX-1 open")\n'
    assert [(s.kind, s.category) for s in scan_source(src, "s.py")] == [("xfail", "tracked-defect"), ("skip", None)]


def test_marker_reasons_resolve_fstrings_and_module_constants() -> None:
    src = (
        "import pytest\n"
        '_WHY = "sqlite-vec did not load"\n'
        "@pytest.mark.skipif(True, reason=_WHY)\n"
        "def a(): ...\n"
        '@pytest.mark.skipif(True, reason=f"{1} POSIX mode bits")\n'
        "def b(): ...\n"
        "@pytest.mark.skip\n"
        "def c(): ...\n"
    )
    assert [(s.kind, s.category) for s in scan_source(src, "s.py")] == [
        ("mark.skipif", "optional-dependency"),
        ("mark.skipif", "platform"),
        ("mark.skip", None),
    ]
