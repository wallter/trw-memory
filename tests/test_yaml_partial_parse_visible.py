"""W15 — a row whose verification evidence did not parse must not read as evidence-free.

Two silent drops lived in the YAML row mapper:

* a malformed assertion was skipped individually, and
* ONE malformed anchor discarded the entire anchor list, because the whole list
  comprehension sat inside a single ``try``.

Either way the row came back as an ordinary :class:`MemoryEntry`, byte-identical
to one stored with no anchors and no assertions. The consequence is not cosmetic:
``YAMLBackend.update`` is a read-modify-write that serialises the PARSED entry
back over the file, so a drop on the read leg deleted the unparsed evidence from
disk on the write leg.

Faults are injected by writing the malformed row to a real entries directory and
driving the real backend.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from structlog.testing import capture_logs

from trw_memory.exceptions import StorageError
from trw_memory.storage.yaml_backend import YAMLBackend

pytestmark = pytest.mark.integration

_NAMESPACE = "project:yaml-partial"
_GOOD_ANCHOR = {"file": "src/app.py", "symbol_name": "handler"}
_BAD_ANCHOR = {"file": "src/app.py"}  # missing symbol_name
_GOOD_ASSERTION = {"type": "grep_present", "pattern": "handler", "target": "src/*.py"}
_BAD_ASSERTION = {"type": "not_a_real_assertion_type", "target": "src/*.py"}


def _write_row(entries_dir: Path, entry_id: str = "M-partial") -> Path:
    """Write an entry file carrying one valid and one malformed item of each kind."""
    entries_dir.mkdir(parents=True, exist_ok=True)
    path = entries_dir / f"{entry_id}.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "id": entry_id,
                "content": "anchored learning",
                "namespace": _NAMESPACE,
                "anchors": [_GOOD_ANCHOR, _BAD_ANCHOR],
                "assertions": [_GOOD_ASSERTION, _BAD_ASSERTION],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_one_malformed_anchor_no_longer_discards_the_valid_ones(tmp_path: Path) -> None:
    """The valid anchor survives its malformed sibling."""
    entries = tmp_path / "entries"
    _write_row(entries)
    backend = YAMLBackend(entries)

    entry = backend.get("M-partial", namespace=_NAMESPACE)

    assert entry is not None
    assert [(a.file, a.symbol_name) for a in entry.anchors] == [("src/app.py", "handler")]
    assert len(entry.assertions) == 1


def test_a_partial_parse_is_logged_with_counts(tmp_path: Path) -> None:
    """The drop is a WARNING naming the row, not silence."""
    entries = tmp_path / "entries"
    _write_row(entries)
    backend = YAMLBackend(entries)

    with capture_logs() as logs:
        backend.get("M-partial", namespace=_NAMESPACE)

    partial = [line for line in logs if line.get("event") == "yaml_row_partial_parse"]
    assert len(partial) == 1
    assert partial[0]["entry_id"] == "M-partial"
    assert (partial[0]["dropped_anchors"], partial[0]["dropped_assertions"]) == (1, 1)
    assert partial[0]["log_level"] == "warning"


def test_update_refuses_to_rewrite_a_partially_parsed_row(tmp_path: Path) -> None:
    """The read-modify-write that would erase the evidence is refused instead.

    Before this change ``update`` parsed the row, dropped what it could not read,
    and wrote the parsed entry back -- so a single malformed anchor cost the row
    every anchor it had, permanently, on the next unrelated field update.
    """
    entries = tmp_path / "entries"
    path = _write_row(entries)
    backend = YAMLBackend(entries)
    before = path.read_text(encoding="utf-8")

    with pytest.raises(StorageError, match="did not parse"):
        backend.update("M-partial", content="updated")

    assert path.read_text(encoding="utf-8") == before
    stored = yaml.safe_load(before)
    assert stored["anchors"] == [_GOOD_ANCHOR, _BAD_ANCHOR]
    assert stored["assertions"] == [_GOOD_ASSERTION, _BAD_ASSERTION]


def test_a_clean_row_still_updates(tmp_path: Path) -> None:
    """The refusal is scoped to unparseable evidence, not to updates in general."""
    entries = tmp_path / "entries"
    entries.mkdir(parents=True)
    (entries / "M-clean.yaml").write_text(
        yaml.safe_dump(
            {
                "id": "M-clean",
                "content": "original",
                "namespace": _NAMESPACE,
                "anchors": [_GOOD_ANCHOR],
                "assertions": [_GOOD_ASSERTION],
            }
        ),
        encoding="utf-8",
    )
    backend = YAMLBackend(entries)

    updated = backend.update("M-clean", content="updated")

    assert updated is not None
    assert updated.content == "updated"
    assert len(updated.anchors) == 1
