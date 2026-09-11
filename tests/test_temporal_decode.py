"""Early temporal decisions match fully mapped rows across legacy formats."""

from datetime import datetime, timezone

import pytest

from trw_memory.models.memory import MemoryEntry
from trw_memory.retrieval.temporal_selection import TemporalSelection
from trw_memory.storage._parsing import parse_dt_safe
from trw_memory.storage._row_mapper import entry_to_row, row_to_entry
from trw_memory.storage._shared import ENTRY_COLUMNS
from trw_memory.storage._temporal_decode import TemporalMaterialization

REFERENCE = datetime(2022, 1, 1, tzinfo=timezone.utc)


@pytest.mark.parametrize("created", [None, "malformed", "2020-01-01"])
@pytest.mark.parametrize("opened", [None, "malformed", "20200101T000000+0000", "2020-01-01T01:00:00+01:00"])
@pytest.mark.parametrize("closed", [None, "malformed", "2024-01-01T00:00:00Z"])
@pytest.mark.parametrize("expires", ["", "malformed", "2022-01-01", "2021-12-31T23:00:00-07:00"])
@pytest.mark.parametrize("historical", [False, True])
def test_early_decision_matches_row_mapper(created, opened, closed, expires, historical) -> None:
    raw = list(entry_to_row(MemoryEntry(id="parity", content="parity")))
    fields = {
        "created_at": created,
        "valid_from": opened,
        "invalid_from": closed,
        "invalidated_by": "replacement" if parse_dt_safe(closed, default=None) else None,
        "expires_at": expires,
    }
    for key, value in fields.items():
        raw[ENTRY_COLUMNS.index(key)] = value
    selection = TemporalSelection(as_of=REFERENCE if historical else None, reference_time=REFERENCE)
    mapped = row_to_entry(tuple(raw), reference_time=REFERENCE)
    assert TemporalMaterialization(selection, limit=1).retain(raw) == selection.eligible(mapped)
