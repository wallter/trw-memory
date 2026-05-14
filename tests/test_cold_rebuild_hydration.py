# ruff: noqa: F401
"""Direct hydrator unit tests for cold rebuild."""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from trw_memory.storage._cold_rebuild import _hydrate_yaml, _HydrationError

from ._test_cold_rebuild_support import _configure_structlog


def test_hydrate_yaml_missing_id_raises() -> None:
    """_hydrate_yaml raises _HydrationError('id') when id missing."""
    with pytest.raises(_HydrationError) as exc_info:
        _hydrate_yaml({"summary": "x", "created": "2026-04-12", "updated": "2026-04-12"})
    assert exc_info.value.field == "id"


def test_hydrate_yaml_hardcodes_type() -> None:
    """_hydrate_yaml directly: the 'type' column slot is always 'pattern'."""
    row = _hydrate_yaml(
        {
            "id": "L-X",
            "summary": "x",
            "created": "2026-04-12",
            "updated": "2026-04-12",
            "source_type": "human",
        }
    )
    assert row is not None
    from trw_memory.storage._cold_rebuild import _INSERT_COLUMNS

    assert row[_INSERT_COLUMNS.index("type")] == "pattern"
    assert row[_INSERT_COLUMNS.index("source")] == "human"


def test_hydrate_yaml_datetime_object_created() -> None:
    """_coerce_ts handles datetime objects (ruamel loads full ISO as datetime)."""
    dt = datetime(2026, 4, 12, 15, 30, 0, tzinfo=timezone.utc)
    row = _hydrate_yaml({"id": "L-DT", "summary": "x", "created": dt, "updated": dt})
    assert row is not None
    from trw_memory.storage._cold_rebuild import _INSERT_COLUMNS

    assert row[_INSERT_COLUMNS.index("created_at")] == dt.isoformat()


def test_hydrate_yaml_date_object_created() -> None:
    """_coerce_ts handles date objects (ruamel loads bare YYYY-MM-DD as date)."""
    created = date(2026, 4, 12)
    row = _hydrate_yaml({"id": "L-DO", "summary": "x", "created": created, "updated": created})
    assert row is not None
    from trw_memory.storage._cold_rebuild import _INSERT_COLUMNS

    assert row[_INSERT_COLUMNS.index("created_at")] == "2026-04-12T00:00:00+00:00"


def test_hydrate_yaml_missing_updated_falls_back_to_created() -> None:
    """Permissive fallback: missing updated reuses created_at."""
    row = _hydrate_yaml({"id": "L-NU", "summary": "x", "created": "2026-04-12"})
    assert row is not None
    from trw_memory.storage._cold_rebuild import _INSERT_COLUMNS

    created = row[_INSERT_COLUMNS.index("created_at")]
    updated = row[_INSERT_COLUMNS.index("updated_at")]
    assert created == updated


def test_hydrate_yaml_bad_impact_raises() -> None:
    """_hydrate_yaml raises on non-float impact."""
    with pytest.raises(_HydrationError) as exc_info:
        _hydrate_yaml(
            {
                "id": "L-B",
                "summary": "x",
                "created": "2026-04-12",
                "updated": "2026-04-12",
                "impact": "not-a-number",
            }
        )
    assert exc_info.value.field == "impact"


def test_hydrate_yaml_bad_recurrence_raises() -> None:
    """_hydrate_yaml raises on non-int recurrence."""
    with pytest.raises(_HydrationError) as exc_info:
        _hydrate_yaml(
            {
                "id": "L-B",
                "summary": "x",
                "created": "2026-04-12",
                "updated": "2026-04-12",
                "recurrence": "abc",
            }
        )
    assert exc_info.value.field == "recurrence"


def test_hydrate_yaml_bad_list_field_raises() -> None:
    """_hydrate_yaml raises _HydrationError with field name when list shape is wrong."""
    with pytest.raises(_HydrationError) as exc_info:
        _hydrate_yaml(
            {
                "id": "L-L",
                "summary": "x",
                "created": "2026-04-12",
                "updated": "2026-04-12",
                "tags": "not-a-list",
            }
        )
    assert exc_info.value.field == "tags"


def test_hydrate_yaml_bad_dict_field_raises() -> None:
    """_hydrate_yaml raises _HydrationError with field name when dict shape is wrong."""
    with pytest.raises(_HydrationError) as exc_info:
        _hydrate_yaml(
            {
                "id": "L-D",
                "summary": "x",
                "created": "2026-04-12",
                "updated": "2026-04-12",
                "metadata": "not-a-dict",
            }
        )
    assert exc_info.value.field == "metadata"
