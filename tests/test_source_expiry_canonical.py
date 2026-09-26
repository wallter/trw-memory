"""Source expiry and temporal eligibility share CORE-244's date convention."""

from datetime import datetime, timezone
from typing import Any

import pytest

from trw_memory.retrieval.source_policy import SourcePolicy, is_expired_result
from trw_memory.retrieval.validity_prior import expiry_has_passed


def _apply(rows: Any, **options: Any) -> list[dict[str, Any]]:
    """Admission plus the source-weighted order, as ``MemoryClient.recall`` applies it.

    Mirrors the removed ``SourcePolicy.apply`` — production now composes
    ``allows``/``rank_key`` inline (see ``_client_recall_helpers.py``).
    """
    policy = SourcePolicy.resolve(**options)
    ranked = [(policy.rank_key(row), row) for row in rows if policy.allows(row)]
    ranked.sort(key=lambda item: item[0])
    return [dict(row, score=-key[1]) for key, row in ranked]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("", False),
        ("invalid", False),
        ("2024-01-01", True),
        ("2024-01-02", False),
        ("2024-01-03", False),
        # PRD-CORE-278 FR05: a value carrying a TIME expires at that instant.
        # This row asserted False — midnight on the reference day treated as
        # unexpired until the following day — which is exactly the defect
        # sub_4-nL1paSXxQx41fH reported: an entry that expired an hour ago kept
        # its slot in recall for the rest of the day. Bare dates below keep the
        # day-exclusive convention.
        ("2024-01-02T00:00:00Z", True),
        ("2024-01-02T23:00:00-06:00", False),
        ("2024-01-01T23:00:00-06:00", True),
        ("20240102", False),
    ],
)
@pytest.mark.parametrize("in_metadata", [False, True])
def test_source_expiry_matches_canonical_predicate(raw: str, expected: bool, in_metadata: bool) -> None:
    reference = datetime(2024, 1, 2, 12, tzinfo=timezone.utc)
    result = {"metadata": {"expires": raw}} if in_metadata else {"expires": raw}
    assert expiry_has_passed(raw, reference_time=reference) is expected
    assert is_expired_result(result, now=reference) is expected


def test_model_expiry_precedes_metadata_expiry() -> None:
    result = {"expires": "2024-01-03", "metadata": {"expires": "2020-01-01"}}
    assert not is_expired_result(result, now=datetime(2024, 1, 2, tzinfo=timezone.utc))


@pytest.mark.parametrize("family", ["episodic", "lifecycle"])
def test_source_policy_filters_against_explicit_reference(family: str) -> None:
    rows = [
        {"memory_id": "expired", "score": 0.9, "tags": [f"source_kind:{family}"], "expires": "2024-01-01"},
        {"memory_id": "same-day", "score": 0.7, "tags": [f"source_kind:{family}"], "expires": "2024-01-02"},
    ]
    reference = datetime(2024, 1, 2, tzinfo=timezone.utc)
    kept = _apply(rows, reference_time=reference)
    assert [row["memory_id"] for row in kept] == ["same-day"]
    all_rows = _apply(rows, exclude_expired=False, reference_time=reference)
    assert [row["memory_id"] for row in all_rows] == ["expired", "same-day"]


@pytest.mark.parametrize(
    ("reference", "expected"),
    [
        ("2024-01-02T00:30:00+02:00", False),
        ("2024-01-01T22:30:00+00:00", False),
        ("2024-01-01T23:30:00-02:00", True),
        ("2024-01-02T01:30:00+00:00", True),
        ("2024-01-02T00:00:00", True),
    ],
)
def test_expiry_reference_is_utc_not_local_calendar(reference: str, expected: bool) -> None:
    instant = datetime.fromisoformat(reference)
    assert expiry_has_passed("2024-01-01", reference_time=instant) is expected
    assert is_expired_result({"expires": "2024-01-01"}, now=instant) is expected
