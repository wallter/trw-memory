"""Coverage for the pure org-shared recall helpers in ``_client_org_shared``."""

from __future__ import annotations

from trw_memory._client_org_shared import coerce_float, matches_query


def _make_result(content: str = "", detail: str = "", tags: list[str] | None = None) -> dict:
    return {
        "memory_id": "MQ-001",
        "content": content,
        "detail": detail,
        "tags": tags or [],
        "score": 0.9,
        "source": "local",
        "importance": 0.5,
        "created_at": "",
        "updated_at": "",
        "namespace": "default",
    }


def test_matches_query_true_when_content_contains_query() -> None:
    result = _make_result(content="information about caching strategies")
    assert matches_query(result, "caching") is True  # type: ignore[arg-type]


def test_matches_query_false_when_no_match() -> None:
    result = _make_result(content="nothing relevant here")
    assert matches_query(result, "database migration") is False  # type: ignore[arg-type]


def test_coerce_float_converts_numeric() -> None:
    assert coerce_float(0.5) == 0.5
    assert coerce_float(1) == 1.0
    assert coerce_float("0.75") == 0.75
    assert coerce_float("invalid") == 0.0
