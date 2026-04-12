from __future__ import annotations

import pytest

from trw_memory.exceptions import SchemaValidationError
from trw_memory.security.poisoning import validate_store_inputs


def test_validate_store_inputs_accepts_valid_payload() -> None:
    validate_store_inputs(
        content="valid content",
        detail="detail",
        tags=["tag"],
        metadata={"source": "test"},
        importance=0.5,
    )


def test_validate_store_inputs_rejects_invalid_fields() -> None:
    with pytest.raises(SchemaValidationError) as excinfo:
        validate_store_inputs(
            content="",
            detail=123,
            tags=["ok", 1],
            metadata={"source": 1},
            importance=1.5,
        )

    assert excinfo.value.failed_fields == ["content", "detail", "tags", "metadata", "importance"]
