"""Tests for trw_memory.adapters.openai_compat."""

from __future__ import annotations

from typing import Any

from trw_memory.adapters.openai_compat import (
    generate_openai_functions,
    get_memory_forget_schema,
    get_memory_recall_schema,
    get_memory_search_schema,
    get_memory_store_schema,
)


def _validate_function_schema(schema: dict[str, Any]) -> None:
    """Assert that a schema has the required top-level fields."""
    assert "name" in schema
    assert isinstance(schema["name"], str)
    assert len(schema["name"]) > 0

    assert "description" in schema
    assert isinstance(schema["description"], str)
    assert len(schema["description"]) > 0

    assert "parameters" in schema
    params = schema["parameters"]
    assert params["type"] == "object"
    assert "properties" in params
    assert isinstance(params["properties"], dict)

    # Required must be a list
    assert "required" in params
    assert isinstance(params["required"], list)


def _validate_property(
    prop: dict[str, Any],
    expected_type: str,
    must_have_description: bool = True,
) -> None:
    """Assert that a property has the expected type and description."""
    assert prop["type"] == expected_type
    if must_have_description:
        assert "description" in prop
        assert len(prop["description"]) > 0


# ---------------------------------------------------------------------------
# Individual schema tests
# ---------------------------------------------------------------------------


class TestStoreSchema:
    def test_structure(self) -> None:
        schema = get_memory_store_schema()
        _validate_function_schema(schema)
        assert schema["name"] == "memory_store"

    def test_content_is_required(self) -> None:
        schema = get_memory_store_schema()
        assert "content" in schema["parameters"]["required"]

    def test_content_property(self) -> None:
        props = get_memory_store_schema()["parameters"]["properties"]
        _validate_property(props["content"], "string")

    def test_tags_property(self) -> None:
        props = get_memory_store_schema()["parameters"]["properties"]
        assert props["tags"]["type"] == "array"
        assert props["tags"]["items"]["type"] == "string"

    def test_importance_bounds(self) -> None:
        props = get_memory_store_schema()["parameters"]["properties"]
        imp = props["importance"]
        assert imp["type"] == "number"
        assert imp["minimum"] == 0.0
        assert imp["maximum"] == 1.0
        assert imp["default"] == 0.5


class TestRecallSchema:
    def test_structure(self) -> None:
        schema = get_memory_recall_schema()
        _validate_function_schema(schema)
        assert schema["name"] == "memory_recall"

    def test_query_is_required(self) -> None:
        schema = get_memory_recall_schema()
        assert "query" in schema["parameters"]["required"]

    def test_limit_property(self) -> None:
        props = get_memory_recall_schema()["parameters"]["properties"]
        limit = props["limit"]
        assert limit["type"] == "integer"
        assert limit["minimum"] == 1
        assert limit["default"] == 10

    def test_graph_depth_and_org_memory_properties(self) -> None:
        props = get_memory_recall_schema()["parameters"]["properties"]
        assert props["graph_depth"]["type"] == "integer"
        assert props["graph_depth"]["maximum"] == 3
        assert props["include_org_memories"]["type"] == "boolean"
        assert props["include_org_memories"]["default"] is True

    def test_source_aware_properties_are_exposed(self) -> None:
        props = get_memory_recall_schema()["parameters"]["properties"]
        assert props["include_source_kinds"]["type"] == "array"
        assert props["exclude_source_kinds"]["type"] == "array"
        assert props["source_weights"]["type"] == "object"
        assert props["exclude_expired"]["type"] == "boolean"
        assert props["exclude_expired"]["default"] is True
        assert props["include_distilled"]["type"] == "boolean"
        assert props["distilled_weight"]["type"] == "number"


class TestSearchSchema:
    def test_structure(self) -> None:
        schema = get_memory_search_schema()
        _validate_function_schema(schema)
        assert schema["name"] == "memory_search"

    def test_no_required_fields(self) -> None:
        schema = get_memory_search_schema()
        assert schema["parameters"]["required"] == []

    def test_since_has_datetime_format(self) -> None:
        props = get_memory_search_schema()["parameters"]["properties"]
        assert props["since"]["format"] == "date-time"

    def test_min_importance_bounds(self) -> None:
        props = get_memory_search_schema()["parameters"]["properties"]
        mi = props["min_importance"]
        assert mi["minimum"] == 0.0
        assert mi["maximum"] == 1.0


class TestForgetSchema:
    def test_structure(self) -> None:
        schema = get_memory_forget_schema()
        _validate_function_schema(schema)
        assert schema["name"] == "memory_forget"

    def test_memory_id_is_required(self) -> None:
        schema = get_memory_forget_schema()
        assert "memory_id" in schema["parameters"]["required"]


# ---------------------------------------------------------------------------
# generate_openai_functions
# ---------------------------------------------------------------------------


class TestGenerateOpenAIFunctions:
    def test_returns_list_of_four(self) -> None:
        fns = generate_openai_functions()
        assert isinstance(fns, list)
        assert len(fns) == 4

    def test_all_have_valid_structure(self) -> None:
        for fn in generate_openai_functions():
            _validate_function_schema(fn)

    def test_function_names(self) -> None:
        names = {fn["name"] for fn in generate_openai_functions()}
        assert names == {"memory_store", "memory_recall", "memory_search", "memory_forget"}

    def test_all_have_descriptions(self) -> None:
        for fn in generate_openai_functions():
            assert len(fn["description"]) > 10

    def test_no_duplicate_names(self) -> None:
        fns = generate_openai_functions()
        names = [fn["name"] for fn in fns]
        assert len(names) == len(set(names))
