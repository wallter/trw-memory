"""OpenAI function calling schema generator for trw-memory.

Generates JSON Schema (draft-07) compatible function definitions that can be
passed directly to the OpenAI Chat Completions API ``functions`` parameter
or the ``tools`` parameter (with ``type: "function"`` wrapper).
"""

from __future__ import annotations

from typing import Any


def get_memory_store_schema() -> dict[str, Any]:
    """Return OpenAI function calling schema for memory_store.

    Returns:
        Function schema dict with name, description, and JSON Schema parameters.
    """
    return {
        "name": "memory_store",
        "description": "Store a new memory entry in the agent's persistent memory",
        "parameters": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "Core knowledge statement to remember",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Categorization tags for retrieval filtering",
                },
                "importance": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.5,
                    "description": "Importance score from 0.0 (trivial) to 1.0 (critical)",
                },
                "detail": {
                    "type": "string",
                    "default": "",
                    "description": "Extended explanation or context",
                },
                "namespace": {
                    "type": "string",
                    "default": "default",
                    "description": "Isolation scope (e.g. project:my-app)",
                },
            },
            "required": ["content"],
        },
    }


def get_memory_recall_schema() -> dict[str, Any]:
    """Return OpenAI function calling schema for memory_recall.

    Returns:
        Function schema dict with name, description, and JSON Schema parameters.
    """
    return {
        "name": "memory_recall",
        "description": "Search memories by keyword query, returning the most relevant entries",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Free-text search query",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "default": 10,
                    "description": "Maximum number of results to return",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Filter results to entries containing all listed tags",
                },
                "min_score": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.0,
                    "description": "Minimum relevance score threshold",
                },
                "namespace": {
                    "type": "string",
                    "default": "default",
                    "description": "Namespace to search within",
                },
                "include_org_memories": {
                    "type": "boolean",
                    "default": True,
                    "description": "Append cross-validated memories from sibling project namespaces",
                },
                "graph_depth": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 3,
                    "default": 0,
                    "description": "When > 0, include graph-related memories up to this traversal depth",
                },
            },
            "required": ["query"],
        },
    }


def get_memory_search_schema() -> dict[str, Any]:
    """Return OpenAI function calling schema for memory_search.

    Returns:
        Function schema dict with name, description, and JSON Schema parameters.
    """
    return {
        "name": "memory_search",
        "description": "Search memories using structured filters (tags, importance, date)",
        "parameters": {
            "type": "object",
            "properties": {
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Filter results to entries containing all listed tags",
                },
                "min_importance": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.0,
                    "description": "Minimum importance threshold",
                },
                "since": {
                    "type": "string",
                    "format": "date-time",
                    "description": "ISO 8601 datetime — only return entries created after this time",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "default": 50,
                    "description": "Maximum number of results to return",
                },
                "namespace": {
                    "type": "string",
                    "default": "default",
                    "description": "Namespace to search within",
                },
            },
            "required": [],
        },
    }


def get_memory_forget_schema() -> dict[str, Any]:
    """Return OpenAI function calling schema for memory_forget.

    Returns:
        Function schema dict with name, description, and JSON Schema parameters.
    """
    return {
        "name": "memory_forget",
        "description": "Delete a specific memory entry by its ID",
        "parameters": {
            "type": "object",
            "properties": {
                "memory_id": {
                    "type": "string",
                    "description": "Unique identifier of the memory entry to delete",
                },
                "namespace": {
                    "type": "string",
                    "default": "default",
                    "description": "Namespace the entry belongs to",
                },
            },
            "required": ["memory_id"],
        },
    }


def generate_openai_functions() -> list[dict[str, Any]]:
    """Return all memory functions in OpenAI function calling format.

    Returns:
        List of 4 function schema dicts: store, recall, search, forget.
    """
    return [
        get_memory_store_schema(),
        get_memory_recall_schema(),
        get_memory_search_schema(),
        get_memory_forget_schema(),
    ]
