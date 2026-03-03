"""Protocol adapters for trw-memory -- OpenAI function calling, etc."""

from trw_memory.adapters.openai_compat import (
    generate_openai_functions,
    get_memory_forget_schema,
    get_memory_recall_schema,
    get_memory_search_schema,
    get_memory_store_schema,
)

__all__ = [
    "generate_openai_functions",
    "get_memory_forget_schema",
    "get_memory_recall_schema",
    "get_memory_search_schema",
    "get_memory_store_schema",
]
