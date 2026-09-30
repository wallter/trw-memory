"""The span keys, types and closed enums trw-memory emits (PRD-CORE-343 NFR05, OTEL-CONVENTIONS NS-3).

The single source for :mod:`trw_memory._otel`; the conformance test checks every emitted
key, type and enum value against :data:`REGISTRY`. Design source: SPEC Part A registry
(``com.trwframework.memory.*``, development stability) and the merged upstream
``gen_ai.memory.client`` span. No other module names these strings.
"""

from __future__ import annotations

SEMCONV_VERSION = "0.12.0-draft"
SCOPE_SEMCONV_VERSION = "com.trwframework.semconv.version"

OPERATION_NAME = "gen_ai.operation.name"
STORE_ID = "gen_ai.memory.store.id"
RECORD_ID = "gen_ai.memory.record.id"
RECORD_COUNT = "gen_ai.memory.record.count"
TOP_K = "gen_ai.retrieval.top_k"
ERROR_TYPE = "error.type"

_P = "com.trwframework.memory."
RECORD_IDS = _P + "record.ids"
RECORD_TRUNCATED = _P + "record.truncated"
FAILED_COUNT = _P + "record.failed_count"
WRITE_OUTCOME = _P + "write.outcome"
DELETE_TARGET = _P + "delete.target"
RECALL_METHOD = _P + "recall.method"
RECALL_SCORE_KIND = _P + "recall.score_kind"
RECALL_TOP_SCORE = _P + "recall.top_score"
RECALL_THRESHOLD = _P + "recall.threshold"
RECALL_RERANKED = _P + "recall.reranked"
RECALL_FILTERED = _P + "recall.filtered_count"

OPERATIONS = frozenset({"create_memory", "upsert_memory", "update_memory", "delete_memory", "search_memory"})
#: MEM-11.2: the documented ``error.type`` list; anything else is ``_OTHER``.
ERROR_TYPES = frozenset(
    {"invalid", "not_found", "conflict", "rate_limited", "blocked", "unauthorized", "storage_error", "_OTHER"}
)

#: key -> (python type, closed enum or None). Arrays are ``tuple`` of ``str``.
REGISTRY: dict[str, tuple[type, frozenset[str] | None]] = {
    OPERATION_NAME: (str, OPERATIONS),
    STORE_ID: (str, None),
    RECORD_ID: (str, None),
    RECORD_COUNT: (int, None),
    TOP_K: (int, None),
    ERROR_TYPE: (str, ERROR_TYPES),
    RECORD_IDS: (tuple, None),
    RECORD_TRUNCATED: (bool, None),
    FAILED_COUNT: (int, None),
    WRITE_OUTCOME: (str, frozenset({"created", "updated", "merged", "noop", "mixed"})),
    DELETE_TARGET: (str, frozenset({"record", "records", "selector", "store_all"})),
    RECALL_METHOD: (str, frozenset({"vector", "keyword", "graph", "hybrid", "other"})),
    RECALL_SCORE_KIND: (str, frozenset({"similarity", "distance", "rrf", "bm25", "probability", "other"})),
    RECALL_TOP_SCORE: (float, None),
    RECALL_THRESHOLD: (float, None),
    RECALL_RERANKED: (bool, None),
    RECALL_FILTERED: (int, None),
}
