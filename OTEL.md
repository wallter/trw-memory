# OpenTelemetry memory spans: conformance note

trw-memory emits one `gen_ai.memory.client` span per public memory operation (PRD-CORE-343). The emitter is `trw_memory/_otel.py`; every key, type and closed enum it may emit is listed in `trw_memory/_otel_keys.py`.

## What is claimed

- The upstream `gen_ai.memory.client` span (Development stability), plus SPEC Part A steps 1 and 2 under `com.trwframework.memory.*`, and profile P (Part C).
- **Identifier mode: omit.** An identifier that fails `[A-Za-z0-9_.:-]{1,128}`, starts with `idh1:`, is longer than 4096 bytes, or is not exactly a `str` or `int` (an `int` of at most 63 bits) is omitted. There are no pseudonyms and no digests of any kind. A `#hype` child id is reported as its parent.
- **Opt-in.** The package depends on `opentelemetry-api` only. Without an SDK provider, with `OTEL_SDK_DISABLED=true`, or with an API older than 1.38, every span is a no-op and no attribute is built. The daemon installs a provider only when its first spawner set `TRW_OTEL_ENABLED=true` (see below).

## Operation mapping

| Entry point (both families) | `gen_ai.operation.name` | Notes |
|---|---|---|
| store without an id | `create_memory` | outcome `created` or `updated`, from the engine status |
| store with a caller id | `upsert_memory` | the engine decides create or update; the outcome says which |
| `bulk_store` | `upsert_memory` | `record.count` = attempted, `record.failed_count` = rejected, outcome `created`/`updated`/`mixed`, omitted when nothing landed |
| `store_many` | `create_memory` | one span; the nested `bulk_store` emits none |
| `memory_update` | `update_memory` | outcome `updated`, or `noop` for `no_changes` |
| forget by id / by query or actor | `delete_memory` | `delete.target` = `record` / `selector`; the query and actor are never emitted |
| recall, search, `memory_similar` | `search_memory` | `record.count` (0 is not an error), `gen_ai.retrieval.top_k` only when `graph_depth == 0` |

- **`gen_ai.memory.store.id`** is the namespace (MEM-08.1). It goes through `export_store_id`, which is the identity locally. A hosted service must replace it with an opaque-id map or a keyed tenant-scoped HMAC; an unkeyed hash is never allowed.
- **Recall diagnostics** come from values the recall already computed. `recall.method` is `hybrid`, `keyword` (no embedder), or `other` (wildcard). `recall.score_kind` is `rrf` for fused scores, and `other` when a rerank or the recency prior turned scores into rank positions, or on the keyword fallback. `recall.top_score` and `recall.threshold` are set only for `rrf` (MEM-13.3). `recall.filtered_count` is the number of rows the single `min_score` filter removed. `recall.reranked` is true when the cross-encoder reordered or cut the list.

## Errors (MEM-11.2)

`error.type` is one of `invalid`, `not_found`, `conflict`, `rate_limited`, `blocked`, `unauthorized`, `storage_error`, `_OTHER`. A refusal returned as a status (never raised, MEM-10.2) sets `error.type` and ERROR with no description and no outcome. A quarantined write is not an error and carries no outcome. A raised exception is re-raised unchanged; the span gets only `error.type` and ERROR. Cancellation is not an error. No exception text, event or status description is ever recorded.

## Batch policy, retries, arrays

- **Arrays** (`record.ids`) are deduplicated in first-seen order and capped at 32, with `record.truncated=true`; collection stops after 33 distinct ids, so the work is bounded whatever the result size.
- **Consolidation successors** (MEM-02.5, 32 successor spans per call) are **not emitted** in this release; FR06 is deferred.
- **Retries** (MEM-07.3): each attempt the engine makes is part of one operation span. A caller that retries makes a new span; a retried or unknown attempt is not deduplicated.
- **Not reported:** reads, lists, status, audit, review, reembed, maintain and decay effects, sync, namespace admin, closures and supersession (steps 3a, 3b and 4).
- **Counts from spans are lower bounds** (MEM-15.4): sampling, a missing provider or a dropped export all lose spans. The store and its audit log stay authoritative.

## Trace context across the daemon

- The daemon's direct JSON-RPC path (`daemon/_direct.py`) injects the caller's `traceparent` into `params._meta`, and only the `traceparent` (never `tracestate` or baggage). It does so only when OpenTelemetry is already imported and the current span is valid, so the idle hook path never imports it. FastMCP's server extracts `_meta.traceparent`; `tests/test_otel_propagation.py` proves the memory span is a child of the `tools/call` SERVER span, which is a child of the caller's span.
- The daemon (`daemon/_serve.py`) calls `configure_tracing("trw-memory", <user memory dir>/traces, enabled=TRW_OTEL_ENABLED == "true")` once before serving, and drops an inherited `TRACEPARENT`.
- **Limitation:** the launcher starts the daemon with its spawner's whole environment. The **first** spawner's `TRW_OTEL_ENABLED` and `OTEL_*` settings decide export for the daemon's whole lifetime, across checkouts and sessions. A trace can span two files, trw-mcp's and the daemon's, joined by trace id.
