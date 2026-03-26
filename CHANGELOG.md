# Changelog

All notable changes to the TRW Memory package.

## [0.4.0] — 2026-03-26

### Added

- **Executable assertions** (PRD-CORE-086) — machine-verifiable grep/glob assertions attached to memory entries that execute against the codebase to detect stale knowledge. Inspired by [membase-4-claude](https://github.com/mike-remakerdigital/membase-4-claude).
  - `AssertionType` enum: `grep_present`, `grep_absent`, `glob_exists`, `glob_absent`
  - `Assertion` Pydantic model with security validators (path traversal rejection, 500-char pattern cap, absolute path rejection)
  - `AssertionResult` model for verification outcomes (passed/failed/unverifiable)
  - `assertions` field on `MemoryEntry` (default empty list, backward-compatible)
  - `verify_assertions()` pure function engine in `lifecycle/verification.py` — uses `pathlib.glob()` + `re.search()` only (no shell commands)
  - Default exclude list: `.git`, `__pycache__`, `node_modules`, `.egg-info`, `dist`
  - Binary file detection (null byte check in first 512 bytes), 1MB file size cap
  - Graceful degradation when `project_root` is unavailable or regex is invalid
  - Structured `structlog` timing instrumentation on verification runs
- **SQLite schema migration** — `assertions TEXT DEFAULT '[]'` column added via existing `_ensure_schema()` migration pattern. Idempotent, backward-compatible with pre-migration databases.
- **76 new tests** across 10 test files covering models, SQLite round-trip, all 4 verification types, graceful degradation, and security.

---

## [0.3.0] — 2026-03-19

### Added

- **Structured logging** — new `trw_memory/_logging.py` module with `configure_logging()`, `-v/-q/--log-level` CLI flags, `NullHandler` in `__init__.py` (library best practice), and `exc_info=True` added to exception handlers in `embeddings/local.py` and `migration/from_trw.py`. All 28 source files normalized to `structlog.get_logger(__name__)`. 10 sentence-style log messages converted to snake_case event names.
- **BSL 1.1 license** — `pyproject.toml` updated from CC BY-NC-SA to `BUSL-1.1` (Change Date 2030-03-21, Change License Apache 2.0) in preparation for open-source publication.
- **CI hardening** — ruff format checks now blocking in the memory CI workflow; pre-existing ruff lint issues resolved. mypy `--strict` fully clean across all 77 files.

### Changed

- **Version aligned to monorepo v24.4_TRW release** — package version bumped from 0.2.0 to 0.3.0 at the v24.4 framework release (`98608d2a`).

---

## [0.2.0] — 2026-03-19

### Added

- **Structured logging foundation** — `_logging.py` introduced as part of the monorepo-wide logging overhaul. Library stays silent by default (`NullHandler`); CLI configures logging at startup only.
- **CLI verbosity flags** — `-v/--verbose`, `-q/--quiet`, `--log-level` added to all CLI subcommands.

### Changed

- **Logger normalization** — all 28 source files migrated from bare `structlog.get_logger()` to `structlog.get_logger(__name__)` for module-level filtering.
- **Log event naming** — 10 sentence-style event strings converted to snake_case.

---

## [0.1.3] — 2026-03-17

### Fixed

- **WAL mode reliability** — enabled `PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;` on `SQLiteBackend` with a warning log if WAL activation fails. Allows concurrent reads without write serialization.
- **Decay loop `datetime.now()` per-iteration** — captured once before the loop instead of calling it on every iteration.
- **`graph.py` thread-safe commits** — `conn.commit()` calls now accept an optional `threading.Lock` parameter; wired in from all callers.

---

## [0.1.2] — 2026-03-15

### Changed

- **mypy `--strict` clean** — resolved all 13 pre-existing type errors: fixed `type: ignore` codes in `sqlite_backend.py`, `local.py`, and `client.py`; widened formatter params to `Sequence[Mapping]` for TypedDict covariance in `cli_formatters.py`; added `_backend_or_raise` property for None safety in `llamaindex.py`. Zero errors across 77 files.
- **Ruff expanded** — rule set expanded from 14 to 26 categories (matching monorepo standard). 244 violations fixed, 0 remaining.
- **`delete_by_namespace` cursor fix** — `cursor.rowcount` moved inside the write lock to prevent a race condition with concurrent deletes.

---

## [0.1.1] — 2026-03-10

### Changed

- **Standalone REST API removed** — the orphaned `trw-memory/api/` package (8 files, ~1,428 LOC) and its 5 matching test files were deleted. The API was superseded by the backend package and was unreachable without a running platform instance. The removal cleans up dead code and reduces confusion for consumers.
- **`_parsing.py` shared helpers** — `storage/_parsing.py` introduced with `parse_dt`, `parse_json_list`, `parse_json_dict_str`, `parse_json_dict_int` to replace duplicated parsing in `sqlite_backend.py` and `yaml_backend.py`. Fixes a subtle UTC normalization bug in `yaml_backend`.
- **Sprint 39/40 quality review** — CLI import merge check fixed (keyword recall replaced with `backend.get(eid)` for reliable ID-based dedup during `--merge` imports); `BackendOwnerMixin` extracted to eliminate identical `close/__enter__/__exit__` boilerplate across LangChain, CrewAI, and VSCode integration adapters; `corpus.py` docstrings corrected (said "YAML" but writes JSON).

---

## [0.1.0] — 2026-02-27

### Added

- **Initial release** — `trw-memory` introduced as a standalone persistent memory engine extracted from `trw-mcp`. Shipped as part of the monorepo alongside trw-mcp's knowledge topology and hybrid retrieval work (commit `b28c2490`).

#### Storage backends

- **SQLite primary backend** — `storage/sqlite_backend.py` with FTS5 full-text search, WAL journal mode, and `check_same_thread=False` for concurrent HTTP client access.
- **YAML secondary backend** — `storage/yaml_backend.py` for backward compatibility with pre-migration entries.

#### Retrieval

- **BM25 sparse retrieval** — `retrieval/bm25.py` via `rank-bm25` with hyphenated-tag expansion and zero-IDF fallback.
- **Dense vector retrieval** — `retrieval/dense.py` via `sqlite-vec` (384-dim all-MiniLM-L6-v2).
- **Hybrid search with RRF** — `retrieval/hybrid.py` combines BM25 and dense rankings via Reciprocal Rank Fusion (k=60). Gracefully degrades to BM25-only when vectors are unavailable.

#### Lifecycle

- **Tiered storage** — `lifecycle/tiers.py`: hot tier (in-memory LRU), warm tier (sqlite-vec + JSONL sidecar), cold tier (YAML archive). Four automatic transitions: Hot→Warm (TTL/overflow), Warm→Cold (idle+low-impact), Cold→Warm (on access), Cold→Purge (365d+low-impact). Purge audit trail at `.trw/memory/purge_audit.jsonl`.
- **Scoring engine** — `lifecycle/scoring.py`: Q-learning with EMA updates, Ebbinghaus forgetting curve, Bayesian MACLA calibration. Stanford Generative Agents importance formula: `w1*relevance + w2*recency + w3*importance`.
- **LLM consolidation** — `lifecycle/consolidation.py`: embedding-based cluster detection (single-linkage agglomerative, pairwise cosine threshold), LLM-powered summarization via `anthropic` SDK (claude-haiku), original entry archival to cold tier with atomic rollback.
- **Semantic dedup** — `lifecycle/dedup.py`: three-tier write-time dedup (skip ≥0.95, merge ≥0.85, store <0.85) via cosine similarity. `merge_entries()` with union tags/evidence, max impact, recurrence increment, and `merged_from` audit trail.

#### Knowledge graph

- **`graph.py`** — similarity, tag co-occurrence, and consolidation edges; BFS traversal; importance boost/decay with outcome history.

#### Remote sync

- **`sync/remote.py`** — publish/fetch with anonymization and fail-open design.
- **`sync/conflict.py`** — vector clock comparison, merge, and conflict resolution.
- **`sync/retry_queue.py`** — thread-safe JSONL queue with 500-entry depth cap.
- **`sync/subscriber.py`** — SSE real-time stream with automatic reconnection.

#### Security

- **`security/`** — AES-256-GCM field-level encryption, PII detection and redaction, memory poisoning detection, RBAC, and full audit trail.

#### Integrations

- **LangChain** — `TRWChatMessageHistory` and `TRWVectorStore` adapters.
- **LlamaIndex** — reader and writer components.
- **CrewAI** — `TRWCrewStorage` component.
- **OpenAI-compatible** — `LocalMemoryAdapter` with 4 function-call operations (store, recall, search, forget).

#### CLI

- **`trw-memory` CLI** — 8 subcommands (`store`, `recall`, `search`, `forget`, `consolidate`, `export`, `import`, `status`) with JSON, table, and YAML output formats. `cli_parser.py` extracted for line-count compliance.

#### MCP tools

- **6 MCP tools** (`tools/`) — `store`, `recall`, `search`, `consolidate`, `forget`, `status` for direct integration into Claude Code sessions.

#### Client SDK

- **`MemoryClient`** (`client.py`) — high-level async Python client with `store()`, `recall()`, `forget()`, and `search()` methods.
- **TypeScript SDK** — `@trw/memory-sdk` with `MemoryClient` class and full TypeScript types.

#### Benchmarks

- **`benchmarks/`** — 4 benchmark modules (latency, throughput, memory, quality) with configurable thresholds and `python -m benchmarks` entry point.
