# Changelog

All notable changes to the TRW Memory package.

## [Unreleased]

### Added

- **2026-04-13 — Recovery is safer after serious corruption incidents** — strict-salvage refusal, cold-tier rebuilds, and restore-from-cold flows make it less likely that a bad SQLite state silently turns into partial data loss (PRD-CORE-138, PRD-CORE-140).
- **2026-04-13 — Operational safety rails for long-lived stores** — timestamped corrupt-backup rotation, scheduled integrity checks, multi-writer advisory tracking, and snapshot rotation make recovery paths more explicit and easier to reason about (PRD-CORE-139, PRD-INFRA-063, PRD-INFRA-064, PRD-INFRA-065).

### Changed

- **2026-04-13 — Durability observability improved** — integrity and snapshot signals are easier to publish and track without reviving the deprecated session-start memory-health surface (PRD-INFRA-066, PRD-INFRA-067, PRD-INFRA-068).

### Fixed

- **2026-04-12 to 2026-04-13 — Recovery and sync edge cases were hardened** — follow-up audit work closed gaps in sync tracking, decay proofing, and team-lifecycle semantics so cross-session behavior is more dependable under stress.

## [0.6.7] — 2026-04-13

### Added
- **Feedback lifecycle** — `recall_count`, `helpful_count`, `unhelpful_count` fields on `MemoryEntry` (PRD-CORE-132)
- **Recall tracking** — `record_recall_access` increments `recall_count` on every recall
- **Dynamic decay scoring** — `feedback_decay_score()` implements `impact × 0.95^(recall_count / max(1, helpful_count))`; wired into `entry_utility()` for recall ranking
- **Schema migration** — `ensure_schema` adds feedback columns via ALTER TABLE with defaults

### Changed
- Removed 150 lines of duplicated `rank_by_utility`/`utility_based_prune_candidates` from `_recall.py` (DRY cleanup)

### Fixed
- `_row_mapper.py` feedback fields properly wired into `MemoryEntry` construction

## [0.6.6] — 2026-04-07

### Added

- **Sync delta tracking** (PHASE-BACKEND-INTELLIGENCE, PRD-INFRA-051)
  - 3 new fields on MemoryEntry: `sync_hash`, `sync_seq`, `last_synced_at`
  - SQLite schema migration with `idx_memories_sync_seq` index
  - ENTRY_COLUMNS expanded from 45 to 48
  - `sync/delta.py` — DeltaTracker with compute_sync_hash, get_dirty_entries, mark_synced, mark_dirty
  - Auto dirty-marking in SQLiteBackend.store() and update()
  - YAML backend preserves sync fields via to_dict()

### Validation

- Full `trw-memory` package suite passed: `1778` passed.

## [0.6.5] — 2026-04-02

### Fixed

- **Typed-learning expiry column normalized** — the SQLite `memories` schema now uses `expires_at` consistently, matching the migration contract and storage tests.
- **Legacy DB migration preserved** — `ensure_schema()` now renames pre-existing `expires` columns to `expires_at` before adding missing typed-learning fields, preserving stored expiry values during upgrade.
- **SQLite API compatibility retained** — the backend maps the physical `expires_at` column back to the public `MemoryEntry.expires` field on reads and updates, so callers do not need to change.

### Validation

- Full `trw-memory` package suite passed: `1633` passed.
- Focused storage regression coverage passed for schema, row mapping, provenance, assertion columns, and PRD-FIX-060 serialization.

---

## [0.6.3] — 2026-04-01

### Improved

- **`compute_anchor_validity()` type safety** — Now accepts both `list[Anchor]` (Pydantic models) and `list[dict[str, object]]` (raw dicts) via `Union` type. Uses attribute access for Anchor instances and `.get()` for dicts, eliminating the need for callers to convert between formats.

---

## [0.6.2] — 2026-03-31

### Added

- **`MemoryConfig.fsync_on_append`** — New config flag to enable `os.fsync()` on audit log appends for durability.
- **`MemoryConfig.__repr__`** — Shows storage_backend, path, encryption, rbac at a glance.
- **`MemoryEntry.__repr__`** — Shows id, content preview, tags, importance.
- **13 new tests** in `test_qual_054_dx_polish.py` covering repr, fsync, and config validation.

---

## [0.6.1] — 2026-03-31

### Fixed

- **SQLite corruption auto-recovery** — `SQLiteBackend` now runs `PRAGMA quick_check` on startup and automatically recovers from "database disk image is malformed" errors. Corrupt databases are renamed to `.corrupt.bak` (with rotation), salvageable rows are copied to a fresh database, and stale WAL/SHM files are cleaned up.
- **Runtime corruption resilience** — `store_learning()` and `recall_learnings()` catch corruption errors mid-session, reset the backend singleton, recover the database, and retry the operation once before propagating the error.

### Added

- **`SQLiteBackend.recover_db()`** — Public static method to recover a corrupt database, salvage rows, and return a fresh connection.
- **`SQLiteBackend.check_integrity()`** — Public static utility to check database health without opening a full backend.
- **10 new tests** in `test_db_recovery.py` covering integrity checks, row salvage, WAL cleanup, backup rotation, and auto-recovery on init.

---

## [0.6.0] — 2026-03-29

### Changed

- **CLI error boundary decorator** — All CLI subcommands now use `_cli_error_boundary` decorator for uniform error handling instead of per-command try/except blocks. Error messages show `Error: <message>` format consistently.
- **Namespace module reorganized** — Removed `trw_memory.namespace` backward-compatibility shim. Canonical imports are now `trw_memory.namespaces.path_mapping` and `trw_memory.namespaces.validation`. Top-level exports (`from trw_memory import validate_namespace`) are unchanged.
- **Storage interface extracted** — New `storage/interface.py` with abstract base protocol for storage backends, improving type safety and testability.
- **YAML backend simplified** — Refactored to use the extracted storage interface, reducing coupling.
- **Security keys refactored** — `security/keys.py` module cleaned up with improved key management.
- **Typed test fixtures** — `conftest.py` fixtures now return concrete types (`SQLiteBackend`, `MemoryClient`, `MemoryConfig`, `MemoryEntry`) instead of `Any`. Imports moved to module level.
- **Memory entry ID length** — IDs now use 16 hex chars (`M-` + 16) instead of 8, reducing collision probability.
- **Source field validation** — Non-standard `source` values (e.g., `"synthetic"`) are coerced to `"agent"` by the model validator.
- **Deprecated lint rules removed** — Removed `ANN101` and `ANN102` from ruff ignore list (deprecated in ruff).

### Added

- **6 new PRD test files** — `test_prd_fix_059_fra.py`, `test_prd_fix_059_frb.py`, `test_prd_fix_060.py`, `test_prd_qual_053.py`, `test_prd_qual_054_a.py`, `test_prd_qual_054_b.py`.
- **CLI export refactored** — `entry_to_export_dict()` uses `MemoryEntry.to_dict()` for consistent serialization.

---

## [0.5.0] — 2026-03-28

### Fixed

- **YAML backend field parity** (PRD-FIX-058) — 7 `MemoryEntry` fields were silently dropped during YAML serialization/deserialization: `vector_clock`, `remote_id`, `published_to_platform`, `pending_delete`, `cross_validated`, `outcome_history`, `assertions`. All 7 now round-trip correctly through both `_entry_to_dict()` and `_dict_to_entry()`. Backward-compatible with pre-fix YAML files (missing fields load as defaults).
- **`serialize_update_value()` Assertion data loss** — `backend.update(id, assertions=[...])` silently dropped all `Assertion` objects by converting them to repr strings via the generic `LIST_FIELDS` path. Now handles `assertions` key with `model_dump()`. Discovered by independent Sonnet audit.
- **`MemoryClient.recall()` scoring** (PRD-QUAL-046 FR06) — Results were ranked by raw `importance` instead of query relevance. Now uses TF-based relevance blended with importance (0.7 TF + 0.3 importance).
- **`MemoryClient.recall()` over-fetch** — Backend search now fetches `limit * 3` to prevent under-delivery when `min_score` post-filtering removes entries.
- **README documentation integrity** (PRD-FIX-057) — Removed non-existent REST API module, `[api]` extra, `trw-memory-api` entry point, and FTS5 claims. Updated architecture tree (verified all 62 paths exist). Updated metrics to 81 files, 1,418 tests, ~90% coverage.
- **Broken test fixtures** (PRD-QUAL-046 FR01) — `conftest.py` fixtures called non-existent `db.initialize()`; fixed to yield/close pattern.
- **tests/CLAUDE.md accuracy** — Removed "No conftest.py" claim, documented actual shared fixtures, corrected test counts.

### Added

- **`py.typed` marker** (PEP 561) — Enables downstream type checking for consumers.
- **`__repr__`** on `MemoryClient`, `SQLiteBackend`, `YAMLBackend` — Shows namespace/mode, db_path/vec, entries_dir for debuggability.
- **22 YAML field parity tests** — Round-trip tests for all 7 fields, backward compatibility, edge cases, and direct assertions update regression test.

### Changed

- **`DedupResult.action`** type from `str` to `Literal["skip", "merge", "store"]` for type safety.
- **14 ruff lint violations resolved** — E402 import ordering in security modules, G201 `logger.error` → `logger.exception` in graph.py, PIE790 unnecessary pass, unused noqa directives.

---

## [0.4.0] — 2026-03-26

### Added

- **Executable assertions** (PRD-CORE-086) — machine-verifiable grep/glob assertions attached to memory entries that execute against the codebase to detect stale knowledge.
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
