# Changelog

All notable changes to the TRW Memory package.

## [Unreleased]

### Changed

- **PRD-DIST-2058 — `MEMORY_RECALL_PRESERVE_HYBRID_ORDER` is now the default.**
  `MemoryConfig.recall_preserve_hybrid_order` now defaults to `True`, so
  `MemoryClient.recall()` preserves the hybrid BM25+dense+RRF ordering whenever
  the hybrid retriever already produced enough local candidates. This avoids the
  c805 score-scale mismatch where the legacy tier merge compared hybrid RRF
  scores with tier-only `entry_utility` scores and pushed high-rank hybrid hits
  out of the top-K. The opt-out remains available:
  set `MEMORY_RECALL_PRESERVE_HYBRID_ORDER=false` to restore the legacy rescore.
  trw-distill c811-c815 validated the flip across 4 curated-query oracles,
  3 languages, and K=10/20/30/50 sweeps; the strongest observed curated-query
  lift was Recall@5 `0.4167 → 1.0000` on the trw-framework oracle.

## [0.8.3] — 2026-05-17

### Fixed

- **Prefer `pysqlite3-binary` over stdlib `sqlite3` to mitigate the SQLite
  WAL-reset bug.** `storage/_dbapi.py` performs a one-time swap of
  `sys.modules["sqlite3"]` to `pysqlite3` at package import so every
  subsequent `import sqlite3` resolves to a modern SQLite build, independent
  of the Python interpreter's bundled version. The dev repo was running
  stdlib SQLite 3.45.1, which carries the WAL-reset bug (fix landed in
  3.51.3 / backports 3.44.6 and 3.50.7). pysqlite3-binary's current wheel
  ships 3.51.1 — short of the fix, but still a multi-release upgrade that
  pulls in unrelated correctness and performance improvements. The shim
  reports the active backend, version, and a conservative
  `is_wal_reset_safe()` verdict so observability can confirm the upgrade
  landed. Absence of the wheel is a silent no-op; behaviour is unchanged in
  that fallback. Dep is required-but-platform-conditional (skipped on
  Windows where pysqlite3 wheels are not currently published).

- **`storage/_snapshot.py` and `storage/_integrity_scheduler.py` now set
  `busy_timeout=30000` on every ad-hoc sqlite connection.** Both paths
  previously relied solely on the `timeout=` connect parameter, which is not
  the same as the SQLite-level busy_timeout that the primary backend honours.
  Under multi-process WAL contention this caused the snapshot path to fail
  fast on `SQLITE_BUSY` and the integrity scheduler to report spurious
  "regression" events whenever a long-running checkpoint happened to overlap
  the probe. Snapshot connect also now passes `check_same_thread=False`
  because `VACUUM INTO` may run from a background thread.

### Changed

- **Module-decomposition campaign (PRD-DIST-245 / PRD-DIST-246).** `storage/sqlite_backend.py`
  (1133 → 343 LOC), `graph.py` (833 → 357 LOC), `security/runtime.py` (715 → 395 LOC), and
  `client.py` (1823 → 428 LOC) were each decomposed below the 350-LOC review gate by extracting
  cohesive `_*.py` helper modules (e.g. `storage/_schema.py`, `_row_mapper.py`, `_recovery.py`,
  `_cold_rebuild.py`, `_snapshot.py`, `_utf8_validator.py`, `_stale_handle*.py`, `_writer_registry.py`,
  `_integrity_scheduler.py`; `_graph_edges.py`, `_graph_clusters.py`, `_graph_conflicts.py`,
  `_graph_decay.py`, `_graph_cross_project.py`, `_graph_primitives.py`; `security/_runtime_anomaly.py`,
  `_runtime_canary.py`, `_runtime_pii.py`, `_runtime_quarantine.py`; `_client_recall.py`,
  `_client_store.py`, `_client_forget_search.py`, `_client_lifecycle.py`, `_client_bulk_store.py`,
  `_client_models.py`, `_client_org_shared.py`, `_client_recall_helpers.py`, `_client_tools_binding.py`,
  `_client_distilled_tiering.py`). Behaviour-preserving — public re-exports unchanged, full suite green.
  Numerous test files were likewise split for `pytest-xdist` parallelism.

### Added

- **`bandit/` package — adaptive-selection primitives.** `BanditSelector` / `BanditDecision`
  (Thompson Sampling with a sliding observation window, cold-start round-robin, and a floor-rate
  exploration guarantee), `ContextualBanditSelector` (LinUCB with Sherman-Morrison incremental
  updates — no per-step matrix inversion), and `PageHinkleyDetector` (change-point detection for
  non-stationary reward streams). Dependency-light building blocks for adaptive learning-selection /
  nudge-selection in the surrounding framework. Exposed via `trw_memory.bandit`.

### Fixed

- **Canary state keyed per `(quarantine, backend)` pair** (commit `4c52caa47`) — canary
  bookkeeping previously collided across quarantine/backend combinations; the key is now the pair.
- **PRD-DIST-255 closed as not-reproducible** (commit `43acb4f19`) — the reported issue could not
  be reproduced; 7 regression tests added to lock the current correct behaviour.

## [0.8.2] — 2026-05-03

### Added

- **`SQLiteBackend.transaction()` context manager** (commit `55aa0bc49`, PRD-FIX-088 FR02
  prerequisite). Re-entrant transaction bracket so callers can collapse N writes into a single
  `BEGIN IMMEDIATE` / `COMMIT` instead of paying a commit per row. The outermost call issues the
  `BEGIN IMMEDIATE` / `COMMIT`; nested calls only increment a `_skip_commit_depth` counter (and
  `update()` honours that counter); exceptions trigger `ROLLBACK` with logged best-effort cleanup.
  The `StorageBackend` ABC default is a no-op pass-through so non-supporting backends (YAML) work
  transparently — callers don't need `hasattr` guards. **Why:** trw-mcp's `_batch_sync_to_sqlite`
  was committing per-row across 2,823 entries during Q-learning outcome correlation (~91 s wall
  time on the dev repo); without a backend-level transaction primitive the trw-mcp side couldn't
  chunk those into a few transactions. Additive API → version bump `0.8.1 → 0.8.2`; existing
  recovery + bulk-update suites (30 tests) remain green.

## [0.8.1] — 2026-04-27

### Fixed

- **`trw_memory.security.observe_clock` raised `ModuleNotFoundError: No module named 'yaml'`**
  on a clean `pip install trw-memory` install. The module imported
  `yaml` (PyYAML) at top-level, but trw-memory's `dependencies` only
  declares `ruamel.yaml`. Converted to use the project's standard
  `ruamel.yaml.YAML(typ="safe")` API. Caught by the public-repo
  Release-to-PyPI smoke-test in CI run `25025008017` against the
  v0.8.0 tag — PyPI publish was correctly gated and 0.8.0 never
  shipped. 0.8.1 supersedes that aborted tag.

## [0.8.0] — 2026-04-26

### Quality

- **Lint, type-check, and format clean across `src/` and `tests/`**
  (release-prep pass). 10 mypy-strict errors fixed across 4 modules:
  optional-dep `Any` fallbacks for PyNaCl `SigningKey`/`VerifyKey`/
  `BadSignatureError` (`security/provenance.py`, `security/keys.py`)
  and torchcodec (`embeddings/local.py`); `Literal["observe","strict"]`
  annotation in `security/recall_filter.py:164`. 82 → 0 ruff errors
  via auto-fixes plus targeted manual edits (`TRY004` on
  `isinstance` failures in `namespaces/manager.py`, SIM102 nested-if
  collapse + RUF021 parenthesize-precedence in `security/keys.py`,
  PERF401 in `integrations/llamaindex.py`, F841 unused-var in
  `lifecycle/tiers/_warm.py`, E402 noqa with rationale in
  `integrations/crewai.py`, S* noqa annotations for the SQLite
  recovery subprocess and placeholder-bound IN-clause in
  `storage/sqlite_backend.py`, S101/S105/S112 noqa for type-
  narrowed asserts, non-credential flag values, and per-iteration
  fallbacks). Project-wide `pyproject.toml [tool.ruff.lint] ignore`
  extended for codebase-intentional patterns (ANN401, PERF203,
  SIM105, C901, RUF002, TRY301) with rationale. Expanded `fixable`
  list so future ruff `--fix` runs cover more rules. No behavioral
  changes; `mypy --strict` clean, `ruff check` clean,
  `ruff format --check` clean.

### Added

- **SEC001 security startup + telemetry + audit/review tools**
  (commit `6d20d2445`). New `security/startup.py` binds the security
  context at server boot (mirrors the trw-mcp wiring), and
  `security/telemetry_emit.py` is the shared fan-out for
  security-relevant events. Two new MCP tools land:
  `tools/audit.py` and `tools/review.py`. Live-paths and server-tool
  smoke tests (`tests/test_sec001_live_paths.py`,
  `tests/test_sec001_server_tools.py`) plus a unit harness for the
  telemetry fan-out (`tests/unit/security/test_security_telemetry.py`).

### Fixed

- **FIX053: torchcodec import guard for text-only embeddings**
  (commit `cbe4a5bcd`,
  `src/trw_memory/embeddings/local.py`). SentenceTransformers 5
  imports optional audio/video helpers at package import time. A
  broken torchcodec wheel can raise non-`ImportError` exceptions
  (e.g. `RuntimeError`/`OSError`) during that optional import and
  prevent `SentenceTransformer` itself from importing — so text
  embeddings broke even though they don't need torchcodec. Added a
  scoped `_hide_broken_torchcodec_for_sentence_transformers` context
  manager that masks a broken torchcodec in `sys.modules` only for
  the duration of the ST import; other application features that
  legitimately use torchcodec are unaffected. New
  `tests/test_embeddings.py` cases pin both the masking and the
  no-mask happy path.

## [0.7.0 → 0.8.0 interim] — 2026-04-19/20 (folded into the 0.8.0 release)

### Added

- **2026-04-19 — `MemoryClient.bulk_store` API** (commits `9a9787b34`,
  `6f8da14b7`). New high-throughput batch write path with structured
  `BulkStoreRequest` / `BulkStoreResult` dataclasses. Per-item
  `PoisoningError` is caught and surfaced on the result so one tainted
  entry cannot fail an entire batch — the remaining items land and the
  caller sees which indices were rejected and why. 7 dedicated tests
  green; complements the existing single-entry `store()` path for
  callers that need batch semantics (distill pipelines, migration
  imports).

### Fixed

Follow-ups to the 2026-04-18 monorepo security audit (learning L-ftMX)
covering three of the eight HIGH findings attributed to trw-memory.

- **2026-04-19 — `code_snippet_flagged` tag bypass closed** (commit
  `668b8c887`, H2 part 1/2). The poisoning gate previously trusted the
  `code_snippet_flagged` tag as a caller-asserted exemption, letting a
  `trw_learn` caller bypass prompt-injection detection by attaching the
  tag themselves. The tag is now ignored for gate purposes on writes;
  it survives as an informational tag on reads. Pairs with trw-mcp's
  `trw_learn` write-path content-policy gate (H2 part 2/2).
- **2026-04-19 — SQL-identifier allowlist on DB recovery** (commit
  `72fc1169a`, H1). `SQLiteBackend.recover_db` previously interpolated
  recovered column names directly into `INSERT` statements, allowing SQL
  injection if a maliciously crafted `.corrupt.bak` was substituted for
  the live DB. Recovered column names are now filtered against a static
  schema allowlist; anything off-list is dropped with a structured warn
  event. Recovery-rate telemetry unchanged.
- **2026-04-19 — Bulk YAML import uses `YAML(typ="safe")`** (commit
  `7e64f84de`, M3). The import path had degraded to the round-trip
  loader, which resolves Python object tags on load. Restored to the
  safe loader, matching the project-wide invariant that all YAML reads
  are `safe`.

### Test hardening

- **2026-04-20 — `mp.get_context('spawn')` on cross-process graph test**
  (commit `e4f9961b5`). Default fork context inherited open SQLite
  handles from the parent test worker and sporadically failed under
  `pytest-xdist`. Spawn context gives each worker a clean interpreter.
  Matches the fix already applied to concurrent-write tests in
  trw-eval.

## [0.7.0] — 2026-04-18 — UTF-8 prevention + per-row quarantine + stale-handle detection

Driven by a 2026-04-18 production incident: two consumer processes held open
file descriptors to a DB inode that had already been moved aside to
`memory.db.corrupt.2026-04-18T18-38-33Z.bak` by the auto-recovery layer. Linux
kept the old inode alive, so those consumers kept reading the corrupt bytes and
every `trw_learn` call failed with
`sqlite3.OperationalError: Could not decode to UTF-8 column 'detail' with text
'...'` until the stale PIDs were manually killed. This release closes three
gaps that let that class of incident happen.

### Added

- **Write-time UTF-8 validation (prevention).** New `Utf8ValidationError`
  (subclass of `SchemaValidationError`) is raised by `SQLiteBackend.store`
  when any TEXT-column string field fails strict UTF-8 round-trip. Lone
  surrogates (`\uD800`–`\uDFFF`) and any other non-encodable char now fail
  fast at write time with `failed_fields=[…]` naming the offending columns,
  so corrupt bytes can no longer land in the DB undetected. Covered fields:
  `content`, `detail`, `nudge_line`, `type`, `namespace`, `source`,
  `source_identity`, `client_profile`, `model_id`, `sync_hash`, and other
  bare-string columns — JSON-serialised fields (`tags`, `evidence`,
  `metadata`) are already safe via `json.dumps` surrogate escaping.
  Implementation: `storage/_utf8_validator.py` (new, 82 LOC, 100% coverage).

- **Per-row quarantine on read (auto-recovery).** `SQLiteBackend.list_entries`,
  `search`, and `entries_with_assertions` now fall back to a
  `text_factory=bytes` cursor when the default decode path raises
  `UnicodeDecodeError` / `sqlite3.OperationalError("Could not decode to UTF-8")`,
  skip the bad row with a `structlog` WARN event `action="memory_row_utf8_quarantined"`
  (column, rowid, table, db_path), and increment the per-backend counter
  `SQLiteBackend.quarantine_count_utf8`. One bad row no longer kills the
  entire scan; callers get a degraded-but-usable result and operators can
  observe the quarantine rate via the counter + structured log.

- **Stale-handle detection with transparent reconnect (coordination).**
  `SQLiteBackend.recover_db` now writes a `memory.db.recovered_at` sentinel
  next to the fresh DB. The backend captures the DB's inode + sentinel
  mtime at connection open, then calls `StaleHandleDetector.is_stale()` at
  the top of every public read method (cached via
  `TRW_MEMORY_STALE_HANDLE_CHECK_SECS`, default 1.0 s — so the check costs
  well under 100 µs on the hot path). On sentinel-newer OR inode-change,
  the backend transparently closes and reopens the connection against the
  current filesystem entry, then resets the detector's baseline. A new
  `StaleConnectionError(StorageError)` surfaces only when the reopen
  attempt itself fails. The existing `IntegrityScheduler`
  observability-only invariant is preserved — the recovery signal lives
  entirely inside `SQLiteBackend` + `recover_db`. Implementation:
  `storage/_stale_handle_detector.py` (new, 181 LOC, 91% coverage) +
  edits to `sqlite_backend.py`.

### Exceptions

- `Utf8ValidationError(SchemaValidationError)` — write-time UTF-8 failure,
  with `failed_fields: list[str]`.
- `StaleConnectionError(StorageError)` — raised only when a detected-stale
  connection cannot be reopened (a normal reconnect is silent).

### Tests

11 new tests under `tests/unit/storage/`:
- `test_utf8_validation.py` (4): lone-surrogate rejection, multi-field
  rejection, valid-unicode acceptance, no double-validation.
- `test_resilient_list_entries.py` (3): skips bad row + counter + log
  capture, all-bad returns empty list, clean DB no-quarantine regression.
- `test_stale_handle_recovery.py` (4): inode-change reconnect, sentinel-mtime
  reconnect, check-budget caching cheap, reopen-failure raises
  `StaleConnectionError`.

Coverage on new code: 92.86% (100% validator, 91% detector — target ≥90%
met). 123 existing `test_sqlite_backend_recovery.py` +
`test_storage_sqlite.py` + `test_sqlite_assertions_column.py` tests remain
green (zero regressions). `mypy --strict` clean on all new + edited files.

### Diagnostic recipe

When `trw_learn` or any `list_entries` call raises
`Could not decode to UTF-8 column '…' with text '…'`:

```bash
# 1. Confirm the live DB is clean:
python3 -c "import sqlite3; c = sqlite3.connect('.trw/memory/memory.db'); c.execute('SELECT COUNT(*) FROM memories').fetchone()"

# 2. Find consumer processes with open handles on a .corrupt.*.bak:
for pid in $(pgrep -f trw-mcp); do
  echo "=== PID $pid ==="
  ls -l /proc/$pid/fd 2>/dev/null | grep -E '\.db'
done

# 3. Kill any PID whose FD points at a .corrupt.*.bak:
kill -TERM <stale-pid>
```

With 0.7.0 installed, this class of incident self-heals: consumers detect
the sentinel on their next read, reconnect, and resume.

### Environment variables

- `TRW_MEMORY_STALE_HANDLE_CHECK_SECS` (default `1.0`) — how long a
  successful stale-handle check is cached before the next `stat` call.

## [0.6.11] — 2026-04-17

### Fixed

- **`import trw_memory` also required `cryptography` on bare install** — same class of latent bug caught by v0.6.10's publish-gating smoke test. `trw_memory.security.encryption` is re-exported unconditionally from `trw_memory.security.__init__`, and that module chain is reached through the package's top-level imports. `cryptography` was listed only under `[encryption]` extras, so fresh `pip install trw-memory` failed with `ModuleNotFoundError: No module named 'cryptography'`. Moved `cryptography>=41.0.0` to base dependencies (it stays in `[encryption]` extras for users who already specify it). Verified locally with a clean venv: `pip install trw-memory && python -c "from trw_memory.storage.sqlite_backend import SQLiteBackend"` now succeeds. v0.6.10 was tagged but never published for the same reason v0.6.9 wasn't — both caught by the smoke gate.

## [0.6.10] — 2026-04-17 (never published)

### Fixed

- **`import trw_memory` failed on a bare install without httpx** — `trw_memory/__init__.py` transitively imports `trw_memory.sync.remote`, which does `import httpx` at module scope. `httpx` was listed only under `dev` extras, so any user running `pip install trw-memory` (without trw-mcp bringing httpx in as a transitive) hit `ModuleNotFoundError: No module named 'httpx'` at first import. This was masked in CI because no cross-platform smoke test exercised the wheel's import path until v0.6.9's release workflow was added. Moved `httpx>=0.27.0` from `[dev]` to base `dependencies`; the v0.6.9 release workflow run (which caught this) is the regression-prevention signal. Retagging as v0.6.10 because v0.6.9 never published a PyPI artifact.

## [0.6.9] — 2026-04-17 (never published)

### Fixed

- **Fresh macOS installs no longer lose learnings when Python was built without SQLite extension support** — on macOS system Python and some python.org builds, `sqlite3` is compiled without `SQLITE_ENABLE_LOAD_EXTENSION`, so `conn.enable_load_extension(True)` raises `AttributeError` (method absent) or `OperationalError` (not authorized). The previous `except (sqlite3.Error, OSError)` clause in `SQLiteBackend.__init__` did not catch `AttributeError`, so the error propagated up through the MCP server and surfaced as "sqlite extension error in the MCP server" at every `trw_learn` call, blocking all learning persistence. `AttributeError` is now caught alongside `sqlite3.Error` and `OSError`; the backend degrades gracefully to BM25-only retrieval and emits a `sqlite_vec_load_failed` warning with the exception type + detail + remediation hint. Added two regression tests (`TestSqliteVecExtensionLoadFailure`) that install a sqlite3 connection proxy raising `AttributeError` and `OperationalError` respectively, and assert backend init succeeds with `_vec_available=False` while metadata operations still round-trip.

## [Unreleased — prior to 0.6.9]

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
