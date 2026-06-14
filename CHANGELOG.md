# Changelog

All notable changes to the TRW Memory package.

## [Unreleased]

### Added

- **`MemoryClient.store_many(entries)` — dict-based bulk insert.** A convenience
  wrapper over `bulk_store` that accepts a list of plain `store()`-shaped dicts
  (rather than `BulkStoreRequest` objects), runs the full validation + security
  gate + FTS dual-write path, and returns the number of rows written. Inserted
  rows are immediately searchable via `search_fts`/`recall`.
- **`MemoryClient.search_fts(query, ...)` — SQLite FTS5 keyword search.** An
  O(log N) inverted-index BM25 lookup over `content`/`detail`/`tags` for
  pure-keyword queries that do not need hybrid ranking, exposed on both the
  client and `SQLiteBackend`. Degrades to an empty result when FTS5 is
  unavailable.
- **Bi-temporal validity fields on `MemoryEntry`.** Entries now carry explicit
  validity-interval metadata so callers can distinguish when a fact was recorded
  from the time window over which it is asserted to hold.
- **CombMAX fusion as a configurable retrieval combiner.** A max-reciprocal-rank
  fusion strategy is selectable alongside the default RRF, preserving
  single-ranker champions on hard-tail queries instead of diluting them.
- **Embedding-download disclosure + offline switch (PRD-QUAL-110).** A
  network-capable embedding load now emits a disclosure log line before any
  huggingface.co fetch. `TRW_OFFLINE=1` / `HF_HUB_OFFLINE=1` (or
  `local_only: true`) force `local_files_only=True`; when the model is not
  cached the standalone engine raises a clear `LocalOnlyViolationError`
  explaining how to pre-download (it does not silently fall back to keyword-only
  recall — that graceful path lives in the trw-mcp embedder wrapper).

### Changed

- **SEC-001 size-anomaly detector now defaults to `observe` mode.** The
  poisoning size-anomaly path no longer quarantines against a cold/empty
  reference distribution by default, so legitimate longer learnings are not
  dropped on the first batch; `strict` remains opt-in.

### Security

- **`memory.db` is created mode `0600` (owner-only) (PRD-QUAL-110).** The
  file-backed SQLite store is `chmod 0600` on creation, mirroring the trw-mcp
  pins-file hardening; a non-POSIX platform degrades to a `db_chmod_failed`
  warning. In-memory (`:memory:`) backends have no file to harden.
- **Caller-controlled anomaly-quarantine bypass removed (`security/runtime.py`,
  commit `209a47853`).** The anomaly quarantine could previously be bypassed by
  any caller setting `metadata['source']` to a configured prefix (e.g.
  `distilled:`). Because `entry.metadata` is caller-supplied, a poisoned outlier
  could skip the detector by spoofing one field. `MemoryEntry` has no
  system-owned trusted-source flag, so the runtime bypass is removed entirely;
  spoofed source metadata can no longer bypass enforce-mode quarantine. The
  `anomaly_bypass_source_prefixes` config field remains for compatibility but no
  longer gates the runtime anomaly path.

### Fixed

- **YAML entry mapper fails open on malformed timestamps.** A single
  unparseable timestamp in a YAML-backed entry no longer aborts the read; the
  field degrades gracefully instead of dropping the row.

## [0.9.6] — 2026-06-09

### Fixed

- **Concurrency hardening across the lifecycle / tier subsystem.**
  Five verified data-safety / resource bugs are fixed:
  - `_sweep_hot_to_warm` no longer iterates and mutates the shared hot
    `OrderedDict` without the manager's `_hot_lock`. The Hot→Warm sweep now
    snapshots the stale candidates under the lock, performs the blocking
    `warm_add` I/O outside it, then re-acquires the lock to evict — re-checking
    that the same entry instance is still resident so a concurrently-refreshed
    entry is never silently dropped (no more `dictionary changed size during
    iteration`).
  - `consolidate_cycle(namespace=None)` now raises `ValueError` on a
    multi-namespace store instead of clustering entries across ALL tenants and
    persisting the merged result into a single namespace (cross-tenant leak).
    The single-namespace and explicit-namespace paths are unchanged.
  - `_TIER_MANAGER_CACHE` is now a bounded LRU (default cap 32) that calls
    `close()` on the evicted manager before dropping it, so it no longer leaks
    one open SQLite connection per namespace forever.
  - The warm-tier JSONL sidecar read-modify-write (`_warm_sidecar_upsert` /
    `purge_sidecar_entry`) is now guarded by the advisory `lock_for_rmw`
    primitive, preventing concurrent upserts (and upsert-vs-purge) from
    clobbering each other's rows and corrupting the file.
  - `_rollback_consolidation` now always restores the originals to ACTIVE
    before surfacing a failed delete of the consolidated entry, so a partial
    consolidation on a YAML backend (whose `transaction()` is a no-op) can no
    longer leave originals archived alongside a surviving consolidated entry.

- **Package lock version now matches the 0.9.6 package bump.** `uv.lock` no
  longer records the previous editable self-package version, restoring the
  package metadata guard.
- **Consolidation lifecycle stays under the effective-LOC ratchet.** The
  consolidated-entry field derivation now stays local to the `MemoryEntry`
  construction path without changing archive/rollback semantics, restoring the
  package maintainability gate.

## [0.9.5] — 2026-06-09

### Fixed

- **Tier erasure now deletes the cold YAML archive copy (GDPR completeness).**
  `remove_entry_from_tiers` previously removed an entry only from the hot and
  warm tiers, so a `forget` / `forget actor=...` bulk erasure left any
  cold-archived copy permanently on disk — a data-deletion / compliance gap.
  A new `ColdTierStore.cold_remove` (exposed via `TierManager.cold_remove`)
  scans the cold partition tree by entry `id` and unlinks the matching file(s);
  `remove_entry_from_tiers` now invokes it so erasure spans all three tiers.

- **FastMCP lock surfaces now require the patched 3.2.x floor.**
  The optional MCP extra and `requirements.lock` no longer allow/pin
  vulnerable FastMCP 3.0.x installs, and package tests now guard the
  requirements lock against regressing below 3.2.0.
- **Requirements-lock security floors were refreshed from the audit backlog.**
  Vulnerable pins with available fixes (`Authlib`, `cryptography`, `idna`,
  `Pygments`, `PyJWT`, `pytest`, `python-dotenv`, `python-multipart`,
  `starlette`) now pin patched versions and have regression guards.
- **Core runtime dependency declarations no longer rely on transitive
  `typing_extensions`.** The package now declares the Python 3.10
  compatibility shim directly, with a metadata regression guard.
- **Deptry static-analysis configuration now matches the package layout.**
  The audit treats `src/trw_memory` as first-party, marks `dev` as a
  development extra, maps LlamaIndex/LangChain package names to their import
  modules, maps optional SQLCipher/CrewAI extras explicitly, and documents
  intentional optional-import seams so `deptry .` reports actionable findings
  instead of optional-extra noise.

## [0.9.4] — 2026-06-09

### Fixed

- **Package lock version now matches the 0.9.4 package bump.** `uv.lock` no longer records the
  previous editable self-package version, restoring the package metadata guard.
- **Tag-filtered hybrid recall no longer silently drops valid hits ranked past
  `top_k` (`_client_recall_hybrid.py`).** The tag filter ran AFTER `hybrid_search`
  truncated the ranking to `top_k` (= `limit * recall_top_k_multiplier`, default
  30), so tag-matching entries ranked beyond that depth were lost — reducing
  recall below the caller-requested count on larger namespaces. When `tags` is
  non-empty, `effective_top_k` is now raised to at least the namespace size (the
  full already-loaded candidate pool) so the post-rank tag filter sees every
  candidate. Tag-free recall is unchanged.
- **`propagate_impact` rolls back partial writes on a mid-loop failure
  (`_graph_clusters.py`).** The BFS importance-propagation loop issued one
  uncommitted `UPDATE` per affected node and only committed after the loop, with
  no exception handling. A failure mid-loop left a corrupt prefix of node-impact
  writes dangling in the connection for a later unrelated commit to flush. The
  loop is now wrapped in `try/except` that calls `conn.rollback()` and re-raises,
  mirroring `memory_decay_pass`.
- **`list_org_shared_entries` pushes `min_importance` into the storage layer
  (`graph.py`, storage backends).** The function loaded up to 10,000 full
  `MemoryEntry` objects per sibling namespace and discarded the low-importance
  ones in Python. `list_entries` now accepts an optional `min_importance` filter
  (threaded through the abstract interface, the SQLite `_build_filter_clause`
  pushdown, and the YAML backend), so only rows the caller can keep are hydrated.
  Result semantics are unchanged; the default `min_importance=0.0` preserves the
  legacy no-filter behaviour for all other callers.

## [0.9.3] — 2026-06-09

### Fixed

- **`entries_with_assertions` no longer leaks cross-namespace rows and is bounded
  (`storage/_query_ops.py`).** The query had no namespace predicate, so the
  session-start assertion-health summary could aggregate assertions from other
  namespaces; it also had no `LIMIT`, allowing an unbounded full-table scan on
  large stores. Added optional `namespace` and `limit` (default 500) parameters,
  mirrored on the `SQLiteBackend.entries_with_assertions` wrapper.
- **Bytes-mode UTF-8 fallback now keys its secondary connection on encrypted
  stores (`storage/_resilient_fetch.py`).** The fallback opened the secondary
  connection without applying the SQLCipher key, so on an encrypted database
  every SELECT ran against a blank handle and returned zero rows — silently
  dropping all data instead of quarantining only the bad-UTF-8 rows. The
  SQLCipher key is now threaded through `FetchQuery` and applied before the read;
  a malformed key raises rather than silently dropping rows.
- **Enum-typed string fields are validated before persistence
  (`storage/_shared.py`).** `serialize_update_value` passed raw strings for
  `status`/`confidence`/`protection_tier`/`type` through unvalidated, so an
  invalid value persisted and made the row un-deserializable (permanent
  quarantine) on the next read. The value now round-trips through the enum
  constructor as a validation gate; the resulting `ValueError` is wrapped into a
  `StorageError`, rejecting the bad write up front.
- **`rotate_key` uses an engine-aware WAL checkpoint and a strict exclusivity
  guard (`security/encryption.py`).** Key rotation always issued
  `PRAGMA wal_checkpoint(TRUNCATE)`, which on a SQLite engine without the
  WAL-reset fix (< 3.51.3) is the documented corruption trigger when racing
  another connection. It now uses `TRUNCATE` only when the active driver is
  WAL-reset-safe and `PASSIVE` (which never resets the WAL) otherwise.
  Separately, the busy-check guard used `checkpoint_row is not None`, so a
  `None` (abnormal/empty) PRAGMA response skipped the exclusivity check; the
  guard now requires `busy == 0` and aborts on any non-zero busy **or** a missing
  row — the safe default for a destructive rotation.

### Changed

- **Effective-LOC ratchet restored for trw-memory source files.** The SQLite transaction
  implementation moved behind a focused `storage/_transaction.py` seam, and the
  `MemoryEntry.to_dict()` field projection moved to `models/_memory_entry_serialization.py`.
  This brings `storage/sqlite_backend.py` and `models/memory.py` back under their committed
  ratchet baselines while preserving the public adapter methods.

### Fixed

- **Package lock version now matches `pyproject.toml`.** `uv.lock` still recorded `trw-memory`
  0.8.5 after the package advanced to 0.9.2, breaking `tests/test_package.py::test_uv_lock_version_matches_pyproject`
  and release lock hygiene. The self package stanza now matches the current project version.

## [0.9.2] — 2026-06-09

### Security

- **SSN/PHONE/CREDIT_CARD in tags are now redacted (`security/_runtime_pii.py`).** Completes the
  v0.9.1 tag-scan: those types were detected in tags but absent from `REDACTED_PII_TYPES`, so
  `replace_pii` fell through and stored the raw value verbatim (surfaced at recall). They now redact
  to `<ssn>`/`<phone>`/`<credit_card>` (redact-not-block, parity with EMAIL/IP).
- **Public `check_entry_pii` now scans `entry.tags` (`security/pii.py`).** The PUBLIC API still
  scanned only `content` + `detail`, so direct callers got a false-clean result for PII hidden in a
  tag even though the internal runtime path was fixed in v0.9.1. Tags are now blocked/redacted per the
  selected action, matching the runtime path.
- **Custom-pattern ReDoS guard now catches quantified alternation (`security/pii.py`).** The v0.9.1
  guard caught nested quantifiers (`(a+)+`) but missed quantified-alternation backtracking
  (`(a|a)*`); such patterns are now rejected as `ConfigError`.
- **`rotate_master_key` converges against the LIVE backend (`security/keys.py`).** v0.9.1 compared
  coverage against a PRE-rotation `count()` snapshot, so entries inserted concurrently during
  re-encryption escaped — left on the OLD key — while the count check stayed satisfied. Rotation now
  sweeps repeatedly, re-reading the live backend and re-encrypting only newly-seen IDs until a pass
  finds nothing new and the post-pass live count is covered. Bounded by `_ROTATION_MAX_SWEEPS`; a
  writer inserting faster than rotation raises `KeyRotationError` instead of spinning.

### Fixed

- **Invalid custom PII pattern raises `ConfigError`, not `re.error` (`security/pii.py`).** The v0.9.1
  ReDoS guard validated length + nested quantifiers but never trial-compiled, so a syntactically
  invalid pattern (e.g. `[`) raised an uncaught `re.error` from `re.compile` and crashed every store
  for the session. The pattern is now trial-compiled at validation time and surfaced as a
  documented `ConfigError`.
- **Public `delete_vector` defers its commit inside `transaction()` (`storage/_vector_ops.py`).** It
  committed unconditionally — the same premature-commit bug class fixed in `_crud_ops` for v0.9.1 but
  missed here — prematurely committing an enclosing caller transaction. It now accepts a `skip_commit`
  flag (matching `upsert_vector`) so the vector delete batches into the caller's outermost COMMIT and
  rolls back with the transaction.
- **Nested `transaction()` depth counter mutated under the lock (`storage/sqlite_backend.py`).** The
  non-outer increment/decrement of `_skip_commit_depth` ran outside `_lock` while other writers read
  that counter under `_lock` to decide whether to commit, racing with no happens-before edge. Both
  nested mutations now run under `_lock`, symmetric with the outer case.

## [0.9.1] — 2026-06-09

### Security

- **Injection scan now covers `entry.tags` (`security/poisoning.py`).** `validate_entry_payload`
  previously scanned only `content` + `detail`, so an injection command placed in a tag bypassed the
  write-time poisoning gate while still surfacing at recall. Tags are now included in the scan.
- **SQLCipher key never leaks through a rekey failure (`security/encryption.py`).** `PRAGMA rekey`
  must embed the key hex in the SQL text; a driver that echoed the failing statement could surface the
  key through the raised exception or its `__context__`/`__cause__` chain. Rekey failures now raise a
  sanitized `KeyRotationError` with the original (key-bearing) exception detached.
- **`rotate_master_key` re-encrypts ALL entries (`security/keys.py`).** The hardcoded
  `list_entries(limit=100_000)` silently left surplus rows encrypted under the OLD key. Re-encryption
  is now count-driven with headroom and raises `KeyRotationError` if coverage is incomplete.
- **Runtime PII policy scans `entry.tags` (`security/_runtime_pii.py`).** A credential (API key) or
  PII in a tag previously bypassed both the PII block gate and redaction. Tags are now blocked and
  redacted alongside `content`/`detail`.
- **ReDoS guard for caller-supplied custom PII patterns (`security/pii.py`).** `detect_pii` compiled
  and ran `custom_patterns` with no safety check; Python `re` has no timeout. Pattern count + length
  are now capped and catastrophic-backtracking (nested-quantifier) constructs are rejected.
- **Keyring master-key cache keyed on identity (`security/keys.py`).** The in-process cache now
  validates the `(service, account)` identity on a keyring hit instead of the bare `source` flag.

### Fixed

- **Transaction atomicity in `storage/_crud_ops.py`.** `increment_session_counts`,
  `increment_access_counts`, and `delete` committed unconditionally, prematurely committing any
  caller-opened `transaction()`. They now defer the commit inside a transaction like
  `store`/`update`/`increment_recall_access`. Recall/session/access counters are also bounded.
- **Tag-filtered search no longer under-delivers (`storage/_query_ops.py`).** `search` applied the
  SQL `LIMIT` before the in-memory tag filter, returning fewer than `top_k` matching entries (often
  zero). The required tags are now pushed into SQL so the limit applies after filtering.
- **`FILE_PATH` PII regex no longer matches URL path components (`security/pii.py`).** A negative
  lookbehind stops false positives on `example.com/api/...` and `https://host/a/b`.

## [0.9.0] — 2026-06-08

### Added

- **User-space (machine-local) memory tier — knowledge-fabric foundation (PRD-CORE-185).**
  trw-memory's latent namespace federation is now driven end-to-end to back a `user:` tier that
  lives outside any single project. A new user store resolves to `~/.trw` (or the XDG base dir when
  set) and is selectable alongside the project-local `default` namespace; the scope resolver and
  config cascade let a caller route a write to the project tier or the machine-local user tier, and
  recall federates across the project ∪ user namespaces in one ranked result set. This unlocks the
  portable, cross-project memory the higher tiers (company knowledge tier, bulk distill) build on.
  The engine-level primitives (namespace-scoped stores, scope resolution, federated recall) are the
  load-bearing half; write routing, the portability classifier, and the `scope=`/`include_tiers=`
  surface live in the trw-mcp layer (see trw-mcp 0.55.0). Additive and backward-compatible: existing
  single-namespace projects are unaffected, and the opt-in backfill is non-destructive.

- **Explicit code index — MCP tools + CLI.** `memory_code_index`, `memory_code_search`, and
  `memory_code_symbol` MCP tools (registered by `tools/code_index.py`) and the matching
  `trw-memory code-index` / `code-search` / `code-symbol` CLI subcommands expose the
  `code_index/` chunker/indexer/symbol/search engine for lexical code search and symbol lookup.
- **Wiki lint — MCP tool + CLI.** `memory_wiki_lint` MCP tool and the `trw-memory wiki-lint`
  CLI subcommand lint wiki page JSON for missing targets, backlinks, and provenance gaps,
  backed by the `wiki/` indexer/lint package.

### Changed

- **Effective-LOC ratchet brought back to green (PRD-DIST-245).** Three modules had grown
  past the 350-effective-LOC module gate; each was split along an existing cohesive seam,
  preserving every public API and the documented monkeypatch seams via re-exports / a mixin:
  - `storage/_recovery.py` (370 → 279): the bounded open-time preflight + advisory
    recovery-state sidecar (`RecoveryPreflight`, `classify_recovery_preflight`,
    `write_recovery_state`, `recovery_state_path`, `_read_persisted_recovery_status`) moved
    to the new `storage/_recovery_preflight.py`; `_recovery.py` re-exports the four public
    names so `_init_helpers` and tests resolve them unchanged.
  - `storage/sqlite_backend.py` (465 → 446, baseline lowered 463 → 446): the standalone
    `check_integrity` probe moved to `storage/_connection.py` (`check_integrity`); the
    backend keeps a `staticmethod` alias so `SQLiteBackend.check_integrity` callers and
    patches are unaffected.
  - `client.py` (388 → 338, removed from the baseline): the org-shared recall alias group
    (`_merge_shared_results`, `_coerce_float`, `_dedupe_cached_shared_results`, …) moved to
    the new `OrgSharedAliasMixin` (`_client_org_shared_aliases.py`), mixed into
    `MemoryClient` so `self._X` / `MemoryClient._X` resolution is unchanged via the MRO.
- **Documentation accuracy pass.** README, `tests/CLAUDE.md`, and this changelog were
  reconciled against the source tree — the MCP tool list (now store/recall/search/forget/
  consolidate/status/audit/review/wiki-lint/code-index/search/symbol), the CLI command set
  (restore/snapshot/wiki-lint/code-*), the `MemoryClient` public surface
  (`bulk_store`/`audit_learning`/`review_quarantined`), optional-dependency extras
  (`[encryption]`, `[all-integrations]`), and the architecture overview were corrected, and
  drift-prone hardcoded file/test/LOC counts were de-quantified.
- **Hybrid recall pipeline extracted to its own deep module.** `_client_recall.py` (482
  effective LOC, over the 350-LOC module gate) had the BM25 + dense + RRF pipeline
  (`try_hybrid_recall`) and its private latency-telemetry helper
  (`_emit_hybrid_recall_telemetry`) split into a new `_client_recall_hybrid.py` (154
  effective LOC). The new module presents one narrow interface —
  `try_hybrid_recall(...) -> list[MemoryResultDict] | None`, where `None` signals
  fall-back — over a deep implementation (candidate-pool sizing, namespace-aware
  BM25/vector candidate auto-scaling, RRF top-K depth, per-recall telemetry).
  `_client_recall.py` re-exports `try_hybrid_recall`, so `MemoryClient._try_hybrid_recall`
  and all public/back-compat imports are unchanged. Brings `_client_recall.py` to 341
  effective LOC; behavior-preserving (no public API or recall-result change).

### Fixed

- **Resolved `_client_recall.py` effective-LOC debt was removed from the root baseline.**
  The module was already split below the 350 effective-LOC gate; the stale grandfathered
  allowance is now gone so future growth above the gate fails instead of passing under an
  obsolete baseline entry.

- **Focused Ruff validation for sync tests is green again.** Removed a stale unused `time` import
  from `tests/test_sync_retry_queue.py` and normalized the `InstrumentedLock` return annotation in
  `tests/test_sync_delta.py` now that `from __future__ import annotations` is active.

- **Lock/version hygiene: uv.lock, requirements.lock, and pyproject realigned.** `pyproject.toml`
  had been bumped to `0.8.5` while `uv.lock` still recorded the package at `0.8.1` (and was missing
  the `pysqlite3-binary` Linux dependency), so `uv lock --check` failed. Regenerated `uv.lock` with
  `uv lock` (no dependency upgrades — only the version bump and the already-declared
  `pysqlite3-binary` pin). `requirements.lock` pinned the editable self-reference to a frozen git
  commit (`...trw-framework.git@0c7d4263...#egg=trw_memory`) that drifts the moment `main` advances;
  normalised it to a path install (`-e .`) so it never goes stale. Added two `tests/test_package.py`
  guards: `test_uv_lock_version_matches_pyproject` and `test_requirements_lock_has_no_stale_self_pin`.
- **`restore --from-snapshot latest` now picks the newest snapshot across BOTH tiers.**
  `handle_restore` resolved `latest` with `listing["daily"] or listing["weekly"]`, so any
  daily snapshot beat every weekly snapshot even when a weekly was strictly newer —
  recovery-hostile, since `latest` could silently restore a stale daily over a fresher
  weekly backup. A new `_snapshot.latest_snapshot(base_dir)` helper compares snapshots
  across the daily and weekly tiers by the calendar date their filename encodes (weekly
  `YYYY-Www` maps to its Sunday, ISO weekday 7), returning the newest; on a same-date tie the
  finer-grained daily is preferred. Explicit-filename restore is unchanged. The `--from-snapshot`
  help text now states `latest` means newest across tiers. (Reviewer P2.)
- **Resilient fetch fast path now quarantines unmappable rows, not just bad-UTF-8 rows.**
  The common (non-fallback) row-materialisation loop in `fetch_rows_resilient` only caught
  `UnicodeDecodeError`/`UnicodeEncodeError`, so a single row whose columns decoded cleanly but
  failed `row_to_entry` — an out-of-range status/type/confidence/tier enum, a non-numeric scalar,
  or schema drift from a newer writer — raised `ValueError`/`TypeError`/`KeyError` out of the whole
  query, collapsing `list_entries`/`search`/recall for every co-resident memory. The fast path now
  quarantines those rows (logging `column='row_to_entry'`, `outcome='quarantined'`) and increments
  the quarantine counter, matching the bytes-mode fallback's existing behaviour.
- **Bandit deserializers fail open per arm.** `BanditSelector.from_json` and
  `ContextualBanditSelector.from_dict` now skip malformed arms while preserving valid arm state,
  parsed hyperparameters, and real `feature_dim`; one corrupt persisted row no longer resets the
  whole selector or causes later dimension-mismatch failures.
- **Vector clock advanced on local update** (commit `b134d9ffc`). The vector clock was being reset
  instead of incremented on local writes, causing team-sync conflict resolution to pick the wrong
  winning value when two nodes updated the same entry. All local stores now call
  `_advance_local_clock()`.
- **Vector dimension mismatch guarded in native sqlite-vec pack path** — an uncaught `struct.error`
  from a mismatched embedding dimension failed the whole store silently; the error is now caught,
  logged with `outcome=dimension_mismatch`, and raised as `DimensionMismatchError`.
- **GitHub PAT and AWS access-key patterns added to PII scan** — secrets matching `ghp_`/`ghs_`/
  `gho_` (GitHub PAT) and `AKIA[0-9A-Z]{16}` (AWS Access Key ID) were not detected by the PII
  scanner, so they could be stored in plaintext.
- **Obsolete neighbours filtered from graph related-recall** — `get_related()` was returning
  `obsolete`/`superseded` neighbour entries the same as main recall; obsolete entries are now
  filtered out before the result is returned.
- **Config-driven tier caps + batch tier convergence** (store-audit S12, recall R-RANK-003).
  Per-tier entry caps are now read from `MemoryConfig` at runtime; the batch tier-assignment loop
  converges in a bounded number of passes instead of potentially oscillating.
- **Recall ranks session-start baseline by utility, not recency, and blends impact into RRF fusion**
  (recall audit R-RANK-002/004, R-FUSION-001). Shared fix with trw-mcp — see trw-mcp changelog.
- **`transaction()` made thread-safe; namespace-delete and consolidation atomicity gaps closed**
  (store-audit S4/S8). `transaction()` now acquires the write-lock before entering the SQLite
  transaction so concurrent threads cannot interleave writes. `delete_namespace` and
  `consolidate_entries` are wrapped in transactions to prevent partial updates.
- **Store + vector writes made atomic; transaction-depth TOCTOU closed** (store-audit S1/S2/S3/S9).
  The primary-store INSERT and the sqlite-vec INSERT are now inside the same `BEGIN IMMEDIATE` block
  so a failure mid-sequence cannot leave orphan vector rows. The transaction-depth check used a
  read-check-write sequence that could race; it is now protected by the write-lock.
- **Dedup works on default installs; merge stops losing data** (store-audit P0/P1). The dedup path
  was gated on optional `[vectors]` being installed; semantic dedup now falls back to BM25-only
  similarity on plain installs. The merge-on-dedup path was discarding the incumbent entry's tags
  and namespace before overwriting; all fields are now merged.
- **Recall correctness and performance** (recall audit C6/C7/C11/P-007/P-008). Expiry filter
  applied before scoring (expired entries no longer appear in results), obsolete/superseded vectors
  pruned from the vector index on recall, batch recall-access uses a single SQL `IN (…)` query
  instead of N round-trips, and missing indexes on `(status, namespace)` and `created_at` added.

## [0.8.5] — 2026-05-29

### Fixed

- **`list_entries`/`search`/`entries_with_assertions` preserve their WHERE filter + LIMIT on the
  UTF-8 bytes-fallback path.** Previously, when a corrupt-UTF-8 row triggered the resilient
  bytes-mode re-execute, the fallback dropped the status/namespace filter and the LIMIT — returning
  rows of all statuses/namespaces, unbounded. The fallback now re-executes the exact query
  (where/params/order/limit). Strong-typed the resilient-fetch helpers (Protocols replace `Any`),
  narrowed broad excepts, and added an `outcome` field to quarantine logs.
- **`pysqlite3-binary` is now a Linux-only dependency.** Upstream removed the macOS-arm64 wheels,
  so `pip install trw-memory` failed with "No matching distribution found for pysqlite3-binary" on
  `macos-latest` (arm64) — which blocked the release smoke matrix. The marker is now
  `platform_system == 'Linux'`; macOS/Windows fall back to stdlib `sqlite3` via the `storage._dbapi`
  shim (with the code-level WAL single-connection mitigation), exactly as the shim already supported.
- **Trust-score-quarantined entries are now signed so they remain auditable** (SEC-001). The
  `prepare_entry_for_store` early-return on the trust-score quarantine path skipped provenance
  signing, so `audit_entry()` reported a quarantined entry as `legacy_unsigned` (no
  `provenance_signature`) instead of `quarantined`. Signing now runs on that path too (best-effort:
  a missing signing key leaves the still-quarantined entry unsigned rather than failing the store).
  The anomaly-quarantine path already signed before quarantining; genuinely-unsigned legacy rows
  still correctly audit as `legacy_unsigned`.

### Added

- **`bytes_fallback_failures` counter** on the resilient-fetch path. When the UTF-8 bytes-mode
  fallback connection itself fails, the fetch fails open (`[]`) but now increments a process-wide
  counter (`get_bytes_fallback_failures()` / `reset_bytes_fallback_failures()`) alongside the
  `outcome=fallback_failed` warning log, so an otherwise-silent secondary drop is countable.


## [0.8.4] — 2026-05-28

### Added

- **Recall-latency telemetry on the hybrid retrieval path** (PRD-DIST-2047 Phase 2,
  commit `a6e756bde`). `try_hybrid_recall` now emits a `hybrid_recall_complete`
  structlog event on every terminating branch (`ok`, `no_candidates`,
  `empty_ranking`, `hybrid_search_failed`). The event carries
  `list_entries_ms`, `hybrid_search_ms`, `total_ms`, `namespace_size`,
  `candidate_pool_size`, `effective_bm25_candidates`, and
  `effective_vector_candidates` so retrieval latency and candidate-pool
  health can be diagnosed from structured logs without instrumentation changes.

### Fixed

- **Restored `SHARED_EVENT_CACHE_MAX` re-export from `trw_memory.client`.**
  The PRD-DIST-246 client.py decomposition moved the constant to the
  `_client_lifecycle` sibling module but the `client.py` facade dropped the
  re-export, breaking `from trw_memory.client import SHARED_EVENT_CACHE_MAX`
  (a public-API regression that also broke `test_client_recall_sync.py`
  collection). The facade re-exports it again and lists it in `__all__`.


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
