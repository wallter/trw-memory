# trw-memory

**Persistent memory engine for AI agents** — local-first storage, hybrid retrieval (BM25 + vectors), lifecycle scoring, tiered storage, and a knowledge graph. The standalone memory backend powering [TRW Framework](https://trwframework.com).

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)
[![License: BSL 1.1](https://img.shields.io/badge/License-BSL_1.1-orange.svg)](https://trwframework.com/license)
[![Docs](https://img.shields.io/badge/docs-trwframework.com-blue)](https://trwframework.com/docs)

> **Release status:** Alpha and source-available under BSL 1.1. The public API may change before 1.0; evaluate upgrades in a test environment before production rollout.

**[Quick start](#quick-start)** · **[Python API](#memoryclient-recommended)** · **[CLI](#cli)** · **[Benchmarks](#benchmarks)** · **[Security and network behavior](#telemetry--network-behavior)** · **[Development](#development)**

## How it fits

trw-memory is the standalone memory engine for [TRW (The Real Work)](https://trwframework.com) — a methodology layer for AI-assisted development that provides stateless agents with a persistent memory layer **designed to enable self-improvement across sessions** via [knowledge compounding](https://trwframework.com/docs). *The outcome effect of cross-session memory on coding tasks is an open empirical question; early SWE-bench single-shot runs (n≥40) produced null. See the [verification docs](https://trwframework.com/docs/verification) for the current methodology and evidence posture.* It works alongside [trw-mcp](https://github.com/wallter/trw-mcp), the MCP server that builds its tooling on this engine.

- **trw-memory** (this repo): Standalone AI agent memory engine with hybrid retrieval, scoring, and lifecycle
- **trw-mcp**: MCP server for AI coding agents — uses trw-memory as its backend

## What it does

TRW-Memory is a standalone **persistent memory engine for AI agents** that gives coding agents searchable, long-lived knowledge storage. It stores learnings (patterns, gotchas, architecture decisions) in SQLite with optional YAML backup, and retrieves them using [hybrid search](https://trwframework.com/docs) that combines keyword matching (BM25) with dense vector similarity.

Designed as the storage backend for [trw-mcp](https://github.com/wallter/trw-mcp) and [TRW Framework](https://trwframework.com), but usable independently by any AI agent framework that needs persistent memory with recall.

## Features

- **MemoryClient SDK** -- High-level async Python client with store/bulk_store/store_many/recall/search/search_fts/forget plus audit_learning and review_quarantined
- **Hybrid Search (BM25 + vector)** -- BM25 keyword matching + dense vector similarity via sqlite-vec, combined with Reciprocal Rank Fusion (RRF). [Learn more](https://trwframework.com/docs)
- **FTS5 keyword search** -- `MemoryClient.search_fts()` runs O(log N) SQLite FTS5 BM25 lookups over content/detail/tags for pure-keyword queries that don't need hybrid ranking; degrades to an empty result when FTS5 is unavailable
- **Hybrid order preservation by default** -- recall preserves the hybrid BM25+dense+RRF order when enough local candidates are already available, avoiding a legacy score-scale mismatch in tier merging. To restore the legacy tier rescore for a workload, set `MEMORY_RECALL_PRESERVE_HYBRID_ORDER=false`.
- **Tiered Storage** -- Hot/warm/cold tiers for fast recall, warm-sidecar persistence, recall-time cold promotion, and explicit sweep-based archiving/purging. [Architecture details](https://trwframework.com/docs)
- **Semantic Deduplication** -- Detects and merges near-duplicate learnings using cosine similarity (0.85 threshold)
- **Knowledge Graph for AI** -- Tag co-occurrence and similarity edges, BFS traversal, importance boost/decay, cross-validation propagation. [Docs](https://trwframework.com/docs)
- **Memory Consolidation** -- Episodic-to-semantic consolidation via clustering with the current shipped path using heuristic/fallback summarization
- **Q-learning Memory Scoring** -- Q-learning with EMA updates, [Ebbinghaus forgetting curve](https://trwframework.com/docs) applied at query time, Bayesian MACLA calibration
- **Remote Sync** -- Publish/fetch learnings across installations with vector clock conflict resolution and SSE live updates
- **Security** -- AES-256-GCM field encryption, PII detection/redaction, memory poisoning detection (z-score anomaly), RBAC, audit trail
- **Agent Integration** -- `register_tools()` for any agent framework, `@auto_recall` decorator
- **Framework Integrations** -- LangChain memory, LlamaIndex reader/writer, CrewAI component, OpenAI-compatible adapter
- **CLI** -- Full command-line interface for store, recall, search, forget, consolidate, export/import
- **MCP Tools** -- store, recall, search, consolidate, forget, status, audit, review, wiki-lint, and an explicit code index (index/search/symbol) — exposed via the optional `[mcp]` extra
- **Dual Storage Backends** -- SQLite with keyword search (primary) + YAML (backup) with one-time migration

## Quick Start

```bash
# Core local engine (SQLite + built-in keyword search)
pip install trw-memory

# Recommended hybrid retrieval
pip install "trw-memory[embeddings,vectors,bm25]"

# Retrieval stack + MCP + LLM-assisted consolidation
pip install "trw-memory[all]"
```

By default, memories are stored in `.memory/` relative to the current directory. Override with `MEMORY_STORAGE_PATH` env var.

For source development, clone the repository and run `pip install -e ".[dev]"` from `trw-memory/`.

### Platform notes

- **SQLite driver** — On Linux, `trw-memory` depends on `pysqlite3-binary` and probes the bundled SQLite at runtime for the WAL-reset corruption fix (3.51.3, or a fixed backport). For example, the observed `0.5.4.post2` wheel reports SQLite 3.51.1 and therefore activates the single-connection WAL-checkpoint mitigation; the runtime probe, not the dependency name or package version, is authoritative. `pysqlite3-binary` publishes manylinux wheels only, so **macOS and Windows fall back to the interpreter's stdlib `sqlite3`** through the same driver shim and runtime safety probe.
- **Vector search is optional** — `[vectors]` (sqlite-vec) and `[embeddings]` (sentence-transformers) are optional extras. When they are unavailable the retrieval pipeline degrades gracefully to BM25 and/or the backend's built-in keyword search rather than failing.

### MemoryClient (recommended)

```python
from trw_memory.client import MemoryClient

async with MemoryClient(namespace="project:my-app") as client:
    stored = await client.store(
        "Pydantic v2 requires use_enum_values=True for YAML round-trip",
        tags=["pydantic", "gotcha"],
        importance=0.8,
    )

    # Uses hybrid retrieval when optional rankers are installed.
    results = await client.recall("pydantic serialization", limit=10)
    high_impact = await client.search(min_importance=0.7, tags=["gotcha"])
```

`MemoryClient` also provides `store_many()` and `bulk_store()` for batch writes, `search_fts()` for keyword-only lookup, `forget()` for deletion, and `audit_learning()` / `review_quarantined()` for lifecycle and security workflows.

### Agent Framework Integration

```python
from trw_memory.client import MemoryClient

client = MemoryClient(namespace="project:my-app")

# Register tools with any agent that has register_tool() or tool() API
client.register_tools(agent)

# Or use the auto_recall decorator
@client.auto_recall(query_from="prompt")
async def handle_prompt(prompt: str, recalled_memories: list | None = None) -> str:
    # recalled_memories is automatically injected with relevant context
    recalled_memories = recalled_memories or []
    return f"Found {len(recalled_memories)} relevant memories"
```

### CLI

```bash
# Store a learning
trw-memory store "Always use connection pooling for PostgreSQL" --tags db,performance --importance 0.8

# Recall by query
trw-memory recall "database optimization" --limit 5

# Search with filters
trw-memory search --tags security --min-importance 0.7

# Consolidate related entries
trw-memory consolidate --namespace project:my-app --dry-run

# Export/import for backup or migration
trw-memory export --format json > memories.json
trw-memory import memories.json --namespace project:new-app

# Forget an entry by ID
trw-memory forget M-abc12345 --namespace project:my-app

# Rebuild the SQLite DB from the cold YAML tier or a snapshot
trw-memory restore --from-cold
trw-memory restore --from-snapshot latest

# Snapshot management (VACUUM INTO rotation)
trw-memory snapshot create --tier daily
trw-memory snapshot list
trw-memory snapshot rotate

# Lint wiki page JSON for missing targets/backlinks/provenance
trw-memory wiki-lint pages.json

# Explicit code index: index, lexical search, and symbol lookup
trw-memory code-index ./src
trw-memory code-search ./src "hybrid_search" --language python --limit 5
trw-memory code-symbol ./src MemoryClient

# Status overview
trw-memory status
```

### Low-Level Backend Access

```python
from trw_memory.storage.sqlite_backend import SQLiteBackend
from trw_memory.models.memory import MemoryEntry

backend = SQLiteBackend(db_path=".trw/memory.db")
entry = MemoryEntry(id="M-abc12345", content="...", namespace="default", ...)
backend.store(entry)
results = backend.search("query", top_k=10, namespace="default")
```

## Benchmarks

Every number below is a **same-harness ablation** — retrieval strategies compared on one fixed corpus and query set, each reported with its sample size and, where the claim is comparative, non-overlapping 95% confidence intervals or a paired test. These are **not** leaderboard claims against other systems. Methodology and raw readouts live in the [verification docs](https://trwframework.com/docs/verification).

### Hybrid retrieval beats either ranker alone

On a gold set of real engineering learnings (**n = 889** typed queries), Reciprocal Rank Fusion of BM25 + dense vectors outranks either single ranker — and the same direction replicates on a second, independent benchmark (LongMemEval_S, **n = 500** questions):

| Retriever | Recall@10 | nDCG@10 |
|-----------|:---------:|:-------:|
| BM25 only | 0.869 | 0.771 |
| Vector only | 0.914 | 0.806 |
| **Hybrid (BM25 + vector, RRF)** | **0.938** | **0.839** |

Fusion earns its keep on the hard questions: exact-match queries are near ceiling for every retriever, so the lift concentrates in the temporal / multi-session discrimination band.

### Better recall prevents re-discovery

On TRW's own active learning store (**n = 175** near-duplicate "rediscoveries"), the share of duplicates a recall *would have caught* before re-deriving them — the Preventable Rediscovery Ratio — is far higher for hybrid than for keyword search alone, with **non-overlapping 95% CIs**:

| Retriever | Preventable Rediscovery Ratio (95% CI) |
|-----------|:--------------------------------------:|
| BM25 only | 0.720 [0.649, 0.781] |
| **Hybrid** | **0.943 [0.898, 0.969]** |

Direct evidence that retrieval quality — not just storage — is what keeps an agent from re-deriving what it already knows.

### Knowledge compounding, measured

On a controlled recall-dependent benchmark (H1-MEMORY-BENCH), agents **with** memory solved every task that required recalling a fact established in an earlier session — **58/58** — while agents **without** memory solved **0/50** (the fact is absent by construction). Paired McNemar **p = 3.6×10⁻¹⁵** across **49 matched pairs** (exceeds the pre-registered n ≥ 30), replicated on a second model family.

> **Scope, honestly.** This demonstrates the *mechanism*: cross-session recall lets an agent complete work it otherwise cannot. Whether that compounds into broad, end-to-end coding-task improvement is a separate, still-open question — early SWE-bench single-shot runs (n ≥ 40) produced null. See the [verification docs](https://trwframework.com/docs/verification) for the full evidence posture.

*Throughput (single-run baseline, not CI-backed): sub-millisecond store (p95 ≈ 0.31 ms) and ~116 ms hybrid recall p95 at 1,000 entries; on-disk footprint ≈ 1.2 MB per 1k entries.*

## Architecture

The engine is organized as a set of focused subpackages under `src/trw_memory/`. (For the
authoritative, always-current layout, browse the source tree directly — file-level listings
drift quickly.)

| Path | Responsibility |
|------|---------------|
| `client.py` (+ `_client_*.py`) | `MemoryClient` SDK — the recommended entry point; store/recall/search/forget/bulk + lifecycle/tiering/org-shared helpers |
| `cli.py`, `cli_parser.py`, `cli_*.py` | `trw-memory` command-line interface and its formatters/storage helpers |
| `server.py`, `tools/` | FastMCP server entry point and the MCP tool implementations (optional `[mcp]` extra) |
| `storage/` | SQLite primary backend (WAL, sqlite-vec vectors, snapshots, recovery, resilient fetch) + YAML backend, behind a shared `StorageBackend` interface; `_dbapi.py` driver shim |
| `retrieval/` | BM25 sparse, dense vector, RRF fusion, and the `hybrid_search()` pipeline + admission/source policies and token budgeting |
| `lifecycle/` | Utility scoring (Q-learning, Ebbinghaus decay, Bayesian calibration), semantic dedup, consolidation, anchor validation, and `tiers/` hot/warm/cold management |
| `graph.py` (+ `_graph_*.py`) | Knowledge graph — similarity/tag edges, BFS traversal, clusters, conflicts, cross-project, decay |
| `bandit/` | Bandit selectors (Thompson, contextual, change-detection) for adaptive ranking |
| `code_index/`, `wiki/` | Explicit code index (chunker/indexer/symbols/search) and wiki page indexing + lint |
| `embeddings/` | Embedding provider protocol + local sentence-transformers provider |
| `sync/` | Remote publish/fetch with vector clocks, three-way merge, retry queue, SSE subscriber |
| `security/` | AES-256-GCM field encryption, PII detection/redaction, poisoning/anomaly defense, RBAC, provenance, audit, trust scoring, quarantine |
| `integrations/`, `adapters/` | LangChain / LlamaIndex / CrewAI / VS Code integrations and an OpenAI-compatible adapter |
| `models/`, `namespaces/`, `migration/`, `utils/` | Pydantic models/config, namespace lifecycle + validation + path mapping, YAML→SQLite migration, and shared utilities |

## API Reference

### Key Modules and Functions

| Name | Module | Description |
|------|--------|-------------|
| `MemoryClient` | `client` | High-level async SDK — `store`, `bulk_store`, `store_many`, `recall`, `search`, `search_fts`, `forget`, `audit_learning`, `review_quarantined`, `register_tools`, `auto_recall` |
| `SQLiteBackend` | `storage.sqlite_backend` | Primary storage with keyword search, WAL, and sqlite-vec vectors |
| `YAMLBackend` | `storage.yaml_backend` | File-based storage (backup/migration) |
| `hybrid_search()` | `retrieval.pipeline` | BM25 + dense vector search with RRF fusion |
| `bm25_search()` | `retrieval.bm25` | BM25Okapi sparse keyword retrieval |
| `dense_search()` | `retrieval.dense` | Cosine similarity vector search |
| `rrf_fuse()` | `retrieval.fusion` | Reciprocal Rank Fusion combiner |
| `KnowledgeGraph` functions | `graph` | Tag/similarity edges, BFS traversal, decay |
| `TierSweepResult` | `lifecycle.tiers` | Hot/warm/cold sweep, promote, demote, purge |
| `DedupResult` | `lifecycle.dedup` | Duplicate detection (skip/merge/store decisions) |
| `compute_utility_score()` | `lifecycle.scoring` | Q-learning + Ebbinghaus + Bayesian scoring |
| `MemoryConfig` | `models.config` | Configuration via env vars or dict |
| `MemoryEntry` | `models.memory` | Core data model for stored memories |

### Storage Backends

**SQLite** (recommended) -- Fast, transactional, supports keyword search, knowledge graph edges, and optional sqlite-vec vector similarity:

```python
from trw_memory.storage.sqlite_backend import SQLiteBackend

backend = SQLiteBackend(db_path=".trw/memory.db")
# Supports: store, get, update, delete, search, count, list_entries,
#           list_namespaces, upsert_vector, search_vectors
```

**YAML** -- Human-readable, git-friendly, used as backup during migration:

```python
from trw_memory.storage.yaml_backend import YAMLBackend

backend = YAMLBackend(entries_dir=".trw/learnings")
```

### Hybrid Search: BM25 + Vector

The hybrid search pipeline combines sparse keyword retrieval with dense semantic search — ensuring strong results for both exact-match queries and conceptually similar queries. [Read the full architecture docs](https://trwframework.com/docs).

```
Query --> BM25 (keyword, rank-bm25) --+
                                       +--> RRF Fusion (k, configurable) --> Ranked Results
Query --> Dense (cosine, sqlite-vec) --+
```

The RRF constant `k` is configurable via `MemoryConfig.rrf_k` (env `MEMORY_RRF_K`); the shipped default is tuned by the memory meta-harness loop and may change between releases, so treat the exact value as a default rather than a contract.

The pipeline gracefully degrades: if BM25 is unavailable, only dense search runs (and vice versa). If neither is available, falls back to the storage backend's built-in keyword search (case-insensitive `LIKE` matching).

### Scoring System

Learning utility is computed from multiple signals. [Full scoring documentation](https://trwframework.com/docs):

- **Q-learning**: Exponential moving average updated from outcome events (success/failure/mixed)
- **Ebbinghaus forgetting curve**: Time-based [Ebbinghaus decay](https://trwframework.com/docs) applied at query time (not mutated in storage) — entries naturally fade unless reinforced by recall
- **Access recency boost**: Recently accessed entries score higher
- **Impact score**: Author-assigned importance (0.0-1.0)
- **Bayesian calibration**: MACLA calibration for impact score accuracy

### Tiered Storage

Hot/warm/cold tiering keeps frequently-used memories fast and archives stale ones. [Architecture overview](https://trwframework.com/docs):

| Tier | Criteria | Storage | Latency |
|------|----------|---------|---------|
| Hot | Recently recalled entries | In-memory LRU cache | <1ms |
| Warm | Active entries mirrored into the tier runtime | SQLite + JSONL sidecar with full entry payloads | <50ms |
| Cold | Archived entries matched by recall or explicit sweep policy | YAML archive (partitioned by year/month) | <200ms |

Store/recall operations keep Hot/Warm in sync, Cold-tier hits are promoted back to Warm within the same recall, and `TierManager.sweep()` applies the configurable archive/purge policy when callers trigger a lifecycle sweep.

### Security

| Feature | Implementation |
|---------|---------------|
| Field encryption | AES-256-GCM with HKDF-SHA256 per-namespace key derivation |
| PII detection | Regex patterns (email, phone, SSN, credit card, API keys) + Shannon entropy analysis. Store path **blocks** API-key/token writes and **records** every other detection as metadata — it does not rewrite your stored text. Masking happens at the publish boundary (`strip_pii`), where the local copy still holds the original |
| Poisoning defense | Z-score anomaly detection on frequency, size, and content patterns — **observe mode by default** (records + telemetry, does not quarantine); `enforce` is opt-in |
| Access control | Role-based (admin/editor/viewer) per namespace |
| Audit trail | Append-only security event log |
| Key management | Master key derivation, per-namespace keys, rotation support |

## Telemetry & network behavior

trw-memory is **local-first**: with the default configuration all data lives in a local SQLite store (and an optional YAML sidecar). It makes **no outbound network calls** except the optional embedding-model download below. There is no usage tracking or content phone-home.

### What can touch the network, when, and how to turn it off

| Surface | When | Default | Opt-out / control |
|---------|------|---------|-------------------|
| **Embedding model download** | Only when `all-MiniLM-L6-v2` is **not** already complete in your local Hugging Face cache. A complete cached snapshot makes **zero** huggingface.co requests — the loader probes the cache before deciding, and forces `local_files_only=True` unconditionally when the snapshot is complete (only with the `[embeddings]` extra installed) | enabled when the extra is present | `TRW_OFFLINE=1` / `HF_HUB_OFFLINE=1`, or `local_only: true` (alias `memory_local_only`) — forces `local_files_only` so no download is attempted; a disclosure log line precedes any network-capable load |
| **Remote sync / publish** | Only when `sync_enabled=true` AND `local_only=false` | **off** (`sync_enabled` defaults `false`) | leave sync disabled, or set `local_only: true` to hard-block all egress |

**A warm cache performs no Hub request, and embedding egress is independent of the consent flags.** A fetch is attempted only when the cached snapshot is incomplete or absent **and** no offline switch is engaged; in exactly that case one structured disclosure log names the host and the switch that would block it. `learning_sharing_enabled` and `platform_telemetry_enabled` govern learning-content publishing and usage telemetry respectively — **neither gates the embedding model fetch**. Embedding egress is governed by the local cache, the offline switches, and `local_only`.

`sync_enabled` defaults `false`, so the engine performs no remote sync out of the box even though `local_only` itself defaults `false`. Setting `local_only: true` is the hard-block: an `@model_validator` forces `sync_enabled=False`, clears `sync_namespace`/`platform_url`, and pins `rbac_mode="local"`, so no remote-capable surface can be re-enabled while it is set.

With an offline switch engaged (`TRW_OFFLINE` / `HF_HUB_OFFLINE`) **or** `local_only: true`, the standalone engine loads the embedding model with `local_files_only=True`; if the model is not already cached it raises a clear `LocalOnlyViolationError` telling you how to pre-download (this is the behaviour in both cases — `local.py` does **not** silently fall back to keyword-only recall here). The graceful "degrade to keyword-only, no crash" path is provided one layer up by trw-mcp's embedder wrapper, which catches that error; a direct trw-memory caller that wants keyword-only recall under an offline switch should pre-download the model or run without the `[embeddings]` extra installed.

### Environment-variable inventory

| Variable | Purpose | Default |
|----------|---------|---------|
| `TRW_OFFLINE` | Master offline switch — blocks the huggingface.co embedding-model download | unset |
| `HF_HUB_OFFLINE` | Upstream huggingface_hub offline switch — also honored | unset |
| `MEMORY_*` | Engine knobs validated by `MemoryConfig` (e.g. `MEMORY_LOCAL_ONLY`, `MEMORY_EMBEDDING_TRUST_REMOTE_CODE`, retrieval + lifecycle tuning) | per-field |

### Security defaults

| Capability | Default | Notes |
|-----------|---------|-------|
| Field-level encryption | **off** (`encryption_enabled=False`) | opt-in (AES-256-GCM per-namespace keys) |
| PII detection | **on** (`pii_enabled=True`) | always scans `content`/`detail`/`tags`/`evidence[]`/`Assertion.last_evidence` on the store path; the configurable `pii_action` default is `warn` for the public `check_entry_pii` helper. On the runtime store path, detected **API keys / tokens block the write** (`PIIBlockError`); every other type is recorded in the `pii_types` metadata and stored **verbatim** — heuristic detectors do not get to irreversibly rewrite local text. Emails, IPs, SSNs, phone numbers and credit-card shapes are masked at the publish boundary instead. Set `pii_custom_patterns` to opt in to local masking with your own regexes |
| Poisoning / size-anomaly detection | **observe** (`poisoning_detection_mode="observe"`) | the SEC-001 statistical size/tag-count detector records anomaly stats + telemetry but does **not** quarantine by default; `enforce` is opt-in. There is no per-source exemption: a caller-supplied `metadata['source']` cannot skip enforce-mode quarantine |
| Trust scoring | **observe** (`trust_scoring_mode="observe"`) | logs intake trust decisions; `enforce`/`strict` are opt-in |
| Provenance signing | **required** (`provenance_required=True`) | persisted rows carry a signed provenance hash-chain |
| Canary tamper response | **halt** (`canary_fail_mode="halt"`) | seeded canaries are probed on recall; tamper detection halts by default (`degrade`/`log-only` opt-in) |
| Remote sync / publishing | **off** (`sync_enabled=False`) | no remote sync out of the box; `local_only=True` hard-blocks it via a validator |
| Model remote-code execution | **off** (`embedding_trust_remote_code=False`) | the ONLY input to sentence-transformers' `trust_remote_code`. Left False, a model repository that ships its own Python modules is refused with `RemoteCodeNotPermittedError` naming this field; set it `true` only for a repository you trust, because its code then runs with your process's privileges. The shipped default model needs no remote code, so the secure default is also the working default |
| `memory.db` permissions | `0600` | the file-backed store is `chmod 0600` (owner-only) on creation; a non-POSIX platform degrades to a `db_chmod_failed` warning |

### Enterprise hardening recipe

```bash
export TRW_OFFLINE=1   # block the huggingface.co model download (local_files_only)
```

```yaml
# MemoryConfig
local_only: true       # hard-block all remote sync + model download
```

With either switch set, pre-download the embedding model (`python -m sentence_transformers download all-MiniLM-L6-v2`) if you want hybrid recall — otherwise the first embedding load raises `LocalOnlyViolationError`. To run keyword-only without that error, omit the `[embeddings]` extra entirely. Verify the on-disk `memory.db` is mode `0600` and that no outbound connection is attempted on first use.

### MCP Tools

When installed with `[mcp]` extra:

```bash
trw-memory-server  # Starts MCP server (stdio transport)
```

| Tool | Purpose |
|------|---------|
| `memory_store` | Store entry with optional embedding/vector persistence |
| `memory_recall` | Hybrid retrieval with optional graph traversal |
| `memory_search` | Filter-based listing (tags, importance, date range) |
| `memory_forget` | Delete entries by ID or bulk search query |
| `memory_consolidate` | Trigger episodic-to-semantic consolidation |
| `memory_status` | Backend stats, entry counts, tier distribution |
| `memory_audit` | Provenance + lifecycle audit data for one entry |
| `memory_review` | Approve/reject a quarantined entry |
| `memory_wiki_lint` | Lint wiki pages for missing targets, backlinks, provenance gaps |
| `memory_code_index` | Index source code into the explicit code index |
| `memory_code_search` | Lexical search over indexed code chunks |
| `memory_code_symbol` | Look up symbols in the explicit code index |

## Integration with trw-mcp

[trw-mcp](https://github.com/wallter/trw-mcp) is the MCP server layer of [TRW Framework](https://trwframework.com) — it exposes a suite of tools, skills, and agents to Claude Code and other AI coding tools (see the [trw-mcp README](https://github.com/wallter/trw-mcp) for current counts). trw-memory serves as its memory backend:

- `trw_learn` delegates to `SQLiteBackend.store()` via `memory_adapter.py` (YAML dual-write as backup)
- `trw_recall` delegates to `SQLiteBackend.search()` / `list_entries()` as the sole query path
- Scoring functions (`compute_utility_score`, `update_q_value`, `apply_time_decay`, `bayesian_calibrate`) are canonical in trw-memory and re-exported by trw-mcp
- One-time YAML-to-SQLite migration runs automatically on first access
- Optional vector search via `LocalEmbeddingProvider` + `rrf_fuse` when `sentence-transformers` is installed

[Read more about the full TRW Framework architecture](https://trwframework.com/docs).

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run full test suite (>=85% coverage required — see fail_under in pyproject.toml)
python -m pytest tests/ -v --cov=trw_memory --cov-report=term-missing

# Type checking (mypy --strict across the package)
python -m mypy --strict src/trw_memory/

# Targeted testing
python -m pytest tests/test_client_*.py -v
python -m pytest tests/test_retrieval_*.py -v
python -m pytest tests/test_storage_sqlite_*.py -v
```

**Quality bar**: a broad pytest suite, mypy `--strict` clean, and a coverage floor of 85% (`fail_under` in `pyproject.toml`).

### Optional Dependencies

| Extra | Packages | Purpose |
|-------|----------|---------|
| `[mcp]` | fastmcp | MCP server tools |
| `[encryption]` | sqlcipher3, keyring, cryptography | Encrypted-at-rest DB (SQLCipher) + key storage |
| `[embeddings]` | sentence-transformers | Dense vector embeddings (all-MiniLM-L6-v2, 384-dim) |
| `[vectors]` | sqlite-vec | Vector similarity search in SQLite |
| `[bm25]` | rank-bm25 | BM25 keyword search |
| `[llm]` | anthropic | LLM-augmented consolidation |
| `[langchain]` | langchain-core | LangChain memory integration |
| `[llamaindex]` | llama-index-core | LlamaIndex reader/writer |
| `[crewai]` | crewai | CrewAI memory component |
| `[all-integrations]` | langchain + llamaindex + crewai | All framework integrations |
| `[all]` | mcp + embeddings + vectors + bm25 + llm | Retrieval stack, MCP server, and LLM consolidation |
| `[dev]` | pytest, mypy, ruff, coverage, pip-audit, vulture, deptry | Testing and linting |

### Entry Points

| Command | Purpose |
|---------|---------|
| `trw-memory` | CLI for store/recall/search/forget/consolidate/export/import, plus restore, snapshot (create/list/rotate), wiki-lint, and code-index/code-search/code-symbol |
| `trw-memory-server` | MCP server (stdio transport) |

## License

[Business Source License 1.1](https://trwframework.com/license) -- source-available, free for non-competing use. Converts to Apache 2.0 on 2030-03-21.

---

Built by [Tyler Wall](http://tylerrwall.com) · [TRW Framework](https://trwframework.com) · [Documentation](https://trwframework.com/docs) · [License](https://trwframework.com/license)
