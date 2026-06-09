# trw-memory

**AI agent memory engine** — persistent memory for AI agents with hybrid search (BM25 + vectors), Q-learning scoring, Ebbinghaus decay curves, tiered storage, and knowledge graph. The standalone memory backend powering [TRW Framework](https://trwframework.com).

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)
[![License: BSL 1.1](https://img.shields.io/badge/License-BSL_1.1-orange.svg)](https://trwframework.com/license)
[![Docs](https://img.shields.io/badge/docs-trwframework.com-blue)](https://trwframework.com/docs)

## Part of TRW Framework

trw-memory is the standalone memory engine for [TRW (The Real Work)](https://trwframework.com) — a methodology layer for AI-assisted development that provides stateless agents with a persistent memory layer **designed to enable self-improvement across sessions** via [knowledge compounding](https://trwframework.com/docs). *The outcome effect of cross-session memory on coding tasks is an open empirical question; iter-9/10 produced null on SWE-bench-single-shot at n≥40. See [docs/eval/iter-notes/iter-11-prospector-analysis.md](https://github.com/wallter/trw-framework/blob/main/docs/eval/iter-notes/iter-11-prospector-analysis.md).* It works alongside [trw-mcp](https://github.com/wallter/trw-mcp), the MCP server that builds its tooling on this engine.

- **trw-memory** (this repo): Standalone AI agent memory engine with hybrid retrieval, scoring, and lifecycle
- **trw-mcp**: MCP server for AI coding agents — uses trw-memory as its backend

## What It Does

TRW-Memory is a standalone **persistent memory engine for AI agents** that gives coding agents searchable, long-lived knowledge storage. It stores learnings (patterns, gotchas, architecture decisions) in SQLite with optional YAML backup, and retrieves them using [hybrid search](https://trwframework.com/docs) that combines keyword matching (BM25) with dense vector similarity.

Designed as the storage backend for [trw-mcp](https://github.com/wallter/trw-mcp) and [TRW Framework](https://trwframework.com), but usable independently by any AI agent framework that needs persistent memory with recall.

## Features

- **MemoryClient SDK** -- High-level async Python client with store/bulk_store/recall/forget/search plus audit_learning and review_quarantined
- **Hybrid Search (BM25 + vector)** -- BM25 keyword matching + dense vector similarity via sqlite-vec, combined with Reciprocal Rank Fusion (RRF). [Learn more](https://trwframework.com/docs)
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
# Install from PyPI
pip install trw-memory

# Or install from source
git clone https://github.com/wallter/trw-memory.git
cd trw-memory
python -m venv .venv && source ../.venv/bin/activate
pip install -e ".[dev]"

# With all optional features (embeddings, vectors, BM25, LLM)
pip install -e ".[all]"
```

By default, memories are stored in `.memory/` relative to the current directory. Override with `MEMORY_STORAGE_PATH` env var.

### Platform notes

- **SQLite driver** — On Linux, `trw-memory` depends on `pysqlite3-binary`, which bundles a recent SQLite (≥3.51) wheel that includes the WAL-reset corruption fix. `pysqlite3-binary` publishes manylinux wheels only, so **macOS and Windows fall back to the interpreter's stdlib `sqlite3`** via the `storage/_dbapi.py` driver shim. Where the bundled SQLite predates the fix, a single-connection WAL-checkpoint window mitigates the concurrent-writer corruption path.
- **Vector search is optional** — `[vectors]` (sqlite-vec) and `[embeddings]` (sentence-transformers) are optional extras. When they are unavailable the retrieval pipeline degrades gracefully to BM25 and/or the backend's built-in keyword search rather than failing.

### MemoryClient (recommended)

```python
from trw_memory.client import MemoryClient

async with MemoryClient(namespace="project:my-app") as client:
    # Store a learning
    await client.store(
        "Pydantic v2 requires use_enum_values=True for YAML round-trip",
        tags=["pydantic", "gotcha"],
        importance=0.8,
    )

    # Recall by keyword query (hybrid BM25 + vector search)
    results = await client.recall("pydantic serialization", limit=10)

    # Search with filters
    high_impact = await client.search(min_importance=0.7, tags=["gotcha"])

    # Forget an entry
    await client.forget(results[0]["memory_id"])

    # Store many entries in one call
    await client.bulk_store([
        {"content": "Use BEGIN IMMEDIATE for write transactions", "tags": ["sqlite"]},
        {"content": "RRF k=60 is the default fusion constant", "tags": ["retrieval"]},
    ])

    # Inspect provenance/lifecycle for one entry
    audit = await client.audit_learning(results[0]["memory_id"])

    # Review entries quarantined by the poisoning/PII defenses
    quarantined = await client.review_quarantined()
```

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
| `MemoryClient` | `client` | High-level async SDK — `store`, `bulk_store`, `recall`, `search`, `forget`, `audit_learning`, `review_quarantined`, `register_tools`, `auto_recall` |
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
                                       +--> RRF Fusion (k=60) --> Ranked Results
Query --> Dense (cosine, sqlite-vec) --+
```

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
| PII detection | Regex patterns (email, phone, SSN, credit card, API keys) + Shannon entropy analysis |
| Poisoning defense | Z-score anomaly detection on frequency, size, and content patterns |
| Access control | Role-based (admin/editor/viewer) per namespace |
| Audit trail | Append-only security event log |
| Key management | Master key derivation, per-namespace keys, rotation support |

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
../.venv/bin/python -m pytest tests/ -v --cov=trw_memory --cov-report=term-missing

# Type checking (mypy --strict across the package)
../.venv/bin/python -m mypy --strict src/trw_memory/

# Targeted testing
../.venv/bin/python -m pytest tests/test_client.py -v
../.venv/bin/python -m pytest tests/test_retrieval_*.py -v
../.venv/bin/python -m pytest tests/test_storage_sqlite.py -v
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
| `[all]` | mcp + embeddings + vectors + bm25 + llm | Full feature set |
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
