# trw-memory

**AI agent memory engine** — persistent memory for AI agents with hybrid search (BM25 + vectors), Q-learning scoring, Ebbinghaus decay curves, tiered storage, and knowledge graph. The standalone memory backend powering [TRW Framework](https://trwframework.com).

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)
[![License: BSL 1.1](https://img.shields.io/badge/License-BSL_1.1-orange.svg)](https://trwframework.com/license)
[![Docs](https://img.shields.io/badge/docs-trwframework.com-blue)](https://trwframework.com/docs)

## Part of TRW Framework

trw-memory is the standalone memory engine for [TRW (The Real Work)](https://trwframework.com) — a methodology layer for AI-assisted development that provides stateless agents with a persistent memory layer **designed to enable self-improvement across sessions** via [knowledge compounding](https://trwframework.com/docs). *The outcome effect of cross-session memory on coding tasks is an open empirical question; iter-9/10 produced null on SWE-bench-single-shot at n≥40. See [docs/eval/iter-notes/iter-11-prospector-analysis.md](https://github.com/wallter/trw-framework/blob/main/docs/eval/iter-notes/iter-11-prospector-analysis.md).* It works alongside [trw-mcp](https://github.com/wallter/trw-mcp), the MCP server that provides 24 tools built on this engine.

- **trw-memory** (this repo): Standalone AI agent memory engine with hybrid retrieval, scoring, and lifecycle
- **trw-mcp**: MCP server with 25 tools, 24 skills, 12 agents — uses trw-memory as its backend

## What It Does

TRW-Memory is a standalone **persistent memory engine for AI agents** that gives coding agents searchable, long-lived knowledge storage. It stores learnings (patterns, gotchas, architecture decisions) in SQLite with optional YAML backup, and retrieves them using [hybrid search](https://trwframework.com/docs) that combines keyword matching (BM25) with dense vector similarity.

Designed as the storage backend for [trw-mcp](https://github.com/wallter/trw-mcp) and [TRW Framework](https://trwframework.com), but usable independently by any AI agent framework that needs persistent memory with recall.

## Features

- **MemoryClient SDK** -- High-level async Python client with store/recall/forget/search
- **Hybrid Search (BM25 + vector)** -- BM25 keyword matching + dense vector similarity via sqlite-vec, combined with Reciprocal Rank Fusion (RRF). [Learn more](https://trwframework.com/docs)
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
- **MCP Tools** -- 6 tools for store, recall, search, consolidate, forget, and status
- **Dual Storage Backends** -- SQLite with keyword search (primary) + YAML (backup) with one-time migration

## Quick Start

```bash
# Install from PyPI
pip install trw-memory

# Or install from source
git clone https://github.com/wallter/trw-memory.git
cd trw-memory
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# With all optional features (embeddings, vectors, BM25, LLM)
pip install -e ".[all]"
```

By default, memories are stored in `.memory/` relative to the current directory. Override with `MEMORY_STORAGE_PATH` env var.

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
```

### Agent Framework Integration

```python
from trw_memory.client import MemoryClient

client = MemoryClient(namespace="project:my-app")

# Register tools with any agent that has register_tool() or tool() API
client.register_tools(agent)

# Or use the auto_recall decorator
@client.auto_recall(query_from="prompt")
async def handle_prompt(prompt: str, recalled_memories: list = []) -> str:
    # recalled_memories is automatically injected with relevant context
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

```
src/trw_memory/
  client.py              # MemoryClient SDK (recommended entry point)
  cli.py                 # CLI entry point (trw-memory command)
  cli_parser.py          # Argument parsing for CLI
  cli_formatters.py      # Output formatting for CLI
  decorators.py          # @auto_recall decorator
  exceptions.py          # Custom exception hierarchy
  graph.py               # Knowledge graph (similarity/tag/consolidation edges, BFS)
  namespace.py           # Namespace validation
  server.py              # FastMCP MCP server entry point
  storage/
    sqlite_backend.py    # Primary: SQLite with keyword search, WAL mode, sqlite-vec vectors
    yaml_backend.py      # Legacy: per-entry YAML files (backup/migration)
    interface.py         # Abstract StorageBackend protocol
    persistence.py       # Atomic read/write helpers
    _schema.py           # DDL schema definitions
    _row_mapper.py       # Row-to-model mapping
    _shared.py           # Shared storage utilities
    _parsing.py          # Query/filter parsing helpers
  retrieval/
    bm25.py              # BM25Okapi sparse retrieval (rank-bm25)
    dense.py             # Cosine similarity dense vector search
    fusion.py            # Reciprocal Rank Fusion (RRF, k=60)
    pipeline.py          # hybrid_search() orchestrator (BM25 + dense + RRF)
  embeddings/
    interface.py         # Embedding provider protocol
    local.py             # Local sentence-transformers provider
  lifecycle/
    scoring.py           # Q-learning, Ebbinghaus decay, Bayesian calibration
    dedup.py             # Semantic dedup (cosine threshold, merge/skip logic)
    consolidation.py     # Consolidation clustering with fallback summarization
    verification.py      # Entry verification utilities
    _utils.py            # Shared lifecycle helpers
    tiers/               # Hot/warm/cold tier management
      _manager.py        # Tier assignment and transitions
      _sweep.py          # Periodic sweep logic (promote/demote/purge)
      _scoring.py        # Tier-specific scoring
      _warm.py           # Warm tier operations
      _cold.py           # Cold tier archival
  sync/
    remote.py            # Publish/fetch to platform backend
    conflict.py          # Vector clock comparison + three-way merge
    retry_queue.py       # Persistent JSONL retry queue
    subscriber.py        # SSE subscriber for live updates
  security/
    encryption.py        # AES-256-GCM with HKDF per-namespace keys
    pii.py               # PII detection (email, phone, SSN, API keys, entropy)
    poisoning.py         # Anomaly detection (z-score frequency/size/pattern)
    audit.py             # Append-only security event audit trail
    rbac.py              # Role-based access control for namespaces
    keys.py              # Master key derivation and rotation
  tools/                 # 6 MCP tool implementations
    store.py             # memory_store (validate + write + optional vector persistence)
    recall.py            # memory_recall (hybrid search + graph traversal)
    search.py            # memory_search (filter-based listing)
    forget.py            # memory_forget (namespace-scoped deletion)
    consolidate.py       # memory_consolidate (trigger consolidation)
    status.py            # memory_status (backend stats + tier info)
  integrations/
    langchain.py         # LangChain memory class
    llamaindex.py        # LlamaIndex reader/writer
    crewai.py            # CrewAI memory component
    factory.py           # Auto-detect framework and create adapter
    vscode.py            # VS Code extension adapter
  adapters/
    openai_compat.py     # OpenAI Memory API compatible endpoint
  models/
    config.py            # MemoryConfig (pydantic-settings)
    memory.py            # MemoryEntry, MemoryStatus
    events.py            # Audit event models
  namespaces/
    manager.py           # Namespace lifecycle (create, expire, list)
    validation.py        # Namespace format validation
    path_mapping.py      # Namespace-to-path resolution
  migration/
    from_trw.py          # YAML-to-SQLite migration from trw-mcp format
```

**92 source files, ~19,900 lines of code**

## API Reference

### Key Modules and Functions

| Name | Module | Description |
|------|--------|-------------|
| `MemoryClient` | `client` | High-level async SDK (store/recall/forget/search) |
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
| `memory_forget` | Namespace-scoped deletion |
| `memory_consolidate` | Trigger episodic-to-semantic consolidation |
| `memory_status` | Backend stats, entry counts, tier distribution |

## Integration with trw-mcp

[trw-mcp](https://github.com/wallter/trw-mcp) is the MCP server layer of [TRW Framework](https://trwframework.com) — it exposes 24 tools, 24 skills, and 18 agents to Claude Code and other AI coding tools. trw-memory serves as its memory backend:

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

# Run full test suite (2,050+ tests, >=85% coverage required)
.venv/bin/python -m pytest tests/ -v --cov=trw_memory --cov-report=term-missing

# Type checking (strict mode, 81 files)
.venv/bin/python -m mypy --strict src/trw_memory/

# Targeted testing
.venv/bin/python -m pytest tests/test_client.py -v
.venv/bin/python -m pytest tests/test_retrieval.py -v
.venv/bin/python -m pytest tests/test_storage_sqlite.py -v
```

**Current metrics**: 2,050+ tests, ~90% coverage, mypy strict clean.

### Optional Dependencies

| Extra | Packages | Purpose |
|-------|----------|---------|
| `[mcp]` | fastmcp | MCP server tools |
| `[embeddings]` | sentence-transformers | Dense vector embeddings (all-MiniLM-L6-v2, 384-dim) |
| `[vectors]` | sqlite-vec | Vector similarity search in SQLite |
| `[bm25]` | rank-bm25 | BM25 keyword search |
| `[llm]` | anthropic | LLM-augmented consolidation |
| `[langchain]` | langchain-core | LangChain memory integration |
| `[llamaindex]` | llama-index-core | LlamaIndex reader/writer |
| `[crewai]` | crewai | CrewAI memory component |
| `[all]` | mcp + embeddings + vectors + bm25 + llm | Full feature set |
| `[dev]` | pytest, mypy, ruff, coverage, etc. | Testing and linting |

### Entry Points

| Command | Purpose |
|---------|---------|
| `trw-memory` | CLI for store/recall/search/forget/consolidate/export/import |
| `trw-memory-server` | MCP server (stdio transport) |

## License

[Business Source License 1.1](https://trwframework.com/license) -- source-available, free for non-competing use. Converts to Apache 2.0 on 2030-03-21.

---

Built by [Tyler Wall](http://tylerrwall.com) · [TRW Framework](https://trwframework.com) · [Documentation](https://trwframework.com/docs) · [License](https://trwframework.com/license)
