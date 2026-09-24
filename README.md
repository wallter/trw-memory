# trw-memory: persistent, local-first memory for AI agents

**trw-memory is a persistent memory engine for AI agents**: an agent memory layer that gives LLM agents long-term memory across sessions, stored locally in SQLite. Use it as an async Python SDK, a CLI, or an MCP memory server. The core install recalls with keyword search; optional extras add hybrid retrieval (BM25 + dense vectors via sqlite-vec, fused with Reciprocal Rank Fusion) and cross-encoder reranking, alongside lifecycle scoring, tiered storage, and a knowledge graph. It is the standalone memory backend of [TRW Framework](https://trwframework.com) and works without it.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)
[![License: BSL 1.1](https://img.shields.io/badge/License-BSL_1.1-orange.svg)](https://trwframework.com/license)
[![Docs](https://img.shields.io/badge/docs-trwframework.com-blue)](https://trwframework.com/docs)

> **Release status:** Alpha and source-available under BSL 1.1. The public API may change before 1.0; evaluate upgrades in a test environment before production rollout.

**[Why trw-memory](#why-trw-memory)** · **[Quick start](#install-and-quick-start)** · **[Python API](#memoryclient-recommended)** · **[Conversation memory](#conversation-memory-without-an-llm-call-at-write-time)** · **[CLI](#cli)** · **[Benchmarks](#benchmarks)** · **[mem0 comparison](#single-conversation-comparison-with-mem0-oss-using-mem0s-evaluation-suite)** · **[MCP server](#mcp-memory-server)** · **[Security and network behavior](#telemetry-and-network-behavior)** · **[FAQ](#faq)** · **[Development](#development)**

## What is trw-memory?

TRW-Memory is a standalone **persistent memory engine for AI agents** that gives coding agents searchable, long-lived knowledge storage. It stores learnings (patterns, gotchas, architecture decisions) in SQLite with optional YAML backup, and retrieves them using [hybrid search](https://trwframework.com/docs) that combines keyword matching (BM25) with dense vector similarity. It also stores conversation memory: `store_conversation()` keeps chat turns verbatim so they can be recalled later, RAG-style, as evidence for an answer.

Designed as the storage backend for [trw-mcp](https://github.com/wallter/trw-mcp) and [TRW Framework](https://trwframework.com), but usable independently by any AI agent framework that needs persistent memory with recall.

## Why trw-memory

- **Local-first.** With the default configuration all data lives in a local SQLite store (plus an optional YAML sidecar). There is no usage tracking or content phone-home; the only network-capable surfaces are optional model downloads and opt-in remote sync. See [Telemetry and network behavior](#telemetry-and-network-behavior).
- **No generative LLM call at write time.** `store_conversation()` stores every turn verbatim with its date and the turn it replied to. Ingest never calls a generative model (the optional local embedding model still encodes each turn); recall does the work.
- **Hybrid retrieval, optional.** With the retrieval extras installed: BM25 (with stemming) + dense vectors via sqlite-vec + Reciprocal Rank Fusion + a cross-encoder re-ranker. Without them, retrieval degrades gracefully to the backend's built-in keyword search.
- **Measured on public benchmarks.** With no LLM in the loop, the gold evidence turn lands in the top 10 for **85.7%** of [LOCOMO](#benchmarks) questions (n = 1,540) and **93.8%** of LongMemEval_S questions (n = 470); top 50: 92.7% and 97.4%. Write and recall costs stay flat as the store and the number of projects sharing it grow. See [Benchmarks](#benchmarks) for method and caveats.
- **Works offline.** `TRW_OFFLINE=1` / `HF_HUB_OFFLINE=1` block model downloads; `local_only: true` hard-blocks all remote sync and model download. Hybrid retrieval offline needs the models already in the local cache.
- **MCP memory server included.** `trw-memory-server` exposes store, recall, search, consolidate, forget and more as MCP tools over stdio, or over a per-user loopback HTTP daemon.
- **Evaluating a mem0 alternative?** Using mem0's open-source evaluation suite on one LOCOMO conversation (n = 152 questions per system, one run each, local `llama3.1` as answerer, judge and mem0's extraction model), neither paired test detected a statistically significant accuracy difference between trw-memory and mem0 (OSS), and trw-memory called no generative LLM to ingest the conversation. That does not establish equivalence or superiority; read the [numbers and caveats](#single-conversation-comparison-with-mem0-oss-using-mem0s-evaluation-suite) first.
- **Source-available.** BSL 1.1, alpha; the public API may change before 1.0.

## Features: hybrid retrieval, knowledge graph, lifecycle, security

- **MemoryClient SDK** -- High-level async Python client with store/bulk_store/store_many/recall/search/search_fts/forget plus audit_learning and review_quarantined
- **Hybrid Search (BM25 + vector + re-ranker)** -- BM25 keyword matching + dense vector similarity via sqlite-vec, combined with Reciprocal Rank Fusion (RRF), then a cross-encoder re-rank with a confidence floor that scales with the requested limit. [Learn more](https://trwframework.com/docs)
- **FTS5 keyword search** -- `MemoryClient.search_fts()` runs indexed SQLite FTS5 keyword search with BM25 ranking over content/detail/tags for pure-keyword queries that don't need hybrid ranking; degrades to an empty result when FTS5 is unavailable
- **Hybrid order preservation by default** -- recall preserves the hybrid BM25+dense+RRF order when enough local candidates are already available, avoiding a legacy score-scale mismatch in tier merging. To restore the legacy tier rescore for a workload, set `MEMORY_RECALL_PRESERVE_HYBRID_ORDER=false`.
- **Tiered Storage** -- Hot/warm/cold tiers for fast recall, warm-sidecar persistence, recall-time cold promotion, and explicit sweep-based archiving/purging. [Architecture details](https://trwframework.com/docs)
- **Semantic Deduplication** -- Detects and merges near-duplicate learnings using cosine similarity (0.85 threshold)
- **Knowledge Graph for AI** -- Tag co-occurrence and similarity edges, BFS traversal, importance boost/decay, cross-validation propagation. [Docs](https://trwframework.com/docs)
- **Memory Consolidation** -- Episodic-to-semantic consolidation via clustering with the current shipped path using heuristic/fallback summarization
- **Utility scoring** -- author-assigned impact with an [Ebbinghaus forgetting curve](https://trwframework.com/docs) applied at query time, boosted by recurrence and access
- **Remote Sync** -- Publish/fetch learnings across installations with vector clock conflict resolution and SSE live updates
- **Security** -- optional SQLCipher whole-database encryption at rest (AES-256-CBC, off by default), PII detection with publish-time masking, memory-poisoning anomaly detection (z-score; enforcement is opt-in), RBAC, audit trail. See [Security defaults](#security-defaults)
- **Agent Integration** -- `register_tools()` for agents that expose a `register_tool()` or `tool()` API, `@auto_recall` decorator
- **CLI** -- Full command-line interface for store, recall, search, forget, consolidate, export/import
- **MCP Tools** -- store, recall, search, consolidate, forget, status, audit, review, wiki-lint, and an explicit code index (index/search/symbol) — served by `trw-memory-server`
- **Dual Storage Backends** -- SQLite with keyword search (primary) + YAML (backup) with one-time migration

## How trw-memory fits into TRW Framework

trw-memory is the standalone memory engine for [TRW (The Real Work)](https://trwframework.com) — a methodology layer for AI-assisted development that provides stateless agents with a persistent memory layer **designed to enable self-improvement across sessions** via [knowledge compounding](https://trwframework.com/docs). *Cross-session recall is measured to let an agent finish work it otherwise cannot; broad coding-task lift is not established, and the measured SWE-bench Verified result is unfavourable (56 vs 79 of 112 paired-valid problems). See the [verification docs](https://trwframework.com/docs/verification) for the current methodology and evidence posture.* It works alongside [trw-mcp](https://github.com/wallter/trw-mcp), the MCP server that builds its tooling on this engine.

- **trw-memory** (this repo): Standalone AI agent memory engine with hybrid retrieval, scoring, and lifecycle
- **trw-mcp**: MCP server for AI coding agents — uses trw-memory as its backend

## Install and quick start

```bash
# Core local engine (SQLite + built-in keyword search)
pip install trw-memory

# Recommended hybrid retrieval
pip install "trw-memory[embeddings,vectors,bm25]"

# The full retrieval stack (same as the line above, one name)
pip install "trw-memory[all]"
```

By default, memories are stored in `.memory/` relative to the current directory. Override with `MEMORY_STORAGE_PATH` env var.

For source development, clone the repository and run `pip install -e ".[dev]"` from `trw-memory/`. Tested on CPython 3.10 through 3.14; see [Platform and interpreter notes](#platform-and-interpreter-notes) for SQLite engine details.

### MemoryClient (recommended)

```python
import asyncio

from trw_memory.client import MemoryClient


async def main() -> None:
    async with MemoryClient(namespace="project:my-app") as client:
        await client.store(
            "Pydantic v2 requires use_enum_values=True for YAML round-trip",
            tags=["pydantic", "gotcha"],
            importance=0.8,
        )

        # Uses hybrid retrieval when the optional rankers are installed.
        results = await client.recall("pydantic serialization", limit=10)
        high_impact = await client.search(min_importance=0.7, tags=["gotcha"])
        print(results, high_impact)


asyncio.run(main())
```

`MemoryClient` also provides `store_many()` and `bulk_store()` for batch writes, `search_fts()` for keyword-only lookup, `forget()` for deletion, and `audit_learning()` / `review_quarantined()` for lifecycle and security workflows.

### Conversation memory without an LLM call at write time

```python
# inside `async with MemoryClient(...) as client:`
turns = [
    {"role": "user", "speaker": "Caroline", "content": "I went to a LGBTQ support group yesterday."},
    {"role": "assistant", "speaker": "Melanie", "content": "That's great! What did it look like?"},
]
summary = await client.store_conversation(turns, observed_at="2023-05-08T13:56:00+00:00", session_id="s1")
rows = await client.recall("what did the support group look like", limit=5)
```

`store_conversation()` stores every turn verbatim and carries the preceding
`context_turns` (default 1) of the same conversation alongside it, so a reply
like "What did it look like?" is retrievable by what it was replying to. No
generative LLM is called at ingest time (the optional local embedding model
still encodes each turn): the raw turn, its date and its neighbourhood are
the evidence, and the reader does the inference at recall time. Feeding a
conversation in chunks? Pass the last turns you already stored as
`preceding=`.

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

# Export/import a namespace's entry data
trw-memory export --format json > memories.json
trw-memory import memories.json --namespace project:new-app

# Forget an entry by ID
trw-memory forget M-abc12345 --namespace project:my-app

# Re-encode stored vectors after an embedding-model change (idempotent, resumable)
trw-memory reembed --namespace project:my-app

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

Export enumerates the requested namespace (`default` unless specified), not the
whole project or every namespace. Use an unchanged store for a consistent export:
pagination is not a snapshot across concurrent writes. JSON/YAML output retains
the entry format and is materialized in memory; it is not a streaming database
backup and does not include arbitrary project files or stored vector indexes.
The separate snapshot commands above serve database snapshot management.

### Low-Level Backend Access

```python
from trw_memory.storage.sqlite_backend import SQLiteBackend
from trw_memory.models.memory import MemoryEntry

backend = SQLiteBackend(db_path=".trw/memory.db")
entry = MemoryEntry(id="M-abc12345", content="Use WAL mode for concurrent readers", namespace="default")
backend.store(entry)
results = backend.search("query", top_k=10, namespace="default")
```

## Benchmarks

Every number below is reported with its sample size; confidence intervals and paired tests are given where they were computed. Apart from the mem0 comparison, these are **same-harness ablations** — retrieval strategies compared on one fixed corpus and query set — not leaderboard claims against other systems. The framework's evidence posture is described in the [verification docs](https://trwframework.com/docs/verification); the mem0 comparison's method and scripts live in [`benchmarks/locomo/`](https://github.com/wallter/trw-memory/blob/main/benchmarks/locomo/README.md).

### Single-conversation comparison with mem0 (OSS), using mem0's evaluation suite

> **Scope first.** One [LOCOMO](https://github.com/snap-research/locomo) conversation of ten, one run per system. The answerer, the judge and mem0's extraction model were all a local 8B `llama3.1`; that judge was not calibrated against the GPT-class judges behind mem0's published numbers, so compare the two columns with each other, not with mem0's website. mem0 was run as its open-source SDK (`mem0ai` 2.0.20), not Mem0 Cloud. Results apply to these configurations only.

We ran [mem0's open-source evaluation suite](https://github.com/mem0ai/memory-benchmarks) (commit `4b61c5d`) **unmodified** against both systems: same dataset parsing, same answer prompt, same LLM judge, same cutoffs, and the same embedding model (`all-MiniLM-L6-v2`) for both. Conversation 0, **n = 152** questions per system, paired by question, trw-memory 0.19 defaults:

| Memories given to the answerer | mem0 (OSS) | trw-memory | McNemar p |
|--------------------------------|:----------:|:----------:|:---------:|
| top 10 | 88.2% [82.1, 92.4] | 91.4% [85.9, 94.9] | 0.38 |
| top 50 | 92.1% [86.7, 95.4] | 91.4% [85.9, 94.9] | 1.00 |

Neither paired test detected a statistically significant accuracy difference. That does **not** establish equivalence or superiority; it means this sample could not tell the two apart.

Ingestion measurements for the same run (419 turns, one machine, single run):

| | mem0 (OSS) | trw-memory |
|---|---|---|
| Generative LLM calls during ingestion | ~2 per turn | none |
| Ingestion wall-clock time | 1 h 28 min | ~75 s |
| Storage approach | LLM-extracted facts | the supplied turns, verbatim, with their dates and the turn they replied to |

These measurements do not establish total operating cost: recall and answer generation are not included, and "top k" counts stored items, not equal token budgets (a verbatim turn and an extracted fact are different units). Verbatim storage avoids generative rewriting during ingestion; it does not guarantee correct input metadata, retrieval, or answers.

How trw-memory gets there without a generative model at write time: `store_conversation()` keeps each turn verbatim with its conversational context, and recall does the work (BM25 with stemming + dense vectors + rank fusion + a cross-encoder re-ranker that drops low-confidence rows). Evidence retrieval over all ten LOCOMO conversations (**n = 1,540** questions, no LLM in the loop): the gold evidence turn is in the top 10 for **85.7%** of questions and in the top 50 for **92.7%** (mean reciprocal rank 60.3).

A second, independent LLM-free benchmark runs the same way: **LongMemEval_S** (cleaned, HF revision `98d7416c`), **n = 470** non-abstention questions, each with its own haystack of 38-62 sessions. An answer-bearing turn is in the top 10 for **93.8%** of questions (Wilson 95% CI 90.6-95.1, first run of this configuration) and in the top 50 for **97.4%**; the answer *session* is in the top 10 for 93.3%. Weakest question type: single-session preference, 73.3% (n = 30). Harness: [`benchmarks/longmemeval/`](https://github.com/wallter/trw-memory/blob/main/benchmarks/longmemeval/README.md).

Both benchmarks are the inner loop for retrieval changes, and a change ships only if it is non-inferior on **both**. Measured that way, paired question-by-question against the previous defaults:

| Change | LOCOMO (n = 1,540) | LongMemEval (n = 470) |
|---|---|---|
| `bge-small-en-v1.5` replaces `all-MiniLM-L6-v2`; session dates indexed with each turn | hit@10 84.0% -> 85.1% (McNemar p = 0.044) | not run for this change |
| Entity-bridge second hop after re-ranking | hit@10 85.1% -> 85.6% (p = 0.022); multi-hop recall@10 49.9% -> 51.3% (p = 0.020) | no question changed outcome |
| Confidence floor scales with the requested limit | hit@50 92.5% -> 92.7% (p = 0.25, **not significant**) | hit@50 95.3% -> 97.4% (p = 0.002); rows returned at limit 50: min 5 -> 25 |

Measured and **rejected** on the same harnesses, so they are not in the product: CombMax fusion, larger candidate pools, a different RRF constant, and MMR / Dartboard diversity re-ranking (no significant gain on both; diversity cut LOCOMO temporal recall), plus pseudo-relevance feedback, larger cross-encoders, `bge-base` and `mxbai-embed-xsmall`.

### Where this sits against other memory systems

Published LOCOMO numbers for mem0, Zep, Memobase, MemOS and LightMem are **LLM-judged answer accuracy** for a whole retrieve-then-generate pipeline. The retrieval rates above measure something narrower: whether the gold evidence reached the top k, with no model in the loop. **They are not comparable**, and a table placing them side by side would mislead. Two further cautions: the LOCOMO judged-accuracy literature is actively disputed between vendors, and in mem0's own paper a full-context baseline with no memory at all (72.9%) scores above mem0 itself (66.9%). The only like-for-like comparison here is the paired table above, where both systems ran inside mem0's harness with the same answerer and judge.

### Hybrid retrieval beats either ranker alone

On a gold set of real engineering learnings (**n = 889** typed queries), Reciprocal Rank Fusion of BM25 + dense vectors outranks either single ranker (the table below is this gold set; point estimates, no intervals computed):

| Retriever | Recall@10 | nDCG@10 |
|-----------|:---------:|:-------:|
| BM25 only | 0.869 | 0.771 |
| Vector only | 0.914 | 0.806 |
| **Hybrid (BM25 + vector, RRF)** | **0.938** | **0.839** |

The same direction was observed on a second, independent benchmark (LongMemEval_S, **n = 500** questions); those figures are not reproduced here. Fusion earns its keep on the hard questions: exact-match queries are near ceiling for every retriever, so the lift concentrates in the temporal / multi-session discrimination band.

### Retrospective retrieval of previously stored duplicates

On TRW's own active learning store (**n = 175** near-duplicate "rediscoveries"), the share of duplicates a recall *would have caught* before re-deriving them — the Preventable Rediscovery Ratio — is far higher for hybrid than for keyword search alone, with **non-overlapping 95% CIs**:

| Retriever | Preventable Rediscovery Ratio (95% CI) |
|-----------|:--------------------------------------:|
| BM25 only | 0.720 [0.649, 0.781] |
| **Hybrid** | **0.943 [0.898, 0.969]** |

Hybrid retrieval surfaced more of these previously stored duplicates; this evaluation did not measure whether agents then avoided re-deriving them.

### Cross-session recall on constructed tasks

On a controlled recall-dependent benchmark (H1-MEMORY-BENCH), agents **with** memory solved every task that required recalling a fact established in an earlier session — **58/58** — while agents **without** memory solved **0/50** (the fact is absent by construction). Paired McNemar **p = 3.6×10⁻¹⁵** across **49 matched pairs** (exceeds the pre-registered n ≥ 30), replicated on a second model family.

> **Scope, honestly.** This demonstrates the *mechanism*: cross-session recall lets an agent complete work it otherwise cannot. Whether that compounds into broad, end-to-end coding-task improvement is a separate question, and the measured answer so far is unfavourable: on SWE-bench Verified, TRW solved 56 of 112 paired-valid problems against the baseline's 79 (McNemar p = 6.6×10⁻⁵). That surface is treated as contaminated and settles nothing in either direction, but no general outcome-lift claim is supported. See the [verification docs](https://trwframework.com/docs/verification) for the full evidence posture.

### Cost that does not grow with the store

Recall and write paths were profiled and the terms that scaled with store size were removed. One Apple-silicon machine, single runs under load, before and after the 2026-09 changes:

| Operation | Before | After |
|---|---|---|
| Ingest per row, 4th LOCOMO conversation into a shared store | 1,464 ms | 15 ms |
| Store one entry with 20 sibling project namespaces present | 708 ms | 15 ms |
| Access-time sidecar write per recall, 5,000 / 20,000 rows | 144 / 289 ms | 1.2 / 1.3 ms |
| Warm-row scan per recall, 5,000 / 20,000 rows | 11.3 / 60.5 ms | 0.3 / 3.2 ms |
| Store + background graph enrichment, rows 901-1,200 | 15.85 +/- 0.98 ms | 8.38 +/- 0.79 ms |
| Similarity-edge enrichment at 10,000 rows | 12.4 ms | 2.3 ms |

The shape matters more than any single row: these costs used to rise with the number of stored rows and with the number of projects sharing a store, and are now flat. Similarity edges are searched over the whole namespace instead of the newest 500 rows, and bytes written per recall fell from megabytes to about 18 KB.

*Throughput (historical single-run baseline, not CI-backed, measured before cross-encoder re-ranking became the default; re-ranking adds roughly 30-300 ms per recall on CPU): sub-millisecond store (p95 ≈ 0.31 ms) and ~116 ms hybrid recall p95 at 1,000 entries; on-disk footprint ≈ 1.2 MB per 1k entries.*

## Architecture

The engine is organized as a set of focused subpackages under `src/trw_memory/`. (For the
authoritative, always-current layout, browse the source tree directly — file-level listings
drift quickly.)

| Path | Responsibility |
|------|---------------|
| `client.py` (+ `_client_*.py`) | `MemoryClient` SDK — the recommended entry point; store/recall/search/forget/bulk + lifecycle/tiering/org-shared helpers |
| `cli.py`, `cli_parser.py`, `cli_*.py` | `trw-memory` command-line interface and its formatters/storage helpers |
| `server.py`, `tools/` | FastMCP server entry point and the MCP tool implementations (`fastmcp` is a core dependency) |
| `storage/` | SQLite primary backend (WAL, sqlite-vec vectors, snapshots, recovery, resilient fetch) + YAML backend, behind a shared `StorageBackend` interface; `_dbapi.py` driver shim |
| `retrieval/` | BM25 sparse, dense vector, RRF fusion, and the `hybrid_search()` pipeline + admission/source policies and token budgeting |
| `lifecycle/` | Utility scoring (Ebbinghaus decay over impact), semantic dedup, consolidation, anchor validation, and `tiers/` hot/warm/cold management |
| `graph.py` (+ `_graph_*.py`) | Knowledge graph — similarity/tag edges, BFS traversal, clusters, conflicts, cross-project, decay |
| `bandit/` | Change-point detection (`PageHinkleyDetector`) for non-stationary reward streams |
| `code_index/`, `wiki/` | Explicit code index (chunker/indexer/symbols/search) and wiki page indexing + lint |
| `embeddings/` | Embedding provider protocol + local sentence-transformers provider |
| `sync/` | Remote publish/fetch with vector clocks, three-way merge, retry queue, SSE subscriber |
| `security/` | SQLCipher whole-database encryption at rest (AES-256-CBC), PII detection/redaction, poisoning/anomaly defense, RBAC, provenance, audit, trust scoring, quarantine |
| `integrations/` | Shared sync backend bridge used internally by `trw_memory` |
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
| `compute_utility_score()` | `lifecycle.scoring` | Ebbinghaus decay over impact, recurrence and access |
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

### How hybrid retrieval works: BM25 + vector search + cross-encoder reranking

The hybrid search pipeline combines sparse keyword retrieval with dense semantic search — ensuring strong results for both exact-match queries and conceptually similar queries. [Read the full architecture docs](https://trwframework.com/docs).

```
Query --> BM25 (keyword, rank-bm25) --+
                                       +--> RRF Fusion (k, configurable) --> Ranked Results
Query --> Dense (cosine, sqlite-vec) --+
```

BM25 drops function words from the query and suffix-stems tokens on both sides ("researched" meets "research"); after fusion a cross-encoder re-ranks the top `recall_rerank_candidates` on every `MemoryClient.recall()` and drops rows it scores below -8, except the top `max(5, ceil(limit / 2))`, which are always kept (`adaptive_rerank_floor`: 5 rows at the default `limit=10`, 25 at `limit=50`). There is no switch to turn re-ranking off; it is skipped only when `sentence-transformers` or the cached model is unavailable, in which case recall keeps fusion order. The RRF constant `k` is configurable via `MemoryConfig.rrf_k` (env `MEMORY_RRF_K`); the shipped default is tuned by the memory meta-harness loop and may change between releases, so treat the exact value as a default rather than a contract.

The pipeline gracefully degrades: if BM25 is unavailable, only dense search runs (and vice versa). If neither is available, falls back to the storage backend's built-in keyword search (case-insensitive `LIKE` matching).

### Scoring System

Learning utility is computed from multiple signals. [Full scoring documentation](https://trwframework.com/docs):

- **Ebbinghaus forgetting curve**: Time-based [Ebbinghaus decay](https://trwframework.com/docs) applied at query time (not mutated in storage) — entries naturally fade unless reinforced by recall
- **Access recency boost**: Recently accessed entries score higher
- **Impact score**: Author-assigned importance (0.0-1.0)

### Tiered Storage

Hot/warm/cold tiering keeps frequently-used memories fast and archives stale ones. [Architecture overview](https://trwframework.com/docs):

| Tier | Criteria | Storage | Latency |
|------|----------|---------|---------|
| Hot | Recently recalled entries | In-memory LRU cache | <1ms |
| Warm | Active entries mirrored into the tier runtime | SQLite + JSONL sidecar with full entry payloads | <50ms |
| Cold | Archived entries matched by recall or explicit sweep policy | YAML archive (partitioned by year/month) | <200ms |

The latency column is the design target for the tier lookup itself, not end-to-end recall latency (hybrid recall with re-ranking is slower; see [Benchmarks](#benchmarks)). Store/recall operations keep Hot/Warm in sync, Cold-tier hits are promoted back to Warm within the same recall, and `TierManager.sweep()` applies the configurable archive/purge policy when callers trigger a lifecycle sweep.

### Security

| Feature | Implementation |
|---------|---------------|
| Encryption at rest | SQLCipher whole-database encryption (AES-256-CBC), keyed by an HKDF-SHA256 per-namespace derivation from the master key |
| PII detection | Regex patterns (email, phone, SSN, credit card, API keys) + Shannon entropy analysis. Store path **blocks** API-key/token writes and **records** every other detection as metadata — it does not rewrite your stored text. Masking happens at the publish boundary (`strip_pii`), where the local copy still holds the original |
| Poisoning defense | Z-score anomaly detection on frequency, size, and content patterns — **observe mode by default** (records + telemetry, does not quarantine); `enforce` is opt-in |
| Access control | Role-based (admin/editor/viewer) per namespace |
| Audit trail | Append-only security event log |
| Key management | Master key derivation, per-namespace keys (no rotation path — retired, the operator does not use key rotation) |

## MCP memory server

The MCP server ships with the core install (`fastmcp` is a core dependency):

```bash
trw-memory-server  # Starts MCP server (stdio transport)
```

To wire it into an MCP client (Claude Code, Cursor, Claude Desktop and others use this shape):

```json
{
  "mcpServers": {
    "memory": { "command": "trw-memory-server" }
  }
}
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

### Loopback daemon (`serve http`)

`trw-memory-server serve http` runs one process per operating-system user, serving
the same MCP tool surface over `streamable-http` on 127.0.0.1, authenticated by
per-checkout namespace grants. The port is ephemeral by default and published in a 0600 `daemon.json` beside
the store, so clients discover it rather than hardcode it.

**Trust boundary: a token reaches only its grant.** `trw-mcp memory token` mints a
token for the calling checkout's pinned project namespace plus `user:local`; the
daemon keeps only its sha256 digest in the 0600 `daemon-grants.json`, and the raw
token lives in that checkout's `.trw/runtime/memory-token`. Every namespaced call is
checked against the grant before RBAC, so a request for any other namespace is
refused even with RBAC off. A Slice A `daemon-token` (one all-namespace bearer) makes
the daemon refuse to start; `trw-mcp memory token --migrate` deletes it.

**Concurrency: four workers.** Each served `memory_recall`, `memory_store` and
`memory_maintain` call runs its synchronous work in a bounded thread pool
(`OFFLOAD_MAX_WORKERS = 4` in `daemon/_offload.py`), opening and closing its own
SQLite connection inside the worker. Four calls make progress at once; the fifth
queues, and that queue is unbounded. A request that is cancelled after it starts
still runs to completion — the result is discarded, not the work.

**Shutdown.** SIGTERM and SIGINT drain the worker pool, remove the discovery record,
and then let the signal take its default disposition, so a service manager stopping
the daemon does not leave clients pointed at a dead endpoint. The record is only ever
removed when it names this process *and* the start time this process wrote, so a
slow exit cannot delete a successor's record.

**Maintenance.** A daemon has no session end, so decay, consolidation and WAL
checkpointing never run on their own. `memory_maintain(namespace)` triggers them and
records `last_attempted_at` / `last_maintained_at` per namespace in `maintenance.json`
beside the store. Scope is not uniform: consolidation is namespace-scoped, while the
decay pass and the WAL checkpoint act on the whole store.

**Recall is bounded.** Each namespace contributes at most
`max(limit * 5, hybrid_search_candidate_pool_size)` entries (default 1000) to a
search, chosen as the most recently updated rows. On a larger namespace, older
entries are not searched, and an empty result is not evidence of absence. Raising
`MEMORY_HYBRID_SEARCH_CANDIDATE_POOL_SIZE` widens it at a real cost: measured on a
6500-row namespace, warm recall was 139.6 ms at 1000 and 1045.8 ms at 10000.

## Integration with trw-mcp

[trw-mcp](https://github.com/wallter/trw-mcp) is the MCP server layer of [TRW Framework](https://trwframework.com) — it exposes a suite of tools, skills, and agents to Claude Code and other AI coding tools (see the [trw-mcp README](https://github.com/wallter/trw-mcp) for current counts). trw-memory serves as its memory backend:

- `trw_learn` delegates to `SQLiteBackend.store()` via `memory_adapter.py` (YAML dual-write as backup)
- `trw_recall` delegates to `SQLiteBackend.search()` / `list_entries()` as the sole query path
- Scoring functions (`compute_utility_score`, `apply_time_decay`) are canonical in trw-memory and re-exported by trw-mcp
- One-time YAML-to-SQLite migration runs automatically on first access
- Optional vector search via `LocalEmbeddingProvider` + `rrf_fuse` when `sentence-transformers` is installed

[Read more about the full TRW Framework architecture](https://trwframework.com/docs).

## Telemetry and network behavior

trw-memory is **local-first**: with the default configuration all data lives in a local SQLite store (and an optional YAML sidecar). It makes **no outbound network calls** except the optional model downloads below (embedding model and cross-encoder re-ranker). There is no usage tracking or content phone-home.

### What can touch the network, when, and how to turn it off

| Surface | When | Default | Opt-out / control |
|---------|------|---------|-------------------|
| **Embedding model download** | Only when the embedding model — `BAAI/bge-small-en-v1.5` by default (33M parameters, 384-dim, about 130 MB of weights; set `MEMORY_EMBEDDING_MODEL` to change it) — is **not** already complete in your local Hugging Face cache. A complete cached snapshot makes **zero** huggingface.co requests — the loader probes the cache before deciding, and forces `local_files_only=True` unconditionally when the snapshot is complete (only with the `[embeddings]` extra installed) | enabled when the extra is present | `TRW_OFFLINE=1` / `HF_HUB_OFFLINE=1`, or `local_only: true` (alias `memory_local_only`) — forces `local_files_only` so no download is attempted; a disclosure log line precedes any network-capable load |
| **Cross-encoder model download** (re-ranker, on by default since 0.19.0) | Only when `cross-encoder/ms-marco-MiniLM-L-6-v2` is not in your local Hugging Face cache and the `[embeddings]` extra is installed; the same offline switches force `local_files_only=True`, in which case an uncached model means recall keeps fusion order (no download, no error) | enabled when the extra is present | `TRW_OFFLINE=1` / `HF_HUB_OFFLINE=1`, or `local_only: true`; a disclosure log line precedes any network-capable load |
| **Remote sync / publish** | Only when `sync_enabled=true` AND `local_only=false` | **off** (`sync_enabled` defaults `false`) | leave sync disabled, or set `local_only: true` to hard-block all egress |

**A warm cache performs no Hub request, and embedding egress is independent of the consent flags.** A fetch is attempted only when the cached snapshot is incomplete or absent **and** no offline switch is engaged; in exactly that case one structured disclosure log names the host and the switch that would block it. `learning_sharing_enabled` and `platform_telemetry_enabled` govern learning-content publishing and usage telemetry respectively — **neither gates the embedding model fetch**. Embedding egress is governed by the local cache, the offline switches, and `local_only`.

`sync_enabled` defaults `false`, so the engine performs no remote sync out of the box even though `local_only` itself defaults `false`. Setting `local_only: true` is the hard-block: an `@model_validator` forces `sync_enabled=False`, clears `sync_namespace`/`platform_url`, and pins `rbac_mode="local"`, so no remote-capable surface can be re-enabled while it is set.

With an offline switch engaged (`TRW_OFFLINE` / `HF_HUB_OFFLINE`) **or** `local_only: true`, the standalone engine loads the embedding model with `local_files_only=True`; if the model is not already cached it raises a clear `LocalOnlyViolationError` telling you how to pre-download (this is the behaviour in both cases — `local.py` does **not** silently fall back to keyword-only recall here). The graceful "degrade to keyword-only, no crash" path is provided one layer up by trw-mcp's embedder wrapper, which catches that error; a direct trw-memory caller that wants keyword-only recall under an offline switch should pre-download the model or run without the `[embeddings]` extra installed.

### Environment-variable inventory

| Variable | Purpose | Default |
|----------|---------|---------|
| `TRW_OFFLINE` | Master offline switch — blocks the huggingface.co embedding-model and re-ranker model downloads | unset |
| `HF_HUB_OFFLINE` | Upstream huggingface_hub offline switch — also honored | unset |
| `MEMORY_EMBEDDING_MODEL` | Sentence-transformers model used for dense vectors. Changing it leaves stored vectors in the old model's space until `trw-memory reembed` re-encodes them (see [Upgrading from all-MiniLM-L6-v2](#upgrading-from-all-minilm-l6-v2)) | `BAAI/bge-small-en-v1.5` (33M params, 384-dim, ~130 MB) |
| `MEMORY_*` | Engine knobs validated by `MemoryConfig` (e.g. `MEMORY_LOCAL_ONLY`, `MEMORY_EMBEDDING_TRUST_REMOTE_CODE`, retrieval + lifecycle tuning) | per-field |

### Security defaults

| Capability | Default | Notes |
|-----------|---------|-------|
| Encryption at rest | **off** (`encryption_enabled=False`) | opt-in — SQLCipher whole-database encryption (AES-256-CBC), HKDF-SHA256 per-namespace keys |
| PII detection | **on** (`pii_enabled=True`) | always scans `content`/`detail`/`tags`/`evidence[]`/`Assertion.last_evidence` on the store path. On the runtime store path, detected **API keys / tokens block the write** (`PIIBlockError`); every other type is recorded in the `pii_types` metadata and stored **verbatim** — heuristic detectors do not get to irreversibly rewrite local text. Emails, IPs, SSNs, phone numbers and credit-card shapes are masked at the publish boundary instead. Set `pii_custom_patterns` to opt in to local masking with your own regexes |
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

For hybrid recall offline, populate the model cache **before** enabling either switch, in the same environment: `python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; SentenceTransformer('BAAI/bge-small-en-v1.5'); CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"`. Otherwise the first embedding load raises `LocalOnlyViolationError` (an uncached re-ranker is skipped silently and recall keeps fusion order). To run keyword-only without that error, omit the `[embeddings]` extra entirely. Verify the on-disk `memory.db` is mode `0600` and that no outbound connection is attempted on first use.

## Migration notes

### Upgrading from all-MiniLM-L6-v2

The default embedding model is now `BAAI/bge-small-en-v1.5` (same 384 dimensions; queries carry the model's search instruction, stored documents do not). Vectors written by `all-MiniLM-L6-v2` — or written before vectors recorded which model produced them — live in a different embedding space, so dense recall **ignores them** rather than scoring a new-model query against old-model vectors. Until they are re-encoded:

- BM25 still ranks every row, so recall keeps working, with keyword-only relevance for the old rows;
- each recall logs one `dense_vectors_excluded_embedding_space` warning with the number of vectors it held back.

Re-encode each namespace once (idempotent and resumable — rows already in the active space are skipped, and each batch commits on its own):

```bash
trw-memory reembed --namespace default            # --batch-size 64, --format json
```

```python
async with MemoryClient(namespace="default") as client:
    counts = await client.reembed()
```

The re-embed honours `TRW_OFFLINE` / `HF_HUB_OFFLINE` / `local_only` like every other load: with a switch set and the model not cached it raises `LocalOnlyViolationError` instead of downloading, so pre-download `BAAI/bge-small-en-v1.5` first. To keep the previous model instead, set `MEMORY_EMBEDDING_MODEL=all-MiniLM-L6-v2` (or `embedding_model` in config); it keeps working, without a query instruction. A process only ever scores vectors from the exact space its embedder reports, so switching models back and forth never mixes spaces — it just needs another `reembed`.

### Retired hypothetical expansion (unreleased)

HyPE question generation and HyDE query expansion are removed. Ordinary
embeddings, lexical/hybrid recall, code/wiki references and Distill data are not
removed. Delete imports of `QuestionGenerator` and `NoOpQuestionGenerator`.
Remove `question_generator`, `query_expansion`, and `collapse_hype` arguments,
and the `hype_enabled`, `hype_questions_per_entry`, `hype_min_question_chars`
settings (including `memory_` aliases and environment/YAML entries).

Explicit neutral legacy settings (`False`, `3`, `8`) and API arguments (`None`
for the generator, `None`/blank expansion, `False` for collapse) warn temporarily;
activation, nondefault values and invalid types fail before the operation.
Retired settings no longer appear in emitted configuration. These tombstones
will disappear in the next declared breaking API release **after** the retirement
release; that release's notes must announce their removal.

Existing derived question vectors are not knowledge records. Recall ignores them
by requiring canonical membership, without excluding real IDs that happen to end
in `#hype0`. Normal update/forget removes only namespace-owned noncanonical
siblings of the selected canonical parent. Orphan/unknown vector rows remain
untouched for a future canonical-only index rebuild. No startup purge occurs.

#### Optional legacy-vector maintenance on a disposable snapshot

Stop old-version writers first; they can regenerate retired vectors. Preserve a
verified backup using SQLite's online backup API (the approach in
`storage/_schema_backup.py`), **not** a copy of a live database without its WAL.
Do not overwrite canonical writes made since a snapshot to recover optional
vectors. The retirement itself changes no schema or historical migration.

This recipe is for an **existing disposable unencrypted snapshot**, not a live
store. Choose its original embedding dimension, namespace and parent IDs
explicitly; opening `SQLiteBackend` can perform normal schema initialization.
Encrypted stores require their existing key-aware backup/open procedure instead.
`apply = False` only enumerates selected vectors; changing it to `True` removes
those derived vectors atomically. It never deletes canonical records or other
namespaces. No vectors installed means unavailable, not a successful cleanup.

```python
from pathlib import Path
from trw_memory.storage.sqlite_backend import SQLiteBackend

snapshot = Path("/absolute/path/to/disposable-snapshot.db")
if not snapshot.is_file():
    raise FileNotFoundError(snapshot)
namespace = "default"  # explicitly selected, locally authorized namespace
parents = ["selected-parent-id"]
apply = False
backend = SQLiteBackend(snapshot, dim=384)  # use this snapshot's dimension
try:
    if not backend.supports_vectors():
        raise RuntimeError("legacy cleanup unavailable: sqlite-vec required")
    with backend.transaction():
        for parent_id in parents:
            siblings = backend.hype_sibling_ids(parent_id, namespace=namespace)
            print(parent_id, siblings)
            if apply:
                backend.delete_hype_siblings(parent_id, namespace=namespace)
finally:
    backend.close()
```

Cleanup is idempotent; interruption rolls the transaction back. Package rollback
can reopen the same canonical store; restoring previous optional ranking also
requires its matching derived-index snapshot. Never discard newer canonical data
for that purpose. Internal cleanup helpers will be removed once the supported
store floor rejects pre-retirement stores unless canonical-only vector rebuilding
has been verified; ordinary orphan-index handling then owns residual derived data.

## Platform and interpreter notes

### Supported interpreters

`trw-memory` is tested on CPython 3.10 through 3.14 (this repository's own development
interpreter is CPython 3.14.7). One property of the interpreter matters beyond the version:
its bundled SQLite. WAL space is only RECLAIMED on SQLite >= 3.51.3 (or the 3.44.6 / 3.50.7
backports) — below that, `storage/_wal_checkpoint.py` coerces resetting checkpoints to
`PASSIVE`, which is correct and safe but lets the `-wal` file grow without shrinking. Check
yours with `python -c "import sqlite3; print(sqlite3.sqlite_version)"`; on macOS, Homebrew's
current Python ships a qualifying build, and `trw-mcp doctor` names the qualifying
interpreters it finds.

The engine is SELECTED at import by `storage/_dbapi.py`, which ranks
the interpreter's SQLite against an installed `pysqlite3` on (carries the fix, version) and
never replaces a newer engine with an older wheel. The optional `[sqlite-fix]` extra pulls
`pysqlite3-binary` on x86_64 Linux only — no published wheel currently bundles a qualifying
SQLite, so it is an engine override, not a fix.

### Platform notes

- **SQLite driver** — `pysqlite3-binary` is no longer a runtime dependency on any platform; it moved to the optional `[sqlite-fix]` extra, marked for x86_64 Linux (the only platform it publishes a wheel for). It used to be a hard Linux dependency, which made aarch64 Linux installs fail outright while delivering SQLite 3.51.1 — below the 3.51.3 fix it existed to provide. The runtime probe in `storage/_dbapi.py`, not the dependency name or the package version, decides and reports which engine is active.
- **Vector search is optional** — `[vectors]` (sqlite-vec) and `[embeddings]` (sentence-transformers) are optional extras. When they are unavailable the retrieval pipeline degrades gracefully to BM25 and/or the backend's built-in keyword search rather than failing.

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
| `[encryption]` | sqlcipher3, keyring, cryptography | Encrypted-at-rest DB (SQLCipher) + key storage |
| `[embeddings]` | sentence-transformers | Dense vector embeddings (`BAAI/bge-small-en-v1.5` by default, 384-dim) |
| `[vectors]` | sqlite-vec | Vector similarity search in SQLite |
| `[bm25]` | rank-bm25 | BM25 keyword search |
| `[all]` | embeddings + vectors + bm25 | The full retrieval stack |
| `[dev]` | pytest, mypy, ruff, coverage, pip-audit, vulture, deptry | Testing and linting |

There is no `[llm]` extra and no LLM-backed consolidation. Consolidation
summarises a cluster with a longest-content heuristic; an earlier revision of
this table advertised `[llm]`/`anthropic` "LLM-augmented consolidation", which
this package never implemented. The `[langchain]`, `[llamaindex]`, `[crewai]`
and `[all-integrations]` extras and their adapter modules were removed as unused
surface — see [CHANGELOG.md](https://github.com/wallter/trw-memory/blob/main/CHANGELOG.md) `[Unreleased]` Removed.

### Entry Points

| Command | Purpose |
|---------|---------|
| `trw-memory` | CLI for store/recall/search/forget/consolidate/export/import, plus restore, snapshot (create/list/rotate), wiki-lint, and code-index/code-search/code-symbol |
| `trw-memory-server` | MCP server (stdio transport) |

## FAQ

### What is trw-memory?

A persistent, local-first memory engine for AI agents. It stores memories in SQLite and recalls them with keyword search, or, with the retrieval extras installed, hybrid retrieval (BM25 + dense vectors, fused with Reciprocal Rank Fusion, then a cross-encoder re-ranker). It ships a Python SDK (`MemoryClient`), a CLI (`trw-memory`), and an MCP server (`trw-memory-server`). You do not need TRW Framework to use it.

### Does trw-memory need an LLM to store memories?

No. `store_conversation()` stores every turn verbatim and calls no generative LLM at ingest time; the reader does the inference at recall time. There is also no `[llm]` extra and no LLM-backed consolidation: consolidation summarises a cluster with a longest-content heuristic. Dense retrieval uses a local sentence-transformers embedding model (`BAAI/bge-small-en-v1.5` by default) and a local cross-encoder re-ranker, both from the optional `[embeddings]` extra.

### Does it work offline?

Yes. With the default configuration all data is local and remote sync is off. `TRW_OFFLINE=1` / `HF_HUB_OFFLINE=1` or `local_only: true` force `local_files_only=True` for the embedding and re-ranker models. If the embedding model is not already cached when one of those switches is set, the first embedding load raises `LocalOnlyViolationError`: pre-download the model, or omit the `[embeddings]` extra to run keyword-only. See [Telemetry and network behavior](#telemetry-and-network-behavior).

### How does trw-memory compare to mem0?

One small, scoped comparison exists. Using mem0's open-source evaluation suite, unmodified, on LOCOMO conversation 0 (n = 152 questions per system, paired by question, one run each): trw-memory 91.4% [85.9, 94.9] vs mem0 (OSS) 88.2% [82.1, 92.4] at top 10 (McNemar p = 0.38), and 91.4% [85.9, 94.9] vs 92.1% [86.7, 95.4] at top 50 (p = 1.00). Neither test detected a statistically significant difference, which does not establish equivalence or superiority. In that run mem0 made ~2 generative LLM calls per turn to ingest the 419-turn conversation (1 h 28 min); trw-memory made none (~75 s); those figures are ingestion only, not total operating cost. Conditions: a local 8B `llama3.1` as answerer, judge and mem0's extraction model, and mem0 run as its open-source SDK, not Mem0 Cloud. See [the benchmark section](#single-conversation-comparison-with-mem0-oss-using-mem0s-evaluation-suite).

### Does recall search every stored memory?

Not on a very large namespace. Each namespace contributes at most `max(limit * 5, hybrid_search_candidate_pool_size)` entries (default 1000) to a search, chosen as the most recently updated rows, so on a larger namespace older entries are not searched and an empty result is not evidence of absence. `MEMORY_HYBRID_SEARCH_CANDIDATE_POOL_SIZE` widens it at a latency cost. `recall(limit=N)` can also return fewer than N rows: results the cross-encoder scores below -8 are dropped, though the top `min(N, max(5, ceil(N / 2)))` are always kept.

### Where is my data stored?

In `.memory/` relative to the current directory by default (override with `MEMORY_STORAGE_PATH`), as a local SQLite database per namespace plus an optional YAML sidecar. Nothing leaves the machine unless you enable remote sync.

### Can I use it as an MCP memory server?

Yes. Run `trw-memory-server` (stdio transport), or `trw-memory-server serve http` for a per-user loopback daemon. See [MCP memory server](#mcp-memory-server).

### What happens if sqlite-vec or sentence-transformers is not installed?

`[vectors]` and `[embeddings]` are optional extras. When they are unavailable the retrieval pipeline degrades gracefully to BM25 and/or the backend's built-in keyword search rather than failing.

### Does agent memory improve coding-task outcomes?

That is an open empirical question. On a controlled recall-dependent benchmark (H1-MEMORY-BENCH), agents with memory solved 58/58 tasks that required a fact from an earlier session and agents without memory solved 0/50 (the fact is absent by construction), which demonstrates the mechanism. Early SWE-bench single-shot runs (n ≥ 40) produced null. See [Knowledge compounding, measured](#cross-session-recall-on-constructed-tasks).

### What license is trw-memory under?

[Business Source License 1.1](https://trwframework.com/license): source-available, free for non-competing use, converting to Apache 2.0 on 2030-03-21. The package is alpha.

## License

[Business Source License 1.1](https://trwframework.com/license) -- source-available, free for non-competing use. Converts to Apache 2.0 on 2030-03-21.

---

Built by [Tyler Wall](http://tylerrwall.com) · [TRW Framework](https://trwframework.com) · [Documentation](https://trwframework.com/docs) · [License](https://trwframework.com/license)
