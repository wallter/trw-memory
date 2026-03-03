# trw-memory

Persistent memory engine with hybrid retrieval, tiered storage, and semantic dedup for AI agents.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC_BY--NC--SA_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

## What It Does

TRW-Memory is a standalone memory engine that gives AI coding agents persistent, searchable knowledge storage. It stores learnings (patterns, gotchas, architecture decisions) in SQLite with optional YAML backup, and retrieves them using hybrid search that combines keyword matching (BM25) with dense vector similarity.

Designed as the storage backend for [trw-mcp](../trw-mcp/), but usable independently by any AI agent framework that needs persistent memory with recall.

## Features

- **MemoryClient SDK** -- High-level async Python client with store/recall/forget/search
- **Hybrid Search** -- BM25 keyword matching + dense vector similarity via sqlite-vec
- **Tiered Storage** -- Automatic promotion/demotion based on access patterns and impact scores
- **Semantic Dedup** -- Detects and merges near-duplicate learnings using cosine similarity
- **Knowledge Graph** -- Tag co-occurrence edges, BFS traversal, importance boost/decay
- **Remote Sync** -- Publish/fetch learnings across installations with vector clock conflict resolution
- **Field Encryption** -- AES-GCM encryption for sensitive entry fields
- **Agent Integration** -- `register_tools()` for any agent framework, `@auto_recall` decorator
- **Dual Storage Backends** -- SQLite (primary) + YAML (backup) with migration support
- **MCP Tools** -- 6 tools for store, recall, search, consolidate, forget, and status

## Quick Start

```bash
# Install from source
cd trw-memory
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# With all optional features
pip install -e ".[all]"
```

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

    # Recall by keyword query
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

### Low-Level Backend Access

```python
from trw_memory.storage.sqlite_backend import SQLiteBackend
from trw_memory.models.memory import MemoryEntry

backend = SQLiteBackend(db_path=".trw/memory.db")
entry = MemoryEntry(id="M-abc12345", content="...", namespace="default", ...)
backend.store(entry)
results = backend.search("query", top_k=10, namespace="default")
```

## API Reference

### Key Classes

| Class | Module | Description |
|-------|--------|-------------|
| `MemoryClient` | `client` | **Recommended** -- high-level async SDK |
| `SQLiteBackend` | `storage.sqlite_backend` | Primary storage with full CRUD |
| `YAMLBackend` | `storage.yaml_backend` | File-based storage (backup/migration) |
| `HybridSearcher` | `retrieval.hybrid` | BM25 + dense vector search |
| `KnowledgeGraph` | `graph` | Tag/similarity edges, BFS traversal |
| `TierManager` | `lifecycle.tiers` | Access-based promotion/demotion |
| `DedupEngine` | `lifecycle.dedup` | Semantic duplicate detection |
| `MemoryConfig` | `config` | Configuration via env vars or dict |

### Storage Backends

**SQLite** (recommended) -- Fast, transactional, supports hybrid search and knowledge graph:
```python
from trw_memory.storage.sqlite_backend import SQLiteBackend
backend = SQLiteBackend(db_path=".trw/memory.db")
```

**YAML** -- Human-readable, git-friendly, used as backup during migration:
```python
from trw_memory.storage.yaml_backend import YAMLBackend
backend = YAMLBackend(entries_dir=".trw/learnings")
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests (1000+ tests, >=80% coverage required)
.venv/bin/python -m pytest tests/ -v --cov=trw_memory --cov-report=term-missing

# Type checking (strict mode)
.venv/bin/python -m mypy --strict src/trw_memory/

# Targeted testing
.venv/bin/python -m pytest tests/test_client.py -v
```

### Optional Dependencies

| Extra | Packages | Purpose |
|-------|----------|---------|
| `[mcp]` | fastmcp | MCP server tools |
| `[embeddings]` | sentence-transformers | Dense vector embeddings |
| `[vectors]` | sqlite-vec | Vector similarity search |
| `[bm25]` | rank-bm25 | BM25 keyword search |
| `[llm]` | anthropic | LLM-augmented features |
| `[api]` | fastapi, uvicorn | REST API server |
| `[all]` | All of the above | Full feature set |
| `[dev]` | pytest, mypy, ruff, etc. | Testing and linting |

## Architecture

```
src/trw_memory/
  client.py              # MemoryClient SDK (recommended entry point)
  config.py              # MemoryConfig (pydantic-settings)
  graph.py               # Knowledge graph with BFS traversal
  server.py              # FastMCP entry point
  storage/               # SQLite + YAML backends
  retrieval/             # Hybrid search (BM25 + dense vectors)
  lifecycle/             # Tier management, dedup, consolidation
  sync/                  # Remote publish/fetch, conflict resolution
  security/              # Field encryption (AES-GCM)
  tools/                 # 6 MCP tools
  api/                   # FastAPI REST server
```

## License

[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) -- see [LICENSE](../LICENSE).
