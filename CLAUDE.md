# trw-memory

Standalone persistent memory engine for AI agents — part of [TRW Framework](https://trwframework.com).

**Public repo**: [github.com/wallter/trw-memory](https://github.com/wallter/trw-memory) | **PyPI**: `pip install trw-memory`

Hybrid retrieval (BM25 + dense vectors), tiered storage (SQLite primary, YAML secondary), semantic dedup, and knowledge-graph traversal. Used by [trw-mcp](https://github.com/wallter/trw-mcp) as its memory backend.

## Build & Test

```bash
cd trw-memory
../.venv/bin/python -m pytest tests/test_specific_file.py -v   # Single file (preferred)
../.venv/bin/python -m pytest tests/ -v                         # Full suite (delivery only)
../.venv/bin/python -m mypy --strict src/trw_memory/            # Type check
pip install -e ".[dev]"                                       # Dev install
pip install -e ".[dev,vectors,bm25]"                         # With optional deps
```

## Architecture

| Module | Purpose |
|--------|---------|
| `storage/` | SQLite + YAML dual backends (`sqlite_backend.py`, `yaml_backend.py`) |
| `retrieval/` | BM25 keyword + dense vector hybrid search |
| `lifecycle/` | Tier promotion/demotion, utility scoring, dedup, consolidation |
| `graph.py` | Knowledge graph with BFS traversal |
| `sync/` | Remote publish/fetch with vector clocks |
| `security/` | Field-level encryption, PII detection, RBAC |
| `tools/` | MCP tool wrappers (optional `[mcp]` dep) |
| `client.py` | Public `MemoryClient` — primary consumer API |

## Testing Patterns

- **In-memory SQLite**: `SQLiteBackend(":memory:")` — use for unit tests, no `tmp_path` needed
- **Disk SQLite**: `SQLiteBackend(tmp_path / "test.db")` — use for integration tests
- **Shared fixtures**: `tests/conftest.py` provides `sqlite_backend`, `sqlite_memory_backend`, `memory_client`, `make_entry()`, `make_entry_dict()`
- `asyncio_mode = "auto"` — write `async def test_foo():` without `@pytest.mark.asyncio`

## Key Gotchas

- `sqlite-vec` is an optional dep (`[vectors]`) — skip dense vector tests with `pytest.importorskip("sqlite_vec")`
- `sentence-transformers` is optional (`[embeddings]`) — embeddings fail gracefully when unavailable
- Pydantic v2: `use_enum_values=True` required for YAML round-trip; `populate_by_name=True` when using `Field(alias=...)`
- structlog: never use `event=` as a kwarg — it's reserved; use `action=` or `operation=`
- Coverage threshold: 85% (fail_under in pyproject.toml)
