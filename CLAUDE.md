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

## Package Info

- Version: `0.9.6` (see `pyproject.toml` — do not hardcode elsewhere)
- ~170 source modules; 240 test files, 214 with test functions; coverage gate 85%
- Valid namespace prefixes: `project:`, `global`, `default`, `team:`, `org:`, `user:` — the `user:` scope was added by PRD-CORE-185 and is live in `namespaces/validation.py`

## Compatibility Notes

- Concurrent-writer fixes (warm-tier sidecar lock, hot-tier sweep race) shipped in **0.9.5**. Operators running concurrent agents against the same memory store should require `trw-memory >= 0.9.5`.
- SQLite WAL-reset corruption guard requires SQLite >= 3.51.3 or the single-connection window mitigation (active by default); see `reference_memory_db_walreset_fix.md`.

## Key Gotchas

- `sqlite-vec` is an optional dep (`[vectors]`) — skip dense vector tests with `pytest.importorskip("sqlite_vec")`
- `sentence-transformers` is optional (`[embeddings]`) — embeddings fail gracefully when unavailable
- Pydantic v2: `use_enum_values=True` required for YAML round-trip; `populate_by_name=True` when using `Field(alias=...)`
- structlog: never use `event=` as a kwarg — it's reserved; use `action=` or `operation=`
- Coverage threshold: 85% (fail_under in pyproject.toml)
- MCP recall path uses `_keyword_search`/`_search_entries` directly (not `MemoryClient.recall`); user-tier federation is handled separately in `_memory_recall.py`

## Releasing

PyPI publishing is **CI-driven** — `.github/workflows/release.yml` in the public [trw-memory repo](https://github.com/wallter/trw-memory) runs on a **`v*` tag push** (`build → multi-OS/Python smoke matrix → PyPI Trusted Publishing` via OIDC). There is **no manual `twine`/upload and no PyPI token** — never add one. A branch push alone does not publish; only a `v*` tag does.

- **Release `trw-memory` BEFORE `trw-mcp`.** trw-mcp's release builds against `trw-memory @ git+…@main`, so this package's public `main` must carry the new version first or trw-mcp's build/smoke-test resolves the wrong version.
- Tag the **subtree-split commit** (the commit that exists in the public repo), not the monorepo commit — a tag on an unreachable SHA is ignored by Actions.
- Full runbook: [`../docs/deployment/CLAUDE.md`](../docs/deployment/CLAUDE.md) §Public PyPI Releases.
