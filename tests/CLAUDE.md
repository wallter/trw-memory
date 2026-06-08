# trw-memory Test Suite Guide

This guide is for AI agents writing or modifying tests in the `trw-memory/tests/` directory.

## Quick Reference

```bash
# During implementation — run ONLY the file you changed:
cd trw-memory
../.venv/bin/python -m pytest tests/test_specific_file.py -v

# Run a single test:
../.venv/bin/python -m pytest tests/test_storage.py::test_store_entry -v

# Full suite with coverage (delivery only):
../.venv/bin/python -m pytest tests/ --cov=trw_memory --cov-report=term-missing

# Type check:
../.venv/bin/python -m mypy --strict src/trw_memory/
```

**NEVER run the full suite during implementation. Target specific files only.**

## Test Count & Performance

- **Test count**: a broad suite spanning the test files in `tests/` (run `pytest --collect-only -q | tail -1` for the live count — it drifts)
- **Full suite**: several minutes — run targeted files during implementation
- **Coverage threshold**: 85% (`fail_under` in `pyproject.toml`)

## Package Architecture

trw-memory is a standalone memory engine with these modules:

| Module | Tests In | Key Patterns |
|--------|----------|-------------|
| `storage/` | `test_storage*.py` | SQLite + YAML dual-write, `tmp_path` for DB files |
| `retrieval/` | `test_retrieval*.py`, `test_search*.py` | BM25 + dense vectors, in-memory index |
| `graph.py` | `test_graph*.py` | Knowledge graph, BFS traversal |
| `sync/` | `test_sync*.py` | Remote publish/fetch, vector clocks |
| `lifecycle/` | `test_lifecycle*.py` | Tier promotion/demotion, decay |
| `security/` | `test_security*.py` | Encryption for entry fields |
| `tools/` | `test_tools*.py` | MCP tool wrappers (store/recall/search/forget/consolidate/status/audit/review/wiki-lint/code-index/search/symbol) |

## Shared Fixtures (conftest.py)

`tests/conftest.py` provides common fixtures — prefer these over per-file factories:

- `sqlite_backend(tmp_path)` — disk-backed SQLiteBackend (integration tests)
- `sqlite_memory_backend()` — in-memory SQLiteBackend (unit tests)
- `memory_client(tmp_path)` — MemoryClient with isolated SQLite storage
- `yaml_memory_client(tmp_path)` — MemoryClient with isolated YAML storage
- `make_entry(**kwargs)` — factory for `MemoryEntry` with sensible defaults
- `make_entry_dict(**kwargs)` — factory for serialised entry dicts
- `memory_config(tmp_path)` — `MemoryConfig` pointing to temp storage

## Testing Patterns

### SQLite Backend Tests
```python
from trw_memory.storage.sqlite_backend import SQLiteBackend
from trw_memory.models.memory import MemoryEntry

def test_store_and_get(tmp_path):
    backend = SQLiteBackend(tmp_path / "test.db")
    entry = MemoryEntry(id="M-001", content="test content", ...)
    backend.store(entry)
    result = backend.get("M-001")
    assert result is not None
    assert result.content == "test content"
```

### Using conftest fixtures
```python
from tests.conftest import make_entry

def test_with_factory(sqlite_backend):
    entry = make_entry(content="use absolute paths", tags=["gotcha"])
    sqlite_backend.store(entry)
    results = sqlite_backend.search("absolute", top_k=10)
    assert len(results) == 1
```

## Known Gotchas

### sqlite-vec Optional Dependency
Dense vector search requires `sqlite-vec` which is an optional dependency (`[vectors]`). Tests that use dense vectors should check for availability:

```python
pytest.importorskip("sqlite_vec")
```

### Pydantic v2 Serialization
- `use_enum_values=True` required on models for YAML round-trip
- `dict[str, object]` values need `str()` cast for mypy --strict
- `Field(alias=...)` requires `populate_by_name=True`

### structlog Reserved Keywords
Never use `event=` as a kwarg in structlog calls — it's reserved. Use alternative names like `action=` or `operation=`.

## Test Classification

Markers are defined in `pyproject.toml` (`[tool.pytest.ini_options] markers`):

- **`unit`**: pure unit tests — no filesystem I/O beyond `tmp_path`, mocks only, fast
- **`integration`**: uses real filesystem / sqlite / network — slower, still isolated
- **`slow`**: individual-test runtime > 5 seconds — excluded from default runs
- **`network`**: requires external network access (embedding models, API calls)
