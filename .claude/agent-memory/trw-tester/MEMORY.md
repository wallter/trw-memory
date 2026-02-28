# trw-tester Memory

## trw-memory Package

### Test Commands
```bash
cd /mnt/c/Users/Tyler/Desktop/trw_framework/trw-memory
.venv/bin/python -m pytest tests/ -v
.venv/bin/python -m pytest tests/ --cov=trw_mcp --cov-report=term-missing
```

### Key Model Facts
- `MemoryEntry`: `strict=True, use_enum_values=True` — passing "0.5" str for float importance raises ValidationError
- `MemoryEntry.status` serializes as plain string (not enum object) because `use_enum_values=True`
- `MemoryIndex.strict=True` but no `use_enum_values` — contains MemoryEntry instances
- `MemoryConfig` uses `MEMORY_` env prefix, pydantic-settings, `extra="ignore"`
- `MemoryEvent`: `strict=True, use_enum_values=True` — same pattern as MemoryEntry

### Migration Facts
- `from_learning_entry`: `impact` → `importance`, `summary` → `content`, `created`/`updated` date → datetime midnight UTC
- Impact values outside [0,1] are **clamped** (not rejected) by the migration helper
- Missing `id` in source dict generates a UUID4
- Unknown source keys are silently ignored (lenient)
- `migrate_entries_dir` returns `[]` for nonexistent directory (no exception)
- Non-dict YAML files (e.g. lists) are skipped without aborting migration

### Test Patterns Confirmed
- `monkeypatch.setenv` works for `MemoryConfig` env var tests
- `ruamel.yaml YAML()` for writing test YAML fixtures in tmp_path
- `pytest.raises(ValidationError)` for Pydantic strict/range failures
- `asyncio_mode = "auto"` is set in pyproject.toml
- SQLiteBackend: pass `tmp_path / "test.db"` and close in `yield` fixture teardown
- YAMLBackend: pass `tmp_path / "entries"` directory — created automatically
- Mock SentenceTransformer.encode shape: returns a **1-D list[float]** for str input,
  **2-D list[list[float]]** for list[str] input — must replicate this in `side_effect`
  or the production `[float(v) for v in vector]` raises TypeError and returns None.
- Use `_load_attempted = True` + `_model = None` to put provider in graceful-degradation state
  without touching imports.
