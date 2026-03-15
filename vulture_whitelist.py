"""Vulture whitelist for trw-memory package.

Items listed here are used at runtime but appear unused to static analysis.
Vulture uses these entries to suppress false positives.
"""

# --- Pydantic BaseModel fields ---
model_config = None
model_post_init = None
model_validator = None
field_validator = None
model_dump = None

# --- CLI entrypoints ---
main = None

# --- structlog / logging ---
log = None
msg = None

# --- pytest fixtures ---
tmp_path = None
