"""The shared write path masks placeholder-prone credential shapes and blocks real tokens."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from trw_memory.exceptions import PIIBlockError
from trw_memory.models.config import MemoryConfig
from trw_memory.models.memory import MemoryEntry
from trw_memory.security._runtime_pii import apply_runtime_pii_policy
from trw_memory.security.credentials import mask_low_confidence


def _entry(**fields: object) -> MemoryEntry:
    return MemoryEntry(id="M-cred", namespace="project:default", **{"content": "note", **fields})  # type: ignore[arg-type]


def test_direct_store_path_masks_key_value_everywhere(tmp_path: Path) -> None:
    cfg = MemoryConfig(storage_path=str(tmp_path / "mem"))
    secured, _ = apply_runtime_pii_policy(
        _entry(
            content="set password=" + "hunter2hunter2 now",
            detail="postgres://" + "app:s3cretpw@db/x",
            tags=["API_KEY=" + "tagsecretvalue"],
            evidence=["token=" + "evsecretvalue1"],
        ),
        cfg,
    )
    dumped = json.dumps(secured.model_dump(mode="json"), default=str)
    for raw in ("hunter2hunter2", "s3cretpw", "tagsecretvalue", "evsecretvalue1"):
        assert raw not in dumped
    assert "<REDACTED:" in secured.content


def test_block_is_decided_on_original_text_not_the_masked_text(tmp_path: Path) -> None:
    cfg = MemoryConfig(storage_path=str(tmp_path / "mem"))
    with pytest.raises(PIIBlockError, match="api_key"):
        apply_runtime_pii_policy(_entry(content="API_KEY=" + "sk-" + "ant-api03-" + "A" * 40), cfg)


@pytest.mark.parametrize(
    ("text", "leftover"),
    [
        ('{"password": "first' + '\\"remainingsecret42"}', "remainingsecret42"),
        ('{"password": "back' + "\\\\" + '", "note": "x"}', "back"),
        ('{"token": "a\\\\\\"b' + 'tail99"}', "tail99"),
    ],
)
def test_json_secret_with_escapes_is_fully_masked(text: str, leftover: str) -> None:
    masked = mask_low_confidence(text)
    assert leftover not in masked
    assert "<REDACTED:json_secret>" in masked


@pytest.mark.parametrize(
    "payload", ['{"password": "' + "a" * 1_000_000, '{"password": "' * 20_000, '{"token": "\\' * 200_000]
)
def test_json_secret_pattern_is_linear_on_unterminated_input(payload: str) -> None:
    started = time.monotonic()
    mask_low_confidence(payload)
    assert time.monotonic() - started < 5.0
