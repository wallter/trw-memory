"""The pulled-recall benchmark must verify the embedded arm before labeling it."""

from __future__ import annotations

import pytest


def test_embedded_arm_requires_confirmed_vectors(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    from benchmarks.bench_pulled_recall import build_store
    from trw_memory.tools import store as store_tools
    from trw_memory.tools import sync as sync_tools

    monkeypatch.setattr(
        sync_tools,
        "resolve_embedder",
        lambda _config, *, surface: {"status": "unavailable", "reason": "offline"},
    )
    monkeypatch.setattr(store_tools, "memory_store_impl", lambda *_args, **_kwargs: None)

    with pytest.raises(RuntimeError, match="embedding was not confirmed"):
        build_store(tmp_path, embed_pulled=True, filler=0)
