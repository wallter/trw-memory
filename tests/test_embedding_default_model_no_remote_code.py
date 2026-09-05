"""PRD-SEC-014-FR03: the shipped default model needs no remote code.

FR02 makes ``embedding_trust_remote_code=False`` the default, which is only a
usable default if the model TRW ships by default loads without it. That was true
by observation; this file makes it an executable invariant so a future default
change cannot silently reintroduce a remote-code dependency.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from trw_memory.embeddings import local as local_mod
from trw_memory.embeddings._hf_cache import CacheState, probe_model_cache

from ._test_hf_cache_support import (
    REMOTE_CODE_SNAPSHOT_FILES,
    NetworkSeam,
    build_model_cache,
    install_fake_sentence_transformers,
    use_fixture_cache,
)

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def _seam(monkeypatch: pytest.MonkeyPatch) -> Iterator[NetworkSeam]:
    installed = NetworkSeam()
    installed.install(monkeypatch)
    yield installed


def test_default_model_needs_no_remote_code(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """FR03: the default snapshot declares no Python module and loads at False."""
    assert local_mod._DEFAULT_MODEL == "all-MiniLM-L6-v2"
    use_fixture_cache(monkeypatch, tmp_path)
    snapshot = build_model_cache(tmp_path)

    probe = probe_model_cache(local_mod._DEFAULT_MODEL)
    assert probe.state is CacheState.COMPLETE
    assert probe.declares_remote_code is False

    captured = install_fake_sentence_transformers(monkeypatch)
    provider = local_mod.LocalEmbeddingProvider()
    assert provider.available() is True
    assert captured["model_name"] == str(snapshot.resolve())
    assert captured["trust_remote_code"] is False


def test_invariant_fails_for_a_default_that_declares_custom_modules(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Non-vacuity: the same guard rejects a remote-code default (FR03 acceptance)."""
    use_fixture_cache(monkeypatch, tmp_path)
    build_model_cache(tmp_path, files=REMOTE_CODE_SNAPSHOT_FILES)

    probe = probe_model_cache(local_mod._DEFAULT_MODEL)
    assert probe.declares_remote_code is True


def test_trw_mcp_default_matches_the_engine_default() -> None:
    """The two shipped defaults must not diverge (FR03 names both surfaces)."""
    config_module = pytest.importorskip("trw_mcp.models.config")
    default = config_module.TRWConfig.model_fields["retrieval_embedding_model"].default
    assert default == local_mod._DEFAULT_MODEL
