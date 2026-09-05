"""PRD-SEC-014-FR02/NFR03: one typed field decides ``trust_remote_code``.

Until this PRD the loader computed the flag from ``"nomic-ai/" in
self._model_name``, so any model identifier carrying that vendor prefix opted
the deployment into executing code fetched from the Hub — reachable by a
``.trw/config.yaml`` edit alone. The gate is now the documented, default-False
``embedding_trust_remote_code`` field, and the substring test is deleted.
"""

from __future__ import annotations

import inspect
from collections.abc import Iterator
from pathlib import Path

import pytest

from trw_memory.embeddings import local as local_mod
from trw_memory.embeddings._hf_cache import probe_model_cache
from trw_memory.exceptions import MemoryError as TRWMemoryError
from trw_memory.exceptions import RemoteCodeNotPermittedError
from trw_memory.models.config import MemoryConfig

from ._test_hf_cache_support import (
    REMOTE_CODE_SNAPSHOT_FILES,
    NetworkSeam,
    build_model_cache,
    install_fake_sentence_transformers,
    use_fixture_cache,
)

pytestmark = pytest.mark.integration

_REMOTE_CODE_REPO = "nomic-ai/nomic-embed-text-v1.5"


@pytest.fixture(autouse=True)
def _seam(monkeypatch: pytest.MonkeyPatch) -> Iterator[NetworkSeam]:
    installed = NetworkSeam()
    installed.install(monkeypatch)
    yield installed


def test_vendor_prefixed_model_gets_false_while_field_unset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """FR02: the model name no longer influences the trust decision at all."""
    use_fixture_cache(monkeypatch, tmp_path)
    build_model_cache(tmp_path, _REMOTE_CODE_REPO)
    captured = install_fake_sentence_transformers(monkeypatch)

    provider = local_mod.LocalEmbeddingProvider(model_name=_REMOTE_CODE_REPO)
    assert provider.available() is True
    assert captured["trust_remote_code"] is False


def test_field_true_passes_true(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """FR02: setting the typed field is the one way to reach True."""
    use_fixture_cache(monkeypatch, tmp_path)
    build_model_cache(tmp_path, _REMOTE_CODE_REPO, files=REMOTE_CODE_SNAPSHOT_FILES)
    monkeypatch.setenv("MEMORY_EMBEDDING_TRUST_REMOTE_CODE", "true")
    captured = install_fake_sentence_transformers(monkeypatch)

    provider = local_mod.LocalEmbeddingProvider(model_name=_REMOTE_CODE_REPO)
    assert provider.available() is True
    assert captured["trust_remote_code"] is True


def test_remote_code_model_fails_closed_naming_field(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """FR02: a repo that ships Python modules is refused, naming the field."""
    use_fixture_cache(monkeypatch, tmp_path)
    build_model_cache(tmp_path, _REMOTE_CODE_REPO, files=REMOTE_CODE_SNAPSHOT_FILES)
    install_fake_sentence_transformers(monkeypatch)

    provider = local_mod.LocalEmbeddingProvider(model_name=_REMOTE_CODE_REPO)
    with pytest.raises(RemoteCodeNotPermittedError) as excinfo:
        provider._load_model()

    message = str(excinfo.value)
    assert "embedding_trust_remote_code" in message
    assert _REMOTE_CODE_REPO in message
    assert issubclass(RemoteCodeNotPermittedError, TRWMemoryError)


def test_loader_refusal_is_reraised_as_named_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """RISK-004: inconclusive pre-load detection still reaches the named error."""
    use_fixture_cache(monkeypatch, tmp_path)
    # Snapshot looks clean, but the loader itself demands the flag.
    build_model_cache(tmp_path, _REMOTE_CODE_REPO)
    install_fake_sentence_transformers(
        monkeypatch,
        error=ValueError("Loading this model requires you to pass `trust_remote_code=True`"),
    )

    provider = local_mod.LocalEmbeddingProvider(model_name=_REMOTE_CODE_REPO)
    with pytest.raises(RemoteCodeNotPermittedError, match="embedding_trust_remote_code"):
        provider._load_model()


def test_no_model_name_substring_gate_remains() -> None:
    """NFR03: no membership test against the model name gates the trust flag."""
    source = inspect.getsource(local_mod)
    assert "nomic-ai/" not in source
    assert "in self._model_name" not in source
    load_source = inspect.getsource(local_mod.LocalEmbeddingProvider._load_model)
    assert "trust_remote_code = bool(config.embedding_trust_remote_code)" in load_source


def test_config_field_defaults_false_and_reads_yaml_alias(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """FR02: default False; readable from .trw/config.yaml and the MEMORY_ env."""
    monkeypatch.delenv("MEMORY_EMBEDDING_TRUST_REMOTE_CODE", raising=False)
    monkeypatch.chdir(tmp_path)
    assert MemoryConfig().embedding_trust_remote_code is False

    (tmp_path / ".trw").mkdir()
    (tmp_path / ".trw" / "config.yaml").write_text(
        "embedding_trust_remote_code: true\n",
        encoding="utf-8",
    )
    assert MemoryConfig().embedding_trust_remote_code is True

    (tmp_path / ".trw" / "config.yaml").write_text(
        "memory_embedding_trust_remote_code: true\n",
        encoding="utf-8",
    )
    assert MemoryConfig().embedding_trust_remote_code is True

    (tmp_path / ".trw" / "config.yaml").unlink()
    monkeypatch.setenv("MEMORY_EMBEDDING_TRUST_REMOTE_CODE", "1")
    assert MemoryConfig().embedding_trust_remote_code is True


def test_non_boolean_config_value_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """A malformed value fails validation rather than being coerced to a truth."""
    with pytest.raises(ValueError, match="embedding_trust_remote_code"):
        MemoryConfig(embedding_trust_remote_code="maybe")  # type: ignore[arg-type]


def test_probe_reports_remote_code_declaration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The probe's typed signal is what the loader gates on."""
    use_fixture_cache(monkeypatch, tmp_path)
    build_model_cache(tmp_path, _REMOTE_CODE_REPO, files=REMOTE_CODE_SNAPSHOT_FILES)
    assert probe_model_cache(_REMOTE_CODE_REPO).declares_remote_code is True
