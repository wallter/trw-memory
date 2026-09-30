"""configure_tracing: the opt-in switch matrix, resource, exporter choice (PRD-CORE-342 FR02, FR04, NFR02)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

from trw_memory import otel_setup


@pytest.fixture
def spy(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Replace the process-global set_tracer_provider with a recorder; reset the once-guard."""
    calls: list[Any] = []
    monkeypatch.setattr(trace, "set_tracer_provider", calls.append)
    monkeypatch.setattr(trace, "get_tracer_provider", trace.ProxyTracerProvider)
    monkeypatch.setattr(otel_setup, "_installed", False)
    for name in ("OTEL_SDK_DISABLED", "OTEL_TRACES_EXPORTER", "OTEL_SEMCONV_STABILITY_OPT_IN"):
        monkeypatch.delenv(name, raising=False)
    yield calls
    for provider in calls:
        provider.shutdown()


def _exporter_of(provider: TracerProvider) -> Any:
    processors = provider._active_span_processor._span_processors  # test-only introspection
    return processors[0].span_exporter._delegate if processors else None


def test_disabled_installs_nothing_and_writes_nothing(spy: list[Any], tmp_path: Path) -> None:
    assert otel_setup.configure_tracing("trw-mcp", tmp_path / "otel", enabled=False) is False
    assert spy == []
    assert not (tmp_path / "otel").exists()


def test_sdk_disabled_env_wins(spy: list[Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
    assert otel_setup.configure_tracing("trw-mcp", tmp_path / "otel", enabled=True) is False
    assert spy == []
    assert not (tmp_path / "otel").exists()


def test_enabled_default_is_bounded_file_exporter(spy: list[Any], tmp_path: Path) -> None:
    assert otel_setup.configure_tracing("trw-mcp", tmp_path / "otel", enabled=True) is True
    (provider,) = spy
    attrs = dict(provider.resource.attributes)
    assert attrs["service.name"] == "trw-mcp"
    assert attrs["service.namespace"] == "trwframework"
    assert attrs["service.version"] == otel_setup.__version__
    assert len(str(attrs["service.instance.id"])) == 36
    assert not any(k.startswith(("host.", "process.")) for k in attrs)
    exporter = _exporter_of(provider)
    assert isinstance(exporter, otel_setup.OtlpJsonFileExporter)
    assert exporter.path.name.startswith("traces-trw-mcp-")
    assert (tmp_path / "otel").stat().st_mode & 0o777 == 0o700
    # once per process
    assert otel_setup.configure_tracing("trw-mcp", tmp_path / "otel", enabled=True) is True
    assert len(spy) == 1


def test_no_file_dir_selects_otlp_http(spy: list[Any]) -> None:
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    assert otel_setup.configure_tracing("trw-memory", None, enabled=True) is True
    assert isinstance(_exporter_of(spy[0]), OTLPSpanExporter)


@pytest.mark.parametrize(("value", "kind"), [("console", "ConsoleSpanExporter"), ("none", None)])
def test_standard_exporter_values(
    spy: list[Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: str, kind: str | None
) -> None:
    monkeypatch.setenv("OTEL_TRACES_EXPORTER", value)
    assert otel_setup.configure_tracing("trw-mcp", tmp_path, enabled=True) is True
    exporter = _exporter_of(spy[0])
    assert (type(exporter).__name__ if exporter else None) == kind


def test_unknown_exporter_fails_open(spy: list[Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTEL_TRACES_EXPORTER", "zipkin")
    assert otel_setup.configure_tracing("trw-mcp", tmp_path, enabled=True) is False
    assert spy == []


def test_semconv_default_set_only_on_install_and_operator_value_wins(
    spy: list[Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os

    otel_setup.configure_tracing("trw-mcp", tmp_path, enabled=False)
    assert "OTEL_SEMCONV_STABILITY_OPT_IN" not in os.environ
    monkeypatch.setenv("OTEL_SEMCONV_STABILITY_OPT_IN", "gen_ai")
    otel_setup.configure_tracing("trw-mcp", tmp_path, enabled=True)
    assert os.environ["OTEL_SEMCONV_STABILITY_OPT_IN"] == "gen_ai"
    monkeypatch.delenv("OTEL_SEMCONV_STABILITY_OPT_IN")
    monkeypatch.setattr(otel_setup, "_installed", False)
    otel_setup.configure_tracing("trw-mcp", tmp_path, enabled=True)
    assert os.environ["OTEL_SEMCONV_STABILITY_OPT_IN"] == "gen_ai_latest_experimental"


def test_sdk_import_error_returns_false(spy: list[Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "opentelemetry.sdk.trace", None)
    assert otel_setup.configure_tracing("trw-mcp", tmp_path / "otel", enabled=True) is False
    assert spy == []
    assert not (tmp_path / "otel").exists()


def test_disabled_call_never_imports_the_sdk(tmp_path: Path) -> None:
    code = (
        "import sys; from pathlib import Path; from trw_memory.otel_setup import configure_tracing;"
        f"assert configure_tracing('trw-mcp', Path({str(tmp_path)!r}) / 'o', enabled=False) is False;"
        "assert not any(m.startswith('opentelemetry.sdk') for m in sys.modules), 'sdk imported'"
    )
    subprocess.run([sys.executable, "-c", code], check=True, timeout=60)


def test_otel_support_copies_are_byte_identical() -> None:
    here = Path(__file__).resolve().parent / "_otel_support.py"
    sibling = Path(__file__).resolve().parents[2] / "trw-mcp" / "tests" / "_otel_support.py"
    if not sibling.is_file():
        pytest.skip("trw-mcp checkout not present beside trw-memory")
    assert here.read_bytes() == sibling.read_bytes()


def test_an_existing_sdk_provider_is_left_alone(
    spy: list[Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    existing = TracerProvider()
    monkeypatch.setattr(trace, "get_tracer_provider", lambda: existing)
    assert otel_setup.configure_tracing("trw-mcp", tmp_path / "otel", enabled=True) is False
    assert spy == []
    assert not (tmp_path / "otel").exists()


def test_console_exporter_writes_to_stderr_never_stdout(
    spy: list[Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OTEL_TRACES_EXPORTER", "console")
    assert otel_setup.configure_tracing("trw-mcp", tmp_path, enabled=True) is True
    assert _exporter_of(spy[0]).out is sys.stderr
