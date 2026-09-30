"""Tests for trw-memory package metadata, packaging, and workflow surfaces."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest
from ruamel.yaml import YAML

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility path
    import tomli as tomllib

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT_PATH = PACKAGE_ROOT / "pyproject.toml"
UV_LOCK_PATH = PACKAGE_ROOT / "uv.lock"
# Live, package-scoped workflows (the monorepo-root memory-ci.yml / memory-cd.yml
# these paths used to point at were deleted 2026-09-06; these are their replacements).
MEMORY_CI_PATH = PACKAGE_ROOT / ".github" / "workflows" / "ci.yml"
MEMORY_RELEASE_PATH = PACKAGE_ROOT / ".github" / "workflows" / "release.yml"


def _load_pyproject() -> dict[str, object]:
    with PYPROJECT_PATH.open("rb") as handle:
        return tomllib.load(handle)


def _load_workflow(path: Path) -> dict[str, object]:
    yaml = YAML(typ="safe")
    loaded = yaml.load(path.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _find_step(job: dict[str, object], name: str) -> dict[str, object]:
    steps = job.get("steps")
    assert isinstance(steps, list)
    for step in steps:
        assert isinstance(step, dict)
        if step.get("name") == name:
            return step
    raise AssertionError(f"Step {name!r} not found")


def test_package_importable() -> None:
    """Package is importable."""
    import trw_memory

    assert hasattr(trw_memory, "__version__")
    assert hasattr(trw_memory, "__all__")


def test_version_accessible() -> None:
    """Version string is accessible and well-formed."""
    from trw_memory import __version__

    assert isinstance(__version__, str)
    # Verify semantic version format: major.minor.patch
    parts = __version__.split(".")
    assert len(parts) >= 2, f"Version {__version__} does not look like semver"
    assert all(part.isdigit() for part in parts[:2])


def test_core_exports_exist() -> None:
    """All core exports from __all__ are importable."""
    from trw_memory import (
        ConfigError,
        MemoryConfig,
        MemoryEntry,
        MemoryError,
        MemoryEvent,
        MemoryEventType,
        MemoryIndex,
        MemoryStatus,
        StorageError,
        namespace_to_path,
        validate_namespace,
    )

    assert issubclass(ConfigError, Exception)
    assert issubclass(MemoryConfig, object)
    assert issubclass(MemoryEntry, object)
    assert issubclass(MemoryError, Exception)
    assert issubclass(MemoryEvent, object)
    assert issubclass(MemoryEventType, object)
    assert issubclass(MemoryIndex, object)
    assert issubclass(MemoryStatus, object)
    assert issubclass(StorageError, Exception)
    assert callable(namespace_to_path)
    assert callable(validate_namespace)


def test_all_exports_valid() -> None:
    """Every name in __all__ actually exists in the module."""
    import trw_memory

    for name in trw_memory.__all__:
        assert hasattr(trw_memory, name), f"{name} listed in __all__ but not found"


def test_all_exports_complete() -> None:
    """Public names in __all__ match the declared set."""
    import trw_memory

    expected = {
        "AuthorizationError",
        "ConfigError",
        "DimensionMismatchError",
        "ModelNotCachedError",
        "MemoryClient",
        "MemoryConfig",
        "MemoryConnectionError",
        "MemoryEntry",
        "MemoryError",
        "MemoryEvent",
        "MemoryEventType",
        "MemoryIndex",
        "MemoryQuarantinedError",
        "MemoryNotFoundError",
        "MemoryStatus",
        "PIIBlockError",
        "PoisoningError",
        "RateLimitError",
        "SchemaValidationError",
        "StorageError",
        "StoreBusyError",
        "StoreOp",
        "ToolAlreadyRegisteredError",
        "UnsafeWriteError",
        "UnsupportedStorageError",
        "__version__",
        "append_beneath",
        "namespace_to_path",
        "store_access",
        "validate_namespace",
        "write_beneath",
    }
    assert set(trw_memory.__all__) == expected


def test_exceptions_inherit_properly() -> None:
    """Custom exceptions have correct hierarchy."""
    from trw_memory import (
        AuthorizationError,
        ConfigError,
        DimensionMismatchError,
        MemoryConnectionError,
        MemoryError,
        MemoryNotFoundError,
        ModelNotCachedError,
        PIIBlockError,
        PoisoningError,
        RateLimitError,
        SchemaValidationError,
        StorageError,
        ToolAlreadyRegisteredError,
    )

    assert issubclass(MemoryError, Exception)
    assert issubclass(StorageError, MemoryError)
    assert issubclass(ConfigError, MemoryError)
    assert issubclass(MemoryConnectionError, MemoryError)
    assert issubclass(MemoryNotFoundError, MemoryError)
    assert issubclass(ToolAlreadyRegisteredError, MemoryError)
    assert issubclass(AuthorizationError, MemoryError)
    assert issubclass(DimensionMismatchError, MemoryError)
    assert issubclass(ModelNotCachedError, MemoryError)
    # Store-path exceptions are now top-level exported so callers can catch them
    # without reaching into trw_memory.exceptions (they all subclass MemoryError).
    assert issubclass(SchemaValidationError, MemoryError)
    assert issubclass(PIIBlockError, MemoryError)
    assert issubclass(PoisoningError, MemoryError)
    assert issubclass(RateLimitError, MemoryError)


def test_memory_status_is_enum() -> None:
    """MemoryStatus is an enum with expected values."""
    from trw_memory import MemoryStatus

    assert hasattr(MemoryStatus, "ACTIVE")
    assert hasattr(MemoryStatus, "RESOLVED")
    assert hasattr(MemoryStatus, "OBSOLETE")


def test_cli_help_returns_zero() -> None:
    """The installed CLI module prints help successfully."""
    result = subprocess.run(
        [sys.executable, "-m", "trw_memory.cli", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "usage:" in result.stdout.lower()


def test_pyproject_declares_current_package_contract() -> None:
    """Package metadata matches the shipped trw-memory contract."""
    pyproject = _load_pyproject()
    project = pyproject["project"]
    assert isinstance(project, dict)
    classifiers = project["classifiers"]
    assert isinstance(classifiers, list)

    assert pyproject["build-system"] == {
        "requires": ["hatchling==1.28.0"],  # R9-8: exact pin, the attested wheel is byte-reproducible
        "build-backend": "hatchling.build",
    }
    assert project["name"] == "trw-memory"
    assert project["license"] == "BUSL-1.1"
    # PRD-INFRA-200 FR04: the manifest floor catches up to what the code already
    # enforces (code_index _require_runtime fails closed below 3.11), and 3.10
    # reaches end of life 2026-10.
    assert project["requires-python"] == ">=3.11"
    assert "Programming Language :: Python :: 3.10" not in classifiers
    assert "Programming Language :: Python :: 3.11" in classifiers
    assert "Programming Language :: Python :: 3.12" in classifiers
    assert "Programming Language :: Python :: 3.13" in classifiers
    # 3.14 is declared because it is the interpreter this repository's own suite
    # runs on (CPython 3.14.7). sub_IdK7P9kY3r8XSdWk asked for the tested range to
    # be readable from metadata, since "move to an interpreter with a newer SQLite"
    # is otherwise an unvalidated experiment for the operator (PRD-INFRA-185 FR06).
    assert "Programming Language :: Python :: 3.14" in classifiers


def test_pyproject_declares_current_optional_extras_and_scripts() -> None:
    """Optional extras and scripts expose the current packaging surfaces."""
    pyproject = _load_pyproject()
    project = pyproject["project"]
    assert isinstance(project, dict)
    optional = project["optional-dependencies"]
    assert isinstance(optional, dict)
    scripts = project["scripts"]
    assert isinstance(scripts, dict)

    # `mcp` is deliberately absent: it was folded into [project.dependencies]
    # because the console script hard-imports fastmcp. See
    # test_fastmcp_is_a_REQUIRED_dependency_not_an_extra.
    # `llm`, `langchain`, `llamaindex`, `crewai` and `all-integrations` are
    # deliberately absent: the `anthropic` SDK was never imported anywhere in
    # this package (consolidation summarises with a longest-content heuristic),
    # and the three framework adapters were removed with zero evidenced
    # consumers. Their removal also retired the chromadb and nltk CVE
    # risk-acceptances. See CHANGELOG.md [Unreleased] Removed.
    assert set(optional) == {
        "sqlite-fix",
        "embeddings",
        "otel",  # PRD-CORE-342 FR01: SDK + OTLP/HTTP exporter for entrypoints; installing it never enables export
        "all",
        "dev",
    }
    assert optional["all"] == ["trw-memory[embeddings]"]
    # rank-bm25 is a base dependency (PRD-CORE-302 FR08): the entity-bridge hop
    # reads its model, and EngMem complete@10 fell 25 pp without it.
    assert "rank-bm25>=0.2.0" in project["dependencies"]
    # sqlite-vec is a base dependency (3.1.0): vectors are on for every install,
    # and a platform without a wheel (musl, Windows ARM) fails at pip time
    # rather than silently losing dense recall.
    assert "sqlite-vec>=0.1.5" in project["dependencies"]
    # The retired extras must not come back by name without a deliberate edit
    # here: each one either shipped a dependency nothing imported or pulled a
    # package with an unpatched advisory into a public install.
    # [encryption] went in 4.0: its SQLCipher path never worked with a real driver.
    for retired in ("llm", "langchain", "llamaindex", "crewai", "all-integrations", "vectors", "bm25", "encryption"):
        assert retired not in optional, f"the [{retired}] extra was removed; re-adding it needs a PRD"
    declared = "\n".join(str(value) for value in optional.values())
    assert "chromadb" not in declared, "chromadb has no patched release for GHSA-36p7-vc44-83pf"
    assert "litellm" not in declared
    assert "anthropic" not in declared
    # fastmcp is REQUIRED now, not an extra -- the console script hard-imports it.
    assert "fastmcp>=3.2.0,<4.0.0" in project["dependencies"]
    assert scripts["trw-memory"] == "trw_memory.cli:main"
    assert scripts["trw-memory-server"] == "trw_memory.server:main"


def test_pyproject_mypy_config_is_strict_python_311() -> None:
    """The package keeps strict mypy settings aligned to the minimum Python version."""
    pyproject = _load_pyproject()
    mypy = pyproject["tool"]["mypy"]
    assert isinstance(mypy, dict)

    assert mypy["strict"] is True
    assert mypy["python_version"] == "3.11"
    assert mypy["plugins"] == ["pydantic.mypy"]


def test_pyproject_deptry_config_keeps_static_audit_signal_focused() -> None:
    """Deptry should scan src-layout code without optional-extra false positives."""
    pyproject = _load_pyproject()
    deptry = pyproject["tool"]["deptry"]
    assert isinstance(deptry, dict)

    assert deptry["known_first_party"] == ["trw_memory"]
    assert deptry["optional_dependencies_dev_groups"] == ["dev"]
    # The [encryption] extra (and with it sqlcipher3) went in 4.0; the SQLCipher code went in 4.1.
    assert "package_module_name_map" not in deptry

    per_rule = deptry["per_rule_ignores"]
    assert isinstance(per_rule, dict)
    assert per_rule["DEP001"] == ["torchcodec"]
    # `anthropic` and `crewai` left with the extras that declared them; the
    # self-referential `trw-memory[...]` aggregate is the only remaining
    # unimported declaration besides numpy's version ceiling.
    assert per_rule["DEP002"] == ["numpy", "trw-memory"]
    assert per_rule["DEP003"] == ["nacl"]


def test_pyproject_coverage_omits_server_module() -> None:
    """Package coverage excludes the server entry-point module from the denominator."""
    pyproject = _load_pyproject()
    coverage_run = pyproject["tool"]["coverage"]["run"]
    assert isinstance(coverage_run, dict)
    omit = coverage_run["omit"]
    assert isinstance(omit, list)

    assert "*/server.py" in omit


# The monorepo-root memory-ci.yml / memory-cd.yml these tests used to read were
# deleted 2026-09-06; the live, package-scoped replacements are ci.yml and
# release.yml below (retargeted 2026-09-26 against their real content -- two
# of the five original assertions described jobs (`compat`, a push-path-filtered
# trigger) that no longer exist in any form and were dropped rather than
# retargeted onto something they don't mean).


def test_memory_ci_test_job_uploads_coverage_artifacts() -> None:
    """The full-suite test job emits XML coverage and enforces both coverage gates."""
    workflow = _load_workflow(MEMORY_CI_PATH)
    jobs = workflow["jobs"]
    assert isinstance(jobs, dict)
    test_job = jobs["test"]
    assert isinstance(test_job, dict)
    coverage_step = _find_step(test_job, "Test with coverage")
    upload_step = _find_step(test_job, "Upload coverage XML")
    coverage_gate_step = _find_step(test_job, "Coverage gate (85%, lines + branches)")
    security_step = _find_step(test_job, "INFRA-020 security coverage gate (88%, lines + branches)")

    assert "--cov-report=xml:coverage.xml" in coverage_step["run"]
    assert "--cov-branch" in coverage_step["run"]
    # PRD-SEC-020: every action is pinned to a full commit SHA, never a mutable tag.
    assert re.fullmatch(r"actions/upload-artifact@[0-9a-f]{40}", upload_step["uses"])
    assert upload_step["if"] == "always()"
    assert upload_step["with"]["name"] == "coverage-xml"
    assert upload_step["with"]["path"] == "coverage.xml"
    assert "--fail-under=85" in coverage_gate_step["run"]
    assert "--include='*/trw_memory/security/*'" in security_step["run"]
    assert "--fail-under=88" in security_step["run"]


def test_memory_ci_test_job_runs_mypy_strict() -> None:
    """The test job still type-checks the package with strict mypy before testing."""
    workflow = _load_workflow(MEMORY_CI_PATH)
    jobs = workflow["jobs"]
    assert isinstance(jobs, dict)
    test_job = jobs["test"]
    assert isinstance(test_job, dict)

    typecheck_step = _find_step(test_job, "Type check")

    assert typecheck_step["run"] == "mypy --strict src/trw_memory/"


def test_memory_release_workflow_matches_current_publish_contract() -> None:
    """The release workflow verifies the tag, builds, smoke-tests once, then publishes.

    The mirror runs no test suite (operator direction 2026-09-29): the full-suite proof is
    the local release-check receipt `cut` enforces for this exact tree, so no job calls
    ``ci.yml`` and the smoke test is one ubuntu-latest job on the requires-python floor.
    """
    workflow = _load_workflow(MEMORY_RELEASE_PATH)
    on_config = workflow["on"]
    assert isinstance(on_config, dict)
    push = on_config["push"]
    assert isinstance(push, dict)
    jobs = workflow["jobs"]
    assert isinstance(jobs, dict)

    build_job = jobs["build"]
    smoke_test_job = jobs["smoke-test"]
    publish_job = jobs["publish"]
    assert isinstance(build_job, dict)
    assert isinstance(smoke_test_job, dict)
    assert isinstance(publish_job, dict)

    assert push["tags"] == ["v*"]
    # PRD-SEC-020: least privilege; only the publish job may mint an OIDC token.
    assert workflow["permissions"] == {"contents": "read"}

    # R9-8 attestation v2: build at the tag's attested SOURCE_DATE_EPOCH, then prove the wheel is the attested one.
    build_run = str(_find_step(build_job, "Build package at the attested epoch")["run"])
    assert "--field TRW-Source-Date-Epoch" in build_run
    assert 'SOURCE_DATE_EPOCH="${epoch}" python -m build --wheel' in build_run
    assert 'SOURCE_DATE_EPOCH="${epoch}" python -m build --sdist' in build_run
    wheel_run = str(_find_step(build_job, "Verify the built wheel is the attested one")["run"])
    assert '--wheel "${wheels[0]}"' in wheel_run and 'if [ "${#wheels[@]}" -ne 1 ]' in wheel_run
    assert smoke_test_job["needs"] == "build"
    assert not [name for name, job in jobs.items() if "ci.yml" in str(job.get("uses", ""))]
    assert "strategy" not in smoke_test_job
    assert smoke_test_job["runs-on"] == "ubuntu-latest"
    # PRD-INFRA-198-FR01: publish also waits for the signed-tag attestation check.
    assert publish_job["needs"] == ["verify-attestation", "build", "smoke-test"]
    verify_job = jobs["verify-attestation"]
    assert isinstance(verify_job, dict)
    # The check is bound to the commit this run builds, not to what the tag name resolves to later.
    verify_run = str(_find_step(verify_job, "Verify the release attestation")["run"])
    assert 'verify_release_tag.py "${GITHUB_REF_NAME}" --commit "${GITHUB_SHA}"' in verify_run
    assert '[ "${head}" != "${GITHUB_SHA}" ]' in verify_run
    for job in (verify_job, build_job):
        assert job["steps"][0]["with"]["ref"] == "${{ github.sha }}"
    assert publish_job["environment"] == "pypi"
    assert publish_job["permissions"] == {"id-token": "write"}
    publish_step = next(
        step for step in publish_job["steps"] if step.get("uses", "").startswith("pypa/gh-action-pypi-publish")
    )
    assert re.fullmatch(r"pypa/gh-action-pypi-publish@[0-9a-f]{40}", publish_step["uses"])


def test_package_version_is_semver_like() -> None:
    """Version string follows a semantic-version style contract."""
    from trw_memory import __version__

    assert re.match(r"^\d+\.\d+\.\d+$", __version__)


def _uv_lock_package_version(name: str) -> str:
    """Return the version recorded for ``name`` in ``uv.lock``."""
    with UV_LOCK_PATH.open("rb") as handle:
        lock = tomllib.load(handle)
    packages = lock["package"]
    assert isinstance(packages, list)
    matches = [pkg for pkg in packages if isinstance(pkg, dict) and pkg.get("name") == name]
    assert matches, f"{name!r} not found in uv.lock"
    assert len(matches) == 1, f"{name!r} appears {len(matches)} times in uv.lock"
    version = matches[0]["version"]
    assert isinstance(version, str)
    return version


def _version_tuple(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.split(r"[.+-]", version) if part.isdigit())


def _dependency_names(dependencies: list[str]) -> set[str]:
    """Return normalized package names from PEP 508 dependency strings."""
    return {re.split(r"[<>=!~;\\[]", dep, maxsplit=1)[0].strip().lower().replace("_", "-") for dep in dependencies}


def test_uv_lock_version_matches_pyproject() -> None:
    """The trw-memory package version in uv.lock tracks pyproject.toml.

    Guards the PRD lock-hygiene regression where pyproject was bumped to
    0.8.5 while uv.lock still recorded 0.8.1, so `uv lock --check` failed.
    """
    pyproject = _load_pyproject()
    project = pyproject["project"]
    assert isinstance(project, dict)
    pyproject_version = project["version"]
    assert isinstance(pyproject_version, str)

    assert _uv_lock_package_version("trw-memory") == pyproject_version


def test_pyproject_declares_core_runtime_direct_dependencies() -> None:
    """Core runtime imports must be declared directly, not through transitive deps."""
    pyproject = _load_pyproject()
    project = pyproject["project"]
    assert isinstance(project, dict)
    dependencies = project["dependencies"]
    assert isinstance(dependencies, list)

    # Python 3.10 remains the minimum runtime and core client models import
    # NotRequired/Self from typing_extensions.
    assert "typing-extensions" in _dependency_names(dependencies)


# ``requirements.lock`` is a monorepo build artifact that is untracked and never
# generated by any gate, so the ``skip_if_requirements_lock_absent``-decorated
# tests that used to read it never ran. Enforce the same security intent
# against pyproject.toml instead, which always exists.
def _dependency_lower_bound(dependencies: list[str], name: str) -> tuple[int, ...] | None:
    """Return the ``>=`` lower bound pinned for ``name`` in a PEP 508 dependency list, or ``None``."""
    normalized = name.strip().lower().replace("_", "-")
    for dep in dependencies:
        dep_name = re.split(r"[<>=!~;\[]", dep, maxsplit=1)[0].strip().lower().replace("_", "-")
        if dep_name != normalized:
            continue
        match = re.search(r">=\s*([0-9][0-9.]*)", dep)
        return _version_tuple(match.group(1)) if match else None
    return None


def test_pyproject_fastmcp_pin_is_patched() -> None:
    """pyproject.toml must not allow a vulnerable FastMCP floor."""
    pyproject = _load_pyproject()
    dependencies = pyproject["project"]["dependencies"]
    assert isinstance(dependencies, list)

    floor = _dependency_lower_bound(dependencies, "fastmcp")
    assert floor is not None, "fastmcp has no declared >= lower bound in pyproject.toml"
    assert floor >= (3, 2, 0)


def test_pyproject_security_pin_floors_are_patched() -> None:
    """pyproject.toml's own dependency floors stay above known-patched advisories.

    Only packages trw-memory pins DIRECTLY (runtime or dev) are checked here.
    B71-127: CI installs `.[dev]` unconstrained, and every published
    `pip install trw-memory` does too, so a floor that lives ONLY in uv.lock
    (a package trw-memory never names directly) protects the locked dev venv
    and nothing else. authlib/idna/pygments/pyjwt/python-dotenv used to be
    exactly that (checked only by test_uv_lock_transitive_security_pins_are_patched,
    below); they are declared directly in pyproject.toml now, so they belong here.
    """
    pyproject = _load_pyproject()
    project = pyproject["project"]
    dependencies = project["dependencies"]
    dev_dependencies = project["optional-dependencies"]["dev"]
    assert isinstance(dependencies, list)
    assert isinstance(dev_dependencies, list)
    all_dependencies = [*dependencies, *dev_dependencies]

    floors = {
        "authlib": (1, 6, 12),
        "cryptography": (48, 0, 1),
        "idna": (3, 15),
        "pydantic-settings": (2, 14, 2),
        "pygments": (2, 20, 0),
        "pyjwt": (2, 13, 0),
        "pytest": (9, 0, 3),
        "python-dotenv": (1, 2, 2),
        "python-multipart": (0, 0, 27),
        "starlette": (1, 0, 1),
    }
    for package, floor in floors.items():
        actual = _dependency_lower_bound(all_dependencies, package)
        assert actual is not None, (
            f"{package!r} has no declared >= lower bound in pyproject.toml "
            "(B71-127: a lock-only floor doesn't protect CI's `.[dev]` install or a published install)"
        )
        assert actual >= floor, f"{package!r} floor {actual} is below the patched floor {floor}"


# B71-127: authlib/idna/pygments/pyjwt/python-dotenv used to be checked ONLY here,
# against uv.lock — a floor that binds the locked dev venv and nothing else (not CI's
# unconstrained `.[dev]` install, not a published `pip install trw-memory`). They are
# now declared directly in pyproject.toml (see the "dependencies" list) and covered by
# test_pyproject_security_pin_floors_are_patched above, which binds every install; a
# separate lock-only test for them would just duplicate that coverage with a weaker
# guarantee, so it was removed rather than kept as a second, redundant check.


def test_pyproject_has_no_self_dependency() -> None:
    """trw-memory's own runtime dependencies must not list itself.

    Mirrors the intent of the old requirements.lock check, which guarded a
    frozen `-e git+...trw-framework.git@<sha>#egg=trw_memory` self-pin that
    drifts the moment main advances past <sha>. pyproject.toml has no
    equivalent lock-drift risk, but a runtime self-dependency would still be a
    packaging error.
    """
    pyproject = _load_pyproject()
    dependencies = pyproject["project"]["dependencies"]
    assert isinstance(dependencies, list)

    assert "trw-memory" not in _dependency_names(dependencies)


def test_retired_hypothetical_generation_exports_and_benchmark_absent():
    import importlib

    import trw_memory

    assert not hasattr(trw_memory, "QuestionGenerator")
    assert not hasattr(trw_memory, "NoOpQuestionGenerator")
    with pytest.raises(ImportError):
        importlib.import_module("trw_memory.hype")
    assert not (PACKAGE_ROOT / "benchmarks/bench_hype.py").exists()


# ---------------------------------------------------------------------------
# The console script must be importable using DECLARED dependencies only
# ---------------------------------------------------------------------------


def test_the_console_script_entry_point_imports() -> None:
    """`pip install trw-memory` shipped a `trw-memory` command that could not run.

    `cli.py` imports `cli_namespace`, which imports `daemon.client`, which does
    a hard module-level `from fastmcp import Client`. fastmcp was in the `[mcp]`
    extra, so the entry point raised ModuleNotFoundError on EVERY invocation of
    a core-only install while the README presented that install as supported.
    `import trw_memory` still worked, which is why it went unnoticed.

    This imports the actual module named by `[project.scripts]` rather than
    asserting on a dependency list, so it fails for a NEW undeclared hard
    import too, not only for the one we just fixed.
    """
    import importlib

    import tomllib

    manifest = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8"))
    scripts = manifest["project"]["scripts"]
    assert scripts, "non-vacuity: the package must declare at least one console script"

    for name, target in scripts.items():
        module_path = target.split(":", 1)[0]
        try:
            importlib.import_module(module_path)
        # try/except inside the loop: one entry point per iteration, and a
        # per-script failure message is clearer than a batched one. (PERF203 is
        # not an enabled rule here, so this needs no suppression.)
        except ImportError as exc:
            raise AssertionError(
                f"console script `{name}` points at `{module_path}`, which cannot be "
                f"imported: {exc}. Every import reachable from an entry point must be "
                f"satisfied by [project.dependencies], never by an extra."
            ) from exc


def test_fastmcp_is_a_REQUIRED_dependency_not_an_extra() -> None:
    """Pins the decision so the import above cannot quietly break again.

    Moving fastmcp back into an extra would restore a broken console script,
    and the import test alone would not say why.
    """
    import tomllib

    manifest = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8"))
    required = " ".join(manifest["project"]["dependencies"])
    assert "fastmcp" in required, "fastmcp must be a required dependency"
    extras = manifest["project"].get("optional-dependencies", {})
    assert "mcp" not in extras, "the [mcp] extra was folded into the required set"


def test_pysqlite3_is_an_optional_extra_not_a_hard_dependency() -> None:
    """PRD-INFRA-185 FR01.

    ``pysqlite3-binary`` 0.5.4.post2 publishes exactly one wheel
    (manylinux2014_x86_64), so requiring it on all of Linux made
    ``pip install trw-memory`` fail outright on aarch64 — Graviton, Ampere,
    Apple-Silicon containers, Raspberry Pi (sub_UDHveA9lOKJXTZma). And it bundles
    SQLite 3.51.1, below the 3.51.3 WAL-reset fix it was added to deliver
    (sub_XT5i7XFXI4HbOb0C), so nothing is lost by making it opt-in.
    """
    pyproject = _load_pyproject()
    project = pyproject["project"]
    assert isinstance(project, dict)
    dependencies = project["dependencies"]
    assert isinstance(dependencies, list)
    assert not [dep for dep in dependencies if "pysqlite3" in str(dep)], (
        "pysqlite3-binary must not be a runtime dependency: it has no aarch64 wheel, "
        "so declaring it makes the whole package uninstallable there"
    )

    optional = project["optional-dependencies"]
    assert isinstance(optional, dict)
    extra = optional["sqlite-fix"]
    assert isinstance(extra, list)
    assert len(extra) == 1
    requirement = str(extra[0])
    assert requirement.startswith("pysqlite3-binary>=0.5.4")
    # Marked for the ONE platform that publishes a wheel. On every other platform
    # the extra resolves to nothing and the install still succeeds -- which is the
    # point, but it also means the extra's name promises more than it delivers,
    # and the README says so.
    assert "platform_machine == 'x86_64'" in requirement
    assert "platform_system == 'Linux'" in requirement


def test_readme_declares_supported_interpreters() -> None:
    """PRD-INFRA-185 FR06 — the tested range and the SQLite floor are published."""
    readme = (PACKAGE_ROOT / "README.md").read_text(encoding="utf-8")
    assert "### Supported interpreters" in readme
    assert "3.51.3" in readme
    assert "[sqlite-fix]" in readme
    # The old paragraph claimed a Linux dependency that no longer exists.
    assert "On Linux, `trw-memory` depends on `pysqlite3-binary`" not in readme


def test_submodules_resolve_as_attributes_after_a_plain_import() -> None:
    """The lazy package init still serves ``trw_memory.exceptions`` style access."""
    import trw_memory
    import trw_memory.storage as storage

    assert trw_memory.exceptions.MemoryError is trw_memory.MemoryError
    assert storage.persistence.read_yaml is storage.read_yaml
    with pytest.raises(AttributeError):
        _ = trw_memory.definitely_not_a_module
