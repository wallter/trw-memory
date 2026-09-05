"""Tests for trw-memory package metadata, packaging, and workflow surfaces."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

from ruamel.yaml import YAML

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility path
    import tomli as tomllib

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT_PATH = PACKAGE_ROOT / "pyproject.toml"
UV_LOCK_PATH = PACKAGE_ROOT / "uv.lock"
REQUIREMENTS_LOCK_PATH = PACKAGE_ROOT / "requirements.lock"
MEMORY_CI_PATH = REPO_ROOT / ".github" / "workflows" / "memory-ci.yml"
MEMORY_CD_PATH = REPO_ROOT / ".github" / "workflows" / "memory-cd.yml"


def _load_pyproject() -> dict[str, object]:
    with PYPROJECT_PATH.open("rb") as handle:
        return tomllib.load(handle)


def _load_workflow(path: Path) -> dict[str, object]:
    yaml = YAML(typ="safe")
    loaded = yaml.load(path.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _workflows_enabled() -> bool:
    """Return True when the memory CI/CD YAML is actually active.

    The repo intentionally keeps GitHub Actions commented-out (see the
    header comment in each `.github/workflows/*.yml`). When that's the
    case, `_load_workflow` returns `None` and the workflow-surface
    assertions below would all fail. Skip them until the workflows are
    explicitly enabled by the maintainer.
    """
    yaml = YAML(typ="safe")
    try:
        return isinstance(yaml.load(MEMORY_CI_PATH.read_text(encoding="utf-8")), dict)
    except Exception:
        return False


import pytest  # noqa: E402  (placed here so the skip decorator can reference it)

_WORKFLOW_DISABLED_REASON = (
    "GitHub Actions workflows are commented out by repo policy; "
    "workflow-surface assertions are skipped until they are re-enabled."
)
skip_if_workflows_disabled = pytest.mark.skipif(
    not _workflows_enabled(),
    reason=_WORKFLOW_DISABLED_REASON,
)


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
        EncryptionUnavailableError,
        KeyRotationError,
        MasterKeyNotFoundError,
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
    assert issubclass(EncryptionUnavailableError, Exception)
    assert issubclass(KeyRotationError, Exception)
    assert issubclass(MemoryConfig, object)
    assert issubclass(MemoryEntry, object)
    assert issubclass(MemoryError, Exception)
    assert issubclass(MemoryEvent, object)
    assert issubclass(MemoryEventType, object)
    assert issubclass(MemoryIndex, object)
    assert issubclass(MasterKeyNotFoundError, Exception)
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
        "LocalOnlyViolationError",
        "EncryptionUnavailableError",
        "KeyRotationError",
        "MasterKeyNotFoundError",
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
        "NoOpQuestionGenerator",
        "PIIBlockError",
        "PoisoningError",
        "QuestionGenerator",
        "RateLimitError",
        "SchemaValidationError",
        "StorageError",
        "ToolAlreadyRegisteredError",
        "__version__",
        "namespace_to_path",
        "validate_namespace",
    }
    assert set(trw_memory.__all__) == expected


def test_exceptions_inherit_properly() -> None:
    """Custom exceptions have correct hierarchy."""
    from trw_memory import (
        AuthorizationError,
        ConfigError,
        DimensionMismatchError,
        EncryptionUnavailableError,
        KeyRotationError,
        LocalOnlyViolationError,
        MasterKeyNotFoundError,
        MemoryConnectionError,
        MemoryError,
        MemoryNotFoundError,
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
    assert issubclass(LocalOnlyViolationError, MemoryError)
    assert issubclass(EncryptionUnavailableError, MemoryError)
    assert issubclass(KeyRotationError, MemoryError)
    assert issubclass(MasterKeyNotFoundError, MemoryError)
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
        "requires": ["hatchling>=1.27,<1.29"],
        "build-backend": "hatchling.build",
    }
    assert project["name"] == "trw-memory"
    assert project["license"] == "BUSL-1.1"
    assert project["requires-python"] == ">=3.10"
    assert "Programming Language :: Python :: 3.10" in classifiers
    assert "Programming Language :: Python :: 3.11" in classifiers
    assert "Programming Language :: Python :: 3.12" in classifiers
    assert "Programming Language :: Python :: 3.13" in classifiers


def test_pyproject_declares_current_optional_extras_and_scripts() -> None:
    """Optional extras and scripts expose the current packaging surfaces."""
    pyproject = _load_pyproject()
    project = pyproject["project"]
    assert isinstance(project, dict)
    optional = project["optional-dependencies"]
    assert isinstance(optional, dict)
    scripts = project["scripts"]
    assert isinstance(scripts, dict)

    assert set(optional) == {
        "mcp",
        "encryption",
        "embeddings",
        "vectors",
        "bm25",
        "llm",
        "langchain",
        "llamaindex",
        "crewai",
        "all-integrations",
        "all",
        "dev",
    }
    assert optional["all"] == ["trw-memory[mcp,embeddings,vectors,bm25,llm]"]
    assert optional["all-integrations"] == ["trw-memory[langchain,llamaindex,crewai]"]
    assert "chromadb<1.0" in optional["crewai"]
    assert "litellm>=1.84.0" in optional["crewai"]
    assert "fastmcp>=3.2.0,<4.0.0" in optional["mcp"]
    assert scripts["trw-memory"] == "trw_memory.cli:main"
    assert scripts["trw-memory-server"] == "trw_memory.server:main"


def test_pyproject_mypy_config_is_strict_python_310() -> None:
    """The package keeps strict mypy settings aligned to the minimum Python version."""
    pyproject = _load_pyproject()
    mypy = pyproject["tool"]["mypy"]
    assert isinstance(mypy, dict)

    assert mypy["strict"] is True
    assert mypy["python_version"] == "3.10"
    assert mypy["plugins"] == ["pydantic.mypy"]


def test_pyproject_deptry_config_keeps_static_audit_signal_focused() -> None:
    """Deptry should scan src-layout code without optional-extra false positives."""
    pyproject = _load_pyproject()
    deptry = pyproject["tool"]["deptry"]
    assert isinstance(deptry, dict)

    assert deptry["known_first_party"] == ["trw_memory"]
    assert deptry["optional_dependencies_dev_groups"] == ["dev"]
    assert deptry["package_module_name_map"] == {
        "llama-index-core": "llama_index",
        "langchain-core": "langchain_core",
        "sqlcipher3": "sqlcipher3",
        "crewai": "crewai",
    }

    per_rule = deptry["per_rule_ignores"]
    assert isinstance(per_rule, dict)
    assert per_rule["DEP001"] == ["torchcodec"]
    assert per_rule["DEP002"] == ["sqlcipher3", "anthropic", "crewai", "trw-memory"]
    assert per_rule["DEP003"] == ["nacl"]


def test_pyproject_coverage_omits_server_module() -> None:
    """Package coverage excludes the server entry-point module from the denominator."""
    pyproject = _load_pyproject()
    coverage_run = pyproject["tool"]["coverage"]["run"]
    assert isinstance(coverage_run, dict)
    omit = coverage_run["omit"]
    assert isinstance(omit, list)

    assert "*/server.py" in omit


@skip_if_workflows_disabled
def test_memory_ci_workflow_covers_package_and_workflow_changes() -> None:
    """CI runs for trw-memory changes and for workflow edits that change the contract."""
    workflow = _load_workflow(MEMORY_CI_PATH)
    on_config = workflow["on"]
    assert isinstance(on_config, dict)
    push = on_config["push"]
    pull_request = on_config["pull_request"]
    assert isinstance(push, dict)
    assert isinstance(pull_request, dict)

    assert workflow["name"] == "memory-ci"
    assert push["branches"] == ["main"]
    assert push["paths"] == ["trw-memory/**", ".github/workflows/memory-ci.yml", ".github/workflows/memory-cd.yml"]
    assert pull_request["paths"] == [
        "trw-memory/**",
        ".github/workflows/memory-ci.yml",
        ".github/workflows/memory-cd.yml",
    ]


@skip_if_workflows_disabled
def test_memory_ci_test_job_uploads_coverage_artifacts() -> None:
    """The matrix test job emits both terminal and XML coverage for each Python version."""
    workflow = _load_workflow(MEMORY_CI_PATH)
    jobs = workflow["jobs"]
    assert isinstance(jobs, dict)
    test_job = jobs["test"]
    assert isinstance(test_job, dict)
    matrix = test_job["strategy"]["matrix"]
    assert isinstance(matrix, dict)
    coverage_step = _find_step(test_job, "Run tests with coverage")
    upload_step = _find_step(test_job, "Upload coverage XML")
    security_step = _find_step(test_job, "Run INFRA-020 security coverage gate")

    assert matrix["python-version"] == ["3.10", "3.11", "3.12", "3.13"]
    assert coverage_step["run"].count("--cov-report") == 2
    assert "--cov-report=xml:coverage.xml" in coverage_step["run"]
    assert "--cov-fail-under=85" in coverage_step["run"]
    assert upload_step["uses"] == "actions/upload-artifact@v4"
    assert upload_step["if"] == "always()"
    assert upload_step["with"]["name"] == "coverage-${{ matrix.python-version }}"
    assert upload_step["with"]["path"] == "trw-memory/coverage.xml"
    assert "--cov-branch" in security_step["run"]
    assert "--cov-fail-under=90" in security_step["run"]


@skip_if_workflows_disabled
def test_memory_ci_compat_job_checks_core_and_sqlite_vec_paths() -> None:
    """Compat covers core-only imports, optional extras, and sqlite-vec loading."""
    workflow = _load_workflow(MEMORY_CI_PATH)
    jobs = workflow["jobs"]
    assert isinstance(jobs, dict)
    compat_job = jobs["compat"]
    assert isinstance(compat_job, dict)

    install_core_step = _find_step(compat_job, "Install core package only")
    verify_core_step = _find_step(compat_job, "Verify core import without optional deps")
    verify_missing_step = _find_step(compat_job, "Verify optional imports fail gracefully")
    install_all_step = _find_step(compat_job, "Install optional compatibility extras")
    sqlite_vec_step = _find_step(compat_job, "Verify sqlite-vec runtime compatibility")
    imports_step = _find_step(compat_job, "Verify representative optional imports")

    assert install_core_step["run"] == "pip install -e ."
    assert "Core exports OK" in verify_core_step["run"]
    assert "ImportError" in verify_missing_step["run"]
    assert install_all_step["run"] == 'pip install -e ".[all]"'
    assert "sqlite_vec.load(conn)" in sqlite_vec_step["run"]
    assert "SKIPPED: sqlite-vec requires SQLite >= 3.35.0" in sqlite_vec_step["run"]
    for module_name in ("anthropic", "sentence_transformers", "sqlite_vec"):
        assert module_name in imports_step["run"]


@skip_if_workflows_disabled
def test_memory_ci_typecheck_job_uses_python_310_and_mypy_strict() -> None:
    """Typecheck is pinned to the minimum supported runtime with strict mypy."""
    workflow = _load_workflow(MEMORY_CI_PATH)
    jobs = workflow["jobs"]
    assert isinstance(jobs, dict)
    typecheck_job = jobs["typecheck"]
    assert isinstance(typecheck_job, dict)

    setup_step = _find_step(typecheck_job, "Set up Python 3.10")
    typecheck_step = _find_step(typecheck_job, "Run mypy --strict")

    assert setup_step["with"]["python-version"] == "3.10"
    assert typecheck_step["run"] == "mypy --strict src/trw_memory/"


@skip_if_workflows_disabled
def test_memory_cd_workflow_matches_current_release_contract() -> None:
    """The release workflow builds, signs, publishes, and updates the changelog."""
    workflow = _load_workflow(MEMORY_CD_PATH)
    on_config = workflow["on"]
    assert isinstance(on_config, dict)
    push = on_config["push"]
    assert isinstance(push, dict)
    jobs = workflow["jobs"]
    assert isinstance(jobs, dict)

    build_job = jobs["build"]
    sign_job = jobs["sign"]
    publish_job = jobs["publish-codeartifact"]
    changelog_job = jobs["changelog"]
    assert isinstance(build_job, dict)
    assert isinstance(sign_job, dict)
    assert isinstance(publish_job, dict)
    assert isinstance(changelog_job, dict)

    assert workflow["name"] == "memory-cd"
    assert push["tags"] == ["trw-memory-v*"]
    assert workflow["permissions"] == {"contents": "write", "id-token": "write"}

    assert _find_step(build_job, "Build wheel and sdist")["run"] == "python -m build"
    assert "find dist" in _find_step(build_job, "Verify build artifacts exist")["run"]
    assert "sha256sum dist/*" in _find_step(build_job, "Compute checksums")["run"]
    assert _find_step(sign_job, "Download build artifacts")["with"]["path"] == "dist/"
    assert "Install sigstore" not in {step.get("name") for step in sign_job["steps"]}
    assert _find_step(sign_job, "Sign artifacts")["uses"] == "sigstore/gh-action-sigstore-python@v3"
    assert _find_step(publish_job, "Download build artifacts")["with"]["path"] == "trw-memory/dist/"
    assert "Missing required secret(s):" in _find_step(publish_job, "Validate required publishing secrets")["run"]
    assert _find_step(publish_job, "Configure AWS credentials")["uses"] == "aws-actions/configure-aws-credentials@v4"
    assert "twine upload dist/*.whl dist/*.tar.gz" in _find_step(publish_job, "Publish to CodeArtifact")["run"]
    changelog_run = _find_step(changelog_job, "Generate changelog")["run"]
    commit_run = _find_step(changelog_job, "Commit changelog")["run"]
    assert "CHANGELOG.md" in changelog_run
    assert '"Features"' in changelog_run
    assert '"Bug Fixes"' in changelog_run
    assert '"Other"' in changelog_run
    assert '"Uncategorized"' in changelog_run
    assert "TODO" not in changelog_run
    assert "git add CHANGELOG.md" in commit_run
    assert "git push origin HEAD:main" in commit_run
    assert "github-actions[bot]" in commit_run


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


def _requirements_lock_package_version(name: str) -> str:
    match = re.search(
        rf"^{re.escape(name)}==([^\s]+)$",
        REQUIREMENTS_LOCK_PATH.read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    assert match is not None, f"{name!r} not found in requirements.lock"
    return match.group(1)


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


# ``requirements.lock`` is a monorepo build artifact; it is NOT shipped in the
# standalone public mirror (github.com/wallter/trw-memory). Guard the lock-pin
# tests so they enforce in the monorepo but skip cleanly in the mirror CI.
skip_if_requirements_lock_absent = pytest.mark.skipif(
    not REQUIREMENTS_LOCK_PATH.exists(),
    reason="requirements.lock is a monorepo build artifact, absent in the standalone mirror",
)


@skip_if_requirements_lock_absent
def test_requirements_lock_fastmcp_pin_is_patched() -> None:
    """requirements.lock must not pin vulnerable FastMCP releases."""
    assert _version_tuple(_requirements_lock_package_version("fastmcp")) >= (3, 2, 0)


@skip_if_requirements_lock_absent
def test_requirements_lock_security_pin_floors_are_patched() -> None:
    """Known-audited requirements.lock pins stay above patched floors."""
    floors = {
        "Authlib": (1, 6, 12),
        "cryptography": (48, 0, 1),
        "idna": (3, 15),
        "Pygments": (2, 20, 0),
        "PyJWT": (2, 13, 0),
        "pydantic-settings": (2, 14, 2),
        "pytest": (9, 0, 3),
        "python-dotenv": (1, 2, 2),
        "python-multipart": (0, 0, 27),
        "starlette": (1, 0, 1),
    }
    for package, floor in floors.items():
        assert _version_tuple(_requirements_lock_package_version(package)) >= floor


@skip_if_requirements_lock_absent
def test_requirements_lock_has_no_stale_self_pin() -> None:
    """requirements.lock must not pin trw-memory to a frozen git commit.

    A `-e git+...trw-framework.git@<sha>#egg=trw_memory` self-pin drifts the
    moment main advances past <sha>; the editable self-reference is normalised
    to a path install (`-e .`) so it never goes stale.
    """
    text = REQUIREMENTS_LOCK_PATH.read_text(encoding="utf-8")

    stale_self_pin = re.compile(
        r"^-e\s+git\+.*trw-framework\.git@[0-9a-f]{7,40}.*egg=trw_memory",
        re.MULTILINE,
    )
    assert not stale_self_pin.search(text), (
        "requirements.lock pins trw-memory to a frozen git commit; "
        "use an editable path reference (`-e .`) instead so it does not drift."
    )
