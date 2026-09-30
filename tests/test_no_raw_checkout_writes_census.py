"""PRD-CORE-337 FR05 -- every raw write into a checkout from trw-memory is accounted for.

``trw_memory.safe_fs`` (``write_beneath``/``append_beneath``) is the one descriptor-anchored,
symlink-refusing way to write into a project checkout. Charter rule Q2 ("class before site") says the
census lands in the SAME slice as the primitive and BEFORE any call site migrates: it starts green by
listing every current raw write as an audited, class-tagged entry. A slice that migrates a site
deletes its row, which is how this test PROVES the migration instead of merely allowing it.

**Scope.** The whole ``trw_memory`` package, except ``safe_fs.py`` itself (the primitive writes with
``os.open``, which the matcher does not count anyway, so the exemption is belt-and-braces). A store
directory is inside a checkout whenever the store is project-scoped (``<project>/.trw/memory``), so
store-dir writes are checkout writes here. trw-mcp's writer trees have their own census
(``trw-mcp/tests/test_no_raw_checkout_writes_census.py``).

**What counts as a raw write** is decided by ``trw_memory._write_census``, the one matcher both
censuses import; its unit tests are the ``test_the_matcher_*`` tests below. Named here, not silently
filtered, are the write shapes it does NOT see: ``os.open`` (flags, not a mode -- including the
``_dir_trust`` descriptor-anchored opens), ``tempfile.mkstemp``/``NamedTemporaryFile`` creation,
``shutil.copy*``/``move`` and ``os.replace``.

**Keying.** ``(relative path from trw_memory, enclosing function qualname, 1-based ordinal within
that scope)`` -- the shape of ``test_direct_sqlite_connect_census.py``.
"""

from __future__ import annotations

import ast
import shutil
from pathlib import Path

import pytest

import trw_memory
from trw_memory._write_census import CLASS_TAGS, Site, census, is_raw_write, ordered_sites, raw_write_sites, report

#: The primitive itself: the one module allowed to write beneath a checkout without an allowlist row.
_PRIMITIVE = "safe_fs.py"


#: (relative path, enclosing qualname, 1-based ordinal) -> (class tag, one-line reason). Every row is a
#: REPORTED residual at the class-before-site stage, not a verified-safe exception (tag legend:
#: ``trw_memory._write_census.CLASS_TAGS``).
_AUDITED_WRITES: dict[Site, tuple[str, str]] = {
    ("cli_storage.py", "handle_import", 1): (
        "operator-named-path",
        "<import-file>.rejected.jsonl beside the file the operator named on the command line.",
    ),
    ("cli_storage.py", "write_export", 1): (
        "operator-named-path",
        "export destination is the operator's --output argument; TRW does not choose a checkout path.",
    ),
    ("cli_storage.py", "write_export", 2): (
        "operator-named-path",
        "export destination is the operator's --output argument; TRW does not choose a checkout path.",
    ),
    ("daemon/_paths.py", "write_secret_file", 1): (
        "own-state-stays",
        "os.fdopen on an O_EXCL|O_NOFOLLOW dir_fd-anchored descriptor in the hardened user daemon dir; not a checkout.",
    ),
    ("integrations/_backend.py", "_write_namespace_metadata", 1): (
        "unscheduled-checkout-write",
        "namespace metadata file inside the store dir, which is under a project's .trw when the store is project-scoped.",
    ),
    ("lifecycle/tiers/_sweep.py", "_sweep_cold_to_purge", 1): (
        "unscheduled-checkout-write",
        "purge_audit.jsonl append via Path.open('a') in the store dir (project .trw when project-scoped).",
    ),
    ("lifecycle/tiers/_warm.py", "WarmTierStore._replace_sidecar", 1): (
        "unscheduled-checkout-write",
        "deterministic .tmp sibling of the warm sidecar written by name, then replaced.",
    ),
    ("lifecycle/tiers/_warm.py", "WarmTierStore._warm_sidecar_upsert_many", 1): (
        "unscheduled-checkout-write",
        "warm-tier sidecar append via Path.open('a') in the store dir.",
    ),
    ("security/audit.py", "AuditLog.compact", 1): (
        "unscheduled-checkout-write",
        "os.fdopen on a mkstemp(dir=log parent) descriptor for the audit log compaction beside the store.",
    ),
    ("storage/_backup_archive.py", "create_backup_archive", 1): (
        "unscheduled-checkout-write",
        "gzip archive into a mkstemp tmp and the .sha256 sidecar beside the backup in the store's backups dir.",
    ),
    ("storage/_backup_archive.py", "create_backup_archive", 2): (
        "unscheduled-checkout-write",
        "gzip archive into a mkstemp tmp and the .sha256 sidecar beside the backup in the store's backups dir.",
    ),
    ("storage/_backup_archive.py", "verified_archive", 1): (
        "unscheduled-checkout-write",
        "NamedTemporaryFile path in the snapshots dir re-opened 'wb' by name for the decompressed (and then verified) restore source.",
    ),
    ("storage/_integrity_scheduler.py", "IntegrityScheduler._write_sentinel", 1): (
        "unscheduled-checkout-write",
        ".integrity_last_check sentinel beside the store db; plain write_text.",
    ),
    ("storage/_recovery_preflight.py", "_write_json_atomic", 1): (
        "unscheduled-checkout-write",
        "os.fdopen on a mkstemp(dir=path.parent) descriptor beside the store; parent re-resolved by name.",
    ),
    ("storage/_stale_handle_detector.py", "write_sentinel", 1): (
        "unscheduled-checkout-write",
        "stale-handle sentinel beside the store db; plain write_text.",
    ),
    ("storage/persistence.py", "append_jsonl", 1): (
        "unscheduled-checkout-write",
        "store JSONL append via Path.open('a'); follows a leaf symlink.",
    ),
    ("storage/persistence.py", "lock_for_rmw", 1): (
        "unscheduled-checkout-write",
        "sibling .lock opened 'a+b' beside a store file; follows a leaf symlink.",
    ),
    ("storage/persistence.py", "write_yaml", 1): (
        "unscheduled-checkout-write",
        "os.fdopen on a mkstemp(dir=path.parent) descriptor for a YAML write beside the store.",
    ),
    ("storage/probe_fixtures.py", "_write", 1): (
        "own-state-stays",
        "contract-test fixture builder, imported by no runtime module; writes only where a test tells it to.",
    ),
    ("tools/checkout_import.py", "_private_checkout_copy", 1): (
        "own-state-stays",
        "os.fdopen on a create_private_file_fd descriptor in the private user import-tmp dir; not a checkout.",
    ),
    ("tools/maintain.py", "_record_stamp", 1): (
        "unscheduled-checkout-write",
        "maintenance-state JSON rewrite beside the store db under lock_for_rmw; plain write_text.",
    ),
    ("tools/namespace_admin.py", "_forget_move_progress", 1): (
        "unscheduled-checkout-write",
        "namespace-move progress JSON rewrite beside the store; plain write_text.",
    ),
    ("tools/namespace_admin.py", "_save_move_progress", 1): (
        "unscheduled-checkout-write",
        "namespace-move progress JSON rewrite beside the store; plain write_text.",
    ),
}


def _sites(package_root: Path | None = None) -> list[Site]:
    """Every raw write in *package_root* (default: the imported ``trw_memory``), the primitive excepted."""
    return raw_write_sites(package_root or Path(trw_memory.__file__).parent, (".",), exclude={_PRIMITIVE})


def _copy_package(destination: Path) -> Path:
    root = destination / "trw_memory"
    shutil.copytree(Path(trw_memory.__file__).parent, root, ignore=shutil.ignore_patterns("__pycache__"))
    return root


def test_every_raw_checkout_write_is_accounted_for() -> None:
    """FR05: a new, unlisted raw write fails by path, qualname and ordinal -- route it through ``safe_fs``."""
    unlisted, _stale = census(_sites(), _AUDITED_WRITES)
    assert unlisted == [], (
        report(unlisted, [])
        + "\nwrite it with trw_memory.safe_fs.write_beneath/append_beneath, or add a class-tagged row"
    )


def test_every_allowlist_row_still_has_its_call_site() -> None:
    """A migrated or moved site must take its row with it: the allowlist tracks source, it does not fossilize."""
    _unlisted, stale = census(_sites(), _AUDITED_WRITES)
    assert stale == [], report([], stale)


def test_every_allowlist_row_is_class_tagged_with_a_one_line_reason() -> None:
    bad = [
        key
        for key, (tag, reason) in _AUDITED_WRITES.items()
        if tag not in CLASS_TAGS or not reason.strip() or "\n" in reason
    ]
    assert bad == []


def test_a_planted_write_text_turns_the_census_red_by_name(tmp_path: Path) -> None:
    """Guard-the-guard, on a COPY of the real package: one planted write is the only failure, named exactly."""
    root = _copy_package(tmp_path)
    with (root / "storage" / "persistence.py").open("a", encoding="utf-8") as handle:
        handle.write("\n\ndef _planted_census_probe(target):\n    target.write_text('planted')\n")

    unlisted, stale = census(_sites(root), _AUDITED_WRITES)

    assert unlisted == [("storage/persistence.py", "_planted_census_probe", 1)]
    assert stale == []
    assert "storage/persistence.py :: _planted_census_probe #1" in report(unlisted, stale)


@pytest.mark.parametrize(("relative", "counted"), [("safe_fs.py", False), ("safe_fs_helpers.py", True)])
def test_only_the_primitive_module_itself_is_exempt(tmp_path: Path, relative: str, counted: bool) -> None:
    root = tmp_path / "pkg"
    root.mkdir()
    (root / relative).write_text("def publish(target):\n    target.write_bytes(b'x')\n", encoding="utf-8")

    assert _sites(root) == ([(relative, "publish", 1)] if counted else [])


def test_dropping_an_allowlist_row_without_its_call_site_turns_the_census_red() -> None:
    if not _AUDITED_WRITES:
        pytest.skip("every audited write has migrated; there is no row left to drop")
    dropped = min(_AUDITED_WRITES)
    remaining = {key: value for key, value in _AUDITED_WRITES.items() if key != dropped}

    unlisted, stale = census(_sites(), remaining)

    assert unlisted == [dropped]
    assert stale == []


def test_deleting_a_call_site_but_keeping_its_row_turns_the_census_red(tmp_path: Path) -> None:
    if not _AUDITED_WRITES:
        pytest.skip("every audited write has migrated; there is no row to go stale")
    root = _copy_package(tmp_path)
    victim = min(_AUDITED_WRITES)[0]
    (root / victim).unlink()

    unlisted, stale = census(_sites(root), _AUDITED_WRITES)

    assert unlisted == []
    assert stale == sorted(key for key in _AUDITED_WRITES if key[0] == victim)


@pytest.mark.parametrize(
    ("source", "counted"),
    [
        ("p.write_text('x')", True),
        ("p.write_bytes(b'x')", True),
        ("open(p, 'a')", True),
        ("open(p, mode='wb')", True),
        ("open(p, 'r+')", True),
        ("open(p, mode)", True),
        ("open(p, mode=mode)", True),
        ("builtins.open('cfg', 'w')", True),
        ("io.open('cfg', 'a')", True),
        ("open('cfg', **{'mode': 'w'})", True),
        ("open('cfg', **options)", True),
        ("open(*args)", True),
        ("os.fdopen(fd, 'w')", True),
        ("os.fdopen(fd, mode)", True),
        ("shutil.copy2(src, dest)", True),
        ("shutil.copyfile(src, dest)", True),
        ("shutil.copy(src, dest)", True),
        ("shutil.copytree(src, dest)", False),
        ("p.open('a+b')", True),
        ("p.open(**options)", True),
        ("gzip.open(p, 'wb')", True),
        ("open(p)", False),
        ("open(p, 'rb')", False),
        ("builtins.open('cfg')", False),
        ("builtins.open('cfg', 'r')", False),
        ("io.open('cfg', 'rb')", False),
        ("os.fdopen(fd)", False),
        ("os.fdopen(fd, 'rb')", False),
        ("p.open()", False),
        ("gzip.open(p, 'rb')", False),
        ("os.open(p, flags)", False),
        ("webbrowser.open('https://example.invalid')", False),
        ("p.read_bytes()", False),
    ],
)
def test_the_matcher_counts_writes_and_only_writes(source: str, counted: bool) -> None:
    tree = ast.parse(f"def f(p, fd, mode, flags, options, args):\n    {source}\n")
    assert ordered_sites(tree, "m.py") == ([("m.py", "f", 1)] if counted else [])
    function = tree.body[0]
    assert isinstance(function, ast.FunctionDef)
    statement = function.body[0]
    assert isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call)
    assert is_raw_write(statement.value) is counted
