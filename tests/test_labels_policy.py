"""PRD-SEC-023 FR01/FR02 + NFR02/NFR03: the user-space label policy, a row's label, and admission against a surface or sink.

Every case points ``TRW_USER_DIR`` at a temporary directory, so the operator's real labels.yaml is never read. No daemon, no embedder.
"""

from __future__ import annotations

import itertools
import os
import statistics
import time
from pathlib import Path

import pytest
import structlog

from tests._timing import assert_budget
from trw_memory.labels import Admission, LabelPolicy, Level, Sink, Surface
from trw_memory.models.memory import MemoryEntry

_RULES_YAML = """\
version: 1
auto_surface_max: team
agent_max: personal
rules:
  - tags_any: [finance, Travel]
    level: personal
  - tags_any: [health]
    level: sensitive
  - namespace: "user:local"
    tags_any: [dots]
    level: personal
"""


@pytest.fixture
def user_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    base = tmp_path / "user-base"
    base.mkdir()
    monkeypatch.setenv("TRW_USER_DIR", str(base))
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    return base


def _write(base: Path, text: str, mode: int = 0o600) -> Path:
    path = base / "labels.yaml"
    path.write_text(text, encoding="utf-8")
    path.chmod(mode)
    return path


def _row(namespace: str = "default", tags: tuple[str, ...] = (), **metadata: str) -> MemoryEntry:
    return MemoryEntry(id="L-x", content="c", namespace=namespace, tags=list(tags), metadata=dict(metadata))


# ── FR01: the policy file ────────────────────────────────────────────────────


def test_with_no_file_the_default_policy_changes_nothing(user_dir: Path) -> None:
    policy = LabelPolicy.current()

    assert policy.source == "default"
    plain, stamped_personal, stamped_sensitive = (
        _row("user:local"),
        _row(trw_label="personal"),
        _row(trw_label="sensitive"),
    )
    assert policy.admit([plain], Surface.AUTO).admitted == [plain]
    assert policy.admit([stamped_personal], Surface.AUTO).withheld == 1, "personal is not shown unasked"
    assert policy.admit([stamped_personal], Surface.AGENT).admitted == [stamped_personal]
    assert policy.admit([stamped_sensitive], Surface.AGENT).withheld == 1, "sensitive reaches no agent in phase 0"


def test_a_valid_file_loads_and_its_rules_apply(user_dir: Path) -> None:
    _write(user_dir, _RULES_YAML)

    policy = LabelPolicy.current()

    assert policy.source == "file"
    assert policy.label_of(_row(tags=("finance",))) is Level.PERSONAL
    assert policy.label_of(_row(tags=("health",))) is Level.SENSITIVE
    assert policy.label_of(_row(tags=("code",))) is Level.TEAM


@pytest.mark.parametrize(
    ("reason", "text", "mode"),
    [
        ("unparseable", "version: 1\nrules: [unclosed", 0o600),
        ("oversize", "version: 1\n# " + "x" * 70_000 + "\n", 0o600),
        ("unknown key", "version: 1\nsecret_extra: 1\n", 0o600),
        ("unknown version", "version: 2\n", 0o600),
        ("team rule", "version: 1\nrules:\n  - tags_any: [a]\n    level: team\n", 0o600),
        ("public rule", "version: 1\nrules:\n  - tags_any: [a]\n    level: public\n", 0o600),
        ("rule with no selector", "version: 1\nrules:\n  - level: personal\n", 0o600),
        ("auto above agent", "version: 1\nauto_surface_max: personal\nagent_max: team\n", 0o600),
        ("sensitive agent", "version: 1\nagent_max: sensitive\n", 0o600),
        ("too many rules", "version: 1\nrules:\n" + "  - tags_any: [t]\n    level: personal\n" * 201, 0o600),
        ("group writable", _RULES_YAML, 0o664),
        ("world writable", _RULES_YAML, 0o602),
    ],
)
def test_every_invalid_or_unsafe_file_loads_strict(user_dir: Path, reason: str, text: str, mode: int) -> None:
    _write(user_dir, text, mode)

    policy = LabelPolicy.current()

    assert policy.source == "strict", reason
    assert policy.label_of(_row("user:local")) >= Level.PERSONAL, "strict: every user: row is at least personal"
    assert policy.label_of(_row("default")) is Level.TEAM


def test_a_symlinked_labels_file_is_strict(user_dir: Path, tmp_path: Path) -> None:
    real = tmp_path / "elsewhere.yaml"
    real.write_text(_RULES_YAML, encoding="utf-8")
    real.chmod(0o600)
    (user_dir / "labels.yaml").symlink_to(real)

    assert LabelPolicy.current().source == "strict"


def test_a_strict_load_warns_once_with_the_path_and_error_class_and_never_the_contents(user_dir: Path) -> None:
    secret = "my-private-category-name"
    _write(user_dir, f"version: 1\nrules:\n  - tags_any: [{secret}]\n    level: team\n")

    with structlog.testing.capture_logs() as logs:
        LabelPolicy.current()
        LabelPolicy.current()

    warnings = [e for e in logs if e.get("log_level") == "warning"]
    assert len(warnings) == 1, warnings
    text = repr(warnings[0])
    assert str(user_dir / "labels.yaml") in text and "error_class" in warnings[0]
    assert secret not in text, "a strict warning must never carry the file's contents"


def test_the_user_base_directory_resolves_trw_user_dir_then_xdg_then_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for var in ("TRW_USER_DIR", "XDG_DATA_HOME"):
        monkeypatch.delenv(var, raising=False)
    home, xdg, explicit = tmp_path / "home", tmp_path / "xdg", tmp_path / "explicit"
    (home / ".trw").mkdir(parents=True)
    (xdg / "trw").mkdir(parents=True)
    explicit.mkdir()
    monkeypatch.setenv("HOME", str(home))
    _write(home / ".trw", "version: 1\nrules:\n  - tags_any: [a]\n    level: personal\n")
    assert LabelPolicy.current().label_of(_row(tags=("a",))) is Level.PERSONAL

    monkeypatch.setenv("XDG_DATA_HOME", str(xdg))
    assert LabelPolicy.current().source == "default", "XDG has no file, and the home file is no longer consulted"

    _write(xdg / "trw", "version: 1\nrules:\n  - tags_any: [b]\n    level: sensitive\n")
    assert LabelPolicy.current().label_of(_row(tags=("b",))) is Level.SENSITIVE

    monkeypatch.setenv("TRW_USER_DIR", str(explicit))
    assert LabelPolicy.current().source == "default", "TRW_USER_DIR wins over XDG_DATA_HOME"


def test_a_labels_file_inside_a_project_checkout_is_never_read(
    user_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = tmp_path / "hostile-checkout"
    (checkout / ".trw").mkdir(parents=True)
    _write(checkout / ".trw", "version: 1\nrules:\n  - tags_any: [a]\n    level: sensitive\n")
    _write(checkout, "version: 1\nrules:\n  - tags_any: [a]\n    level: sensitive\n")
    monkeypatch.chdir(checkout)

    assert LabelPolicy.current().source == "default"


def test_an_unchanged_file_costs_one_stat_and_no_parse(user_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from trw_memory.labels import _policy

    _write(user_dir, _RULES_YAML)
    LabelPolicy.current()
    stats, parses = [], []
    real_stat, real_parse = _policy._stat_file, _policy._parse_file
    monkeypatch.setattr(_policy, "_stat_file", lambda path: stats.append(path) or real_stat(path))
    monkeypatch.setattr(_policy, "_parse_file", lambda *a, **k: parses.append(a) or real_parse(*a, **k))

    LabelPolicy.current()
    LabelPolicy.current()

    assert len(stats) == 2 and parses == []


def test_a_changed_file_is_re_read_and_a_deleted_one_falls_back_to_the_default(user_dir: Path) -> None:
    path = _write(user_dir, _RULES_YAML)
    assert LabelPolicy.current().label_of(_row(tags=("finance",))) is Level.PERSONAL

    _write(user_dir, "version: 1\nrules:\n  - tags_any: [finance]\n    level: sensitive\n")
    assert LabelPolicy.current().label_of(_row(tags=("finance",))) is Level.SENSITIVE

    path.unlink()
    assert LabelPolicy.current().source == "default"


# ── FR02: a row's label ──────────────────────────────────────────────────────


def test_label_of_is_the_maximum_of_the_floor_the_namespace_the_rules_and_the_stamp(user_dir: Path) -> None:
    _write(user_dir, _RULES_YAML)
    policy = LabelPolicy.current()

    assert policy.label_of(_row("default")) is Level.TEAM
    assert policy.label_of(_row("user:local")) is Level.TEAM, "user:local is today's behaviour"
    assert policy.label_of(_row("user:alice")) is Level.PERSONAL, "any other user:<name> is inferred personal"
    assert policy.label_of(_row("user:local", tags=("DOTS",))) is Level.PERSONAL, "tag match is case-insensitive"
    assert policy.label_of(_row("default", tags=("dots",))) is Level.TEAM, "that rule is bound to user:local"
    assert policy.label_of(_row("default", tags=("travel",))) is Level.PERSONAL
    assert policy.label_of(_row("default", tags=("finance", "health"))) is Level.SENSITIVE
    assert policy.label_of(_row("default", trw_label="personal")) is Level.PERSONAL
    assert policy.label_of(_row("user:alice", trw_label="team")) is Level.PERSONAL, "a stamp can only raise"


def test_a_namespace_glob_matches_with_fnmatchcase(user_dir: Path) -> None:
    _write(user_dir, 'version: 1\nrules:\n  - namespace: "project:Secret-*"\n    level: sensitive\n')
    policy = LabelPolicy.current()

    assert policy.label_of(_row("project:Secret-abc")) is Level.SENSITIVE
    assert policy.label_of(_row("project:secret-abc")) is Level.TEAM, "the match is case-sensitive"


@pytest.mark.parametrize("stamp", ["", "Secret", "nope", "4", "PERSONAL "])
def test_an_unknown_stamp_is_sensitive(user_dir: Path, stamp: str) -> None:
    assert LabelPolicy.current().label_of(_row(trw_label=stamp)) is Level.SENSITIVE


def test_a_row_whose_evaluation_raises_is_sensitive_and_the_log_names_no_content(user_dir: Path) -> None:
    class _Broken:
        namespace = "default"
        tags: list[str] = []
        metadata = None  # reading the stamp raises
        content = "private-category-text"

    with structlog.testing.capture_logs() as logs:
        level = LabelPolicy.current().label_of(_Broken())  # type: ignore[arg-type]

    assert level is Level.SENSITIVE
    assert "private-category-text" not in repr(logs)


def test_adding_a_rule_or_a_stamp_never_lowers_a_label(user_dir: Path) -> None:
    """Exhaustive: 4 stamp values x 3 policy sources x every subset of a 6-rule fixture."""
    rules = [
        ("tags_any: [a]", "personal"),
        ("tags_any: [b]", "sensitive"),
        ('namespace: "user:*"', "personal"),
        ('namespace: "user:local"\n    tags_any: [c]', "sensitive"),
        ("tags_any: [d, e]", "personal"),
        ('namespace: "default"\n    tags_any: [f]', "sensitive"),
    ]
    entries = [
        _row(ns, tags=tags)
        for ns in ("default", "user:local", "user:bob")
        for tags in ((), ("a",), ("b", "c"), ("d",), ("f",))
    ]
    stamps = [None, "team", "personal", "sensitive"]

    def labels(mask: int, stamp: str | None) -> list[Level]:
        body = "version: 1\nrules:\n" + "".join(
            f"  - {sel}\n    level: {lvl}\n" for i, (sel, lvl) in enumerate(rules) if mask >> i & 1
        )
        if mask == 0:
            body = "version: 1\n"
        _write(user_dir, body)
        policy = LabelPolicy.current()
        assert policy.source == "file"
        out = []
        for entry in entries:
            row = entry if stamp is None else entry.model_copy(update={"metadata": {"trw_label": stamp}})
            out.append(policy.label_of(row))
        return out

    for stamp in stamps:
        by_mask = {mask: labels(mask, stamp) for mask in range(1 << len(rules))}
        for mask, bit in itertools.product(range(1 << len(rules)), range(len(rules))):
            if not mask >> bit & 1:
                assert all(hi >= lo for hi, lo in zip(by_mask[mask | 1 << bit], by_mask[mask], strict=True)), (
                    stamp,
                    mask,
                    bit,
                )

    # a stamp never lowers either, in every source (default, file, strict)
    for source_text in (None, "version: 1\n", "garbage: ["):
        if source_text is None:
            (user_dir / "labels.yaml").unlink(missing_ok=True)
        else:
            _write(user_dir, source_text)
        policy = LabelPolicy.current()
        for entry in entries:
            plain = policy.label_of(entry)
            for stamp in ("team", "personal", "sensitive", "junk"):
                assert policy.label_of(entry.model_copy(update={"metadata": {"trw_label": stamp}})) >= plain


# ── admission against a surface or a sink ────────────────────────────────────


def test_admit_withholds_by_surface_and_counts_without_naming(user_dir: Path) -> None:
    _write(user_dir, _RULES_YAML)
    policy = LabelPolicy.current()
    team, personal, sensitive = _row(tags=("code",)), _row(tags=("finance",)), _row(tags=("health",))
    rows = [team, personal, sensitive]

    auto = policy.admit(rows, Surface.AUTO)
    agent = policy.admit(rows, Surface.AGENT)
    platform = policy.admit(rows, Sink.PLATFORM)

    assert isinstance(auto, Admission)
    assert (auto.admitted, auto.withheld, auto.top) == ([team], 2, Level.TEAM)
    assert (agent.admitted, agent.withheld, agent.top) == ([team, personal], 1, Level.PERSONAL)
    assert (platform.admitted, platform.withheld) == ([team], 2), "sinks clear team only"
    assert policy.admit([], Surface.AGENT) == Admission([], 0, Level.TEAM)


def test_a_user_who_opts_in_can_show_personal_rows_unasked(user_dir: Path) -> None:
    _write(
        user_dir,
        "version: 1\nauto_surface_max: personal\nagent_max: personal\nrules:\n  - tags_any: [finance]\n    level: personal\n",
    )
    personal = _row(tags=("finance",))

    assert LabelPolicy.current().admit([personal], Surface.AUTO).admitted == [personal]


# ── NFR02: the cost of a decision ────────────────────────────────────────────


def _p95_ms(policy: LabelPolicy, rows: list[MemoryEntry], batches: int = 3) -> float:
    """The best of a few batches, so a loaded host cannot fail a budget the code meets: 20 warm-up runs, then 200 timed runs per batch."""
    best = float("inf")
    for _ in range(batches):
        for _ in range(20):
            policy.admit(rows, Surface.AGENT)
        samples = []
        for _ in range(200):
            start = time.perf_counter()
            policy.admit(rows, Surface.AGENT)
            samples.append((time.perf_counter() - start) * 1000)
        best = min(best, statistics.quantiles(samples, n=20)[-1])
    return best


def _bulk_rows() -> list[MemoryEntry]:
    namespaces = ["default", "project:alpha", "project:beta", "user:local"]
    return [_row(namespaces[i % 4], tags=(f"t{i % 40}", f"u{i % 17}", "code")) for i in range(500)]


def _fifty_rules() -> str:
    return "version: 1\nrules:\n" + "".join(
        f"  - tags_any: [t{i}]\n    level: personal\n"
        if i % 2
        else f'  - namespace: "project:*"\n    tags_any: [u{i}]\n    level: sensitive\n'
        for i in range(50)
    )


def test_the_fifty_rule_fixture_file_is_a_valid_policy(user_dir: Path) -> None:
    """The timing test below measures a real file policy, not a strict fallback; that check lives here, unmarked, so the gate runs it."""
    _write(user_dir, _fifty_rules())
    assert LabelPolicy.current().source == "file"


@pytest.mark.requires_local_timing
def test_admit_over_500_rows_and_50_rules_is_fast(user_dir: Path) -> None:
    _write(user_dir, _fifty_rules())
    policy = LabelPolicy.current()

    assert_budget("labels.admit p95, 500 rows, 50 rules", _p95_ms(policy, _bulk_rows()), 2.0, "ms")


@pytest.mark.requires_local_timing
def test_admit_with_the_default_policy_is_faster_still(user_dir: Path) -> None:
    assert_budget("labels.admit p95, 500 rows, default policy", _p95_ms(LabelPolicy.current(), _bulk_rows()), 0.5, "ms")


def test_a_file_is_not_re_read_by_admit_itself(user_dir: Path) -> None:
    _write(user_dir, _RULES_YAML)
    policy = LabelPolicy.current()
    os.remove(user_dir / "labels.yaml")

    assert policy.admit([_row(tags=("finance",))], Surface.AGENT).top is Level.PERSONAL, (
        "a policy object is immutable once built"
    )


def test_a_file_swapped_between_the_check_and_the_open_is_judged_by_what_was_opened(
    user_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """codex r1: the mode, size and kind were checked on one path lookup and the bytes read from another. The opened descriptor is judged."""
    from trw_memory.labels import _file

    _write(user_dir, _RULES_YAML)
    other = tmp_path / "other.yaml"
    other.write_text(_RULES_YAML, encoding="utf-8")
    other.chmod(0o666)  # world-writable: the file the policy must not trust
    real_open = os.open
    monkeypatch.setattr(_file.os, "open", lambda path, flags, *a, **k: real_open(other, flags))

    assert LabelPolicy.current().source == "strict"


@pytest.mark.timeout(10)
def test_a_fifo_swapped_in_after_the_check_cannot_hang_the_load(
    user_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """codex r2: opening a FIFO blocks until a writer appears. The load must not wait; the opened descriptor is judged and found not regular."""
    from trw_memory.labels import _file

    _write(user_dir, _RULES_YAML)
    fifo = tmp_path / "swapped.fifo"
    os.mkfifo(fifo, 0o600)
    real_open = os.open
    monkeypatch.setattr(_file.os, "open", lambda path, flags, *a, **k: real_open(fifo, flags))

    assert LabelPolicy.current().source == "strict"
