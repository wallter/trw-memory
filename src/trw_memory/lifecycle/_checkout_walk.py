"""The no-follow, dir_fd-anchored checkout enumeration primitive -- split out of ``verification.py``.

PRD-SEC-016 round-8/round-9: this is the ONE place ``verification.py``'s
grep/glob assertion verifiers reach the filesystem. Everything here operates
on an already-open, no-follow-verified directory descriptor (the caller's
``anchor_fd``, opened once per batch via
:func:`trw_memory._dir_trust.open_anchored_walk`) -- nothing in this module
ever re-resolves a path string, so nothing after that anchor open can be
redirected by a later ancestor swap.
"""

from __future__ import annotations

import fnmatch
import itertools
import os
import stat
import time
from dataclasses import dataclass, field
from pathlib import Path

from trw_memory._dir_trust import open_component_fd
from trw_memory._live_stores import close_reader_fd
from trw_memory.exceptions import UntrustedDirectoryError

__all__ = ["_Budget", "_WalkResult", "_read_bytes_through_checkout", "_walk_checkout"]


@dataclass
class _Budget:
    """What one ``verify_assertions`` call may still spend, shared by every walk and match in it (C12).

    ``deadline`` is a ``time.monotonic()`` instant. ``work`` counts down one per directory listing
    and one per listed entry, so neither many small directories nor one huge one runs unbounded.
    """

    deadline: float
    work: int


@dataclass
class _WalkResult:
    """Outcome of one :func:`_walk_checkout` call.

    ``files`` are relative paths whose ENTIRE walk -- every directory
    component and the leaf -- was verified no-follow: nothing in this list
    was reached by resolving a path string, so no downstream containment
    re-check is needed. ``glob_only`` is the same guarantee for a pattern-matching
    candidate that is a real (non-symlink) directory, socket, FIFO or device,
    never opened (C12 rc3: dropping the special ones let ``glob_absent`` pass
    over an existing socket) -- round-10
    review, item 1: ``pathlib.Path.glob`` matches a directory leaf (e.g. a
    bare ``target="src"``), and a glob assertion must too; a directory is
    never useful to ``grep`` (there is nothing to read), so ``_verify_grep``
    consults ``files`` only and ``_verify_glob`` consults both. ``refused``
    names any pattern-matching candidate
    (leaf or an intermediate directory) that was a symlink and so was never
    opened, read, or enumerated through. ``incomplete`` names anything the
    walk encountered but could not fully classify or enumerate -- a
    pattern-matching candidate that vanished (or otherwise failed a fresh
    stat) between being listed and being classified, or a subtree ``**``
    reached but could not scan -- round-9 review: a walk that silently
    treats "could not tell" as "not a match" lets a negative assertion
    (``grep_absent``/``glob_absent``) pass over ground it never actually
    inspected, this framework's own "absence of a measurement is not a
    measurement of absence" rule applied to itself.
    """

    files: list[Path] = field(default_factory=list)
    glob_only: list[Path] = field(default_factory=list)
    refused: list[str] = field(default_factory=list)
    incomplete: list[str] = field(default_factory=list)


def _walk_checkout(anchor_fd: int, target: str, excludes: frozenset[str], budget: _Budget) -> _WalkResult | None:
    """Enumerate files reachable from *anchor_fd* matching the glob-style *target*.

    Replaces ``project_root.glob(target)`` (PRD-SEC-016 round-8 finding 1) and
    the after-the-fact ``path.resolve(strict=True)`` containment check
    (round-4 finding 1, tightened by round-8 finding 2): both were ordinary,
    symlink-following path operations, computed independently of each other
    and of *anchor_fd*'s own no-follow open, so a component swapped for a
    symlink between them was invisible to either. This walks with
    ``os.scandir(dir_fd)`` at each level and descends into a matched
    subdirectory only via :func:`trw_memory._dir_trust.open_component_fd`
    (``O_NOFOLLOW``, anchored on the parent's already-open descriptor) --
    nothing here ever re-resolves a path string, so nothing after
    *anchor_fd*'s own open can redirect the walk.

    A pattern-matching candidate that is a symlink -- checked with
    ``os.DirEntry.is_symlink()``, which never stats what the link points at
    -- is recorded in ``_WalkResult.refused`` and neither opened nor
    descended into, whether it is the final match or an intermediate
    directory component, and regardless of whether its target exists,
    dangles, or is itself unreadable: the classification never depends on
    what is on the other side of the link.

    Supports a literal name, an ``fnmatch``-style wildcard component, and
    ``**`` (zero or more directories, mirroring ``pathlib``'s own meaning --
    including its refusal to descend into a symlinked directory for ``**``).

    A target ending in ``/`` matches directories only, as ``pathlib`` (3.11+) does: splitting
    on ``/`` drops that empty last component, so it is read off *target* first (C12-R).

    C12-R: the walk visits each (directory, remaining pattern) state once, so its work is
    linear in directories times pattern components rather than combinatorial in the ``**``
    count; and every listing is charged to *budget*.

    Returns ``None`` for a target that parses to zero path components
    (empty, ``"."``, or ``"/"``), a top-of-walk failure, or an exhausted *budget* -- the same
    "could not enumerate, so this proves nothing in either direction" contract the old
    ``_iter_files`` upheld for a failed glob.
    """
    raw_parts = tuple(p for p in target.split("/") if p not in ("", "."))
    if not raw_parts:
        return None
    # Round-10 review, item 2: two adjacent "**" components both mean "zero
    # or more directories" -- collapsing a run of them to one BEFORE the walk
    # starts means the recursive-descent branch below is only ever entered
    # once per subtree, so a directory is scanned once, not once per
    # redundant "**" in the pattern.
    parts = tuple(p for i, p in enumerate(raw_parts) if not (p == "**" and i and raw_parts[i - 1] == "**"))
    seen: set[tuple[Path, tuple[str, ...]]] = set()

    files: list[Path] = []
    glob_only: list[Path] = []
    refused: list[str] = []
    incomplete: list[str] = []

    def entries_of(dir_fd: int, prefix: Path, *, strict: bool) -> list[os.DirEntry[str]]:
        """Directory entries of *dir_fd*, filtered by *excludes*.

        *strict* mirrors ``pathlib.Path.glob``'s own tolerance: a failure to
        scan a NESTED directory reached mid-walk (``strict=False``) does not
        abort the whole call the way a TOP-of-walk failure does -- but round-9
        review: it must not be spent as "this subtree has zero matches"
        either, because a ``grep_absent``/``glob_absent`` claim needs to know
        the difference between "verified empty" and "could not look." A
        failure here is recorded in ``incomplete`` (keyed on *prefix*, the
        directory that could not be scanned) and yields no entries -- the
        caller cannot enumerate further, but the RESULT reflects that instead
        of silently agreeing with what an attacker-controlled failure implies.
        A failure at the TOP of the walk (*anchor_fd* itself, ``strict=True``)
        still propagates outright, so the caller reports "could not
        enumerate ... unverified" for the WHOLE assertion rather than a
        partial result.

        C12-R: every listing is charged to *budget* (one unit, plus one per entry, read no further
        than the units left). Running out raises ``TimeoutError`` AFTER the tolerant ``try``, so
        it is never recorded as a merely incomplete subtree: it ends the whole walk unverified.
        """
        try:
            with os.scandir(dir_fd) as listing:
                # Stops at the units left (0 once an earlier walk spent them) or the deadline, whichever first.
                listed = list(itertools.takewhile(_in_time, itertools.islice(listing, max(budget.work, 0))))
        except OSError:
            if strict:
                raise
            incomplete.append(str(prefix))
            return []
        budget.work -= 1 + len(listed)
        if budget.work < 0 or time.monotonic() >= budget.deadline:
            raise TimeoutError
        return [e for e in listed if not (e.name in excludes or e.name.endswith(".egg-info"))]

    def _in_time(_entry: object) -> bool:
        return time.monotonic() < budget.deadline

    def fresh_kind(dir_fd: int, name: str) -> tuple[bool, bool, bool, bool]:
        """A FRESH, no-follow ``fstatat`` of *name* relative to *dir_fd*, taken at the moment of use.

        PRD-SEC-016 round-8 review, item 2: ``os.DirEntry.is_symlink()``/
        ``is_file(follow_symlinks=False)`` describe the entry as ``scandir``
        saw it at LISTING time; classifying a candidate from that cached
        snapshot rather than from a check taken right before it is acted on
        leaves a window between the two for whatever replaced it. This
        issues its own ``os.stat(name, dir_fd=dir_fd, follow_symlinks=False)``
        -- anchored on the SAME already-verified parent descriptor, so
        nothing about *name*'s ancestors can be redirected here -- and reads
        the kind from THAT result, not from ``entries_of``'s listing. A stat
        needs no read/execute permission on *name* itself (only search
        permission on the parent, which the open of *dir_fd* already
        proved), so a permission-denied-but-otherwise-ordinary file is still
        correctly classified as a regular-file candidate here -- it fails
        later, at the actual read, the same way it always has.

        Returns ``(is_regular_file, is_dir, is_symlink, could_not_classify)``.
        The first three are all ``False`` alongside ``could_not_classify=True``
        when *name* could not be stat'd at all (vanished between listing and
        this call, or genuinely inaccessible even to ``lstat``) -- round-9
        review: this is DISTINCT from "stat succeeded and it is simply the
        wrong kind" (a genuine non-match), because a candidate that already
        matched the pattern here and then could not be classified is ground a
        negative assertion never actually inspected, not ground it correctly
        found clean.
        """
        try:
            mode = os.stat(name, dir_fd=dir_fd, follow_symlinks=False).st_mode
        except OSError:
            return False, False, False, True
        return stat.S_ISREG(mode), stat.S_ISDIR(mode), stat.S_ISLNK(mode), False

    def open_subdir(dir_fd: int, name: str) -> int | None:
        try:
            return open_component_fd(dir_fd, name, directory=True)
        except UntrustedDirectoryError:  # trw-fail-silent-allow: every caller of this helper independently records a `refused`/`incomplete` entry for a None return when the candidate already matched the pattern -- nothing about the walk's overall verdict is silently dropped by THIS function
            return None

    def walk(dir_fd: int, remaining: tuple[str, ...], prefix: Path, *, strict: bool) -> None:
        if (prefix, remaining) in seen:  # reached again through another "**" branch: same answer
            return
        seen.add((prefix, remaining))
        head, tail = remaining[0], remaining[1:]
        if head == "**":
            # Zero directories: continue matching the rest of the pattern
            # right here. A trailing "**" alone (no ``tail``) matches every
            # file below this point, which is the same as continuing with a
            # single "*" component. Still the same directory, so still
            # *strict* if the caller was.
            walk(dir_fd, tail or ("*",), prefix, strict=strict)
            # One or more directories: descend into every real directory,
            # keeping "**" (plus whatever follows it) active for the
            # recursive call -- mirrors pathlib's own refusal to follow a
            # symlinked directory for "**" ("**" never names a specific
            # candidate to REPORT as refused, so a symlink -- or a genuine
            # non-directory -- here is simply not descended into, same as
            # any other non-match). A candidate that could not be classified
            # AT ALL, or that looked like a directory but failed the actual
            # no-follow open, is different: "**" reached it but could not
            # confirm what is inside it, so it is recorded as incomplete
            # rather than silently treated as an empty, fully-scanned
            # subtree. A descent is never strict: a permission-denied
            # subtree is tolerated (recorded, not fatal), matching historic
            # glob behavior for the fatal/non-fatal split while still being
            # honest about what it does and does not prove.
            for entry in entries_of(dir_fd, prefix, strict=strict):
                _, is_dir, is_symlink, unclassified = fresh_kind(dir_fd, entry.name)
                if unclassified:
                    incomplete.append(str(prefix / entry.name))
                    continue
                if is_symlink or not is_dir:
                    continue
                sub_fd = open_subdir(dir_fd, entry.name)
                if sub_fd is None:
                    incomplete.append(str(prefix / entry.name))
                    continue
                try:
                    walk(sub_fd, remaining, prefix / entry.name, strict=False)
                finally:
                    os.close(sub_fd)
            return

        for entry in entries_of(dir_fd, prefix, strict=strict):
            if not fnmatch.fnmatch(entry.name, head):
                continue
            rel = prefix / entry.name
            is_regular, is_dir, is_symlink, unclassified = fresh_kind(dir_fd, entry.name)
            if unclassified:
                # This candidate MATCHED the pattern -- a negative assertion
                # needs an answer about it, and "could not stat it" is not
                # one. Round-9 review: the old code silently dropped it here,
                # letting `grep_absent`/`glob_absent` pass over ground it
                # never actually inspected.
                incomplete.append(str(rel))
                continue
            if is_symlink:
                refused.append(str(rel))
                continue
            if tail:
                if not is_dir:
                    continue
                sub_fd = open_subdir(dir_fd, entry.name)
                if sub_fd is None:
                    # The fresh stat above said "directory, not a symlink,"
                    # but the actual no-follow open -- the real gate for
                    # descending -- failed anyway (a race in the gap between
                    # the two, or a kind neither stat nor this open agree on).
                    # Conservative: refuse rather than silently skip a
                    # candidate two independent checks disagreed about.
                    refused.append(str(rel))
                    continue
                try:
                    walk(sub_fd, tail, rel, strict=False)
                finally:
                    os.close(sub_fd)
            elif is_regular and not target.endswith("/"):  # a trailing "/" matches directories only
                files.append(rel)
            elif is_dir or not (is_regular or target.endswith("/")):
                # Round-10 review, item 1: the leaf of the pattern matched a
                # real (non-symlink, already ruled out above) directory --
                # `pathlib.Path.glob` matches a directory leaf too (e.g. a
                # bare `target="src"`), so a glob assertion must as well; so
                # does a socket, FIFO or device (C12 rc3), which is never
                # opened. `grep`'s own caller (`_verify_grep`) never reads
                # `glob_only` -- there is nothing to read -- so this only
                # changes `_verify_glob`'s view.
                glob_only.append(rel)

    try:
        walk(anchor_fd, parts, Path("."), strict=True)
        if time.monotonic() >= budget.deadline:  # per-entry classification ran past it: no verdict
            raise TimeoutError
    except OSError:  # trw-fail-silent-allow: the caller (_verify_grep/_verify_glob) turns a None _WalkResult into "could not enumerate ... unverified" -- never spent as a passing verdict
        return None
    # Evidence counts distinct paths, not distinct WALK PATHS to them (round-9 review, P2). A leaf
    # is collected only in the pattern's one terminal state, which `seen` runs once per directory
    # (C12-R), so `files` and `glob_only` hold no repeats; an intermediate component can be refused or
    # left unclassified by two different pattern heads, so those two are deduplicated here.
    return _WalkResult(files, glob_only, list(dict.fromkeys(refused)), list(dict.fromkeys(incomplete)))


def _read_bytes_through_checkout(anchor_fd: int, rel_path: Path, *, max_bytes: int | None = None) -> bytes | None:
    """*rel_path*'s bytes, read via a fresh no-follow, component-by-component walk off *anchor_fd*; ``None`` if it could not be opened that way.

    PRD-SEC-016 round-8 review item 3: the previous version re-derived its
    own anchor per file via ``open_checkout_file_fd(str(project_root), ...)``,
    which calls :func:`trw_memory._dir_trust.open_anchored_walk` on
    ``project_root``'s STRING fresh EVERY read -- quietly re-opening the
    exact check-then-use window this module's single, batch-lifetime
    *anchor_fd* otherwise closes for enumeration. Walking *rel_path*'s
    components directly off the SAME already-open *anchor_fd* (the identical
    per-component ``open_component_fd`` primitive ``open_checkout_file_fd``
    uses internally, minus the redundant anchor re-open) keeps one anchor as
    the trust boundary for both enumeration and content reads: a component
    swapped for a symlink between ``_walk_checkout``'s enumeration and this
    read is still refused here, not followed, and *project_root*'s own
    ancestors are never re-resolved a second time to do it.

    Round-10 (gpt-6-sol) review: reading a whole matched file before the
    caller's own size cap ran meant an attacker-sized file inside the
    checkout (no symlink needed at all -- this is not a containment bug, just
    an unbounded read) could exhaust the verifier's memory regardless of
    that cap. When *max_bytes* is given, at most ``max_bytes + 1`` bytes are
    read -- one byte past the cap is enough for the caller to detect
    "oversized" without this function needing to know the cap's exact value
    or duplicate the caller's classification.
    """
    parts = rel_path.parts
    if not parts:
        return None
    opened: list[int] = []
    dir_fd = anchor_fd
    try:
        for index, part in enumerate(parts):
            last = index == len(parts) - 1
            try:
                fd = open_component_fd(dir_fd, part, directory=not last)
            except UntrustedDirectoryError:  # trw-fail-silent-allow: caller (_verify_grep) folds a None read into "unsearched", which already makes the assertion unverified rather than silently passing -- see the module's own NEGATIVE-assertion rule
                return None
            opened.append(fd)
            dir_fd = fd
        leaf_fd = opened.pop()
        try:
            # closefd=False: closed through close_reader_fd, releasing its read lease (C15).
            with os.fdopen(leaf_fd, "rb", closefd=False) as handle:
                return handle.read() if max_bytes is None else handle.read(max_bytes + 1)
        except OSError:  # trw-fail-silent-allow: the caller (_verify_grep) folds a None read into "unsearched", which already makes the assertion unverified rather than silently passing -- see the module's own NEGATIVE-assertion rule
            return None
        finally:
            close_reader_fd(leaf_fd)
    finally:
        for fd in opened:
            os.close(fd)
