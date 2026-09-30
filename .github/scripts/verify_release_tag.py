#!/usr/bin/env python3
"""Publish only a tag the release tool signed and attested.

A release tag is an annotated tag signed with the release SSH key. Its message
carries an attestation block (``TRW-Attestation: 1``, the version, the package
tree and the local release-check evidence behind it). Before anything is built,
the publish workflow runs this check: the tag must peel to ``--commit`` (the commit
the workflow checked out and builds, ``GITHUB_SHA``), the signature must verify
against the committed ``.github/release-signers``, the attested version must be the
tag's and the attested tree that commit's, and the attested evidence must license a
publish (a passing receipt with full-suite Linux proof or a recorded override, or a
recorded bypass). A v2 attestation also names the interpreter the evidence ran under,
the ``uv.lock`` blob (which must be this commit's) and the wheel built from the tree at
``TRW-Source-Date-Epoch``; ``--wheel`` checks a built wheel against it. A tag pushed by hand, a signed tag moved onto another tree, or a
tag replaced after the run was triggered is refused with a named reason: the check
is bound to what is built, never to what the tag name resolves to when it runs. The
release tool runs the same file on the tag object before pushing it, so a tag this
check would refuse never leaves the release host.

    python3 .github/scripts/verify_release_tag.py v1.2.3 --commit SHA [--signers .github/release-signers] [--object SHA]
        [--wheel dist/pkg.whl] [--field TRW-Source-Date-Epoch]
"""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
from pathlib import Path

ATTESTATION = "TRW-Attestation"
ATTESTATION_VERSION = "2"  # ATTEST-V2-CUTOVER: v1 binds no build, so it can no longer publish
SIGNATURE_START = "-----BEGIN SSH SIGNATURE-----"


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=False)  # noqa: S603,S607


def attestation(tag_object: str) -> dict[str, str]:
    """The ``TRW-*`` fields of a tag object's message (the signature block excluded)."""
    message = tag_object.split("\n\n", 1)[1] if "\n\n" in tag_object else ""
    fields: dict[str, str] = {}
    for line in message.split(SIGNATURE_START, 1)[0].splitlines():
        key, sep, value = line.partition(": ")
        if sep and key.startswith("TRW-"):
            fields[key] = value.strip()
    return fields


def _listed_signers(signers: Path) -> bool:
    lines = signers.read_text(encoding="utf-8").splitlines() if signers.is_file() else []
    return any(line.strip() and not line.lstrip().startswith("#") for line in lines)


def verify(
    repo: Path, tag: str, signers: Path, *, commit: str, obj: str | None = None, wheel: Path | None = None
) -> str | None:
    """Why *tag* (the object *obj*, default ``refs/tags/<tag>``) must not publish as *commit*, or ``None``.

    *commit* is the full id of the commit that will be built; the tag must peel to it and
    attest its tree, so a tag that names another commit never vouches for this build.
    """
    ref = obj or f"refs/tags/{tag}"
    kind = _git(repo, "cat-file", "-t", ref).stdout.strip()
    if kind != "tag":
        return f"{tag} is {'a ' + kind if kind else 'missing'}, not an annotated tag: it carries no release attestation"
    if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", commit):
        return f"{tag}: {commit!r} is not a full commit id to bind the check to"
    target = _git(repo, "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}").stdout.strip()
    if target != commit:
        return f"{tag} names commit {target or '(none)'}, not the commit this run checked out ({commit})"
    if not _listed_signers(signers):
        return f"{signers} lists no release signer, so no tag can verify"
    signed = _git(repo, "-c", f"gpg.ssh.allowedSignersFile={signers.resolve()}", "verify-tag", ref)
    if signed.returncode != 0:
        why = " ".join((signed.stdout + signed.stderr).split())[-300:]
        return f"{tag}: the signature does not verify against {signers} ({why or 'unsigned'})"
    fields = attested_fields(repo, ref)
    if fields.get(ATTESTATION) == "1":
        return f"{tag}: a v1 attestation is no longer accepted; re-cut the release to attest v2 (the built wheel)"
    if fields.get(ATTESTATION) != ATTESTATION_VERSION:
        return f"{tag}: signed, but carries no '{ATTESTATION}: {ATTESTATION_VERSION}' block"
    if f"v{fields.get('TRW-Version', '')}" != tag:
        return f"{tag}: the attestation is for version {fields.get('TRW-Version')!r}"
    tree = _git(repo, "rev-parse", "--verify", "--quiet", f"{commit}^{{tree}}").stdout.strip()
    if not tree or fields.get("TRW-Tree") != tree:
        return f"{tag}: attested tree {fields.get('TRW-Tree')!r} is not the checked-out tree {tree!r}"
    unproven = proof_problem(fields) or build_problem(repo, commit, fields, wheel)
    return f"{tag}: {unproven}" if unproven else None


def attested_fields(repo: Path, ref: str) -> dict[str, str]:
    return attestation(_git(repo, "cat-file", "tag", ref).stdout)


def build_problem(repo: Path, commit: str, fields: dict[str, str], wheel: Path | None) -> str | None:
    """Why the attested build inputs/outputs do not match this commit (and *wheel*), or ``None``."""
    if not fields.get("TRW-Interpreter"):
        return "the attestation names no interpreter"
    listed = _git(repo, "ls-tree", "--object-only", commit, "--", "uv.lock")
    if listed.returncode != 0:  # a git error is not "no lock" (ATTEST-V2-CUTOVER)
        return f"cannot list this commit's uv.lock: {' '.join(listed.stderr.split())[-200:]}"
    lock = listed.stdout.strip() or "none"
    if fields.get("TRW-Lock") != lock:
        return f"attested lock {fields.get('TRW-Lock')!r} is not this commit's uv.lock ({lock})"
    if not re.fullmatch(r"\d+", fields.get("TRW-Source-Date-Epoch", "")):
        return "the attestation names no source date epoch"
    attested = re.fullmatch(r"(\S+\.whl) sha256:([0-9a-f]{64})", fields.get("TRW-Wheel", ""))
    if attested is None:
        return "the attestation names no wheel"
    if wheel is None:
        return None
    if not wheel.is_file():
        return f"no built wheel at {wheel}"
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    if (wheel.name, digest) != (attested.group(1), attested.group(2)):
        return f"built {wheel.name} sha256:{digest} is not the attested {fields['TRW-Wheel']}"
    return None


def proof_problem(fields: dict[str, str]) -> str | None:
    """Why the attested evidence does not license a publish, or ``None``.

    A release-check receipt whose host replay passed, and either a full-suite Linux
    pass or a recorded override; or an explicit, recorded ``--no-ci-replay`` bypass
    (which the release tool accepts only together with a recorded override).
    """
    receipt, override = fields.get("TRW-Receipt", ""), fields.get("TRW-Linux-Override", "").strip()
    if receipt.startswith("none (--no-ci-replay"):
        if not re.fullmatch(r"none \(--no-ci-replay: \S.*\)", receipt):
            return "a --no-ci-replay bypass that records no reason"
        return None if override else "a --no-ci-replay bypass without a recorded Linux override"
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", receipt):
        return "the tag attests no release-check receipt"
    if fields.get("TRW-Host-Replay") != "pass":
        return "the receipt's host replay is not attested as a pass"
    if not re.fullmatch(r"pass \(full suite; \S.*\)", fields.get("TRW-Linux-Leg", "")) and not override:
        return "the receipt has no full-suite Linux proof and no recorded override"
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("tag")
    parser.add_argument("--commit", required=True, help="the full id of the commit being built (GITHUB_SHA)")
    parser.add_argument("--signers", type=Path, default=Path(".github/release-signers"))
    parser.add_argument("--object", default=None, help="verify this tag object instead of refs/tags/<tag>")
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--wheel", type=Path, default=None, help="also require this built wheel to be the attested one")
    parser.add_argument("--field", default=None, help="after verifying, print only this TRW-* field's value")
    args = parser.parse_args(argv)
    problem = verify(args.repo, args.tag, args.signers, commit=args.commit, obj=args.object, wheel=args.wheel)
    if problem is not None:
        print(f"::error::release attestation refused: {problem}")
        return 1
    if args.field is not None:
        value = attested_fields(args.repo, args.object or f"refs/tags/{args.tag}").get(args.field)
        if value is None:
            print(f"::error::release attestation refused: {args.tag} attests no {args.field}")
            return 1
        print(value)
        return 0
    print(f"{args.tag}: signed by a listed release signer; attests {args.commit}, the commit being built")
    return 0


if __name__ == "__main__":
    sys.exit(main())
