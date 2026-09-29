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
recorded bypass). A tag pushed by hand, a signed tag moved onto another tree, or a
tag replaced after the run was triggered is refused with a named reason: the check
is bound to what is built, never to what the tag name resolves to when it runs. The
release tool runs the same file on the tag object before pushing it, so a tag this
check would refuse never leaves the release host.

    python3 .github/scripts/verify_release_tag.py v1.2.3 --commit SHA [--signers .github/release-signers] [--object SHA]
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ATTESTATION = "TRW-Attestation"
ATTESTATION_VERSION = "1"
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


def verify(repo: Path, tag: str, signers: Path, *, commit: str, obj: str | None = None) -> str | None:
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
    fields = attestation(_git(repo, "cat-file", "tag", ref).stdout)
    if fields.get(ATTESTATION) != ATTESTATION_VERSION:
        return f"{tag}: signed, but carries no '{ATTESTATION}: {ATTESTATION_VERSION}' block"
    if f"v{fields.get('TRW-Version', '')}" != tag:
        return f"{tag}: the attestation is for version {fields.get('TRW-Version')!r}"
    tree = _git(repo, "rev-parse", "--verify", "--quiet", f"{commit}^{{tree}}").stdout.strip()
    if not tree or fields.get("TRW-Tree") != tree:
        return f"{tag}: attested tree {fields.get('TRW-Tree')!r} is not the checked-out tree {tree!r}"
    unproven = proof_problem(fields)
    return f"{tag}: {unproven}" if unproven else None


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
    args = parser.parse_args(argv)
    problem = verify(args.repo, args.tag, args.signers, commit=args.commit, obj=args.object)
    if problem is not None:
        print(f"::error::release attestation refused: {problem}")
        return 1
    print(f"{args.tag}: signed by a listed release signer; attests {args.commit}, the commit being built")
    return 0


if __name__ == "__main__":
    sys.exit(main())
