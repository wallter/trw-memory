"""Deterministic linting for wiki page references and confidence metadata."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from trw_memory.wiki.models import WikiPage, validate_wiki_slug

__all__ = [
    "WikiLintFinding",
    "WikiLintReport",
    "lint_wiki_pages",
    "summarize_lint",
]

Severity = Literal["error", "warning"]
FindingCode = Literal[
    "missing_target",
    "asymmetric_ref",
    "invalid_slug",
    "orphan_page",
    "provenance_gap",
    "confidence_gap",
]


class WikiLintFinding(BaseModel):
    """A structured, stable wiki lint finding."""

    model_config = ConfigDict(extra="forbid", strict=True)

    code: FindingCode
    severity: Severity
    page_slug: str
    target_slug: str | None = None
    message: str = Field(min_length=1)


class WikiLintReport(BaseModel):
    """Complete lint report with stable summary and bounded top findings."""

    model_config = ConfigDict(extra="forbid", strict=True)

    findings: list[WikiLintFinding] = Field(default_factory=list)
    summary: dict[str, int] = Field(default_factory=dict)
    top_findings: list[WikiLintFinding] = Field(default_factory=list)


def lint_wiki_pages(
    pages: Sequence[WikiPage | Mapping[str, object]],
    *,
    top_limit: int = 20,
) -> WikiLintReport:
    """Lint wiki pages for bidirectional refs, slug validity, and confidence gaps."""

    validated_pages, findings = _coerce_pages(pages)
    pages_by_slug = {page.slug: page for page in validated_pages if _slug_is_valid(page.slug)}
    inbound: dict[str, set[str]] = {slug: set() for slug in pages_by_slug}
    outbound: dict[str, set[str]] = {slug: set() for slug in pages_by_slug}

    for page in sorted(validated_pages, key=lambda item: item.slug):
        if not _slug_is_valid(page.slug):
            findings.append(
                WikiLintFinding(
                    code="invalid_slug",
                    severity="error",
                    page_slug=page.slug,
                    message="page slug is invalid or unsafe",
                )
            )
            continue
        for ref in sorted(page.outbound_refs, key=lambda item: (item.target_slug, item.ref_type)):
            outbound[page.slug].add(ref.target_slug)
            if ref.target_slug not in pages_by_slug:
                findings.append(
                    WikiLintFinding(
                        code="missing_target",
                        severity="error",
                        page_slug=page.slug,
                        target_slug=ref.target_slug,
                        message=f"wiki page {page.slug!r} references missing target {ref.target_slug!r}",
                    )
                )
                continue
            inbound[ref.target_slug].add(page.slug)
            if ref.bidirectional and page.slug not in {backref.target_slug for backref in pages_by_slug[ref.target_slug].outbound_refs}:
                findings.append(
                    WikiLintFinding(
                        code="asymmetric_ref",
                        severity="warning",
                        page_slug=page.slug,
                        target_slug=ref.target_slug,
                        message=f"wiki page {page.slug!r} references {ref.target_slug!r} without backlink",
                    )
                )

    for page in sorted(validated_pages, key=lambda item: item.slug):
        if not _slug_is_valid(page.slug):
            continue
        if not outbound[page.slug] and not inbound[page.slug]:
            findings.append(
                WikiLintFinding(
                    code="orphan_page",
                    severity="warning",
                    page_slug=page.slug,
                    message=f"wiki page {page.slug!r} has no inbound or outbound references",
                )
            )
        findings.extend(_confidence_findings(page))

    sorted_findings = sorted(findings, key=_finding_sort_key)
    summary: dict[str, int] = dict(sorted(Counter(finding.code for finding in sorted_findings).items()))
    return WikiLintReport(
        findings=sorted_findings,
        summary=summary,
        top_findings=sorted_findings[: max(0, top_limit)],
    )


def summarize_lint(report: WikiLintReport, *, top_limit: int = 20) -> dict[str, object]:
    """Return a stable JSON-compatible summary for CLI/MCP wrappers."""

    top_findings = [
        {
            "code": finding.code,
            "severity": finding.severity,
            "page_slug": finding.page_slug,
            "target_slug": finding.target_slug,
        }
        for finding in report.findings[: max(0, top_limit)]
    ]
    return {
        "summary": dict(sorted(report.summary.items())),
        "total": len(report.findings),
        "top_findings": top_findings,
    }


def _coerce_pages(
    pages: Sequence[WikiPage | Mapping[str, object]],
) -> tuple[list[WikiPage], list[WikiLintFinding]]:
    validated_pages: list[WikiPage] = []
    findings: list[WikiLintFinding] = []
    for index, raw_page in enumerate(pages):
        if isinstance(raw_page, WikiPage):
            validated_pages.append(raw_page)
            continue
        try:
            validated_pages.append(WikiPage.model_validate(raw_page))
        except ValidationError as err:
            raw_slug = str(raw_page.get("slug", f"<page-{index}>"))
            findings.append(
                WikiLintFinding(
                    code="invalid_slug",
                    severity="error",
                    page_slug=raw_slug,
                    message=f"wiki page metadata failed validation: {err.errors()[0]['msg']}",
                )
            )
    return validated_pages, findings


def _confidence_findings(page: WikiPage) -> list[WikiLintFinding]:
    findings: list[WikiLintFinding] = []
    confidence = str(page.confidence)
    promoted = confidence in {"high", "verified"}
    has_source_ref = any(ref.ref_type == "source" for ref in page.outbound_refs)
    if promoted and not page.provenance:
        findings.append(
            WikiLintFinding(
                code="provenance_gap",
                severity="error",
                page_slug=page.slug,
                message="high or verified wiki pages require provenance",
            )
        )
    elif not page.provenance:
        findings.append(
            WikiLintFinding(
                code="provenance_gap",
                severity="warning",
                page_slug=page.slug,
                message="wiki page has no provenance",
            )
        )
    if promoted and (not page.evidence or not has_source_ref):
        findings.append(
            WikiLintFinding(
                code="confidence_gap",
                severity="error",
                page_slug=page.slug,
                message="high or verified wiki pages require evidence and a source reference",
            )
        )
    return findings


def _slug_is_valid(slug: str) -> bool:
    try:
        validate_wiki_slug(slug)
    except ValueError:
        return False
    return True


def _finding_sort_key(finding: WikiLintFinding) -> tuple[str, str, str, str]:
    return (
        finding.page_slug,
        finding.code,
        finding.target_slug or "",
        finding.severity,
    )
