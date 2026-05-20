"""Tests for deterministic wiki linting and proposal helpers."""

from __future__ import annotations

from trw_memory.wiki.indexer import WikiIndexCandidate, propose_wiki_pages
from trw_memory.wiki.lint import lint_wiki_pages, summarize_lint
from trw_memory.wiki.models import WikiPage, WikiProvenance, WikiReference


def _page(
    slug: str,
    *,
    refs: list[WikiReference] | None = None,
    confidence: str = "medium",
    provenance: list[WikiProvenance] | None = None,
    evidence: list[str] | None = None,
) -> WikiPage:
    return WikiPage(
        kind="topic",
        slug=slug,
        title=slug.replace("/", " ").title(),
        provenance=provenance if provenance is not None else [WikiProvenance(source="human", source_id="fixture")],
        confidence=confidence,
        evidence=evidence if evidence is not None else ["fixture evidence"],
        outbound_refs=refs or [],
    )


def test_lint_reports_missing_targets_missing_backlinks_and_orphans_deterministically() -> None:
    pages = [
        _page(
            "topic/a",
            refs=[
                WikiReference(target_slug="topic/b", ref_type="related"),
                WikiReference(target_slug="topic/missing", ref_type="related"),
            ],
        ),
        _page("topic/b"),
        _page("topic/orphan"),
    ]

    report = lint_wiki_pages(pages)

    assert [(finding.code, finding.page_slug, finding.target_slug) for finding in report.findings] == [
        ("asymmetric_ref", "topic/a", "topic/b"),
        ("missing_target", "topic/a", "topic/missing"),
        ("orphan_page", "topic/orphan", None),
    ]
    assert report.summary == {"asymmetric_ref": 1, "missing_target": 1, "orphan_page": 1}


def test_lint_reports_invalid_slugs_without_shell_execution_or_mutation() -> None:
    invalid = WikiPage.model_construct(kind="topic", slug="../bad;rm", title="Bad", outbound_refs=[])

    report = lint_wiki_pages([invalid])

    assert [(finding.code, finding.severity, finding.page_slug) for finding in report.findings] == [
        ("invalid_slug", "error", "../bad;rm")
    ]


def test_lint_reports_provenance_and_confidence_gaps_for_high_confidence_pages() -> None:
    page = _page("topic/unproven", confidence="verified", provenance=[], evidence=[])

    report = lint_wiki_pages([page])

    assert [(finding.code, finding.severity, finding.page_slug) for finding in report.findings] == [
        ("confidence_gap", "error", "topic/unproven"),
        ("orphan_page", "warning", "topic/unproven"),
        ("provenance_gap", "error", "topic/unproven"),
    ]


def test_lint_allows_verified_page_with_evidence_source_ref_and_provenance() -> None:
    source = WikiPage(
        kind="source",
        slug="sources/prd-core-169",
        title="PRD-CORE-169",
        provenance=[WikiProvenance(source="human", source_id="prd")],
        confidence="verified",
        evidence=["docs/requirements-aare-f/prds/PRD-CORE-169.md"],
        outbound_refs=[WikiReference(target_slug="topic/proven", ref_type="source")],
    )
    page = _page(
        "topic/proven",
        confidence="verified",
        provenance=[WikiProvenance(source="human", source_id="curator")],
        evidence=["focused pytest"],
        refs=[WikiReference(target_slug="sources/prd-core-169", ref_type="source")],
    )

    report = lint_wiki_pages([page, source])

    assert report.findings == []


def test_lint_summary_bounds_top_findings_in_stable_order() -> None:
    report = lint_wiki_pages([_page("topic/a"), _page("topic/b")], top_limit=1)
    summary = summarize_lint(report, top_limit=1)

    assert summary["summary"] == {"orphan_page": 2}
    assert summary["total"] == 2
    assert summary["top_findings"] == [
        {"code": "orphan_page", "severity": "warning", "page_slug": "topic/a", "target_slug": None}
    ]


def test_disabled_proactive_indexing_returns_no_mutation_and_no_proposals() -> None:
    result = propose_wiki_pages(
        [WikiIndexCandidate(title="Should Not Index", source_slug="memory/source")],
        enabled=False,
        dry_run=True,
    )

    assert result.mutated is False
    assert result.proposals == []


def test_dry_run_proactive_indexing_returns_explicit_proposals_without_mutation() -> None:
    result = propose_wiki_pages(
        [
            WikiIndexCandidate(
                title="Bidirectional Reference Linting",
                source_slug="memory/prd-core-169",
                evidence=["PRD-CORE-169"],
            )
        ],
        enabled=True,
        dry_run=True,
        existing_slugs={"topic/already-indexed"},
    )

    assert result.enabled is True
    assert result.dry_run is True
    assert result.mutated is False
    assert [proposal.slug for proposal in result.proposals] == ["topic/bidirectional-reference-linting"]
    assert result.proposals[0].outbound_refs == [
        WikiReference(target_slug="memory/prd-core-169", ref_type="source", bidirectional=False)
    ]
