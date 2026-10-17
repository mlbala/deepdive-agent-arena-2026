"""Agent 7 — ReportApproverAgent: the final approver that re-verifies the synthesized report.

Every claim is verified before the Synthesizer sees it, but synthesis rewrites claims into prose,
and a reworded sentence can drift from what its source says. The Report Approver closes that gap:
it checks every cited sentence of the report (executive summary, key findings, sections) against
the spans of the sources it cites, with the Verifier's own guards (verbatim quote, numbers locked,
citation scope, authority floor).

    synthesizer -> report_approver -> synthesizer   fix the failing sentences (REPORT_AUDIT_FIXES rounds)
                                   -> output_guard  approve, or remove what still fails, then publish

A sentence that is word for word a verified claim (same text, citations within that claim's)
carries the claim's verdict and verbatim quote: the Verifier already checked it against the same
source with the same guards, and asking the model again only adds noise. Everything the
Synthesizer reworded is judged again, and identical sentences are judged once per check so
repeated sentences always get the same verdict.

Sentences without citations are meta statements (the citation guard already removed uncited
facts), and sentences that state a conflict between sources were quote-validated by the
contradiction radar; both are recorded as skipped rather than judged, so conflicts are never
stripped from the report.

The checks fail open on infrastructure, never on evidence: a sentence the model could not judge
(the Mesh call failed) restates verified claims, so it is kept and flagged `unchecked` (and
re-checked if another round happens) instead of being removed as if its source disagreed.
"""

from __future__ import annotations

import re

from src.agents.base import EventCollector
from src.agents.verifier import NO_JUDGEMENT, VerifierAgent
from src.core.config import Settings
from src.core.mesh_gateway import MeshGateway
from src.schemas.domain import (
    AtomicClaim,
    AuditedSentence,
    AuditEventType,
    FinalReport,
    ReportAudit,
    SourceDocument,
    SubQuestion,
    Verdict,
    VerificationResult,
    extract_citations,
)
from src.storage.vector_store import HybridVectorStore

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
_CITATION = re.compile(r"\s*\[[^\]]*SRC[^\]]*\]")
_QUOTE_TAG = re.compile(r"\s*\[quote match: [^\]]*\]\s*$")
_CONFLICT_MARKERS = ("conflict", "contradict", "contested", "whereas", "disagree", "on the other hand", "however,")


def _split(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(text.strip()) if s.strip()]


def report_sentences(report: FinalReport) -> list[tuple[str, int, str]]:
    """Every sentence of the published narrative as (field, index, sentence), in reading order.

    field is `executive_summary`, `key_finding:<i>` or `section:<i>` (index into `report.sections`);
    the second element is the sentence's position inside that field.
    """
    out: list[tuple[str, int, str]] = [("executive_summary", i, s) for i, s in enumerate(_split(report.executive_summary))]
    for f, finding in enumerate(report.key_findings):
        out += [(f"key_finding:{f}", i, s) for i, s in enumerate(_split(finding))]
    for k, section in enumerate(report.sections):
        out += [(f"section:{k}", i, s) for i, s in enumerate(_split(section.body))]
    return out


def _norm(text: str) -> str:
    return " ".join(_CITATION.sub("", text).lower().split()).rstrip(" .")


def verified_index(
    claims: dict[str, list[AtomicClaim]], verifications: dict[str, list[VerificationResult]]
) -> dict[str, tuple[AtomicClaim, VerificationResult]]:
    """Supported, quote-verified claims keyed by their normalised text."""
    verdict_of = {v.claim_id: v for vs in verifications.values() for v in vs}
    index: dict[str, tuple[AtomicClaim, VerificationResult]] = {}
    for sq_claims in claims.values():
        for claim in sq_claims:
            v = verdict_of.get(claim.claim_id)
            if v is not None and v.verdict is Verdict.SUPPORTED and v.quote_verified:
                index.setdefault(_norm(claim.text), (claim, v))
    return index


def failure_reason(result: VerificationResult) -> str:
    """Why a sentence failed, in words, even when the model gave no rationale."""
    if result.downgrade_reason:
        return result.downgrade_reason
    rationale = _QUOTE_TAG.sub("", result.rationale).strip()
    if rationale:
        return rationale
    source = result.evidence_source_id or "its cited source"
    return f"contradicted by {source}" if result.verdict is Verdict.CONTRADICTED else f"{source} does not state this"


def location_label(report: FinalReport, field: str) -> str:
    if field.startswith("section:"):
        return f"section:{report.sections[int(field.split(':')[1])].sub_question_id}"
    return field


def rewrite_report(report: FinalReport, changes: dict[str, str | None]) -> FinalReport:
    """Apply sentence replacements (str) and removals (None), keyed by sentence id S1, S2, ...

    Sentence ids are positional over `report_sentences`, so they are resolved against this exact report.
    """
    positions = {f"S{n}": (field, idx) for n, (field, idx, _s) in enumerate(report_sentences(report), 1)}
    by_field: dict[str, list[str]] = {}
    for field, _idx, sentence in report_sentences(report):
        by_field.setdefault(field, []).append(sentence)
    for sid, new in changes.items():
        if sid in positions:
            field, idx = positions[sid]
            by_field[field][idx] = (new or "").strip()

    def joined(field: str) -> str:
        return " ".join(s for s in by_field.get(field, []) if s)

    findings = [joined(f"key_finding:{f}") for f in range(len(report.key_findings))]
    sections = []
    for k, section in enumerate(report.sections):
        body = joined(f"section:{k}") or "The available evidence was insufficient to answer this sub-question."
        sections.append(section.model_copy(update={"body": body, "cited_source_ids": extract_citations(body)}))
    summary = joined("executive_summary") or "The available evidence was insufficient to answer the research question."
    return report.model_copy(update={
        "executive_summary": summary,
        "key_findings": [f for f in findings if f],
        "sections": sections,
    })


class _ReportVerifier(VerifierAgent):
    """The Verifier's entailment machinery, billed to the Report Approver in traces and budgets."""

    name = "report_approver"


class ReportApproverAgent:
    name = "report_approver"

    def __init__(self, gateway: MeshGateway, settings: Settings, vector_store: HybridVectorStore) -> None:
        self.settings = settings
        self.checker = _ReportVerifier(gateway, settings, vector_store)

    def audit(
        self,
        run_id: str,
        report: FinalReport,
        sources: dict[str, SourceDocument],
        *,
        audit_round: int,
        known: dict[str, AuditedSentence] | None = None,
        verified: dict[str, tuple[AtomicClaim, VerificationResult]] | None = None,
        collector: EventCollector | None = None,
    ) -> ReportAudit:
        """Check every cited sentence; reuse verdicts for sentences already checked word for word."""
        known, verified = known or {}, verified or {}
        first_of: dict[str, AuditedSentence] = {}  # identical sentences are judged once per check
        copies: list[tuple[AuditedSentence, AuditedSentence]] = []
        sentences: list[AuditedSentence] = []
        pending: list[tuple[AuditedSentence, AtomicClaim, SubQuestion]] = []
        sq_by_id = {sq.id: sq for sq in report.sub_questions}
        overview = SubQuestion(id="SQ-0", question=(report.query or "research question")[:300] or "research question",
                               search_queries=["overview"])
        for n, (field, _idx, sentence) in enumerate(report_sentences(report), 1):
            cited = extract_citations(sentence)
            location = location_label(report, field)
            item = AuditedSentence(sid=f"S{n}", location=location, text=sentence, cited_source_ids=cited)
            body = _CITATION.sub("", sentence).strip()
            if not cited:
                item.status, item.reason = "skipped", "meta statement without a factual claim"
            elif any(marker in sentence.lower() for marker in _CONFLICT_MARKERS):
                item.status, item.reason = "skipped", "states a conflict between sources, quote-validated by the contradiction radar"
            elif sentence in known:
                prior = known[sentence]
                item = prior.model_copy(update={"sid": item.sid, "location": location})
            elif (hit := verified.get(_norm(sentence))) and set(cited) <= {*hit[0].cited_source_ids, hit[1].evidence_source_id}:
                claim, result = hit
                item.verdict, item.quote_verified = Verdict.SUPPORTED, result.quote_verified
                item.evidence_source_id, item.evidence_quote = result.evidence_source_id, result.evidence_quote
                item.status = "fixed" if audit_round > 1 else "supported"
                item.reason = f"word for word verified claim {claim.claim_id}"
            elif sentence in first_of:
                copies.append((item, first_of[sentence]))
            else:
                first_of[sentence] = item
                sq_id = location.split(":", 1)[1] if location.startswith("section:") else "SQ-0"
                claim = AtomicClaim(claim_id=f"RPT-{audit_round}.{n}", sub_question_id=sq_id, iteration=max(1, audit_round),
                                    text=body if len(body) >= 3 else sentence, cited_source_ids=cited)
                pending.append((item, claim, sq_by_id.get(sq_id, overview)))
            sentences.append(item)

        llm_reviewed = True
        scratch = EventCollector(run_id, "report_approver", audit_round)  # verdict events stay out of the claim matrix
        for sq_id in dict.fromkeys(sq.id for _i, _c, sq in pending):
            group = [(i, c, sq) for i, c, sq in pending if sq.id == sq_id]
            results = self.checker.judge_claims(run_id, group[0][2], [c for _i, c, _sq in group], sources, scratch)
            for (item, _claim, _sq), result in zip(group, results, strict=True):
                if result.rationale == NO_JUDGEMENT:
                    item.status = "unchecked"
                    item.reason = "not checked: the Mesh entailment call failed; kept because it restates verified claims"
                    continue
                item.verdict = result.verdict
                item.quote_verified = result.quote_verified
                item.evidence_source_id = result.evidence_source_id
                item.evidence_quote = result.evidence_quote
                ok = result.verdict is Verdict.SUPPORTED
                # after the first check, a newly judged sentence is the Synthesizer's fix
                item.status = ("fixed" if audit_round > 1 else "supported") if ok else "failed"
                item.reason = result.rationale if ok else failure_reason(result)
        for item, original in copies:
            item.verdict, item.quote_verified, item.status, item.reason = (
                original.verdict, original.quote_verified, original.status, original.reason)
            item.evidence_source_id, item.evidence_quote = original.evidence_source_id, original.evidence_quote
        for event in scratch.events:
            if event.event_type is AuditEventType.WARNING and collector is not None:
                llm_reviewed = False
                collector.emit(AuditEventType.WARNING, f"Report Approver: {event.message}")

        judged = [s for s in sentences if s.status in ("supported", "fixed", "failed")]
        supported = sum(s.status != "failed" for s in judged)
        failed = len(judged) - supported
        return ReportAudit(
            round=audit_round, judged=len(judged), supported=supported, failed=failed,
            skipped=sum(s.status == "skipped" for s in sentences),
            unchecked=sum(s.status == "unchecked" for s in sentences),
            groundedness=round(supported / len(judged), 4) if judged else 1.0,
            passed=failed == 0, decision="approve", sentences=sentences, llm_reviewed=llm_reviewed,
        )


def finalize_audit(audit: ReportAudit, *, fixes_left: bool) -> ReportAudit:
    """Decide what happens next: approve, send failing sentences back for a fix, or remove them."""
    if audit.passed:
        return audit.model_copy(update={"decision": "approve"})
    if fixes_left:
        return audit.model_copy(update={"decision": "fix"})
    sentences = [s.model_copy(update={"status": "removed"}) if s.status == "failed" else s for s in audit.sentences]
    return audit.model_copy(update={"decision": "remove", "sentences": sentences})
