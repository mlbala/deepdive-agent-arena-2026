"""Agent 5 — SynthesizerAgent: compile the verified executive report.

Only SUPPORTED claims reach the model. Everything else is stripped and preserved
as `RemovedClaim` entries (and `claims_audit` rows). After generation a
deterministic citation guard removes any sentence that cites a source outside the
verified set, or that asserts a fact without a citation, so the model cannot
re-introduce unverified content during synthesis.
"""

from __future__ import annotations

import re
from statistics import mean

from src.agents.base import BaseAgent, EventCollector, render_input
from src.schemas.domain import (
    AtomicClaim,
    AuditEventType,
    ContradictionItem,
    FinalReport,
    IterationMetrics,
    RemovedClaim,
    ReportMetrics,
    ReportSection,
    SourceDocument,
    SourceReference,
    SubQuestion,
    SynthesisOutput,
    Verdict,
    VerificationResult,
    extract_citations,
)

SYSTEM_PROMPT = """ROLE: synthesizer
You are DeepDive's SYNTHESIZER agent, an executive research editor.

Compile a decision-grade report from the VERIFIED claims you receive. Each verified claim has
passed sentence-level entailment checking against its cited source.

Hard rules:
- Use ONLY the verified claims. Do not add facts, numbers, dates, names or sources that are not
  in them. You may merge and paraphrase claims, but never strengthen or generalise them.
- Every sentence that states a fact must carry the [SRC-XXX] citation(s) of the claims it uses.
- Surface every listed contradiction neutrally: give both positions with citations and say which
  source carries more authority (use the resolution hint).
- If a sub-question has no verified claims, write exactly one sentence saying the evidence was
  insufficient to answer it; do not speculate.
- limitations: reflect removed claims, low coverage, reliance on low-authority sources, and
  unresolved contradictions.
- Write one section per sub-question, using its sub_question_id. Plain prose, no markdown."""

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
_META_MARKERS = (
    "insufficient", "not established", "does not establish", "no verified", "could not", "unclear",
    "not enough evidence", "evidence was", "evidence is", "contested", "conflict", "contradict",
    "disagree", "limited evidence", "no evidence", "not available", "remains uncertain",
)


def _is_meta_sentence(sentence: str) -> bool:
    lowered = sentence.lower()
    return any(marker in lowered for marker in _META_MARKERS) and not re.search(r"\d", sentence)


def guard_text(text: str, allowed_sources: set[str]) -> tuple[str, list[str]]:
    """Keep sentences whose citations are all verified; keep uncited sentences only if meta."""
    kept: list[str] = []
    removed: list[str] = []
    for sentence in _SENTENCE_SPLIT.split(text.strip()):
        sentence = sentence.strip()
        if not sentence:
            continue
        cites = extract_citations(sentence)
        if (cites and set(cites) <= allowed_sources) or (not cites and _is_meta_sentence(sentence)):
            kept.append(sentence)
        else:
            removed.append(sentence)
    return " ".join(kept), removed


class SynthesizerAgent(BaseAgent):
    name = "synthesizer"

    def synthesize(
        self,
        run_id: str,
        *,
        query: str,
        sub_questions: list[SubQuestion],
        claims: dict[str, list[AtomicClaim]],
        verifications: dict[str, list[VerificationResult]],
        contradictions: list[ContradictionItem],
        sources: dict[str, SourceDocument],
        iteration_metrics: list[IterationMetrics],
        collector: EventCollector,
    ) -> FinalReport:
        verdict_of = {v.claim_id: v for vs in verifications.values() for v in vs}
        verified: dict[str, list[AtomicClaim]] = {}
        removed: list[RemovedClaim] = []
        for sq in sub_questions:
            for claim in claims.get(sq.id, []):
                v = verdict_of.get(claim.claim_id)
                if v is not None and v.verdict is Verdict.SUPPORTED:
                    verified.setdefault(sq.id, []).append(claim)
                    continue
                verdict = v.verdict if v else Verdict.UNSUPPORTED
                reason = (v.downgrade_reason or v.rationale) if v else "never verified"
                removed.append(
                    RemovedClaim(
                        claim_id=claim.claim_id,
                        sub_question_id=sq.id,
                        text=claim.text,
                        verdict=verdict,
                        reason=reason or verdict.value.lower(),
                    )
                )
                collector.emit(
                    AuditEventType.CLAIM_REMOVED,
                    f"{claim.claim_id} stripped from report ({verdict.value})",
                    claim_id=claim.claim_id,
                    claim=claim.text,
                    verdict=verdict.value,
                    reason=reason,
                )

        def evidence_source(claim: AtomicClaim) -> list[str]:
            v = verdict_of[claim.claim_id]
            return [v.evidence_source_id] if v.evidence_source_id else claim.cited_source_ids

        allowed = {s for cl in verified.values() for c in cl for s in evidence_source(c)}
        allowed |= {c.source_a for c in contradictions if c.source_a in sources}
        allowed |= {c.source_b for c in contradictions if c.source_b in sources}

        payload = {
            "research_question": query,
            "sub_questions": [
                {
                    "sub_question_id": sq.id,
                    "question": sq.question,
                    "verified_claims": [
                        {
                            "claim_id": c.claim_id,
                            "claim": c.text,
                            "citations": evidence_source(c),
                            "authority": max(
                                (sources[s].authority.score for s in evidence_source(c) if s in sources),
                                default=0.0,
                            ),
                        }
                        for c in verified.get(sq.id, [])
                    ],
                    "claims_removed_by_verifier": sum(r.sub_question_id == sq.id for r in removed),
                }
                for sq in sub_questions
            ],
            "contradictions": [
                {
                    "topic": c.topic,
                    "position_a": c.statement_a,
                    "source_a": c.source_a,
                    "position_b": c.statement_b,
                    "source_b": c.source_b,
                    "severity": c.severity.value,
                    "resolution_hint": c.resolution_hint,
                }
                for c in contradictions
                if c.origin == "cross_source"
            ],
            "low_authority_sources": sorted(
                s.source_id for s in sources.values() if s.source_id in allowed and s.authority.score < 0.65
            ),
        }
        output = self._llm(
            run_id,
            SYSTEM_PROMPT,
            render_input("Compile the verified executive research report.", payload),
            SynthesisOutput,
        )

        guard_removed: list[str] = []
        summary, dropped = guard_text(output.executive_summary, allowed)
        guard_removed += dropped
        findings: list[str] = []
        for finding in output.key_findings:
            text, dropped = guard_text(finding, allowed)
            guard_removed += dropped
            if text:
                findings.append(text)

        llm_sections = {s.sub_question_id: s for s in output.sections}
        sections: list[ReportSection] = []
        for sq in sub_questions:
            sq_claims = verified.get(sq.id, [])
            drafted = llm_sections.get(sq.id)
            body, dropped = guard_text(drafted.body, allowed) if drafted else ("", [])
            guard_removed += dropped
            if not body:
                body = (
                    " ".join(f"{c.text} [{', '.join(evidence_source(c))}]." for c in sq_claims)
                    if sq_claims
                    else "The available evidence was insufficient to answer this sub-question."
                )
            sections.append(
                ReportSection(
                    sub_question_id=sq.id,
                    heading=(drafted.heading if drafted and drafted.heading else sq.question),
                    body=body,
                    claim_ids=[c.claim_id for c in sq_claims],
                    cited_source_ids=extract_citations(body),
                    confidence=round(mean(verdict_of[c.claim_id].confidence for c in sq_claims), 3)
                    if sq_claims
                    else 0.0,
                )
            )

        if not summary:
            summary = " ".join(s.body.split(". ")[0].rstrip(".") + "." for s in sections if s.claim_ids)[:1200]
            summary = summary or "The available evidence was insufficient to answer the research question."
        if guard_removed:
            collector.emit(
                AuditEventType.WARNING,
                f"Citation guard removed {len(guard_removed)} synthesized sentence(s) lacking verified citations",
                removed_sentences=guard_removed,
            )

        cited_in_report = set(extract_citations(" ".join([summary, *findings, *(s.body for s in sections)])))
        report_sentences = [
            s for s in _SENTENCE_SPLIT.split(" ".join([summary, *(s.body for s in sections)])) if s.strip()
        ]
        cited_sentences = sum(bool(extract_citations(s)) for s in report_sentences)
        first = iteration_metrics[0] if iteration_metrics else None
        last = iteration_metrics[-1] if iteration_metrics else None
        usage = self.gateway.usage_for(run_id)

        metrics = ReportMetrics(
            grounding_rate_pre=first.grounding_rate if first else 0.0,
            grounding_rate_final=last.grounding_rate if last else 0.0,
            coverage_score=last.coverage_score if last else 0.0,
            citation_precision=last.citation_precision if last else 0.0,
            quote_verification_rate=last.quote_verification_rate if last else 0.0,
            report_citation_coverage=round(cited_sentences / len(report_sentences), 4) if report_sentences else 0.0,
            iterations=len(iteration_metrics),
            loop_backs=max(0, len(iteration_metrics) - 1),
            total_claims_evaluated=sum(m.total_claims for m in iteration_metrics),
            claims_in_report=sum(len(v) for v in verified.values()),
            claims_removed=len(removed),
            contradictions=len(contradictions),
            sources_indexed=len(sources),
            sources_cited=len(cited_in_report),
            llm_calls=usage["llm_calls"],
            total_tokens=usage["total_tokens"],
            llm_latency_ms=usage["latency_ms"],
        )
        report = FinalReport(
            run_id=run_id,
            query=query,
            title=output.title,
            executive_summary=summary,
            key_findings=findings,
            sections=sections,
            contradictions=contradictions,
            removed_claims=removed,
            sources=[
                SourceReference(
                    source_id=s.source_id,
                    title=s.title,
                    url=s.url,
                    domain=s.domain,
                    authority_score=s.authority.score,
                    tier=s.authority.tier,
                    retrieval_tier=s.retrieval_tier,
                    cited_in_report=s.source_id in cited_in_report,
                )
                for s in sorted(sources.values(), key=lambda s: s.source_id)
            ],
            limitations=output.limitations,
            sub_questions=sub_questions,
            metrics=metrics,
            guard_removed_sentences=guard_removed,
        )
        collector.emit(
            AuditEventType.REPORT_COMPILED,
            f"Report compiled: {metrics.claims_in_report} verified claims kept, "
            f"{metrics.claims_removed} stripped, {metrics.contradictions} contradiction(s) surfaced",
            metrics=metrics.model_dump(),
        )
        return report
