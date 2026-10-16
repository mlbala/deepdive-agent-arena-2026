"""The agents' group chat: each node posts what it actually hands to the next agent.

Every message is composed from the structured result the receiving agent consumes (the plan,
the Approver's review, the evidence pool, the drafts, the verdicts and gaps, the gate decision),
so the chat is a readable view of the real hand-offs rather than narration. The orchestrator
emits each message as an AGENT_MESSAGE audit event, so it is persisted, streamed live over
SSE and covered by the run's SHA-256 hash chain.

Two rooms iterate until their quality gate passes:

* ``plan_review``   Planner ⇄ Approver, until the plan scores ≥ PLAN_APPROVAL_THRESHOLD
* ``evidence_loop`` Researcher → Drafter → Verifier → quality gate, until grounding and
                    coverage clear their thresholds (or the iteration/budget limit is hit)
"""

from __future__ import annotations

from collections.abc import Sequence

from src.schemas.domain import (
    AgentMessage,
    ContradictionItem,
    DraftSection,
    FinalReport,
    GovernanceSummary,
    IterationMetrics,
    PlanReview,
    RefinementQuery,
    SourceDocument,
    SubQuestion,
    VerificationGap,
)

INSUFFICIENT_PREFIX = "The available evidence does not establish"


def _clip(text: str, n: int = 150) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def plan_proposal(
    run_id: str, *, sub_questions: Sequence[SubQuestion], strategy: str, plan_round: int, notes: Sequence[str] = ()
) -> AgentMessage:
    if plan_round == 1:
        text = f"Here is my plan: {len(sub_questions)} sub-questions. {strategy}".strip()
        items = [f"{sq.id}: {_clip(sq.question, 170)}" for sq in sub_questions]
        kind = "proposal"
    else:
        text = f"Revised plan (version {plan_round}): {len(sub_questions)} sub-questions, addressing your review."
        items = [*(f"Changed: {_clip(n, 170)}" for n in notes), *(f"{sq.id}: {_clip(sq.question, 170)}" for sq in sub_questions)]
        kind = "revision"
    return AgentMessage(
        run_id=run_id, channel="plan_review", round=plan_round, sender="planner", to=["approver"], kind=kind,
        text=text, items=items, refs=[sq.id for sq in sub_questions],
    )


def plan_verdict(run_id: str, review: PlanReview) -> AgentMessage:
    blocking = [i for i in review.issues if i.blocking]
    status = f"Score {review.score:.2f} (needs {review.threshold:.2f})"
    if review.passed:
        text = f"Approved. {status}. {review.summary} Researcher, go ahead."
        to, kind = ["planner", "researcher"], "approval"
    elif review.forced:
        text = (f"{status}, but the revision budget is spent. Proceeding with {len(review.issues)} open issue(s) "
                f"on record. Researcher, go ahead.")
        to, kind = ["planner", "researcher"], "approval"
    else:
        reason = f"{len(blocking)} blocking issue(s)" if blocking else "below the bar"
        text = f"Sent back: {status[0].lower()}{status[1:]}, {reason}. {review.summary}"
        to, kind = ["planner"], "critique"
    items = [
        f"{i.target} · {i.severity.value}{' · blocking' if i.blocking else ''}{' · check' if i.origin == 'check' else ''}: {_clip(i.problem, 140)}"
        + (f" → {_clip(i.fix, 120)}" if i.fix else "")
        for i in review.issues
    ]
    return AgentMessage(
        run_id=run_id, channel="plan_review", round=review.round, sender="approver", to=to, kind=kind,
        text=text.strip(), items=items, score=review.score, threshold=review.threshold,
        passed=review.passed, refs=sorted({i.target for i in review.issues if i.target != "plan"}),
    )


def evidence_handoff(
    run_id: str,
    *,
    iteration: int,
    targets: Sequence[str],
    new_sources: dict[str, SourceDocument],
    total_sources: int,
    refinements: Sequence[RefinementQuery],
    shielded: int,
) -> AgentMessage:
    tier1 = sum(s.authority.tier.value == "TIER_1" for s in new_sources.values())
    lead = "Re-searched" if iteration > 1 else "Researched"
    text = (f"{lead} {', '.join(targets)}: {len(new_sources)} new source(s), {tier1} Tier 1, "
            f"{total_sources} in the evidence pool. Drafter, evidence is indexed with exact spans.")
    top = sorted(new_sources.values(), key=lambda s: s.authority.score, reverse=True)[:3]
    items = [*(f"Refined {q.sub_question_id}: “{_clip(q.query, 90)}”" for q in refinements[:4]),
             *(f"{s.source_id} · {_clip(s.title, 80)} · authority {s.authority.score:.2f}" for s in top)]
    if shielded:
        items.append(f"Content and URL shields acted {shielded} time(s) on retrieved pages")
    return AgentMessage(
        run_id=run_id, channel="evidence_loop", round=iteration, sender="researcher", to=["drafter"], kind="evidence",
        text=text, items=items, refs=[s.source_id for s in top],
    )


def drafts_handoff(
    run_id: str, *, iteration: int, drafts: Sequence[DraftSection], gaps: Sequence[VerificationGap]
) -> AgentMessage:
    cited = sorted({c for d in drafts for c in d.cited_source_ids})
    if iteration > 1:
        notes = sum(len(g.feedback) for g in gaps if g.sub_question_id in {d.sub_question_id for d in drafts})
        text = (f"Re-drafted {', '.join(d.sub_question_id for d in drafts)} using your notes on {notes} rejected "
                f"claim(s). Verifier, please re-check.")
    else:
        text = f"Drafted {len(drafts)} section(s) citing {len(cited)} distinct source(s). Verifier, over to you."
    items = []
    for d in drafts:
        line = f"{d.sub_question_id}: {len(d.text.split())} words"
        line += f" · cites {', '.join(d.cited_source_ids[:4])}" if d.cited_source_ids else " · no citations"
        if d.text.startswith(INSUFFICIENT_PREFIX) or not d.cited_source_ids:
            line += " · evidence insufficient"
        if d.invalid_citations:
            line += f" · {len(d.invalid_citations)} citation(s) outside the evidence pool"
        items.append(line)
    return AgentMessage(
        run_id=run_id, channel="evidence_loop", round=iteration, sender="drafter", to=["verifier"], kind="draft",
        text=text, items=items, refs=cited[:8],
    )


def verdicts_handoff(
    run_id: str,
    *,
    metrics: IterationMetrics,
    gaps: Sequence[VerificationGap],
    contradictions: Sequence[ContradictionItem],
) -> AgentMessage:
    text = (f"Checked {metrics.total_claims} claim(s): {metrics.supported} supported, {metrics.unsupported} "
            f"unsupported, {metrics.contradicted} contradicted. Grounding {metrics.grounding_rate:.0%}, "
            f"coverage {metrics.coverage_score:.0%}.")
    if gaps:
        text += f" Researcher and Drafter: {', '.join(g.sub_question_id for g in gaps)} need work."
    items = []
    for g in gaps[:4]:
        items.append(f"{g.sub_question_id}: {g.reason}")
        items.extend(f"   {_clip(f, 170)}" for f in g.feedback[:2])
    cross = [c for c in contradictions if c.origin == "cross_source"]
    items.extend(f"Conflict: {_clip(c.topic, 80)} ({c.source_a} vs {c.source_b})" for c in cross[:3])
    return AgentMessage(
        run_id=run_id, channel="evidence_loop", round=metrics.iteration, sender="verifier",
        to=["researcher", "drafter"] if gaps else ["synthesizer"], kind="verdict", text=text, items=items,
        score=metrics.grounding_rate, refs=[g.sub_question_id for g in gaps],
    )


def gate_decision(
    run_id: str,
    *,
    metrics: IterationMetrics,
    route: str,
    coverage_threshold: float,
    grounding_threshold: float,
    budget_exhausted: bool,
    max_iterations: int,
) -> AgentMessage:
    g, c = metrics.grounding_rate, metrics.coverage_score
    passed = g >= grounding_threshold and c >= coverage_threshold
    bar = f"grounding {g:.0%} {'≥' if g >= grounding_threshold else '<'} {grounding_threshold:.0%}, " \
          f"coverage {c:.0%} {'≥' if c >= coverage_threshold else '<'} {coverage_threshold:.0%}"
    if route == "researcher":
        text = (f"Below the bar ({bar}). Round {metrics.iteration + 1}: Researcher re-searches "
                f"{', '.join(metrics.target_sub_question_ids)}, then the Drafter re-drafts them.")
        to = ["researcher", "drafter"]
    elif passed:
        text = f"Quality bar met ({bar}). Synthesizer, write the report from the supported claims."
        to = ["synthesizer"]
    elif budget_exhausted:
        text = f"Budget guard: the run budget's soft limit is reached ({bar}). Publishing verified claims only."
        to = ["synthesizer"]
    else:
        reason = (f"after {metrics.iteration} of {max_iterations} round(s)" if metrics.iteration >= max_iterations
                  else "with no sub-question left to re-research")
        text = f"Still below the bar {reason} ({bar}). Publishing verified claims only and disclosing the gaps."
        to = ["synthesizer"]
    return AgentMessage(
        run_id=run_id, channel="evidence_loop", round=metrics.iteration, sender="quality_gate", to=to, kind="decision",
        text=text, items=[f"Round {metrics.iteration} of at most {max_iterations}"],
        score=g, threshold=grounding_threshold, passed=passed, refs=list(metrics.target_sub_question_ids),
    )


def report_handoff(run_id: str, report: FinalReport) -> AgentMessage:
    m = report.metrics
    text = (f"Report written from {m.claims_in_report} verified claim(s); {m.claims_removed} withheld; "
            f"{m.contradictions} conflict(s) disclosed. Output shield, please review before publishing.")
    return AgentMessage(
        run_id=run_id, channel="publish", round=1, sender="synthesizer", to=["output_guard"], kind="report",
        text=text, items=[_clip(x, 170) for x in report.limitations[:3]],
    )


def publish_notice(run_id: str, governance: GovernanceSummary) -> AgentMessage:
    text = (f"Published. {governance.pii_redactions} PII and {governance.secret_redactions} secret redaction(s); "
            f"{len(governance.findings)} guardrail decision(s) on record; the audit chain is sealed.")
    return AgentMessage(
        run_id=run_id, channel="publish", round=1, sender="output_guard", to=[], kind="decision", text=text,
        passed=True,
    )
