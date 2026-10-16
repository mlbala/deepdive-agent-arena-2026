"""Agent 2 — ApproverAgent: the quality gate between planning and retrieval.

The Approver scores each version of the plan on a five-part rubric (via Mesh) and runs
deterministic plan checks. The plan goes back to the Planner with concrete fixes until

    score >= PLAN_APPROVAL_THRESHOLD (0.80)  and  no blocking issue

A deterministic high-severity issue always blocks. A model-raised high-severity issue blocks
only when the model's own rubric corroborates it (some criterion below 0.70): severity labels
from a model are noisy, and a relevance quibble must not bounce a 0.9-scoring plan.

or MAX_PLAN_REVISIONS is spent, in which case it proceeds with its open issues on record
(`forced`). Retrieval is the expensive part of a run, so a weak plan is cheaper to fix here.
"""

from __future__ import annotations

import re
from itertools import combinations
from statistics import mean

from src.agents.base import BaseAgent, EventCollector, render_input
from src.core.exceptions import MeshStructuredOutputError, MeshTransientError
from src.schemas.domain import (
    AuditEventType,
    PlanIssue,
    PlanReview,
    PlanReviewOutput,
    Severity,
    SubQuestion,
)
from src.storage.vector_store import tokenize

SYSTEM_PROMPT = """ROLE: approver
You are DeepDive's APPROVER agent, a demanding research director. You sign off on a research plan
before any retrieval runs. You never answer the research question yourself.

Score the plan from 0.0 to 1.0 on each rubric criterion:
- coverage: together, the sub-questions fully answer the research question, including every entity,
  metric, period and sub-part it asks about.
- distinctness: no two sub-questions ask for the same facts.
- searchability: the search queries are concise (3-10 words) and likely to surface primary, citable sources.
- premise_check: if the question embeds a doubtful or contested premise, one sub-question tests it directly.
  Score 1.0 when the question has no doubtful premise.
- specificity: entities, metrics, jurisdictions and time periods are named precisely; nothing is vague.

List concrete issues (at most 6). Tie each one to a sub-question id (e.g. SQ-2) or to "plan", give a
severity (low, medium, high) and a specific fix the Planner can apply.
Severity: high = a part of the question is not covered, a doubtful premise is not tested, or a
sub-question would lead the research in the wrong direction. Wording, relevance quibbles and style
are low or medium. Your rubric scores must be consistent with your issues: a high issue means the
matching criterion scores below 0.7.
If previous_review_issues are given, check whether each was fixed. Do not repeat an issue the
revision addressed; raise it again only if it is clearly still unresolved.
Write a 1-3 sentence summary addressed to the Planner."""

PENALTY = {Severity.HIGH: 0.10, Severity.MEDIUM: 0.05, Severity.LOW: 0.0}
CORROBORATION_FLOOR = 0.70  # a model-raised high issue blocks only if some rubric criterion is below this
_QUESTION_WORDS = frozenset(
    "what why how when where which who whom whose is are was were do does did can could should would will "
    "the a an in of and or for to i".split()
)
_TERM = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-]*")


def salient_terms(question: str) -> list[str]:
    """Names, codes and figures the plan must keep: capitalised words (not the first), anything with a digit."""
    terms: list[str] = []
    for index, word in enumerate(_TERM.findall(question)):
        has_digit = any(ch.isdigit() for ch in word)
        proper = index > 0 and word[0].isupper() and word.lower() not in _QUESTION_WORDS
        if (has_digit or proper) and word.lower() not in (t.lower() for t in terms):
            terms.append(word)
    return terms


def focus_overlap(a: str, b: str, question: str) -> float:
    """Overlap of what two sub-questions add beyond the research question's own words.

    Sub-questions legitimately repeat the topic ("RBI", "repo rate"); they duplicate each
    other only when the facts they ask for beyond it are the same.
    """
    topic = set(tokenize(question))
    fa, fb = set(tokenize(a)) - topic, set(tokenize(b)) - topic
    if len(fa) < 2 or len(fb) < 2:
        return 0.0
    return len(fa & fb) / len(fa | fb)


def plan_checks(question: str, sub_questions: list[SubQuestion]) -> list[PlanIssue]:
    """Deterministic checks that need no model: count, overlap, dropped entities, query shape."""
    issues: list[PlanIssue] = []
    if len(sub_questions) < 3:
        issues.append(PlanIssue(
            severity=Severity.HIGH, origin="check",
            problem=f"only {len(sub_questions)} distinct sub-question(s) remain after removing near-duplicates",
            fix="Add sub-questions until the plan covers the question with 3-5 distinct parts.",
        ))
    for a, b in combinations(sub_questions, 2):
        overlap = focus_overlap(a.question, b.question, question)
        if overlap >= 0.6:
            issues.append(PlanIssue(
                target=b.id, severity=Severity.MEDIUM, origin="check",
                problem=f"{a.id} and {b.id} overlap heavily (word overlap {overlap:.0%})",
                fix=f"Merge them or refocus {b.id} on facts {a.id} does not ask for.",
            ))
    plan_text = " ".join(f"{sq.question} {' '.join(sq.search_queries)}" for sq in sub_questions).lower()
    missing = [t for t in salient_terms(question) if t.lower() not in plan_text]
    if missing:
        issues.append(PlanIssue(
            severity=Severity.MEDIUM, origin="check",
            problem=f"the plan never mentions {', '.join(repr(t) for t in missing[:4])} from the question",
            fix="Name these entities, codes or figures in the sub-questions or their search queries.",
        ))
    for sq in sub_questions:
        bad = [q for q in sq.search_queries if not 2 <= len(q.split()) <= 12]
        if bad:
            issues.append(PlanIssue(
                target=sq.id, severity=Severity.LOW, origin="check",
                problem=f"search query {bad[0]!r} is {'too short' if len(bad[0].split()) < 2 else 'too long'}",
                fix="Use 3-10 words naming the precise entity, metric and period.",
            ))
    return issues


class ApproverAgent(BaseAgent):
    name = "approver"

    def review(
        self,
        run_id: str,
        query: str,
        *,
        sub_questions: list[SubQuestion],
        strategy: str,
        plan_round: int,
        previous: PlanReview | None = None,
        collector: EventCollector | None = None,
    ) -> PlanReview:
        threshold = self.settings.plan_approval_threshold
        checks = plan_checks(query, sub_questions)
        rubric: dict[str, float] = {}
        llm_issues: list[PlanIssue] = []
        summary = ""
        llm_reviewed = True
        try:
            output = self._llm(
                run_id,
                SYSTEM_PROMPT,
                render_input(
                    "Review this research plan and decide whether it is ready for retrieval.",
                    {
                        "research_question": query,
                        "round": plan_round,
                        "plan": {
                            "research_strategy": strategy,
                            "sub_questions": [
                                {"id": sq.id, "question": sq.question, "search_queries": sq.search_queries}
                                for sq in sub_questions
                            ],
                        },
                        "previous_review_issues": [
                            f"{i.target}: {i.problem}" for i in (previous.issues if previous else [])
                        ],
                    },
                ),
                PlanReviewOutput,
            )
            rubric = output.rubric.model_dump()
            known = {sq.id for sq in sub_questions} | {"plan"}
            llm_issues = [
                PlanIssue(
                    target=i.target if i.target in known else "plan",
                    severity=i.severity, problem=i.problem, fix=i.fix, origin="approver",
                )
                for i in output.issues[:6]
            ]
            summary = output.summary
        except (MeshTransientError, MeshStructuredOutputError) as exc:
            # fail open on the model, never on the checks: the deterministic issues still gate the plan
            llm_reviewed = False
            summary = f"The Approver model was unavailable ({type(exc).__name__}); only the deterministic plan checks ran."
            if collector is not None:
                collector.emit(AuditEventType.WARNING, summary)

        corroborated = bool(rubric) and min(rubric.values()) < CORROBORATION_FLOOR
        for issue in llm_issues:
            issue.blocking = issue.severity is Severity.HIGH and corroborated
        for issue in checks:
            issue.blocking = issue.severity is Severity.HIGH
        issues = [*llm_issues, *checks]
        base = mean(rubric.values()) if rubric else 1.0
        score = round(max(0.0, min(1.0, base - sum(PENALTY[i.severity] for i in checks))), 4)
        blocking = [i for i in issues if i.blocking]
        passed = score >= threshold and not blocking
        forced = not passed and plan_round > self.settings.max_plan_revisions
        if not summary:
            summary = "The plan is ready for retrieval." if passed else "The plan needs another revision."
        review = PlanReview(
            round=plan_round, score=score, threshold=threshold, passed=passed, forced=forced,
            rubric=rubric, issues=issues, summary=summary, llm_reviewed=llm_reviewed,
        )
        if collector is not None:
            verdict = "approved" if passed else ("proceeding with open issues" if forced else "sent back to the Planner")
            collector.emit(
                AuditEventType.PLAN_REVIEWED,
                f"Approver round {plan_round}: score {score:.2f} (needs {threshold:.2f}), {len(blocking)} blocking issue(s) -> {verdict}",
                **review.model_dump(mode="json"),
            )
        return review
