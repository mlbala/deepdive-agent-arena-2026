"""Agent 1 — PlannerAgent: MECE decomposition of the research question, revised until the Approver signs off."""

from __future__ import annotations

import re
from datetime import UTC, datetime

from src.agents.base import BaseAgent, EventCollector, render_input
from src.schemas.domain import AuditEventType, PlannerOutput, PlanReview, SubQuestion
from src.storage.vector_store import tokenize

SYSTEM_PROMPT = """ROLE: planner
You are DeepDive's PLANNER agent, a principal research strategist.

Decompose the user's research question into 3-5 sub-questions that are:
- Mutually exclusive: no two sub-questions ask for the same facts.
- Collectively exhaustive: together they fully answer the original question.
- Evidence-seeking: each can be answered from public, citable sources (official statistics,
  regulators, peer-reviewed research, reputable journalism).
- Self-contained: a researcher can act on each one without seeing the others.

If the question embeds a premise that may be false or contested (e.g. "Why did X happen?"),
dedicate one sub-question to verifying that premise directly.

For each sub-question write 1-3 concise web search queries (3-10 words) using the precise
entities, metrics, jurisdictions and time periods involved. Prefer phrasing that surfaces primary
sources (e.g. "IEA renewables 2023 solar additions", "RBI monetary policy framework repo rate").
Never answer the question yourself."""

REVISION_PROMPT = """ROLE: planner.revise
You are DeepDive's PLANNER agent. The APPROVER agent reviewed your research plan and sent it back.

Revise the plan:
- Fix every high- and medium-severity issue in the approver's review; fix low-severity issues when that improves the plan.
- Keep sub-questions that drew no criticism, wording and all.
- The revised plan must still follow the original rules: 3-5 mutually exclusive, collectively exhaustive,
  evidence-seeking, self-contained sub-questions, each with 1-3 concise search queries (3-10 words) naming
  precise entities, metrics, jurisdictions and time periods. If the question embeds a doubtful premise,
  one sub-question tests it directly.
In revision_notes, list each change you made and the issue it fixes, one short line per change.
Never answer the question yourself."""


_RESTATED = re.compile(r"^SQ-\d+\s*:.*\?$")  # a note that only repeats a sub-question, not a change


def _jaccard(a: str, b: str) -> float:
    ta, tb = set(tokenize(a)), set(tokenize(b))
    return len(ta & tb) / len(ta | tb) if ta and tb else 0.0


class PlannerAgent(BaseAgent):
    name = "planner"

    def plan(
        self,
        run_id: str,
        query: str,
        *,
        document_titles: list[str] | None = None,
        collector: EventCollector | None = None,
    ) -> tuple[list[SubQuestion], str]:
        user_prompt = render_input(
            "Decompose this research question into MECE sub-questions with targeted search queries.",
            {
                "research_question": query,
                "today": datetime.now(UTC).date().isoformat(),
                "user_uploaded_documents": document_titles or [],
            },
        )
        output = self._llm(run_id, SYSTEM_PROMPT, user_prompt, PlannerOutput)
        sub_questions = self._to_sub_questions(output)
        if collector is not None:
            collector.emit(
                AuditEventType.PLAN_CREATED,
                f"Planner decomposed the question into {len(sub_questions)} sub-questions",
                round=1,
                strategy=output.research_strategy,
                sub_questions=[sq.model_dump() for sq in sub_questions],
                revision_notes=[],
            )
        return sub_questions, output.research_strategy

    def revise(
        self,
        run_id: str,
        query: str,
        *,
        current: list[SubQuestion],
        strategy: str,
        review: PlanReview,
        collector: EventCollector | None = None,
    ) -> tuple[list[SubQuestion], str, list[str]]:
        """Rewrite the plan to address the Approver's review. Returns (sub_questions, strategy, notes)."""
        user_prompt = render_input(
            "Revise your research plan so that it addresses the approver's review.",
            {
                "research_question": query,
                "today": datetime.now(UTC).date().isoformat(),
                "current_plan": {
                    "research_strategy": strategy,
                    "sub_questions": [
                        {"id": sq.id, "question": sq.question, "search_queries": sq.search_queries} for sq in current
                    ],
                },
                "approver_review": {
                    "score": review.score,
                    "required_score": review.threshold,
                    "summary": review.summary,
                    "issues": [i.model_dump(mode="json", exclude={"origin"}) for i in review.issues],
                },
            },
        )
        output = self.gateway.complete_structured(
            self.name, REVISION_PROMPT, user_prompt, PlannerOutput, run_id=run_id
        )
        sub_questions = self._to_sub_questions(output)
        notes = [n.strip() for n in output.revision_notes if n.strip() and not _RESTATED.match(n.strip())][:6]
        if collector is not None:
            collector.emit(
                AuditEventType.PLAN_CREATED,
                f"Planner revised the plan (round {review.round + 1}): {len(sub_questions)} sub-questions",
                round=review.round + 1,
                strategy=output.research_strategy,
                sub_questions=[sq.model_dump() for sq in sub_questions],
                revision_notes=notes,
            )
        return sub_questions, output.research_strategy or strategy, notes

    @staticmethod
    def _to_sub_questions(output: PlannerOutput) -> list[SubQuestion]:
        sub_questions: list[SubQuestion] = []
        for planned in sorted(output.sub_questions, key=lambda p: p.priority):
            if any(_jaccard(planned.question, sq.question) > 0.8 for sq in sub_questions):
                continue  # enforce mutual exclusivity deterministically
            queries = list(dict.fromkeys(q.strip() for q in planned.search_queries if q.strip()))[:3]
            sub_questions.append(
                SubQuestion(
                    id=f"SQ-{len(sub_questions) + 1}",
                    question=planned.question,
                    rationale=planned.rationale,
                    search_queries=queries or [planned.question],
                    priority=planned.priority,
                )
            )
        return sub_questions
