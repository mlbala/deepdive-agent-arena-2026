"""Agent 6 — FollowUpAgent: conversational Q&A grounded strictly in a run's evidence pool.

The agent retrieves from the run's own indexed spans (rehydrated from SQLite after a
restart), answers only from them, and passes the answer through a citation guard:
citations outside the retrieved pool are stripped, and an "answerable" response with no
surviving citation is converted into an explicit "cannot be grounded" answer.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence

from src.agents.base import BaseAgent, EventCollector, render_input
from src.core.config import Settings
from src.core.mesh_gateway import MeshGateway
from src.schemas.domain import (
    AuditEventType,
    EvidenceChunk,
    FinalReport,
    FollowUpAnswer,
    FollowUpOutput,
    extract_citations,
)
from src.storage.vector_store import HybridVectorStore

SYSTEM_PROMPT = """ROLE: followup
You are DeepDive's FOLLOW-UP agent. You answer questions about a completed research run.

- Answer ONLY from the evidence spans and verified findings provided. No outside knowledge.
- Cite every factual sentence with [SRC-XXX] ids taken from the evidence.
- If the evidence does not contain the answer, set answerable=false and say plainly what is
  missing; suggest what the user could research next. Never guess.
- Be concise (at most 150 words). Plain prose."""

UNGROUNDED_ANSWER = (
    "I could not ground an answer to this question in this run's verified evidence pool. "
    "Consider starting a new research run focused on it."
)
FOLLOWUP_EVIDENCE_K = 8


class FollowUpAgent(BaseAgent):
    name = "followup"

    def __init__(self, gateway: MeshGateway, settings: Settings, vector_store: HybridVectorStore) -> None:
        super().__init__(gateway, settings)
        self.vector_store = vector_store

    def answer(
        self,
        run_id: str,
        question: str,
        *,
        report: FinalReport | None,
        evidence_loader: Callable[[], Sequence[EvidenceChunk]] | None = None,
        history: Sequence[dict[str, str]] = (),
        collector: EventCollector | None = None,
    ) -> FollowUpAnswer:
        if not self.vector_store.has_namespace(run_id) and evidence_loader is not None:
            self.vector_store.add_chunks(run_id, list(evidence_loader()))

        hits = self.vector_store.search(run_id, question, k=FOLLOWUP_EVIDENCE_K)
        authority = {s.source_id: s for s in report.sources} if report else {}
        payload = {
            "question": question,
            "research_question": report.query if report else "",
            "conversation": list(history)[-4:],
            "verified_findings": (
                [s.body for s in report.sections if s.claim_ids] + report.key_findings if report else []
            ),
            "evidence": [
                {
                    "source_id": h.source_id,
                    "title": authority[h.source_id].title if h.source_id in authority else h.title,
                    "authority": round(
                        authority[h.source_id].authority_score if h.source_id in authority else h.authority_score, 2
                    ),
                    "text": h.text,
                }
                for h in hits
            ],
        }

        if not hits:
            result = FollowUpAnswer(
                run_id=run_id, question=question, answer=UNGROUNDED_ANSWER, answerable=False, confidence=0.0
            )
        else:
            output = self._llm(
                run_id,
                SYSTEM_PROMPT,
                render_input("Answer the follow-up question from this run's evidence.", payload),
                FollowUpOutput,
            )
            allowed = {h.source_id for h in hits}
            if report:
                allowed |= {sid for s in report.sections for sid in s.cited_source_ids}
            cited = list(dict.fromkeys([*output.cited_source_ids, *extract_citations(output.answer)]))
            valid = [c for c in cited if c in allowed]
            removed = [c for c in cited if c not in allowed]
            text = output.answer
            for bad in removed:
                text = re.sub(rf"\s*\[{bad}\]", "", text)
            answerable = output.answerable and bool(valid)
            result = FollowUpAnswer(
                run_id=run_id,
                question=question,
                answer=text if answerable or not output.answerable else UNGROUNDED_ANSWER,
                answerable=answerable,
                cited_source_ids=valid if answerable else [],
                evidence_chunk_ids=[h.chunk_id for h in hits],
                confidence=output.confidence if answerable else min(output.confidence, 0.2),
                removed_citations=removed,
            )

        if collector is not None:
            collector.emit(
                AuditEventType.FOLLOWUP_ANSWERED,
                f"Follow-up answered ({'grounded' if result.answerable else 'not answerable'}) "
                f"citing {len(result.cited_source_ids)} source(s)",
                question=question,
                answerable=result.answerable,
                cited_source_ids=result.cited_source_ids,
                removed_citations=result.removed_citations,
            )
        return result
