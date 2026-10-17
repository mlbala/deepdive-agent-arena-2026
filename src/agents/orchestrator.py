"""LangGraph orchestrator: a Master Approver supervising a group of agents, plus guardrails.

    START -> input_guard --(blocked)--> END
                 |
                 v
              planner <--revise-- master (Master Approver) --approve--> synthesizer <-> report_approver -> output_guard -> END
                 |                  ^  |
                 +------plan--------+  | dispatch / another round
                                    |  v
                                    |  researcher -> drafter <-> verifier   (quick fixes inside the round)
                                    +------------ round report ------+      (evidence group chat)

The Master Approver (`master`) is the hub. It reviews every version of the plan (Mesh rubric +
deterministic checks) and sends it back to the Planner until it clears PLAN_APPROVAL_THRESHOLD
or MAX_PLAN_REVISIONS is spent. It then dispatches the Researcher -> Drafter -> Verifier group
and reviews each round's report: below the quality gate's bar it must send the group back (with
instructions per agent); above it, it approves the work for the Synthesizer or adds up to
MASTER_EXTRA_ROUNDS targeted passes. Every node posts its real hand-off to the agents' group chat
(`state["group_chat"]`, AGENT_MESSAGE audit events), so the whole conversation is auditable.

`report_approver` re-checks every cited sentence of the synthesized report against its sources (the
Verifier's guards) and sends failing sentences back to the Synthesizer for REPORT_AUDIT_FIXES fix
rounds; whatever still fails is removed before the output shield publishes the report.

`input_guard` runs the deterministic input shield (prompt injection, harmful requests,
PII/secret redaction) and the Mesh-backed Guardian classifier; a blocked question ends the
run before any research agent sees it. `output_guard` redacts the published report and
attaches the governance summary (findings, budgets, models, audit-chain head).

`route_verification` is the deterministic floor under the Master Approver's round decision: another
round — with targeted refinement queries for only the failing sub-questions — is mandatory while

    coverage_score < COVERAGE_THRESHOLD (0.70)  or  grounding_rate < GROUNDING_THRESHOLD (0.75)

and `iteration_count < max_iterations` and the per-run budget's soft limit has not been
reached.

Every node emits typed `AuditEvent`s that are (a) appended to `state["audit_log"]`,
(b) persisted to SQLite immediately, and (c) pushed to an optional live sink (SSE).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator, Sequence
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph

from src.agents import group_chat as chat
from src.agents.base import EventCollector, EventSink
from src.agents.baseline import BaselineAgent
from src.agents.drafter import DrafterAgent
from src.agents.followup import FollowUpAgent
from src.agents.guardian import GuardianAgent
from src.agents.master import MasterApproverAgent
from src.agents.planner import PlannerAgent
from src.agents.report_approver import (
    ReportApproverAgent,
    finalize_audit,
    rewrite_report,
    verified_index,
)
from src.agents.researcher import ResearcherAgent
from src.agents.synthesizer import SynthesizerAgent, guard_text, verified_material
from src.agents.verifier import VerifierAgent
from src.core.config import Settings
from src.core.exceptions import (
    DeepDiveError,
    GuardrailBlockedError,
    MeshStructuredOutputError,
    MeshTransientError,
    OrchestrationError,
)
from src.core.mesh_gateway import MeshGateway
from src.guardrails.shields import InputGuard, OutputGuard
from src.schemas.domain import (
    AgentMessage,
    AuditEvent,
    AuditEventType,
    BaselineComparison,
    FinalReport,
    FollowUpAnswer,
    GovernanceSummary,
    GuardrailAction,
    GuardrailFinding,
    GuardrailSeverity,
    RunStatus,
    VerificationGap,
    new_run_id,
)
from src.schemas.state import ResearchState, initial_state
from src.storage.database import AuditStore
from src.storage.vector_store import HybridVectorStore
from src.tools.web_search import WebRetriever

logger = logging.getLogger(__name__)

Route = Literal["researcher", "synthesizer"]
NODES = ("input_guard", "planner", "master", "researcher", "drafter", "verifier", "synthesizer", "report_approver", "output_guard")
LLM_AGENTS = ("guardian", "planner", "master", "researcher", "drafter", "verifier", "synthesizer", "report_approver")
DISCLAIMER = (
    "AI-generated research brief. Every published claim passed entailment checking against a verbatim span of "
    "its cited source, but sources can be wrong or outdated: verify decision-critical facts against the cited originals."
)


def route_verification(
    state: ResearchState,
    *,
    coverage_threshold: float = 0.70,
    grounding_threshold: float = 0.75,
) -> Route:
    """Conditional edge: loop back for more evidence, or synthesize the verified report."""
    below_bar = (
        state.get("coverage_score", 0.0) < coverage_threshold
        or state.get("grounding_rate", 0.0) < grounding_threshold
    )
    budget_left = state.get("iteration_count", 0) < state.get("max_iterations", 1)
    if below_bar and budget_left and not state.get("budget_exhausted") and state.get("target_sub_question_ids"):
        return "researcher"
    return "synthesizer"


def route_after_guard(state: ResearchState) -> Literal["planner", "__end__"]:
    """Conditional edge: a blocked question never reaches the research agents."""
    return "__end__" if state.get("blocked") else "planner"


def quick_fix_targets(gaps: Sequence[VerificationGap], evidence_by_sq: dict[str, list[str]]) -> list[VerificationGap]:
    """Gaps a quick fix can repair: claims were rejected and the sub-question has evidence to fix them from.
    A sub-question without evidence needs new research, which is the Master Approver's call, not the Drafter's."""
    return [g for g in gaps if g.unsupported_claims and evidence_by_sq.get(g.sub_question_id)]


def route_group(state: ResearchState) -> Literal["drafter", "master"]:
    """Conditional edge inside the group chat: a quick fix back to the Drafter, or the round report to the Master."""
    return "drafter" if state.get("group_next") == "drafter" else "master"


def route_report(state: ResearchState) -> Literal["synthesizer", "output_guard"]:
    """Conditional edge out of the Report Approver: a fix round with the Synthesizer, or publication."""
    return "synthesizer" if state.get("audit_next") == "synthesizer" else "output_guard"


def route_master(state: ResearchState) -> Literal["planner", "researcher", "synthesizer"]:
    """Conditional edge out of the Master Approver: revise the plan, run a group round, or synthesize."""
    decision = state.get("master_decision")
    if decision == "revise":
        return "planner"
    if decision in ("dispatch", "another_round"):
        return "researcher"
    return "synthesizer"


class ResearchOrchestrator:
    """Builds the agents, wires the StateGraph, and runs research end-to-end."""

    def __init__(
        self,
        settings: Settings,
        gateway: MeshGateway,
        retriever: WebRetriever,
        vector_store: HybridVectorStore,
        *,
        store: AuditStore | None = None,
        event_sink: EventSink | None = None,
    ) -> None:
        self.settings = settings
        self.gateway = gateway
        self.vector_store = vector_store
        self.store = store
        self.event_sink = event_sink
        loader: Callable[[Sequence[str]], list[dict[str, Any]]] | None = store.get_documents if store else None
        self.planner = PlannerAgent(gateway, settings)
        self.master = MasterApproverAgent(gateway, settings)
        self.researcher = ResearcherAgent(gateway, settings, retriever, vector_store, document_loader=loader)
        self.drafter = DrafterAgent(gateway, settings)
        self.verifier = VerifierAgent(gateway, settings, vector_store)
        self.synthesizer = SynthesizerAgent(gateway, settings)
        self.report_approver = ReportApproverAgent(gateway, settings, vector_store)
        self.followup_agent = FollowUpAgent(gateway, settings, vector_store)
        self.baseline_agent = BaselineAgent(gateway, settings, vector_store)
        self.guardian = GuardianAgent(gateway, settings)
        self.input_guard = InputGuard(settings)
        self.output_guard = OutputGuard(settings)
        self.graph = self._build_graph()

    # -- graph ---------------------------------------------------------------
    def _route(self, state: ResearchState) -> Route:
        return route_verification(
            state,
            coverage_threshold=self.settings.coverage_threshold,
            grounding_threshold=self.settings.grounding_threshold,
        )

    def _build_graph(self) -> Any:
        graph = StateGraph(ResearchState)
        graph.add_node("input_guard", self.input_guard_node)
        graph.add_node("planner", self.planner_node)
        graph.add_node("master", self.master_node)
        graph.add_node("researcher", self.researcher_node)
        graph.add_node("drafter", self.drafter_node)
        graph.add_node("verifier", self.verifier_node)
        graph.add_node("synthesizer", self.synthesizer_node)
        graph.add_node("output_guard", self.output_guard_node)
        if self.settings.report_audit:
            graph.add_node("report_approver", self.report_approver_node)
        graph.add_edge(START, "input_guard")
        graph.add_conditional_edges("input_guard", route_after_guard, {"planner": "planner", "__end__": END})
        graph.add_edge("planner", "master")
        graph.add_conditional_edges(
            "master", route_master, {"planner": "planner", "researcher": "researcher", "synthesizer": "synthesizer"}
        )
        graph.add_edge("researcher", "drafter")
        graph.add_edge("drafter", "verifier")
        graph.add_conditional_edges("verifier", route_group, {"drafter": "drafter", "master": "master"})
        if self.settings.report_audit:
            graph.add_edge("synthesizer", "report_approver")
            graph.add_conditional_edges(
                "report_approver", route_report, {"synthesizer": "synthesizer", "output_guard": "output_guard"}
            )
        else:
            graph.add_edge("synthesizer", "output_guard")
        graph.add_edge("output_guard", END)
        return graph.compile()

    def mermaid(self) -> str:
        return str(self.graph.get_graph().draw_mermaid())

    # -- event plumbing --------------------------------------------------------
    def _emit(self, event: AuditEvent) -> None:
        if self.store is not None:
            try:
                self.store.record_audit_events([event])
            except Exception:
                logger.exception("Failed to persist audit event")
        if self.event_sink is not None:
            try:
                self.event_sink(event)
            except Exception:
                logger.exception("Live event sink failed")

    def _collector(self, state: ResearchState, node: str, iteration: int | None = None) -> EventCollector:
        it = state.get("iteration_count", 0) if iteration is None else iteration
        return EventCollector(state["run_id"], node, it, sink=self._emit)

    @staticmethod
    def _instructions_for(state: ResearchState, agent: str) -> dict[str, list[str]]:
        """The Master Approver's instructions for one agent, keyed by sub-question id."""
        out: dict[str, list[str]] = {}
        for item in state.get("master_instructions", []):
            if item.agent == agent:
                out.setdefault(item.sub_question_id, []).append(item.instruction)
        return out

    @staticmethod
    def _post(col: EventCollector, message: AgentMessage) -> AgentMessage:
        """Publish one group-chat hand-off as an audit event (persisted, streamed, hash-chained)."""
        to = ", ".join(message.to) or "team"
        col.emit(AuditEventType.AGENT_MESSAGE, f"{message.sender} -> {to}: {message.text[:160]}", **message.model_dump(mode="json"))
        return message

    @staticmethod
    def _emit_finding(col: EventCollector, finding: GuardrailFinding) -> None:
        col.emit(
            AuditEventType.GUARDRAIL_FINDING,
            f"{finding.guard}: {finding.kind} -> {finding.action.value} ({finding.location})",
            **finding.model_dump(mode="json"),
        )

    # -- nodes ---------------------------------------------------------------
    def input_guard_node(self, state: ResearchState) -> dict[str, Any]:
        col = self._collector(state, "input_guard", 0)
        col.emit(AuditEventType.NODE_STARTED, "Input shield screening the research question")
        screen = self.input_guard.screen(state["query"])
        for finding in screen.findings:
            self._emit_finding(col, finding)
        blocked_reason = screen.blocked_reason

        llm_screen = "skipped"
        if (
            blocked_reason is None
            and self.settings.guardrails_enabled
            and self.settings.guardrail_llm_screen
            and self.gateway.configured
        ):
            try:
                verdict = self.guardian.classify(state["run_id"], screen.text)
                if self.guardian.should_block(verdict):
                    llm_screen = "blocked"
                    blocked_reason = f"Request blocked by the Guardian classifier: {verdict.rationale or ', '.join(verdict.categories)}"
                elif verdict.verdict != "allow":
                    llm_screen = "flagged"
                else:
                    llm_screen = "passed"
                if verdict.verdict != "allow":
                    self._emit_finding(
                        col,
                        GuardrailFinding(
                            guard="llm_screen",
                            kind=(verdict.categories or ["policy"])[0],
                            severity=GuardrailSeverity.HIGH if llm_screen == "blocked" else GuardrailSeverity.LOW,
                            action=GuardrailAction.BLOCK if llm_screen == "blocked" else GuardrailAction.FLAG,
                            detail=f"Guardian verdict '{verdict.verdict}' (confidence {verdict.confidence:.2f})",
                            location="query",
                        ),
                    )
            except Exception as exc:  # fail-open: the deterministic shield already passed this input
                llm_screen = "error"
                col.emit(AuditEventType.WARNING, f"Guardian classifier unavailable ({type(exc).__name__}); continuing")

        status = "blocked" if blocked_reason else ("passed_with_redactions" if screen.redacted else "passed")
        col.emit(
            AuditEventType.GUARDRAIL_CHECK,
            f"Input shield: {status.replace('_', ' ')} · Guardian: {llm_screen}",
            stage="input",
            status=status,
            llm_screen=llm_screen,
            findings=len(screen.findings),
        )
        if blocked_reason:
            col.emit(AuditEventType.GUARDRAIL_BLOCKED, blocked_reason, stage="input")
        col.emit(AuditEventType.NODE_COMPLETED, f"Input shield finished ({status.replace('_', ' ')})")
        return {
            "query": screen.text,
            "blocked": blocked_reason is not None,
            "block_reason": blocked_reason or "",
            "llm_screen": llm_screen,
            "audit_log": col.events,
        }

    def planner_node(self, state: ResearchState) -> dict[str, Any]:
        plan_round = state.get("plan_round", 0) + 1
        col = self._collector(state, "planner", plan_round)
        review = state.get("plan_review")
        notes: list[str] = []
        if plan_round == 1 or review is None:
            col.emit(AuditEventType.NODE_STARTED, "Planner decomposing the research question")
            titles = (
                [d["filename"] for d in self.store.get_documents(state.get("document_ids", []))]
                if self.store and state.get("document_ids")
                else []
            )
            sub_questions, strategy = self.planner.plan(
                state["run_id"], state["query"], document_titles=titles, collector=col
            )
        else:
            col.emit(AuditEventType.NODE_STARTED, f"Planner revising the plan after the Master Approver's review (round {plan_round})")
            sub_questions, strategy, notes = self.planner.revise(
                state["run_id"], state["query"], current=state["sub_questions"],
                strategy=state.get("research_strategy", ""), review=review, collector=col,
            )
        message = self._post(col, chat.plan_proposal(
            state["run_id"], sub_questions=sub_questions, strategy=strategy, plan_round=plan_round, notes=notes
        ))
        col.emit(AuditEventType.NODE_COMPLETED, f"Planner produced {len(sub_questions)} sub-questions (version {plan_round})")
        return {
            "sub_questions": sub_questions,
            "research_strategy": strategy,
            "target_sub_question_ids": [sq.id for sq in sub_questions],
            "iteration_count": 0,
            "plan_round": plan_round,
            "group_chat": [message],
            "audit_log": col.events,
        }

    def master_node(self, state: ResearchState) -> dict[str, Any]:
        """The Master Approver: reviews the plan until approved, then reviews every evidence round."""
        if not state.get("plan_approved"):
            return self._master_plan_review(state)
        return self._master_round_review(state)

    def _master_plan_review(self, state: ResearchState) -> dict[str, Any]:
        plan_round = state.get("plan_round", 1)
        col = self._collector(state, "master", plan_round)
        col.emit(AuditEventType.NODE_STARTED, f"Master Approver reviewing plan version {plan_round}", mode="plan")
        review = self.master.review_plan(
            state["run_id"], state["query"], sub_questions=state["sub_questions"],
            strategy=state.get("research_strategy", ""), plan_round=plan_round,
            previous=state.get("plan_review"), collector=col,
        )
        approved = review.passed or review.forced
        message = self._post(col, chat.plan_verdict(state["run_id"], review))
        outcome = "plan approved, group dispatched" if approved else "revision requested"
        col.emit(AuditEventType.NODE_COMPLETED, f"Master Approver finished plan review {plan_round}: {outcome}", mode="plan")
        return {
            "plan_review": review, "plan_reviews": [review], "plan_approved": approved,
            "master_decision": "dispatch" if approved else "revise", "group_chat": [message], "audit_log": col.events,
        }

    def _master_round_review(self, state: ResearchState) -> dict[str, Any]:
        iteration = state["iteration_count"]
        col = self._collector(state, "master", iteration)
        col.emit(AuditEventType.NODE_STARTED, f"Master Approver reviewing round {iteration}", mode="round")
        metrics = state["iteration_metrics"][-1]
        floor = self._route(state)
        budget = self.gateway.budget_status(state["run_id"])
        budget_exhausted = bool(state.get("budget_exhausted"))
        if floor == "researcher" and budget["soft_limit_reached"]:
            budget_exhausted, floor = True, "synthesizer"
            col.emit(
                AuditEventType.BUDGET_GUARD,
                f"Budget guard: {budget['fraction_used']:.0%} of the run budget used -> loop-back suppressed",
                **budget,
            )
        gate_passed = (
            metrics.coverage_score >= self.settings.coverage_threshold
            and metrics.grounding_rate >= self.settings.grounding_threshold
        )
        review = self.master.review_round(
            state["run_id"], state["query"], sub_questions=state["sub_questions"], metrics=metrics,
            gaps=state.get("verification_gaps", []), contradictions=state.get("contradictions", []),
            floor=floor, gate_passed=gate_passed,
            budget_exhausted=budget_exhausted or bool(budget["soft_limit_reached"]),
            max_iterations=state.get("max_iterations", 1),
            extra_rounds_used=sum(r.extra_round for r in state.get("round_reviews", [])), collector=col,
        )
        route = "researcher" if review.decision == "another_round" else "synthesizer"
        col.emit(
            AuditEventType.ROUTE_DECISION,
            f"Master Approver -> {route} ({review.reason})",
            route=route,
            decided_by="master",
            reason=review.reason,
            coverage_score=metrics.coverage_score,
            grounding_rate=metrics.grounding_rate,
            coverage_threshold=self.settings.coverage_threshold,
            grounding_threshold=self.settings.grounding_threshold,
            iteration=iteration,
            max_iterations=state.get("max_iterations"),
        )
        if route == "researcher":
            gaps = [g for g in state.get("verification_gaps", []) if g.sub_question_id in review.target_sub_question_ids]
            col.emit(
                AuditEventType.LOOP_BACK,
                f"Loop-back {iteration}: re-researching {', '.join(review.target_sub_question_ids)}",
                gaps=[g.model_dump() for g in gaps],
                extra_round=review.extra_round,
            )
        message = self._post(col, chat.round_decision(
            state["run_id"], review, max_iterations=state.get("max_iterations", 1)
        ))
        col.emit(AuditEventType.NODE_COMPLETED, f"Master Approver finished round {iteration} review: {review.decision.replace('_', ' ')}", mode="round")
        return {
            "master_decision": review.decision,
            "round_reviews": [review],
            "route": route,
            "target_sub_question_ids": review.target_sub_question_ids,
            "master_instructions": review.instructions,
            "budget_exhausted": budget_exhausted,
            "group_chat": [message],
            "audit_log": col.events,
        }

    def researcher_node(self, state: ResearchState) -> dict[str, Any]:
        iteration = state.get("iteration_count", 0) + 1
        col = self._collector(state, "researcher", iteration)
        targets = state.get("target_sub_question_ids") or [sq.id for sq in state["sub_questions"]]
        col.emit(
            AuditEventType.NODE_STARTED,
            f"Researcher pass {iteration}: retrieving evidence for {', '.join(targets)}",
            target_sub_question_ids=targets,
        )
        previous = [q for sq in state["sub_questions"] for q in sq.search_queries] + [
            rq.query for rq in state.get("refinement_queries", [])
        ]
        result = self.researcher.research(
            state["run_id"],
            query=state["query"],
            sub_questions=state["sub_questions"],
            target_ids=targets,
            existing_sources=state.get("sources", {}),
            gaps=state.get("verification_gaps", []),
            previous_queries=previous,
            document_ids=state.get("document_ids", []),
            iteration=iteration,
            collector=col,
            instructions=self._instructions_for(state, "researcher") if iteration > 1 else None,
        )
        if self.store is not None and result.sources:
            self.store.record_sources(state["run_id"], list(result.sources.values()))
            self.store.record_evidence(state["run_id"], list(result.evidence.values()))
        total_sources = len(state.get("sources", {})) + len(result.sources)
        message = self._post(col, chat.evidence_handoff(
            state["run_id"], iteration=iteration, targets=targets, new_sources=result.sources,
            total_sources=total_sources, refinements=result.refinement_queries,
            shielded=sum(e.event_type is AuditEventType.GUARDRAIL_FINDING for e in col.events),
            instructions=state.get("master_instructions", []) if iteration > 1 else [],
        ))
        col.emit(
            AuditEventType.NODE_COMPLETED,
            f"Researcher indexed {len(result.sources)} new source(s), {len(result.evidence)} evidence spans",
            new_sources=len(result.sources),
            total_sources=total_sources,
        )
        return {
            "iteration_count": iteration,
            "sources": result.sources,
            "evidence": result.evidence,
            "evidence_by_sq": result.evidence_by_sq,
            "refinement_queries": result.refinement_queries,
            "exchange": 0,
            "group_next": "",
            "round_first_check": None,
            "group_chat": [message],
            "audit_log": col.events,
        }

    def drafter_node(self, state: ResearchState) -> dict[str, Any]:
        iteration = state["iteration_count"]
        exchange = state.get("exchange", 0)
        col = self._collector(state, "drafter")
        targets = [sq for sq in state["sub_questions"] if sq.id in set(state["target_sub_question_ids"])]
        col.emit(
            AuditEventType.NODE_STARTED,
            f"Drafter quick fix {exchange}: revising {len(targets)} section(s) the Verifier sent back" if exchange
            else f"Drafter writing {len(targets)} cited section(s)",
            exchange=exchange,
        )
        previous = state.get("drafts", {})
        gaps = {g.sub_question_id: g for g in state.get("verification_gaps", [])}
        evidence = state.get("evidence", {})
        briefs = self._instructions_for(state, "drafter")

        def draft(sq: Any) -> Any:
            chunks = [evidence[c] for c in state.get("evidence_by_sq", {}).get(sq.id, []) if c in evidence]
            return self.drafter.draft(
                state["run_id"],
                query=state["query"],
                sub_question=sq,
                evidence=chunks,
                sources=state.get("sources", {}),
                iteration=iteration,
                gap=gaps.get(sq.id) if iteration > 1 or exchange else None,
                collector=col,
                instructions=briefs.get(sq.id) if iteration > 1 else None,
                attempt=exchange + 1,
                previous_draft=previous[sq.id].text if exchange and sq.id in previous else None,
            )

        drafts = self.drafter._parallel(draft, targets)
        message = self._post(col, chat.drafts_handoff(
            state["run_id"], iteration=iteration, drafts=drafts, gaps=state.get("verification_gaps", []),
            instructions=state.get("master_instructions", []) if iteration > 1 else [], exchange=exchange,
        ))
        col.emit(AuditEventType.NODE_COMPLETED, f"Drafter produced {len(drafts)} section(s)", exchange=exchange)
        return {"drafts": {d.sub_question_id: d for d in drafts}, "group_chat": [message], "audit_log": col.events}

    def verifier_node(self, state: ResearchState) -> dict[str, Any]:
        iteration = state["iteration_count"]
        exchange = state.get("exchange", 0)
        col = self._collector(state, "verifier")
        targets = set(state["target_sub_question_ids"])
        drafts = [d for sq_id, d in state["drafts"].items() if sq_id in targets]
        col.emit(
            AuditEventType.NODE_STARTED,
            f"Verifier re-checking {len(drafts)} section(s) after quick fix {exchange}" if exchange
            else f"Verifier checking {len(drafts)} draft section(s)",
            exchange=exchange,
        )
        sources = state.get("sources", {})

        outcome = self.verifier.verify_sections(
            state["run_id"], sub_questions=state["sub_questions"], drafts=drafts, sources=sources, collector=col
        )
        claims = {**state.get("claims", {}), **outcome.claims}
        verifications = {**state.get("verifications", {}), **outcome.verifications}
        evidence = state.get("evidence", {})
        contradictions = self.verifier.detect_contradictions(
            state["run_id"],
            query=state["query"],
            evidence_by_sq={
                sq: [evidence[c] for c in ids if c in evidence]
                for sq, ids in state.get("evidence_by_sq", {}).items()
            },
            sources=sources,
            claims=claims,
            verifications=verifications,
            collector=col,
            # a quick fix re-drafts from the same evidence, so the cross-source scan cannot change
            reuse_cross_source=state.get("contradictions", []) if exchange else None,
        )
        metrics, gaps = self.verifier.compute_metrics(
            iteration=iteration,
            sub_questions=state["sub_questions"],
            claims=claims,
            verifications=verifications,
            sources_indexed=len(sources),
            grounding_threshold=self.settings.grounding_threshold,
        )
        kept_claims, kept_verifications = outcome.claims, outcome.verifications
        previous_grounding = state.get("grounding_rate", 0.0)
        if exchange and metrics.grounding_rate < previous_grounding:
            # the quick fix made things worse: keep the previous version (the attempt stays in the audit history)
            kept_claims = {sq: state["claims"][sq] for sq in outcome.claims if sq in state.get("claims", {})}
            kept_verifications = {sq: state["verifications"][sq] for sq in outcome.verifications if sq in state.get("verifications", {})}
            claims = {**claims, **kept_claims}
            verifications = {**verifications, **kept_verifications}
            contradictions = state.get("contradictions", [])
            col.emit(
                AuditEventType.WARNING,
                f"Quick fix {exchange} lowered grounding ({metrics.grounding_rate:.0%} < {previous_grounding:.0%}); "
                f"keeping the previous version of {', '.join(sorted(kept_claims))}",
                exchange=exchange, reverted=sorted(kept_claims),
            )
            metrics, gaps = self.verifier.compute_metrics(
                iteration=iteration,
                sub_questions=state["sub_questions"],
                claims=claims,
                verifications=verifications,
                sources_indexed=len(sources),
                grounding_threshold=self.settings.grounding_threshold,
            )
        first = state.get("round_first_check") if exchange else metrics
        fixable = quick_fix_targets(gaps, state.get("evidence_by_sq", {}))
        progressed = exchange == 0 or metrics.grounding_rate > previous_grounding
        quick_fix = (
            bool(fixable)
            and exchange < self.settings.group_chat_exchanges
            and progressed
            and not self.gateway.budget_status(state["run_id"])["soft_limit_reached"]
        )
        col.emit(
            AuditEventType.METRICS_COMPUTED,
            f"Round {iteration}{f' quick fix {exchange}' if exchange else ''}: grounding {metrics.grounding_rate:.0%}, "
            f"coverage {metrics.coverage_score:.0%}, citation precision {metrics.citation_precision:.0%}",
            **metrics.model_dump(), exchange=exchange, final=not quick_fix,
        )
        common: dict[str, Any] = {
            "claims": kept_claims,
            "verifications": kept_verifications,
            "contradictions": contradictions,
            "verification_gaps": gaps,
            "coverage_score": metrics.coverage_score,
            "grounding_rate": metrics.grounding_rate,
            "claim_history": [c for cl in outcome.claims.values() for c in cl],
            "verification_history": [v for vs in outcome.verifications.values() for v in vs],
        }
        if quick_fix:
            nxt = exchange + 1
            col.emit(
                AuditEventType.QUICK_FIX,
                f"Quick fix {nxt}: {len(fixable)} section(s) sent back to the Drafter before the round report",
                exchange=nxt, targets=[g.sub_question_id for g in fixable], gaps=[g.model_dump() for g in fixable],
                grounding_rate=metrics.grounding_rate,
            )
            message = self._post(col, chat.fix_request(
                state["run_id"], metrics=metrics, targets=fixable, exchange=nxt,
                max_exchanges=self.settings.group_chat_exchanges, grounding_threshold=self.settings.grounding_threshold,
            ))
            col.emit(AuditEventType.NODE_COMPLETED, f"Verifier sent {len(fixable)} section(s) back to the Drafter", exchange=exchange)
            return {
                **common,
                "target_sub_question_ids": [g.sub_question_id for g in fixable],
                "exchange": nxt,
                "group_next": "drafter",
                "round_first_check": first,
                "group_chat": [message],
                "audit_log": col.events,
            }
        final = metrics.model_copy(update={
            "exchanges": exchange,
            "first_check_grounding": first.grounding_rate if first else metrics.grounding_rate,
            "first_check_coverage": first.coverage_score if first else metrics.coverage_score,
        })
        message = self._post(col, chat.verdicts_handoff(
            state["run_id"], metrics=final, gaps=gaps, contradictions=contradictions
        ))
        col.emit(AuditEventType.NODE_COMPLETED, f"Verifier finished round {iteration}: report sent to the Master Approver", exchange=exchange)
        return {
            **common,
            "target_sub_question_ids": metrics.target_sub_question_ids,
            "iteration_metrics": [final],
            "group_next": "master",
            "group_chat": [message],
            "audit_log": col.events,
        }

    def synthesizer_node(self, state: ResearchState) -> dict[str, Any]:
        if state.get("report_fix_request"):
            return self._synthesizer_fix(state)
        col = self._collector(state, "synthesizer")
        col.emit(AuditEventType.NODE_STARTED, "Synthesizer compiling the verified report")
        report = self.synthesizer.synthesize(
            state["run_id"],
            query=state["query"],
            sub_questions=state["sub_questions"],
            claims=state.get("claims", {}),
            verifications=state.get("verifications", {}),
            contradictions=state.get("contradictions", []),
            sources=state.get("sources", {}),
            iteration_metrics=state.get("iteration_metrics", []),
            collector=col,
            claim_history=state.get("claim_history", []),
            verification_history=state.get("verification_history", []),
        )
        if self.store is not None:
            included = {cid for section in report.sections for cid in section.claim_ids}
            self.store.record_claims(
                state["run_id"],
                state.get("claim_history", []),
                state.get("verification_history", []),
                included,
            )
        reviews = state.get("plan_reviews", [])
        report = report.model_copy(update={
            "metrics": report.metrics.model_copy(update={
                "plan_rounds": len(reviews), "plan_score": reviews[-1].score if reviews else 0.0,
                "quick_fixes": sum(m.exchanges for m in state.get("iteration_metrics", [])),
            }),
            "plan_reviews": reviews,
            "round_reviews": state.get("round_reviews", []),
        })
        message = self._post(col, chat.report_handoff(
            state["run_id"], report, to="report_approver" if self.settings.report_audit else "output_guard"
        ))
        col.emit(AuditEventType.NODE_COMPLETED, "Synthesizer finished")
        return {"final_report": report, "group_chat": [message], "audit_log": col.events}

    def _synthesizer_fix(self, state: ResearchState) -> dict[str, Any]:
        audit_round = state.get("audit_round", 1)
        col = self._collector(state, "synthesizer", audit_round)
        failing = state["report_fix_request"]
        col.emit(AuditEventType.NODE_STARTED, f"Synthesizer fixing {len(failing)} sentence(s) the Report Approver rejected",
                 mode="fix", audit_round=audit_round)
        verified, allowed = verified_material(state.get("claims", {}), state.get("verifications", {}), state.get("contradictions", []))
        try:
            fixes = self.synthesizer.revise(state["run_id"], query=state["query"], failing=failing, verified=verified)
        except (MeshTransientError, MeshStructuredOutputError) as exc:
            col.emit(AuditEventType.WARNING, f"Report fix unavailable ({type(exc).__name__}); the Report Approver will remove what fails")
            fixes = {}
        changes: dict[str, str | None] = {}
        dropped: list[str] = []
        unchanged = 0
        for item in failing:
            kept, _ = guard_text(fixes.get(item.sid, ""), allowed) if fixes.get(item.sid) else ("", [])
            changes[item.sid] = kept or None  # a fix must still cite only verified sources, else it is dropped
            if not kept:
                dropped.append(item.text)
            elif kept.strip() == item.text.strip():
                unchanged += 1  # the Synthesizer stands by it; the Report Approver judges it again
        revised = len(failing) - len(dropped) - unchanged
        report = rewrite_report(state["final_report"], changes)  # type: ignore[arg-type]
        message = self._post(col, chat.report_fix_handoff(
            state["run_id"], audit_round=audit_round, revised=revised, dropped=len(dropped), unchanged=unchanged
        ))
        col.emit(AuditEventType.NODE_COMPLETED, f"Synthesizer revised {revised}, kept {unchanged}, dropped {len(dropped)}",
                 mode="fix", audit_round=audit_round, revised=revised, dropped=len(dropped), unchanged=unchanged)
        return {"final_report": report, "report_fix_request": [], "report_dropped": dropped, "group_chat": [message],
                "audit_log": col.events}

    def report_approver_node(self, state: ResearchState) -> dict[str, Any]:
        audit_round = state.get("audit_round", 0) + 1
        col = self._collector(state, "report_approver", audit_round)
        col.emit(AuditEventType.NODE_STARTED, f"Report Approver checking every report sentence (round {audit_round})",
                 audit_round=audit_round)
        report: FinalReport = state["final_report"]  # type: ignore[assignment]
        known = {s.text: s for a in state.get("report_audits", []) for s in a.sentences if s.status in ("supported", "fixed", "skipped")}
        audit = self.report_approver.audit(
            state["run_id"], report, state.get("sources", {}), audit_round=audit_round, known=known,
            verified=verified_index(state.get("claims", {}), state.get("verifications", {})), collector=col,
        )
        fixes_left = (
            audit_round <= self.settings.report_audit_fixes
            and not self.gateway.budget_status(state["run_id"])["soft_limit_reached"]
        )
        audit = finalize_audit(audit, fixes_left=fixes_left)
        update: dict[str, Any] = {}
        if audit.decision == "remove":
            report = rewrite_report(report, {s.sid: None for s in audit.sentences if s.status == "removed"})
        if audit.decision in ("approve", "remove"):
            audits = [*state.get("report_audits", []), audit]
            usage = self.gateway.usage_for(state["run_id"])  # include the Report Approver's own Mesh calls
            removed = sum(s.status == "removed" for s in audit.sentences) + len(state.get("report_dropped", []))
            judged_left = audit.judged - sum(s.status == "removed" for s in audit.sentences)
            report = report.model_copy(update={
                "report_audits": audits,
                "metrics": report.metrics.model_copy(update={
                    "report_groundedness_initial": audits[0].groundedness,
                    # nothing judged because Mesh failed is unknown grounding, not 100%
                    "report_groundedness": (round(audit.supported / judged_left, 4) if judged_left
                                            else None if audit.unchecked else 1.0),
                    "report_sentences_fixed": sum(s.status == "fixed" for s in audit.sentences),
                    "report_sentences_removed": removed,
                    "llm_calls": usage["llm_calls"],
                    "total_tokens": usage["total_tokens"],
                    "llm_latency_ms": usage["latency_ms"],
                }),
            })
            update["final_report"] = report
        col.emit(
            AuditEventType.REPORT_AUDITED,
            f"Report Approver round {audit_round}: {audit.supported}/{audit.judged} cited sentences supported -> {audit.decision}",
            **audit.model_dump(mode="json"),
        )
        message = self._post(col, chat.audit_verdict(state["run_id"], audit, max_fixes=self.settings.report_audit_fixes))
        col.emit(AuditEventType.NODE_COMPLETED, f"Report Approver finished round {audit_round}: {audit.decision}",
                 audit_round=audit_round)
        return {
            **update,
            "report_audits": [audit],
            "audit_round": audit_round,
            "audit_next": "synthesizer" if audit.decision == "fix" else "output_guard",
            "report_fix_request": [s for s in audit.sentences if s.status == "failed"] if audit.decision == "fix" else [],
            "group_chat": [message],
            "audit_log": col.events,
        }

    def output_guard_node(self, state: ResearchState) -> dict[str, Any]:
        col = self._collector(state, "output_guard")
        col.emit(AuditEventType.NODE_STARTED, "Output shield reviewing the report before publication")
        report: FinalReport = state["final_report"]  # type: ignore[assignment]
        report, findings = self.output_guard.apply_to_report(report)
        for finding in findings:
            self._emit_finding(col, finding)
        governance = self._governance_summary(state, col.events)
        report = report.model_copy(update={"governance": governance})
        col.emit(
            AuditEventType.GUARDRAIL_CHECK,
            f"Output shield: {governance.pii_redactions} PII / {governance.secret_redactions} secret redaction(s); "
            f"{len(governance.findings)} guardrail finding(s) in this run",
            stage="output",
            findings=len(governance.findings),
        )
        if self.store is not None:
            self.store.save_final_report(report)
        message = self._post(col, chat.publish_notice(state["run_id"], governance))
        col.emit(AuditEventType.NODE_COMPLETED, "Report published")
        return {"final_report": report, "group_chat": [message], "audit_log": col.events}

    def _governance_summary(self, state: ResearchState, extra_events: list[AuditEvent]) -> GovernanceSummary:
        events = [*state.get("audit_log", []), *extra_events]
        findings = [
            GuardrailFinding.model_validate(e.payload)
            for e in events
            if e.event_type is AuditEventType.GUARDRAIL_FINDING
        ]
        by_guard: dict[str, int] = {}
        for f in findings:
            by_guard[f.guard] = by_guard.get(f.guard, 0) + 1
        redactions = [f for f in findings if f.action is GuardrailAction.REDACT]

        def redaction_count(kind: str) -> int:
            total = 0
            for f in redactions:
                if f.kind == kind:
                    try:
                        total += int(f.detail.split(" ", 1)[0])
                    except ValueError:
                        total += 1
            return total

        input_status = next(
            (e.payload.get("status", "passed") for e in events
             if e.event_type is AuditEventType.GUARDRAIL_CHECK and e.payload.get("stage") == "input"),
            "passed",
        )
        dense = self.vector_store.dense_enabled
        models = sorted(
            {self.settings.model_for(a) for a in LLM_AGENTS if a != "guardian" or state.get("llm_screen") not in ("skipped", None)}
            | ({self.settings.mesh_embedding_model} if dense else set())
        )
        head = self.store.audit_chain_head(state["run_id"]) if self.store is not None else None
        budget = self.gateway.budget_status(state["run_id"]) | {"loop_backs_suppressed": bool(state.get("budget_exhausted"))}
        return GovernanceSummary(
            policy_version=self.settings.policy_version,
            input_screen=input_status,
            llm_screen=state.get("llm_screen", "skipped"),
            findings=findings,
            findings_by_guard=by_guard,
            injections_neutralized=sum(f.kind == "indirect_prompt_injection" for f in findings),
            urls_blocked=sum(f.guard == "url_shield" for f in findings),
            pii_redactions=redaction_count("pii"),
            secret_redactions=redaction_count("secret"),
            claims_downgraded=sum(v.downgraded for v in state.get("verification_history", [])),
            budget=budget,
            models_used=models,
            audit_chain_head=head,
            audit_events=len(events),
            disclaimer=DISCLAIMER,
        )

    # -- public API ------------------------------------------------------------
    def run(
        self,
        query: str,
        *,
        run_id: str | None = None,
        max_iterations: int | None = None,
        document_ids: list[str] | None = None,
    ) -> ResearchState:
        """Execute the full graph synchronously and return the final state."""
        final: ResearchState = {}
        for _node, _update, state in self._execute(query, run_id, max_iterations, document_ids):
            final = state
        return final

    def stream(
        self,
        query: str,
        *,
        run_id: str | None = None,
        max_iterations: int | None = None,
        document_ids: list[str] | None = None,
    ) -> Iterator[tuple[str, dict[str, Any]]]:
        """Yield `(node_name, partial_update)` as each LangGraph node completes."""
        for node, update, _state in self._execute(query, run_id, max_iterations, document_ids):
            yield node, update

    def _execute(
        self,
        query: str,
        run_id: str | None,
        max_iterations: int | None,
        document_ids: list[str] | None,
    ) -> Iterator[tuple[str, dict[str, Any], ResearchState]]:
        run_id = run_id or new_run_id()
        iterations = max_iterations or self.settings.max_iterations
        docs = list(document_ids or [])
        if self.store is not None:
            # data minimisation: only the redacted question is ever persisted
            persisted_query = self.input_guard.redact_only(query) if self.settings.guardrails_enabled else query
            self.store.ensure_run(run_id, persisted_query, max_iterations=iterations, document_ids=docs)
            self.store.update_run_status(run_id, RunStatus.RUNNING)
        state = initial_state(run_id, query, max_iterations=iterations, document_ids=docs)
        start = EventCollector(run_id, "orchestrator", 0, sink=self._emit)
        start.emit(
            AuditEventType.RUN_STARTED,
            f"Research run started (max {iterations} iteration(s))",
            query=self.input_guard.redact_only(query),
            document_ids=docs,
            config=self.settings.public_summary(),
        )
        try:
            current: dict[str, Any] = dict(state)
            for chunk in self.graph.stream(
                state,
                config={"recursion_limit": 10 + (5 + 2 * self.settings.group_chat_exchanges) * iterations
                        + 2 * (self.settings.max_plan_revisions + 1) + 2 * (self.settings.report_audit_fixes + 1)},
                stream_mode=["updates", "values"],
            ):
                mode, payload = chunk
                if mode == "values":
                    current = payload
                    continue
                for node, update in payload.items():
                    yield node, update, current  # type: ignore[misc]
            final: ResearchState = current  # type: ignore[assignment]
        except Exception as exc:
            message = exc.message if isinstance(exc, DeepDiveError) else f"{type(exc).__name__}: {exc}"
            EventCollector(run_id, "orchestrator", 0, sink=self._emit).emit(
                AuditEventType.RUN_FAILED, f"Run failed: {message}", error_type=type(exc).__name__
            )
            if self.store is not None:
                self.store.update_run_status(run_id, RunStatus.FAILED, error=message)
            if isinstance(exc, DeepDiveError):
                raise
            raise OrchestrationError(f"Research run {run_id} failed: {message}") from exc

        closing = EventCollector(run_id, "orchestrator", final.get("iteration_count", 0), sink=self._emit)
        if final.get("blocked"):
            if self.store is not None:
                self.store.update_run_status(run_id, RunStatus.BLOCKED, error=final.get("block_reason"))
            closing.emit(AuditEventType.RUN_BLOCKED, f"Run blocked by guardrails: {final.get('block_reason')}")
            yield "__end__", {}, final
            return
        if self.store is not None:
            self.store.update_run_status(run_id, RunStatus.COMPLETED)
        report = final.get("final_report")
        closing.emit(
            AuditEventType.RUN_COMPLETED,
            "Research run completed",
            iterations=final.get("iteration_count", 0),
            metrics=report.metrics.model_dump() if report else None,
        )
        yield "__end__", {}, final

    def compare_baseline(self, run_id: str) -> BaselineComparison:
        """Answer the run's question with one plain Mesh call and check it against the run's own evidence."""
        report = self.store.get_report(run_id) if self.store else None
        if report is None or self.store is None:
            raise OrchestrationError("Only a completed run with a stored report can be compared")
        store = self.store
        collector = EventCollector(run_id, "baseline", 0, sink=self._emit)
        before = self.gateway.usage_for(run_id)
        comparison = self.baseline_agent.compare(
            run_id, report=report, sources=store.load_sources(run_id),
            published_claims=[c["claim_text"] for c in store.list_claims(run_id) if c.get("included_in_report")],
            evidence_loader=lambda: store.list_evidence(run_id), collector=collector,
        )
        after = self.gateway.usage_for(run_id)
        comparison = comparison.model_copy(update={
            "llm_calls": after["llm_calls"] - before["llm_calls"],
            "total_tokens": after["total_tokens"] - before["total_tokens"],
        })
        comparison, findings = self.output_guard.apply_to_comparison(comparison)
        for finding in findings:
            self._emit_finding(collector, finding)
        rate = "n/a" if comparison.traceable_rate is None else f"{comparison.traceable_rate:.0%}"
        collector.emit(
            AuditEventType.BASELINE_COMPARED,
            f"Plain AI answer: {comparison.supported} of {comparison.checked} claim(s) traceable to a source span ({rate}); "
            f"{comparison.contradicted} contradicted by the sources",
            **comparison.model_dump(mode="json"),
        )
        return comparison

    def followup(
        self, run_id: str, question: str, *, history: Sequence[dict[str, str]] = ()
    ) -> FollowUpAnswer:
        """Answer a follow-up question grounded in a completed run's evidence pool."""
        collector = EventCollector(run_id, "followup", 0, sink=self._emit)
        screen = self.input_guard.screen(question, location="followup_question")
        for finding in screen.findings:
            self._emit_finding(collector, finding)
        if not screen.allowed:
            collector.emit(AuditEventType.GUARDRAIL_BLOCKED, screen.blocked_reason or "blocked", stage="followup")
            raise GuardrailBlockedError(
                screen.blocked_reason or "Follow-up blocked by guardrails",
                details={"findings": [f.model_dump(mode="json") for f in screen.findings]},
            )
        report = self.store.get_report(run_id) if self.store else None
        loader = (lambda: self.store.list_evidence(run_id)) if self.store else None  # type: ignore[union-attr]
        answer = self.followup_agent.answer(
            run_id, screen.text, report=report, evidence_loader=loader, history=history, collector=collector
        )
        answer, findings = self.output_guard.apply_to_answer(answer)
        for finding in findings:
            self._emit_finding(collector, finding)
        if self.store is not None:
            self.store.record_followup(answer)
        return answer
