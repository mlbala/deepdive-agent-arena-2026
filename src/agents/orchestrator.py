"""LangGraph cyclic orchestrator: two quality-gated agent rooms plus guardrail nodes.

    START -> input_guard --(blocked)--> END
                 |
                 v
              planner <--revise-- approver                       (plan review room)
                 |                   ^ |
                 +-------------------+ | route_plan_review
                                       v
                                  researcher -> drafter -> verifier --route_verification--> synthesizer -> output_guard -> END
                                       ^                                   |
                                       +------------ loop-back ------------+   (evidence room)

`approver` scores each version of the plan (Mesh rubric + deterministic checks) and sends it
back to the Planner with concrete fixes until it clears PLAN_APPROVAL_THRESHOLD or the revision
budget (MAX_PLAN_REVISIONS) is spent. Every node posts its real hand-off to the agents' group
chat (`state["group_chat"]`, AGENT_MESSAGE audit events), so the conversation is auditable.

`input_guard` runs the deterministic input shield (prompt injection, harmful requests,
PII/secret redaction) and the Mesh-backed Guardian classifier; a blocked question ends the
run before any research agent sees it. `output_guard` redacts the published report and
attaches the governance summary (findings, budgets, models, audit-chain head).

`route_verification` sends the run back to the Researcher — with targeted refinement
queries for only the failing sub-questions — while

    coverage_score < COVERAGE_THRESHOLD (0.70)  or  grounding_rate < GROUNDING_THRESHOLD (0.75)

and `iteration_count < max_iterations` and the per-run budget's soft limit has not been
reached. Otherwise the verified state goes to the Synthesizer.

Every node emits typed `AuditEvent`s that are (a) appended to `state["audit_log"]`,
(b) persisted to SQLite immediately, and (c) pushed to an optional live sink (SSE).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator, Sequence
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph

from src.agents import group_chat as chat
from src.agents.approver import ApproverAgent
from src.agents.base import EventCollector, EventSink
from src.agents.drafter import DrafterAgent
from src.agents.followup import FollowUpAgent
from src.agents.guardian import GuardianAgent
from src.agents.planner import PlannerAgent
from src.agents.researcher import ResearcherAgent
from src.agents.synthesizer import SynthesizerAgent
from src.agents.verifier import VerifierAgent
from src.core.config import Settings
from src.core.exceptions import DeepDiveError, GuardrailBlockedError, OrchestrationError
from src.core.mesh_gateway import MeshGateway
from src.guardrails.shields import InputGuard, OutputGuard
from src.schemas.domain import (
    AgentMessage,
    AuditEvent,
    AuditEventType,
    FinalReport,
    FollowUpAnswer,
    GovernanceSummary,
    GuardrailAction,
    GuardrailFinding,
    GuardrailSeverity,
    RunStatus,
    new_run_id,
)
from src.schemas.state import ResearchState, initial_state
from src.storage.database import AuditStore
from src.storage.vector_store import HybridVectorStore
from src.tools.web_search import WebRetriever

logger = logging.getLogger(__name__)

Route = Literal["researcher", "synthesizer"]
NODES = ("input_guard", "planner", "approver", "researcher", "drafter", "verifier", "synthesizer", "output_guard")
LLM_AGENTS = ("guardian", "planner", "approver", "researcher", "drafter", "verifier", "synthesizer")
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


def route_plan_review(state: ResearchState) -> Literal["planner", "researcher"]:
    """Conditional edge: an approved plan (or one whose revision budget is spent) goes to retrieval."""
    review = state.get("plan_review")
    return "researcher" if review is None or review.passed or review.forced else "planner"


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
        self.approver = ApproverAgent(gateway, settings)
        self.researcher = ResearcherAgent(gateway, settings, retriever, vector_store, document_loader=loader)
        self.drafter = DrafterAgent(gateway, settings)
        self.verifier = VerifierAgent(gateway, settings, vector_store)
        self.synthesizer = SynthesizerAgent(gateway, settings)
        self.followup_agent = FollowUpAgent(gateway, settings, vector_store)
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
        graph.add_node("approver", self.approver_node)
        graph.add_node("researcher", self.researcher_node)
        graph.add_node("drafter", self.drafter_node)
        graph.add_node("verifier", self.verifier_node)
        graph.add_node("synthesizer", self.synthesizer_node)
        graph.add_node("output_guard", self.output_guard_node)
        graph.add_edge(START, "input_guard")
        graph.add_conditional_edges("input_guard", route_after_guard, {"planner": "planner", "__end__": END})
        graph.add_edge("planner", "approver")
        graph.add_conditional_edges("approver", route_plan_review, {"planner": "planner", "researcher": "researcher"})
        graph.add_edge("researcher", "drafter")
        graph.add_edge("drafter", "verifier")
        graph.add_conditional_edges(
            "verifier", self._route, {"researcher": "researcher", "synthesizer": "synthesizer"}
        )
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
            col.emit(AuditEventType.NODE_STARTED, f"Planner revising the plan after the Approver's review (round {plan_round})")
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

    def approver_node(self, state: ResearchState) -> dict[str, Any]:
        plan_round = state.get("plan_round", 1)
        col = self._collector(state, "approver", plan_round)
        col.emit(AuditEventType.NODE_STARTED, f"Approver reviewing plan version {plan_round}")
        review = self.approver.review(
            state["run_id"], state["query"], sub_questions=state["sub_questions"],
            strategy=state.get("research_strategy", ""), plan_round=plan_round,
            previous=state.get("plan_review"), collector=col,
        )
        message = self._post(col, chat.plan_verdict(state["run_id"], review))
        outcome = "approved" if review.passed else ("proceeding with open issues" if review.forced else "revision requested")
        col.emit(AuditEventType.NODE_COMPLETED, f"Approver finished round {plan_round}: {outcome}")
        return {"plan_review": review, "plan_reviews": [review], "group_chat": [message], "audit_log": col.events}

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
        )
        if self.store is not None and result.sources:
            self.store.record_sources(state["run_id"], list(result.sources.values()))
            self.store.record_evidence(state["run_id"], list(result.evidence.values()))
        total_sources = len(state.get("sources", {})) + len(result.sources)
        message = self._post(col, chat.evidence_handoff(
            state["run_id"], iteration=iteration, targets=targets, new_sources=result.sources,
            total_sources=total_sources, refinements=result.refinement_queries,
            shielded=sum(e.event_type is AuditEventType.GUARDRAIL_FINDING for e in col.events),
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
            "group_chat": [message],
            "audit_log": col.events,
        }

    def drafter_node(self, state: ResearchState) -> dict[str, Any]:
        iteration = state["iteration_count"]
        col = self._collector(state, "drafter")
        targets = [sq for sq in state["sub_questions"] if sq.id in set(state["target_sub_question_ids"])]
        col.emit(AuditEventType.NODE_STARTED, f"Drafter writing {len(targets)} cited section(s)")
        gaps = {g.sub_question_id: g for g in state.get("verification_gaps", [])}
        evidence = state.get("evidence", {})

        def draft(sq: Any) -> Any:
            chunks = [evidence[c] for c in state.get("evidence_by_sq", {}).get(sq.id, []) if c in evidence]
            return self.drafter.draft(
                state["run_id"],
                query=state["query"],
                sub_question=sq,
                evidence=chunks,
                sources=state.get("sources", {}),
                iteration=iteration,
                gap=gaps.get(sq.id) if iteration > 1 else None,
                collector=col,
            )

        drafts = self.drafter._parallel(draft, targets)
        message = self._post(col, chat.drafts_handoff(
            state["run_id"], iteration=iteration, drafts=drafts, gaps=state.get("verification_gaps", [])
        ))
        col.emit(AuditEventType.NODE_COMPLETED, f"Drafter produced {len(drafts)} section(s)")
        return {"drafts": {d.sub_question_id: d for d in drafts}, "group_chat": [message], "audit_log": col.events}

    def verifier_node(self, state: ResearchState) -> dict[str, Any]:
        iteration = state["iteration_count"]
        col = self._collector(state, "verifier")
        targets = set(state["target_sub_question_ids"])
        drafts = [d for sq_id, d in state["drafts"].items() if sq_id in targets]
        col.emit(AuditEventType.NODE_STARTED, f"Verifier checking {len(drafts)} draft section(s)")
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
        )
        metrics, gaps = self.verifier.compute_metrics(
            iteration=iteration,
            sub_questions=state["sub_questions"],
            claims=claims,
            verifications=verifications,
            sources_indexed=len(sources),
            grounding_threshold=self.settings.grounding_threshold,
        )
        col.emit(
            AuditEventType.METRICS_COMPUTED,
            f"Iteration {iteration}: grounding {metrics.grounding_rate:.0%}, coverage {metrics.coverage_score:.0%}, "
            f"citation precision {metrics.citation_precision:.0%}",
            **metrics.model_dump(),
        )
        decision_state: ResearchState = {
            **state,
            "coverage_score": metrics.coverage_score,
            "grounding_rate": metrics.grounding_rate,
            "target_sub_question_ids": metrics.target_sub_question_ids,
        }
        route = self._route(decision_state)
        budget = self.gateway.budget_status(state["run_id"])
        budget_exhausted = bool(state.get("budget_exhausted"))
        if route == "researcher" and budget["soft_limit_reached"]:
            budget_exhausted = True
            route = "synthesizer"
            col.emit(
                AuditEventType.BUDGET_GUARD,
                f"Budget guard: {budget['fraction_used']:.0%} of the run budget used -> loop-back suppressed",
                **budget,
            )
        col.emit(
            AuditEventType.ROUTE_DECISION,
            f"route_verification -> {route}",
            route=route,
            coverage_score=metrics.coverage_score,
            grounding_rate=metrics.grounding_rate,
            coverage_threshold=self.settings.coverage_threshold,
            grounding_threshold=self.settings.grounding_threshold,
            iteration=iteration,
            max_iterations=state.get("max_iterations"),
        )
        if route == "researcher":
            col.emit(
                AuditEventType.LOOP_BACK,
                f"Loop-back {iteration}: re-researching {', '.join(metrics.target_sub_question_ids)}",
                gaps=[g.model_dump() for g in gaps],
            )
        messages = [
            self._post(col, chat.verdicts_handoff(state["run_id"], metrics=metrics, gaps=gaps, contradictions=contradictions)),
            self._post(col, chat.gate_decision(
                state["run_id"], metrics=metrics, route=route,
                coverage_threshold=self.settings.coverage_threshold,
                grounding_threshold=self.settings.grounding_threshold,
                budget_exhausted=budget_exhausted, max_iterations=state.get("max_iterations", 1),
            )),
        ]
        col.emit(AuditEventType.NODE_COMPLETED, f"Verifier finished iteration {iteration}")
        return {
            "budget_exhausted": budget_exhausted,
            "claims": outcome.claims,
            "verifications": outcome.verifications,
            "contradictions": contradictions,
            "verification_gaps": gaps,
            "target_sub_question_ids": metrics.target_sub_question_ids,
            "coverage_score": metrics.coverage_score,
            "grounding_rate": metrics.grounding_rate,
            "route": route,
            "iteration_metrics": [metrics],
            "claim_history": [c for cl in outcome.claims.values() for c in cl],
            "verification_history": [v for vs in outcome.verifications.values() for v in vs],
            "group_chat": messages,
            "audit_log": col.events,
        }

    def synthesizer_node(self, state: ResearchState) -> dict[str, Any]:
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
        report = report.model_copy(update={"metrics": report.metrics.model_copy(update={
            "plan_rounds": len(reviews), "plan_score": reviews[-1].score if reviews else 0.0,
        })})
        message = self._post(col, chat.report_handoff(state["run_id"], report))
        col.emit(AuditEventType.NODE_COMPLETED, "Synthesizer finished")
        return {"final_report": report, "group_chat": [message], "audit_log": col.events}

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
                config={"recursion_limit": 10 + 4 * iterations + 2 * (self.settings.max_plan_revisions + 1)},
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
