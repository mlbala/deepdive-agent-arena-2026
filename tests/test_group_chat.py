"""The Team Lead (master agent) and the agents' group chat: plan approval, round reviews that
cannot go below the quality gate, bounded extra rounds, fail-open behaviour, instructions the
other agents really receive, and chat messages that mirror the real hand-offs."""

from __future__ import annotations

from typing import Any

from eval.simulated_llm import HeuristicMeshSimulator
from src.agents.lead import focus_overlap, plan_checks, salient_terms
from src.schemas.domain import AuditEventType, Severity, SubQuestion

SOLAR_Q = "How much solar PV capacity was added globally in 2023?"
RBI_Q = "How does the Reserve Bank of India use the repo rate to control inflation?"


class _AlwaysReject(HeuristicMeshSimulator):
    def _lead_plan(self, p: dict[str, Any]) -> dict[str, Any]:
        body = super()._lead_plan(p)
        body["rubric"]["coverage"] = 0.5  # the rubric corroborates the high-severity issue
        body["issues"].append({"target": "plan", "severity": "high", "problem": "never good enough", "fix": "try again"})
        return body


class _NoisySeverity(HeuristicMeshSimulator):
    def _lead_plan(self, p: dict[str, Any]) -> dict[str, Any]:
        body = super()._lead_plan(p)  # every rubric criterion stays >= 0.85
        body["issues"].append({"target": "SQ-2", "severity": "high", "problem": "could be phrased more tightly", "fix": "reword"})
        return body


class _LeadDown(HeuristicMeshSimulator):
    def _lead_plan(self, p: dict[str, Any]) -> dict[str, Any]:
        return {}  # fails schema validation on every repair round

    def _lead_review(self, p: dict[str, Any]) -> dict[str, Any]:
        return {}


class _EagerApprover(HeuristicMeshSimulator):
    def _lead_review(self, p: dict[str, Any]) -> dict[str, Any]:
        return {"decision": "approve", "rationale": "looks fine to me", "focus_sub_question_ids": [], "instructions": []}


class _Perfectionist(HeuristicMeshSimulator):
    """Asks for another pass on every gap, even when the gate has passed."""

    def _lead_review(self, p: dict[str, Any]) -> dict[str, Any]:
        gaps = [g["sub_question_id"] for g in p.get("gaps", [])]
        return {"decision": "another_round", "rationale": "one more pass on the weak parts",
                "focus_sub_question_ids": gaps, "instructions": [
                    {"agent": "researcher", "sub_question_id": g, "instruction": "find the primary filing"} for g in gaps]}


class _Recording(HeuristicMeshSimulator):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.drafter_payloads: list[dict[str, Any]] = []
        self.refinement_payloads: list[dict[str, Any]] = []

    def _drafter(self, p: dict[str, Any]) -> dict[str, Any]:
        self.drafter_payloads.append(p)
        return super()._drafter(p)

    def _researcher(self, p: dict[str, Any]) -> dict[str, Any]:
        self.refinement_payloads.append(p)
        return super()._researcher(p)


def test_team_lead_sends_a_weak_plan_back_until_it_passes(make_orchestrator):
    simulator = HeuristicMeshSimulator(reject_first_plan=True)
    orchestrator, _store, live = make_orchestrator(client=simulator)
    state = orchestrator.run(RBI_Q)

    first, second = state["plan_reviews"]
    assert not first.passed and any(i.severity is Severity.HIGH and i.blocking for i in first.issues)
    assert second.passed and second.score >= second.threshold and second.round == 2
    assert simulator.calls_by_role["planner"] == 1 and simulator.calls_by_role["planner.revise"] == 1
    assert len(state["sub_questions"]) == 4, "the revision added the sub-question the Team Lead asked for"

    started = [e.node for e in live if e.event_type is AuditEventType.NODE_STARTED]
    assert started[:6] == ["input_guard", "planner", "lead", "planner", "lead", "researcher"]
    room = [(m.sender, m.kind, m.round) for m in state["group_chat"] if m.channel == "plan_review"]
    assert room == [("planner", "proposal", 1), ("lead", "critique", 1), ("planner", "revision", 2), ("lead", "approval", 2)]
    revision, approval = state["group_chat"][2], state["group_chat"][3]
    assert any(item.startswith("Changed:") for item in revision.items)
    assert set(approval.to) == {"planner", "researcher", "drafter", "verifier"} and "Team brief" in approval.text
    assert state["final_report"].metrics.plan_rounds == 2 and len(state["final_report"].plan_reviews) == 2


def test_plan_revisions_are_bounded_and_open_issues_stay_on_record(make_orchestrator):
    orchestrator, _store, _live = make_orchestrator(client=_AlwaysReject(), max_plan_revisions=1)
    state = orchestrator.run(RBI_Q)
    reviews = state["plan_reviews"]
    assert len(reviews) == 2, "one revision, then the run proceeds"
    assert not reviews[-1].passed and reviews[-1].forced
    assert state["final_report"] is not None and state["iteration_metrics"]
    verdict = [m for m in state["group_chat"] if m.sender == "lead" and m.channel == "plan_review"][-1]
    assert verdict.kind == "approval" and "revision budget is spent" in verdict.text
    assert any("never good enough" in item for item in verdict.items)


def test_model_raised_high_issue_blocks_only_when_the_rubric_corroborates_it(make_orchestrator):
    orchestrator, _store, _live = make_orchestrator(client=_NoisySeverity())
    state = orchestrator.run(RBI_Q)
    (review,) = state["plan_reviews"]
    noisy = next(i for i in review.issues if i.problem == "could be phrased more tightly")
    assert review.passed and not noisy.blocking, "a 0.9-scoring plan is not bounced on a severity label alone"
    rejected, *_ = make_orchestrator(client=_AlwaysReject(), max_plan_revisions=0)[0].run(RBI_Q)["plan_reviews"]
    assert not rejected.passed and rejected.forced
    assert any(i.blocking and i.problem == "never good enough" for i in rejected.issues)


def test_team_lead_cannot_approve_below_the_quality_gate(make_orchestrator):
    orchestrator, _store, _live = make_orchestrator(client=_EagerApprover(hallucinate_first_draft=True), grounding_threshold=0.9)
    state = orchestrator.run(SOLAR_Q)
    first, second = state["round_reviews"]
    assert not first.gate_passed and first.floor == "researcher"
    assert first.decision == "another_round" and first.overruled, "the gate keeps the loop going"
    assert first.instructions, "targets still get instructions derived from the Verifier's gaps"
    assert all(i.origin == "gate" for i in first.instructions)
    assert second.gate_passed and second.decision == "approve"
    assert len(state["iteration_metrics"]) == 2


def test_team_lead_extra_rounds_are_targeted_and_bounded(make_settings):
    from src.agents.lead import TeamLeadAgent
    from src.core.mesh_gateway import MeshGateway
    from src.schemas.domain import IterationMetrics, VerificationGap

    settings = make_settings(lead_extra_rounds=1)
    lead = TeamLeadAgent(MeshGateway(settings, client=_Perfectionist(), sleep=lambda _s: None), settings)
    subs = [SubQuestion(id=f"SQ-{i}", question=f"Sub-question number {i}?", search_queries=["q one"]) for i in (1, 2, 3)]
    metrics = IterationMetrics(iteration=1, total_claims=10, supported=9, contradicted=0, unsupported=1, grounding_rate=0.9,
                               coverage_score=1.0, citation_precision=0.9, quote_verification_rate=1.0, sources_indexed=5,
                               target_sub_question_ids=["SQ-2"])
    gaps = [VerificationGap(sub_question_id="SQ-2", reason="1 of 2 claims failed", unsupported_claims=["x"], feedback=["x — no span"])]

    def review(**overrides: Any):
        kwargs: dict[str, Any] = dict(sub_questions=subs, metrics=metrics, gaps=gaps, contradictions=[], floor="synthesizer",
                                      gate_passed=True, budget_exhausted=False, max_iterations=3, extra_rounds_used=0)
        return lead.review_round("run_x", "q", **(kwargs | overrides))

    extra = review()
    assert extra.decision == "another_round" and extra.extra_round and extra.target_sub_question_ids == ["SQ-2"]
    assert {i.agent for i in extra.instructions} == {"researcher", "drafter"}, "both agents get a brief"
    assert review(extra_rounds_used=1).decision == "approve", "LEAD_EXTRA_ROUNDS caps the extra passes"
    assert review(max_iterations=1).decision == "approve", "never beyond MAX_ITERATIONS"
    assert review(budget_exhausted=True).decision == "approve", "never past the budget soft limit"
    assert review(gaps=[]).decision == "approve", "an extra pass needs a sub-question that still has a gap"


def test_team_lead_fails_open_on_the_model_but_not_on_the_checks_or_gate(make_orchestrator):
    orchestrator, _store, live = make_orchestrator(client=_LeadDown(hallucinate_first_draft=True), grounding_threshold=0.9)
    state = orchestrator.run(RBI_Q)
    plan = state["plan_reviews"][0]
    assert plan.llm_reviewed is False and plan.rubric == {} and plan.passed
    assert all(not r.llm_reviewed for r in state["round_reviews"])
    final = state["round_reviews"][-1]
    if not final.gate_passed:  # an approval below the bar must never claim the evidence is complete
        assert "not met" in final.rationale and "unproven" in final.rationale
    assert any(e.event_type is AuditEventType.WARNING and e.node == "lead" for e in live)
    assert state["final_report"] is not None


def test_plan_checks_flag_dropped_entities_overlap_and_bad_queries():
    question = "What were Zentrix Biologics' audited 2023 revenues and the ZX-9 Phase 3 results?"
    assert {"Zentrix", "Biologics", "2023", "ZX-9", "Phase", "3"} <= set(salient_terms(question))
    plan = [
        SubQuestion(id="SQ-1", question="What were Zentrix's audited revenues in 2023?", search_queries=["Zentrix revenue 2023"]),
        SubQuestion(id="SQ-2", question="How did market analysts react to the announcement?", search_queries=["revenue"]),
        SubQuestion(id="SQ-3", question="How did market analysts react to the news announcement?", search_queries=["Zentrix analysts"]),
    ]
    issues = plan_checks(question, plan)
    problems = " | ".join(i.problem for i in issues)
    assert "'ZX-9'" in problems and "SQ-2 and SQ-3 overlap" in problems and "too short" in problems
    assert all(i.origin == "check" for i in issues) and not any(i.severity is Severity.HIGH for i in issues)
    assert any(i.severity is Severity.HIGH and "only 2" in i.problem for i in plan_checks(question, plan[:2]))
    # sub-questions sharing only the topic's own words are not duplicates
    assert focus_overlap("What target does the RBI repo rate serve?", "How has the RBI repo rate changed since 2020?", RBI_Q) < 0.6


def test_group_chat_mirrors_the_real_handoffs_and_instructions_are_used(make_orchestrator):
    simulator = _Recording(hallucinate_first_draft=True)
    orchestrator, store, live = make_orchestrator(client=simulator, grounding_threshold=0.9)
    state = orchestrator.run(SOLAR_Q)
    chat = state["group_chat"]
    m1, m2 = state["iteration_metrics"]

    # every hand-off is an audit event, in the same order (so it is persisted and hash-chained)
    persisted = [e["payload"]["message_id"] for e in store.list_audit_events(state["run_id"]) if e["event_type"] == "agent_message"]
    assert persisted == [m.message_id for m in chat]
    assert store.verify_audit_chain(state["run_id"])["valid"]

    reports = [m for m in chat if m.sender == "verifier"]
    decisions = [m for m in chat if m.sender == "lead" and m.channel == "evidence_loop"]
    assert [m.round for m in reports] == [1, 2] and [d.round for d in decisions] == [1, 2]
    assert all(m.to == ["lead"] for m in reports), "the Verifier reports to the Team Lead"
    assert f"{m1.supported} supported" in reports[0].text and reports[0].score == m1.grounding_rate
    assert decisions[0].passed is False and set(decisions[0].to) == {"researcher", "drafter", "verifier"}
    assert decisions[1].passed is True and decisions[1].to == ["synthesizer"] and m2.grounding_rate >= 0.9
    assert any(item.startswith("Researcher · ") for item in decisions[0].items)

    # the Researcher and the Drafter really receive the Team Lead's instructions on the next round
    briefed = [g for p in simulator.refinement_payloads for g in p.get("gaps", []) if g.get("team_lead_instructions")]
    assert briefed, "refinement planning saw the Team Lead's instructions"
    redrafts = [p for p in simulator.drafter_payloads if "previous_attempt_feedback" in p]
    assert redrafts and all(p.get("team_lead_instructions") for p in redrafts)
    assert all(p["previous_attempt_feedback"]["verifier_notes"] for p in redrafts)
    gap_feedback = [line for e in live if e.event_type is AuditEventType.LOOP_BACK for g in e.payload["gaps"] for line in g["feedback"]]
    assert any("2031" in line for line in gap_feedback)

    assert [m.sender for m in chat if m.channel == "publish"] == ["synthesizer", "output_guard"]
    assert [r.decision for r in state["final_report"].round_reviews] == ["another_round", "approve"]
