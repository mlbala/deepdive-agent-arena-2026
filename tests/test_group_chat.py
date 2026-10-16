"""Plan-review room (Planner <-> Approver) and the agents' group chat: quality gates, bounded
revisions, fail-open behaviour, and messages that mirror the real hand-offs."""

from __future__ import annotations

from typing import Any

from eval.simulated_llm import HeuristicMeshSimulator
from src.agents.approver import focus_overlap, plan_checks, salient_terms
from src.schemas.domain import AuditEventType, Severity, SubQuestion

SOLAR_Q = "How much solar PV capacity was added globally in 2023?"
RBI_Q = "How does the Reserve Bank of India use the repo rate to control inflation?"


class _AlwaysReject(HeuristicMeshSimulator):
    def _approver(self, p: dict[str, Any]) -> dict[str, Any]:
        body = super()._approver(p)
        body["rubric"]["coverage"] = 0.5  # the rubric corroborates the high-severity issue
        body["issues"].append({"target": "plan", "severity": "high", "problem": "never good enough", "fix": "try again"})
        return body


class _NoisySeverity(HeuristicMeshSimulator):
    def _approver(self, p: dict[str, Any]) -> dict[str, Any]:
        body = super()._approver(p)  # every rubric criterion stays >= 0.85
        body["issues"].append({"target": "SQ-2", "severity": "high", "problem": "could be phrased more tightly", "fix": "reword"})
        return body


class _ApproverDown(HeuristicMeshSimulator):
    def _approver(self, p: dict[str, Any]) -> dict[str, Any]:
        return {}  # fails schema validation on every repair round


class _RecordingDrafter(HeuristicMeshSimulator):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.drafter_payloads: list[dict[str, Any]] = []

    def _drafter(self, p: dict[str, Any]) -> dict[str, Any]:
        self.drafter_payloads.append(p)
        return super()._drafter(p)


def test_approver_sends_a_weak_plan_back_until_it_passes(make_orchestrator):
    simulator = HeuristicMeshSimulator(reject_first_plan=True)
    orchestrator, _store, live = make_orchestrator(client=simulator)
    state = orchestrator.run(RBI_Q)

    first, second = state["plan_reviews"]
    assert not first.passed and any(i.severity is Severity.HIGH and i.blocking for i in first.issues)
    assert second.passed and second.score >= second.threshold and second.round == 2
    assert simulator.calls_by_role["planner"] == 1 and simulator.calls_by_role["planner.revise"] == 1
    assert len(state["sub_questions"]) == 4, "the revision added the sub-question the Approver asked for"

    started = [e.node for e in live if e.event_type is AuditEventType.NODE_STARTED]
    assert started[:6] == ["input_guard", "planner", "approver", "planner", "approver", "researcher"]
    room = [(m.sender, m.kind, m.round) for m in state["group_chat"] if m.channel == "plan_review"]
    assert room == [("planner", "proposal", 1), ("approver", "critique", 1), ("planner", "revision", 2), ("approver", "approval", 2)]
    revision = state["group_chat"][2]
    assert any(item.startswith("Changed:") for item in revision.items)
    assert state["final_report"].metrics.plan_rounds == 2


def test_plan_revisions_are_bounded_and_open_issues_stay_on_record(make_orchestrator):
    orchestrator, _store, _live = make_orchestrator(client=_AlwaysReject(), max_plan_revisions=1)
    state = orchestrator.run(RBI_Q)
    reviews = state["plan_reviews"]
    assert len(reviews) == 2, "one revision, then the run proceeds"
    assert not reviews[-1].passed and reviews[-1].forced
    assert state["final_report"] is not None and state["iteration_metrics"]
    verdict = [m for m in state["group_chat"] if m.sender == "approver"][-1]
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


def test_approver_fails_open_on_the_model_but_not_on_the_checks(make_orchestrator):
    orchestrator, _store, live = make_orchestrator(client=_ApproverDown())
    state = orchestrator.run(RBI_Q)
    review = state["plan_reviews"][0]
    assert review.llm_reviewed is False and review.rubric == {}
    assert review.passed, "the deterministic checks found nothing blocking"
    assert any(e.event_type is AuditEventType.WARNING and e.node == "approver" for e in live)


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


def test_group_chat_mirrors_the_real_handoffs_through_a_loop_back(make_orchestrator):
    simulator = _RecordingDrafter(hallucinate_first_draft=True)
    orchestrator, store, live = make_orchestrator(client=simulator, grounding_threshold=0.9)
    state = orchestrator.run(SOLAR_Q)
    chat = state["group_chat"]
    m1, m2 = state["iteration_metrics"]

    # every hand-off is an audit event, in the same order (so it is persisted and hash-chained)
    persisted = [e["payload"]["message_id"] for e in store.list_audit_events(state["run_id"]) if e["event_type"] == "agent_message"]
    assert persisted == [m.message_id for m in chat]
    assert store.verify_audit_chain(state["run_id"])["valid"]

    verdicts = [m for m in chat if m.sender == "verifier"]
    gates = [m for m in chat if m.sender == "quality_gate"]
    assert [m.round for m in verdicts] == [1, 2] and [g.round for g in gates] == [1, 2]
    assert f"{m1.supported} supported" in verdicts[0].text and verdicts[0].score == m1.grounding_rate
    assert gates[0].passed is False and gates[0].to == ["researcher", "drafter"] and "Below the bar" in gates[0].text
    assert gates[1].passed is True and gates[1].to == ["synthesizer"] and m2.grounding_rate >= 0.9
    redraft = next(m for m in chat if m.sender == "drafter" and m.round == 2)
    assert redraft.text.startswith("Re-drafted")

    # the Drafter really receives the Verifier's per-claim reasons on the re-draft
    gap_feedback = [line for e in live if e.event_type is AuditEventType.LOOP_BACK for g in e.payload["gaps"] for line in g["feedback"]]
    assert any("2031" in line for line in gap_feedback)
    redraft_payloads = [p for p in simulator.drafter_payloads if "previous_attempt_feedback" in p]
    assert redraft_payloads and all(p["previous_attempt_feedback"]["verifier_notes"] for p in redraft_payloads)

    assert [m.sender for m in chat if m.channel == "publish"] == ["synthesizer", "output_guard"]
