"""The Report Approver: the final approver that re-verifies every sentence of the synthesized report,
sends failures back to the Synthesizer, removes what still fails, and never strips conflicts."""

from __future__ import annotations

from eval.simulated_llm import HeuristicMeshSimulator
from src.agents.report_approver import report_sentences, rewrite_report
from src.core.exceptions import MeshTransientError
from src.schemas.domain import AuditEventType

SOLAR_Q = "How much new solar PV capacity did the world add in 2023, and what drove that growth?"
EMBELLISHMENT = "reshape every global market"


def test_report_approver_sends_unsupported_sentences_back_to_the_synthesizer(make_orchestrator):
    simulator = HeuristicMeshSimulator(embellish_report=True)
    orchestrator, store, live = make_orchestrator(client=simulator)
    state = orchestrator.run(SOLAR_Q)
    report = state["final_report"]

    first, second = state["report_audits"]
    bad = [s for s in first.sentences if s.status == "failed"]
    assert first.decision == "fix" and any(EMBELLISHMENT in s.text for s in bad)
    assert second.decision == "approve" and second.passed
    assert EMBELLISHMENT not in report.executive_summary
    m = report.metrics
    assert m.report_groundedness_initial is not None and m.report_groundedness_initial < 1.0
    assert m.report_groundedness == 1.0
    assert m.report_sentences_fixed + m.report_sentences_removed >= 1
    assert len(report.report_audits) == 2, "the published report carries its audit trail"

    publish = [(c.sender, c.kind) for c in state["group_chat"] if c.channel == "publish"]
    assert publish == [("synthesizer", "report"), ("report_approver", "fix_request"), ("synthesizer", "fix"),
                       ("report_approver", "approval"), ("output_guard", "decision")]
    assert simulator.calls_by_role["synthesizer.revise"] == 1
    assert "report_approver" in {t["agent_name"] for t in store.list_traces(state["run_id"])}, "checks billed to the Report Approver"
    assert sum(e.event_type is AuditEventType.REPORT_AUDITED for e in live) == 2
    assert not any(e.event_type is AuditEventType.CLAIM_VERIFIED and str(e.payload.get("claim_id", "")).startswith("RPT-")
                   for e in live), "sentence checks do not pollute the claim matrix"
    assert store.verify_audit_chain(state["run_id"])["valid"]
    usage = store.trace_summary(state["run_id"])
    assert usage["total_tokens"] == m.total_tokens and usage["llm_calls"] == m.llm_calls


def test_what_still_fails_after_the_last_fix_is_removed(make_orchestrator):
    orchestrator, _store, _live = make_orchestrator(client=HeuristicMeshSimulator(embellish_report=True), report_audit_fixes=0)
    state = orchestrator.run(SOLAR_Q)
    (audit,) = state["report_audits"]
    assert audit.decision == "remove"
    removed = [s for s in audit.sentences if s.status == "removed"]
    assert removed and all(s.text not in state["final_report"].executive_summary for s in removed)
    assert state["final_report"].metrics.report_sentences_removed == len(removed)
    assert state["final_report"].metrics.report_groundedness == 1.0


def test_conflict_statements_are_never_stripped(make_orchestrator):
    orchestrator, _store, _live = make_orchestrator(client=HeuristicMeshSimulator(embellish_report=True))
    state = orchestrator.run(SOLAR_Q)
    conflicts = [s for a in state["report_audits"] for s in a.sentences if "conflict" in s.text.lower()]
    assert conflicts, "the planted solar conflict is stated in the report"
    assert all(s.status == "skipped" for s in conflicts)
    narrative = " ".join(s.body for s in state["final_report"].sections)
    assert all(s.text in narrative for s in conflicts)


def test_a_failed_mesh_check_keeps_sentences_as_unchecked_instead_of_removing_them(make_orchestrator):
    """A live run lost six good sentences to a Mesh outage once: no judgement is not a failed check."""
    orchestrator, _store, live = make_orchestrator(client=HeuristicMeshSimulator(embellish_report=True))

    def mesh_down(*_args, **_kwargs):
        raise MeshTransientError("Connection error.")

    orchestrator.report_approver.checker._llm = mesh_down
    state = orchestrator.run(SOLAR_Q)
    (audit,) = state["report_audits"]
    assert audit.decision == "approve" and not audit.llm_reviewed
    assert audit.failed == 0 and audit.unchecked == 1, "only the reworded sentence needed the model"
    report = state["final_report"]
    assert report.metrics.report_sentences_removed == 0
    unchecked = [s for s in audit.sentences if s.status == "unchecked"]
    assert EMBELLISHMENT in unchecked[0].text and "Mesh" in unchecked[0].reason, "flagged, not passed off as supported"
    published = " ".join([report.executive_summary, *report.key_findings, *(s.body for s in report.sections)])
    assert all(s.text in published for s in unchecked), "nothing is removed because the checker was down"
    assert any(e.event_type is AuditEventType.WARNING and "Report Approver" in e.message for e in live)
    (verdict,) = [c for c in state["group_chat"] if c.sender == "report_approver"]
    assert "could not be checked" in verdict.text


def test_verbatim_verified_claims_carry_their_verdict_and_repeats_are_judged_once(make_orchestrator):
    simulator = HeuristicMeshSimulator(embellish_report=True)
    orchestrator, _store, _live = make_orchestrator(client=simulator)
    state = orchestrator.run(SOLAR_Q)
    first = state["report_audits"][0]
    carried = [s for s in first.sentences if s.reason.startswith("word for word verified claim CLM-")]
    assert carried and all(s.status == "supported" and s.quote_verified and s.evidence_quote for s in carried)
    judged = [s for s in first.sentences if s.status in ("supported", "failed") and s not in carried]
    assert [s for s in judged if EMBELLISHMENT in s.text], "the reworded sentence is still judged by the model"
    by_text: dict[str, set[str]] = {}
    for s in first.sentences:
        by_text.setdefault(s.text, set()).add(s.status)
    assert all(len(statuses) == 1 for statuses in by_text.values()), "identical sentences get one verdict"


def test_failure_reasons_stay_readable_without_a_model_rationale():
    from src.agents.report_approver import failure_reason
    from src.schemas.domain import Verdict, VerificationResult

    bare = VerificationResult(claim_id="RPT-1.2", sub_question_id="SQ-0", iteration=1, verdict=Verdict.UNSUPPORTED,
                              confidence=0.4, evidence_source_id="SRC-004", rationale=" [quote match: normalized]")
    assert failure_reason(bare) == "SRC-004 does not state this"
    assert failure_reason(bare.model_copy(update={"downgrade_reason": "claim figure(s) 473 absent"})) == "claim figure(s) 473 absent"
    assert failure_reason(bare.model_copy(update={"rationale": "Source gives 2022 figures. [quote match: exact]"})) == \
        "Source gives 2022 figures."


def test_report_audit_can_be_switched_off(make_orchestrator):
    orchestrator, _store, _live = make_orchestrator(client=HeuristicMeshSimulator(embellish_report=True), report_audit=False)
    edges = {(e.source, e.target) for e in orchestrator.graph.get_graph().edges}
    assert ("synthesizer", "output_guard") in edges and "report_approver" not in orchestrator.graph.get_graph().nodes
    state = orchestrator.run(SOLAR_Q)
    assert state["final_report"].report_audits == [] and EMBELLISHMENT in state["final_report"].executive_summary


def test_rewrite_report_replaces_and_removes_sentences_by_id(make_orchestrator):
    orchestrator, _store, _live = make_orchestrator()
    report = orchestrator.run(SOLAR_Q)["final_report"]
    sentences = report_sentences(report)
    assert len(sentences) >= 2
    rewritten = rewrite_report(report, {"S1": "Replaced sentence [SRC-001].", "S2": None})
    after = [s for _f, _i, s in report_sentences(rewritten)]
    assert after[0] == "Replaced sentence [SRC-001]."
    assert len(after) == len(sentences) - 1, "S2 was removed"
    assert after[1:] == [s for _f, _i, s in sentences[2:]], "every other sentence is untouched"
