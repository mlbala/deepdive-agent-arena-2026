"""Verifier + self-correction loop tests: quote localisation, deterministic guards,
contradiction validation, metrics/gaps, `route_verification`, and the loop-back cycle."""

from __future__ import annotations

import json

import pytest

from eval.simulated_llm import HeuristicMeshSimulator
from src.agents.base import EventCollector
from src.agents.orchestrator import route_verification
from src.agents.verifier import VerifierAgent, locate_quote, ungrounded_numbers
from src.core.mesh_gateway import MeshGateway
from src.schemas.domain import (
    AtomicClaim,
    AuditEventType,
    RetrievalTier,
    SourceDocument,
    SubQuestion,
    Verdict,
    VerificationResult,
)
from src.storage.vector_store import HybridVectorStore, build_chunks
from src.tools.document_ingestor import chunk_text
from src.tools.source_scorer import score_source
from tests.conftest import FakeMeshClient

IEA_TEXT = (
    "Global annual renewable capacity additions increased by almost 50% to nearly 510 gigawatts (GW) in 2023, "
    "the fastest growth rate in the past two decades. Spot prices for solar PV modules declined by almost 50% "
    "year-on-year in 2023 as manufacturing capacity expanded rapidly. Global renewable capacity is forecast "
    "to reach 7 300 GW over the 2023-2028 period."
)
BLOG_TEXT = (
    "In our view, rising module prices made panels more expensive than ever, and global solar installations "
    "fell compared with 2022 as developers cancelled projects."
)
SQ = SubQuestion(id="SQ-1", question="How much solar capacity was added in 2023?", search_queries=["solar 2023"])


# ---------------------------------------------------------------------------
# Deterministic quote localisation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("quote", "method"),
    [
        ("increased by almost 50% to nearly 510 gigawatts (GW) in 2023", "exact"),
        ("“Spot prices for solar PV modules  declined by ALMOST 50% year-on-year in 2023.”", "normalized"),
        ("Global annual renewable capacity additions ... the fastest growth rate in the past two decades", "elliptical"),
        ("Global annual renewable capacity additions increased by nearly 50% to almost 510 gigawatts (GW) in 2023", "fuzzy"),
    ],
)
def test_locate_quote_methods_return_exact_spans(quote, method):
    match = locate_quote(IEA_TEXT, quote)
    assert match is not None and match.method == method
    span = IEA_TEXT[match.start : match.end]
    assert span and IEA_TEXT.find(span) == match.start


@pytest.mark.parametrize(
    "quote",
    [
        "Global annual renewable capacity additions increased by almost 60% to nearly 510 gigawatts (GW) in 2023",
        "Solar installations collapsed in 2023 because module prices soared",
        "short",
        None,
    ],
)
def test_locate_quote_rejects_altered_or_absent_quotes(quote):
    assert locate_quote(IEA_TEXT, quote) is None


def test_ungrounded_numbers_handles_separators():
    assert ungrounded_numbers("Capacity reaches 7,300 GW by 2028", [IEA_TEXT]) == set()
    assert ungrounded_numbers("Additions rose 65% to 510 GW", [IEA_TEXT]) == {"65"}


# ---------------------------------------------------------------------------
# Entailment guards (scripted Mesh responses)
# ---------------------------------------------------------------------------
def _source(sid: str, url: str, text: str) -> SourceDocument:
    return SourceDocument(
        source_id=sid,
        url=url,
        title=sid,
        domain=url.split("/")[2],
        content=text,
        retrieval_tier=RetrievalTier.OFFLINE,
        authority=score_source(url, content=text),
    )


@pytest.fixture
def verifier_setup(make_settings):
    settings = make_settings()
    sources = {
        "SRC-001": _source("SRC-001", "https://www.iea.org/reports/renewables-2023", IEA_TEXT),
        "SRC-002": _source("SRC-002", "https://solar-blog.example/blog/stall", BLOG_TEXT),
    }
    store = HybridVectorStore(settings)
    for sid, src in sources.items():
        store.add_chunks(
            "run_v",
            build_chunks(
                source_id=sid, content=src.content, spans=chunk_text(src.content, chunk_size=400, overlap=60),
                authority_score=src.authority.score, url=src.url, title=sid,
            ),
        )

    def make(script):
        client = FakeMeshClient(script)
        return VerifierAgent(MeshGateway(settings, client=client), settings, store), client

    return sources, make


def _claim(n: int, text: str, cited: list[str]) -> AtomicClaim:
    return AtomicClaim(claim_id=f"CLM-1.1.{n}", sub_question_id="SQ-1", iteration=1, text=text, cited_source_ids=cited)


def test_guards_downgrade_hallucinated_low_confidence_and_miscited_verdicts(verifier_setup):
    sources, make = verifier_setup
    claims = [
        _claim(1, "Renewable additions reached nearly 510 GW in 2023", ["SRC-001"]),
        _claim(2, "Module prices declined by almost 50% in 2023", ["SRC-001"]),
        _claim(3, "Solar is the cheapest energy source in history", []),
        _claim(4, "Growth was the fastest in two decades", ["SRC-001"]),
        _claim(5, "Renewable additions grew by 65% in 2023", ["SRC-001"]),
        _claim(6, "Module prices rose sharply in 2023", ["SRC-001"]),
        _claim(7, "Capacity will reach 7 300 GW by 2028", ["SRC-001"]),
    ]
    quote1 = "increased by almost 50% to nearly 510 gigawatts (GW) in 2023"
    judgements = {
        "judgements": [
            {"claim_id": "CLM-1.1.1", "verdict": "SUPPORTED", "confidence": 0.95, "evidence_source_id": "SRC-001", "evidence_quote": quote1},
            {"claim_id": "CLM-1.1.2", "verdict": "SUPPORTED", "confidence": 0.9, "evidence_source_id": "SRC-001",
             "evidence_quote": "The IEA confirms module prices halved in 2023."},
            {"claim_id": "CLM-1.1.4", "verdict": "SUPPORTED", "confidence": 0.3, "evidence_source_id": "SRC-001",
             "evidence_quote": "the fastest growth rate in the past two decades"},
            {"claim_id": "CLM-1.1.5", "verdict": "SUPPORTED", "confidence": 0.9, "evidence_source_id": "SRC-001", "evidence_quote": quote1},
            {"claim_id": "CLM-1.1.6", "verdict": "CONTRADICTED", "confidence": 0.9, "evidence_source_id": "SRC-001",
             "evidence_quote": "Spot prices for solar PV modules declined by almost 50% year-on-year in 2023"},
            {"claim_id": "CLM-1.1.7", "verdict": "SUPPORTED", "confidence": 0.9, "evidence_source_id": "SRC-002",
             "evidence_quote": "global solar installations fell compared with 2022"},
        ]
    }
    verifier, client = make([json.dumps(judgements), json.dumps({"judgements": []})])
    collector = EventCollector("run_v", "verifier", 1)
    results = {r.claim_id: r for r in verifier.judge_claims("run_v", SQ, claims, sources, collector)}

    ok = results["CLM-1.1.1"]
    assert ok.verdict is Verdict.SUPPORTED and ok.quote_verified and not ok.downgraded
    assert IEA_TEXT[ok.evidence_char_start : ok.evidence_char_end] == quote1

    assert results["CLM-1.1.2"].verdict is Verdict.UNSUPPORTED
    assert "hallucinated" in (results["CLM-1.1.2"].downgrade_reason or "")
    assert results["CLM-1.1.3"].verdict is Verdict.UNSUPPORTED  # uncited: no LLM call needed
    assert "no citation" in results["CLM-1.1.3"].rationale
    assert "confidence" in (results["CLM-1.1.4"].downgrade_reason or "")
    assert "65" in (results["CLM-1.1.5"].downgrade_reason or "")  # numeric grounding guard
    assert results["CLM-1.1.6"].verdict is Verdict.CONTRADICTED and results["CLM-1.1.6"].quote_verified
    assert "does not cite" in (results["CLM-1.1.7"].downgrade_reason or "")

    sent_claims = [c["claim_id"] for c in json.loads(client.calls[0]["messages"][1]["content"].split("INPUT (JSON):")[1])["claims"]]
    assert "CLM-1.1.3" not in sent_claims, "uncited claims are decided deterministically"
    downgrades = [e for e in collector.events if e.event_type is AuditEventType.CLAIM_DOWNGRADED]
    assert len(downgrades) == 4


def test_contradiction_radar_validates_quotes_and_ranks_authority(verifier_setup):
    sources, make = verifier_setup
    scan = {
        "contradictions": [
            {"topic": "module price direction", "source_a": "SRC-001",
             "quote_a": "Spot prices for solar PV modules declined by almost 50% year-on-year in 2023",
             "source_b": "SRC-002", "quote_b": "rising module prices made panels more expensive than ever",
             "severity": "high", "explanation": "Opposite price directions."},
            {"topic": "fabricated", "source_a": "SRC-001", "quote_a": "Solar demand collapsed in 2023",
             "source_b": "SRC-002", "quote_b": "fell compared with 2022", "severity": "low"},
            {"topic": "self", "source_a": "SRC-001", "quote_a": "x" * 20, "source_b": "SRC-001", "quote_b": "y" * 20},
        ]
    }
    verifier, _ = make([json.dumps(scan)])
    collector = EventCollector("run_v", "verifier", 1)
    evidence = {"SQ-1": [c for c in verifier.vector_store.all_chunks("run_v")]}
    contradicted = VerificationResult(
        claim_id="CLM-1.1.9", verdict=Verdict.CONTRADICTED, confidence=0.9, evidence_source_id="SRC-001",
        evidence_quote="declined by almost 50%", quote_verified=True,
    )
    items = verifier.detect_contradictions(
        "run_v", query="solar", evidence_by_sq=evidence, sources=sources,
        claims={"SQ-1": [_claim(9, "Module prices rose in 2023", ["SRC-001"])]},
        verifications={"SQ-1": [contradicted]}, collector=collector,
    )
    cross = [i for i in items if i.origin == "cross_source"]
    assert len(cross) == 1
    assert cross[0].quote_a_verified and cross[0].quote_b_verified
    assert cross[0].resolution_hint.startswith("Weight SRC-001")
    assert any(i.origin == "claim_verification" and i.source_b == "SRC-001" for i in items)
    assert any(e.event_type is AuditEventType.WARNING and "Discarded" in e.message for e in collector.events)


def test_compute_metrics_and_gap_targets():
    sqs = [SQ, SQ.model_copy(update={"id": "SQ-2"}), SQ.model_copy(update={"id": "SQ-3"})]
    claims = {
        "SQ-1": [_claim(1, "a b c d", ["SRC-001"]), _claim(2, "e f g h", ["SRC-001"])],
        "SQ-2": [AtomicClaim(claim_id="CLM-2.1.1", sub_question_id="SQ-2", iteration=1, text="i j k", cited_source_ids=["SRC-002"])],
        "SQ-3": [],
    }

    def v(cid, verdict):
        return VerificationResult(claim_id=cid, verdict=verdict, confidence=0.9, evidence_quote="q" if verdict is Verdict.SUPPORTED else None, quote_verified=verdict is Verdict.SUPPORTED)

    verifications = {
        "SQ-1": [v("CLM-1.1.1", Verdict.SUPPORTED), v("CLM-1.1.2", Verdict.SUPPORTED)],
        "SQ-2": [v("CLM-2.1.1", Verdict.UNSUPPORTED)],
    }
    metrics, gaps = VerifierAgent.compute_metrics(
        iteration=1, sub_questions=sqs, claims=claims, verifications=verifications,
        sources_indexed=2, grounding_threshold=0.75,
    )
    assert metrics.grounding_rate == pytest.approx(2 / 3, abs=1e-3)
    assert metrics.coverage_score == pytest.approx(1 / 3, abs=1e-3)
    assert metrics.citation_precision == pytest.approx(2 / 3, abs=1e-3)
    assert metrics.target_sub_question_ids == ["SQ-2", "SQ-3"]
    assert gaps[0].unsupported_claims == ["i j k"]


# ---------------------------------------------------------------------------
# route_verification (conditional loop-back edge)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("coverage", "grounding", "iteration", "targets", "expected"),
    [
        (0.50, 0.90, 1, ["SQ-2"], "researcher"),   # coverage below 0.70
        (0.90, 0.60, 1, ["SQ-1"], "researcher"),   # grounding below 0.75
        (0.69, 0.74, 2, ["SQ-1"], "researcher"),   # both low, budget left
        (0.50, 0.50, 3, ["SQ-1"], "synthesizer"),  # iteration budget exhausted
        (0.70, 0.75, 1, [], "synthesizer"),        # thresholds met exactly
        (0.95, 0.95, 1, [], "synthesizer"),
    ],
)
def test_route_verification(coverage, grounding, iteration, targets, expected):
    state = {
        "coverage_score": coverage, "grounding_rate": grounding, "iteration_count": iteration,
        "max_iterations": 3, "target_sub_question_ids": targets,
    }
    assert route_verification(state) == expected


# ---------------------------------------------------------------------------
# End-to-end self-correction cycle
# ---------------------------------------------------------------------------
def test_loop_back_repairs_hallucinated_draft(make_orchestrator):
    simulator = HeuristicMeshSimulator(hallucinate_first_draft=True)
    orchestrator, store, _live = make_orchestrator(client=simulator, grounding_threshold=0.9)
    state = orchestrator.run("How much solar PV capacity was added globally in 2023?")

    history = state["iteration_metrics"]
    assert len(history) == 2, "one loop-back expected"
    assert history[0].grounding_rate < 0.9 <= history[1].grounding_rate
    assert history[0].unsupported > 0 and history[1].unsupported == 0

    types = [e.event_type for e in state["audit_log"]]
    assert AuditEventType.LOOP_BACK in types and AuditEventType.REFINEMENT_PLANNED in types
    assert state["refinement_queries"], "loop-back must use targeted refinement queries"
    assert simulator.calls_by_role["researcher"] == 1  # refinement planning ran exactly once

    report = state["final_report"]
    assert report.metrics.loop_backs == 1
    assert report.metrics.grounding_rate_pre < report.metrics.grounding_rate_final
    fabricated = [r for r in report.removed_claims if "2031" in r.text]
    assert fabricated and all("superseded" in r.reason for r in fabricated)
    assert all("2031" not in s.body for s in report.sections)

    audit = store.list_claims(state["run_id"])
    assert {r["iteration"] for r in audit} == {1, 2}
    assert all(not r["included_in_report"] for r in audit if "2031" in r["claim_text"])


class _AlwaysUnsupported(HeuristicMeshSimulator):
    def _verifier_entailment(self, p):
        return {"judgements": [
            {"claim_id": c["claim_id"], "verdict": "SUPPORTED", "confidence": 0.99, "evidence_source_id": c["cited_source_ids"][0],
             "evidence_quote": "This sentence was invented by the verifier and appears in no source."}
            for c in p.get("claims", [])
        ]}


def test_hallucinated_evidence_is_never_trusted_and_loop_is_bounded(make_orchestrator):
    orchestrator, _store, _live = make_orchestrator(client=_AlwaysUnsupported(), max_iterations=2)
    state = orchestrator.run("How does the Reserve Bank of India use the repo rate to control inflation?")
    assert len(state["iteration_metrics"]) == 2, "loop must stop at max_iterations"
    assert all(m.supported == 0 for m in state["iteration_metrics"])
    report = state["final_report"]
    assert report.metrics.claims_in_report == 0
    assert report.metrics.coverage_score == 0.0
    assert all("insufficient" in s.body.lower() for s in report.sections)
    assert all("hallucinated" in r.reason for r in report.removed_claims if r.verdict is Verdict.UNSUPPORTED)
