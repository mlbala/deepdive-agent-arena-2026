"""Plain AI vs DeepDive: one plain Mesh answer to the run's question, checked with DeepDive's own machinery
against the run's evidence pool, persisted on the hash chain and served by the API."""

from __future__ import annotations

from fastapi.testclient import TestClient

from eval.simulated_llm import HeuristicMeshSimulator
from src.agents.orchestrator import ResearchOrchestrator
from src.api.main import ServiceContainer, create_app
from src.core.mesh_gateway import MeshGateway
from src.schemas.domain import AuditEventType
from src.storage.vector_store import HybridVectorStore
from src.tools.web_search import WebRetriever

SOLAR_Q = "How much new solar PV capacity did the world add in 2023, and what drove that growth?"


def test_a_plain_answer_is_decomposed_and_checked_against_the_runs_evidence(make_orchestrator):
    simulator = HeuristicMeshSimulator()
    orchestrator, store, live = make_orchestrator(client=simulator)
    run_id = orchestrator.run(SOLAR_Q)["run_id"]
    comparison = orchestrator.compare_baseline(run_id)

    assert comparison.answer and comparison.query == SOLAR_Q
    assert simulator.calls_by_role["baseline.answer"] == 1 and simulator.calls_by_role["baseline.claims"] == 1
    by_status = {c.status for c in comparison.claims}
    assert "supported" in by_status, "what the sources state is credited"
    supported = [c for c in comparison.claims if c.status == "supported"]
    assert all(c.quote_verified and c.evidence_quote and c.evidence_source_id in c.matched_source_ids for c in supported)
    drifted = [c for c in comparison.claims if "666" in c.text]
    assert drifted and drifted[0].status in ("unsupported", "contradicted"), "a drifted figure is caught"
    forecast = [c for c in comparison.claims if "double again" in c.text]
    assert forecast and forecast[0].status in ("unsupported", "not_found"), "a forecast no source makes is not credited"
    assert comparison.checked == len(comparison.claims) and comparison.traceable_rate is not None
    assert comparison.traceable_rate < 1.0
    assert comparison.supported + comparison.contradicted + comparison.unsupported + comparison.not_found == comparison.checked

    assert comparison.with_figures == 1, "the drifted sentence states figures; the other two don't"
    side = comparison.deepdive
    assert 0 < side.claims_with_figures <= side.claims_published
    report = store.get_report(run_id)
    assert side.claims_published == report.metrics.claims_in_report > 0
    assert side.report_sentences_checked and side.report_sentences_supported == side.report_sentences_checked

    assert comparison.llm_calls >= 3 and comparison.total_tokens > 0
    assert "baseline" in {t["agent_name"] for t in store.list_traces(run_id)}, "billed to the baseline"
    assert [e.event_type for e in live].count(AuditEventType.BASELINE_COMPARED) == 1
    assert not any(e.event_type is AuditEventType.CLAIM_VERIFIED and str(e.payload.get("claim_id", "")).startswith("BSL-")
                   for e in live), "baseline verdicts stay out of the run's claim matrix"
    assert store.verify_audit_chain(run_id)["valid"]


def test_the_comparison_rehydrates_evidence_after_a_restart(make_orchestrator, make_settings, audit_store):
    orchestrator, _store, _live = make_orchestrator()
    run_id = orchestrator.run(SOLAR_Q)["run_id"]
    settings = make_settings()
    gateway = MeshGateway(settings, client=HeuristicMeshSimulator(), audit_store=audit_store)
    fresh = ResearchOrchestrator(settings, gateway, WebRetriever(settings), HybridVectorStore(settings, gateway),
                                 store=audit_store)
    comparison = fresh.compare_baseline(run_id)
    assert comparison.supported >= 1, "sources and spans reload from SQLite"


def test_compare_api_creates_serves_and_guards_the_comparison(make_settings):
    container = ServiceContainer.build(make_settings(), mesh_client=HeuristicMeshSimulator())
    with TestClient(create_app(services=container)) as client:
        run_id = client.post("/api/research", json={"query": SOLAR_Q}).json()["run_id"]
        assert client.get(f"/api/runs/{run_id}/compare").status_code in (404, 409)
        container.runs.wait(run_id, timeout=60)
        assert client.get(f"/api/runs/{run_id}/compare").status_code == 404, "nothing until one is requested"
        created = client.post(f"/api/runs/{run_id}/compare")
        assert created.status_code == 200, created.text
        body = created.json()
        assert body["claims"] and body["deepdive"]["claims_published"] > 0
        assert client.get(f"/api/runs/{run_id}/compare").json()["answer"] == body["answer"]
        import io

        from pypdf import PdfReader

        pdf = client.get(f"/api/runs/{run_id}/report.pdf").content
        text = "\n".join(page.extract_text() for page in PdfReader(io.BytesIO(pdf)).pages)
        assert "Plain AI answer vs DeepDive" in text and "CLAIM IN THE PLAIN ANSWER" in text
        assert client.post("/api/runs/run_missing00/compare").status_code == 404
