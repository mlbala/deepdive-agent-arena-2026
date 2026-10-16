"""LangGraph orchestrator tests: topology, persistence, audit trail, uploads, failures,
follow-up grounding, and runtime-computed (non-hardcoded) outputs."""

from __future__ import annotations

import pytest

from src.agents.orchestrator import NODES, ResearchOrchestrator
from src.core.exceptions import MeshNotConfiguredError
from src.core.mesh_gateway import MeshGateway
from src.schemas.domain import AuditEventType, RetrievalTier, extract_citations
from src.storage.vector_store import HybridVectorStore
from src.tools.document_ingestor import ingest_bytes
from src.tools.web_search import WebRetriever

SOLAR_Q = "How much solar PV capacity was added globally in 2023 and what drove the growth?"
RBI_Q = "How does the Reserve Bank of India use the repo rate to control inflation?"


def test_graph_topology_has_cyclic_verification_edge(make_orchestrator):
    orchestrator, _, _ = make_orchestrator()
    graph = orchestrator.graph.get_graph()
    assert set(NODES) <= set(graph.nodes)
    edges = {(e.source, e.target, e.conditional) for e in graph.edges}
    assert ("__start__", "planner", False) in edges
    assert ("verifier", "researcher", True) in edges, "loop-back edge"
    assert ("verifier", "synthesizer", True) in edges
    assert ("synthesizer", "__end__", False) in edges
    assert "verifier -.-> researcher" in orchestrator.mermaid()


def test_full_run_persists_a_consistent_audit_record(make_orchestrator):
    orchestrator, store, _ = make_orchestrator()
    state = orchestrator.run(SOLAR_Q)
    run_id = state["run_id"]
    report = state["final_report"]

    run = store.get_run(run_id)
    assert run["status"] == "completed" and run["report"]["run_id"] == run_id

    # Every citation in the report resolves to an indexed source.
    known = {s.source_id for s in report.sources}
    for section in report.sections:
        assert set(extract_citations(section.body)) <= known
    assert set(extract_citations(report.executive_summary)) <= known

    # Evidence spans are exact substrings of their parent documents.
    sources = {s["source_id"]: s for s in store.list_sources(run_id, include_content=True)}
    assert set(sources) == set(state["sources"])
    for chunk in store.list_evidence(run_id):
        assert sources[chunk.source_id]["content"][chunk.char_start : chunk.char_end] == chunk.text

    # Each verdict in claims_audit has an exact quote span when verified.
    for row in store.list_claims(run_id):
        if row["quote_verified"]:
            src = sources[row["evidence_source_id"]]["content"]
            assert src[row["evidence_char_start"] : row["evidence_char_end"]] == row["evidence_quote"]

    # One immutable trace per Mesh call, for every LLM-backed agent.
    summary = store.trace_summary(run_id)
    assert {"planner", "drafter", "verifier", "synthesizer"} <= set(summary["per_agent"])
    assert summary["total_tokens"] == report.metrics.total_tokens > 0
    assert summary["llm_calls"] == report.metrics.llm_calls


def test_audit_trail_is_ordered_live_and_persisted(make_orchestrator):
    orchestrator, store, live = make_orchestrator()
    state = orchestrator.run(RBI_Q)
    persisted = store.list_audit_events(state["run_id"])
    assert [e.event_id for e in live] == [e["event_id"] for e in persisted]
    assert live[0].event_type is AuditEventType.RUN_STARTED
    assert live[-1].event_type is AuditEventType.RUN_COMPLETED
    started = [e.node for e in live if e.event_type is AuditEventType.NODE_STARTED]
    assert started[:4] == ["planner", "researcher", "drafter", "verifier"] and started[-1] == "synthesizer"
    assert any(e.event_type is AuditEventType.ROUTE_DECISION for e in live)
    assert {e.event_type for e in live} >= {
        AuditEventType.PLAN_CREATED, AuditEventType.SEARCH_EXECUTED, AuditEventType.SOURCE_INDEXED,
        AuditEventType.DRAFT_CREATED, AuditEventType.CLAIMS_EXTRACTED, AuditEventType.CLAIM_VERIFIED,
        AuditEventType.METRICS_COMPUTED, AuditEventType.REPORT_COMPILED,
    }


def test_stream_yields_nodes_in_graph_order(make_orchestrator):
    orchestrator, _, _ = make_orchestrator()
    nodes = [node for node, _update in orchestrator.stream(RBI_Q)]
    assert nodes[:4] == ["planner", "researcher", "drafter", "verifier"]
    assert nodes[-2:] == ["synthesizer", "__end__"]


def test_uploaded_documents_join_the_evidence_pool(make_orchestrator):
    orchestrator, store, _ = make_orchestrator()
    doc = ingest_bytes(
        "board-memo.md",
        b"# Board memo\n\nOur internal analysis estimates that the repo rate transmission to our loan book "
        b"takes about two quarters for external-benchmark linked loans and repo rate changes.",
    )
    store.save_document(
        doc.doc_id, filename=doc.filename, content_type=doc.content_type, content=doc.content,
        page_count=doc.page_count, content_hash=doc.content_hash,
    )
    state = orchestrator.run(RBI_Q, document_ids=[doc.doc_id])
    uploads = [s for s in state["sources"].values() if s.retrieval_tier is RetrievalTier.UPLOAD]
    assert len(uploads) == 1
    assert uploads[0].authority.rule == "uploaded_document"
    assert uploads[0].url == f"upload://{doc.doc_id}/board-memo.md"


def test_unconfigured_mesh_fails_cleanly_and_is_audited(make_settings, audit_store):
    settings = make_settings()
    gateway = MeshGateway(settings, audit_store=audit_store)  # no key, no client
    orchestrator = ResearchOrchestrator(
        settings, gateway, WebRetriever(settings), HybridVectorStore(settings, gateway), store=audit_store
    )
    with pytest.raises(MeshNotConfiguredError):
        orchestrator.run(SOLAR_Q, run_id="run_nokey")
    assert audit_store.get_run("run_nokey")["status"] == "failed"
    events = audit_store.list_audit_events("run_nokey")
    assert events[-1]["event_type"] == AuditEventType.RUN_FAILED.value


def test_followup_is_grounded_and_survives_restart(make_orchestrator, make_settings, audit_store):
    orchestrator, store, _ = make_orchestrator()
    run_id = orchestrator.run(RBI_Q)["run_id"]

    answer = orchestrator.followup(run_id, "What is the RBI inflation target for CPI inflation?")
    assert answer.answerable and answer.cited_source_ids
    assert set(answer.cited_source_ids) <= {s["source_id"] for s in store.list_sources(run_id)}

    off_topic = orchestrator.followup(run_id, "Which football club won the 1998 Champions League final?")
    assert not off_topic.answerable and off_topic.cited_source_ids == []

    # A fresh process: new vector store, same SQLite -> evidence rehydrates from claims/evidence tables.
    from eval.simulated_llm import HeuristicMeshSimulator

    settings = make_settings()
    gateway = MeshGateway(settings, client=HeuristicMeshSimulator(), audit_store=audit_store)
    fresh = ResearchOrchestrator(
        settings, gateway, WebRetriever(settings), HybridVectorStore(settings, gateway), store=audit_store
    )
    again = fresh.followup(run_id, "What is the RBI inflation target for CPI inflation?")
    assert again.answerable and again.cited_source_ids
    assert len(store.list_followups(run_id)) == 3


def test_outputs_are_computed_from_the_query_not_hardcoded(make_orchestrator):
    orchestrator, _, _ = make_orchestrator()
    solar = orchestrator.run(SOLAR_Q)
    rbi = orchestrator.run(RBI_Q)
    assert [s.question for s in solar["sub_questions"]] != [s.question for s in rbi["sub_questions"]]
    solar_domains = {s.domain for s in solar["sources"].values()}
    rbi_domains = {s.domain for s in rbi["sources"].values()}
    assert "iea.org" in solar_domains and "rbi.org.in" in rbi_domains
    assert solar_domains != rbi_domains
    assert solar["run_id"] != rbi["run_id"]
