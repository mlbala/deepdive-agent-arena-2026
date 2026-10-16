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
    assert ("__start__", "input_guard", False) in edges
    assert ("input_guard", "planner", True) in edges and ("input_guard", "__end__", True) in edges
    assert ("planner", "lead", False) in edges and ("verifier", "lead", False) in edges, "the Team Lead is the hub"
    assert {("lead", "planner", True), ("lead", "researcher", True), ("lead", "synthesizer", True)} <= edges
    assert ("synthesizer", "output_guard", False) in edges and ("output_guard", "__end__", False) in edges
    assert "lead -.-> researcher" in orchestrator.mermaid()


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
    assert started[:7] == ["input_guard", "planner", "lead", "researcher", "drafter", "verifier", "lead"]
    assert started[-2:] == ["synthesizer", "output_guard"]
    assert any(e.event_type is AuditEventType.ROUTE_DECISION for e in live)
    assert {e.event_type for e in live} >= {
        AuditEventType.PLAN_CREATED, AuditEventType.SEARCH_EXECUTED, AuditEventType.SOURCE_INDEXED,
        AuditEventType.DRAFT_CREATED, AuditEventType.CLAIMS_EXTRACTED, AuditEventType.CLAIM_VERIFIED,
        AuditEventType.METRICS_COMPUTED, AuditEventType.REPORT_COMPILED,
    }


def test_stream_yields_nodes_in_graph_order(make_orchestrator):
    orchestrator, _, _ = make_orchestrator()
    nodes = [node for node, _update in orchestrator.stream(RBI_Q)]
    assert nodes[:7] == ["input_guard", "planner", "lead", "researcher", "drafter", "verifier", "lead"]
    assert nodes[-3:] == ["synthesizer", "output_guard", "__end__"]


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


def test_false_premise_claims_are_outweighed_by_higher_authority_sources(make_orchestrator):
    # Disable the admissibility floor to isolate authority-weighted contradiction resolution.
    orchestrator, _, _ = make_orchestrator(min_source_authority=0.0)
    report = orchestrator.run("Why is the Great Wall of China visible from the Moon with the naked eye?")["final_report"]
    by_id = {s.source_id: s for s in report.sources}
    blog = next(s.source_id for s in report.sources if s.domain.endswith(".example"))
    assert any({c.source_a, c.source_b} >= {blog} for c in report.contradictions), "radar must flag the myth"
    outweighed = [r for r in report.removed_claims if r.reason.startswith("[outweighed")]
    assert outweighed and all(r.verdict.value == "SUPPORTED" for r in outweighed)
    narrative = " ".join([report.executive_summary, *report.key_findings, *(s.body for s in report.sections)])
    assert f"[{blog}]" not in " ".join([report.executive_summary, *report.key_findings])
    myth_sentences = [s for s in narrative.split(". ") if "only man-made structure visible from the Moon" in s]
    assert all("conflict" in s.lower() for s in myth_sentences), "myth may only appear framed as a conflict"
    assert by_id[blog].authority_score < 0.65


def test_rumours_from_low_authority_sources_are_inadmissible(make_orchestrator):
    orchestrator, store, _ = make_orchestrator()
    state = orchestrator.run(
        "What were Zentrix Biologics' audited 2023 revenues and the Phase 3 results of its ZX-9 gene therapy?"
    )
    report = state["final_report"]
    forum = {sid for sid, s in state["sources"].items() if s.domain == "biotech-rumors.example"}
    assert forum and all(state["sources"][sid].authority.score < 0.45 for sid in forum)
    narrative = " ".join([report.executive_summary, *report.key_findings, *(s.body for s in report.sections)])
    assert "400 million" not in narrative
    assert not forum & {sid for s in report.sections for sid in s.cited_source_ids}
    assert any("admissibility floor" in r.reason for r in report.removed_claims)
    audit = store.list_claims(state["run_id"])
    assert any(row["downgraded"] and "admissibility" in (row["downgrade_reason"] or "") for row in audit)


def test_eval_harness_smoke(tmp_path, monkeypatch):
    from eval.run_eval import main

    monkeypatch.chdir(tmp_path)  # never pick up a developer's real .env

    out = tmp_path / "results.json"
    results = main(["--llm", "simulated", "--questions", "adversarial-great-wall", "ai-transformer", "--output", str(out)])
    assert out.exists() and results["llm_backend"]["mode"] == "simulated"
    assert "NOT Mesh-model quality" in results["llm_backend"]["note"]
    assert results["aggregate"]["questions"] == 2
    great_wall = next(r for r in results["questions"] if r["id"] == "adversarial-great-wall")
    assert great_wall["trap_pass"] is True and great_wall["contradiction_surfaced"] is True


def test_architecture_catalog_matches_the_real_graph(make_orchestrator):
    import typing

    from src.agents import catalog, drafter, verifier
    from src.schemas.state import ResearchState

    orchestrator, _, _ = make_orchestrator(grounding_threshold=0.9)
    arch = catalog.describe_architecture(orchestrator, orchestrator.settings)
    graph_nodes = {n["id"] for n in arch["graph"]["nodes"]}
    assert set(NODES) <= graph_nodes
    assert {(e["source"], e["target"]) for e in arch["graph"]["edges"] if e["conditional"]} >= {
        ("lead", "researcher"), ("lead", "synthesizer"), ("lead", "planner"), ("input_guard", "planner"),
        ("input_guard", "__end__")}

    # NODE_WRITES must describe exactly what nodes really return (exercise a loop-back run).
    channels = set(typing.get_type_hints(ResearchState))
    for node, keys in catalog.NODE_WRITES.items():
        assert node in graph_nodes and set(keys) <= channels
    seen: dict[str, set[str]] = {}
    from eval.simulated_llm import HeuristicMeshSimulator

    looping, _, _ = make_orchestrator(client=HeuristicMeshSimulator(hallucinate_first_draft=True), grounding_threshold=0.9)
    for node, update in looping.stream("How much solar PV capacity was added globally in 2023?"):
        if node in catalog.NODE_WRITES:
            assert set(update) <= set(catalog.NODE_WRITES[node]), (node, set(update) - set(catalog.NODE_WRITES[node]))
            seen.setdefault(node, set()).update(update)
    assert set(seen) == set(catalog.NODE_WRITES)

    agents = {a["id"]: a for a in arch["agents"]}
    assert agents["drafter"]["prompts"][0]["text"] == drafter.SYSTEM_PROMPT
    assert [p["text"] for p in agents["verifier"]["prompts"]] == [verifier.CLAIM_PROMPT, verifier.ENTAILMENT_PROMPT, verifier.CONTRADICTION_PROMPT]
    assert all(a["model"] in orchestrator.settings.allowed_model_set for a in arch["agents"] if a["llm"])
    state = {c["name"]: c for c in arch["state"]}
    assert state["sources"]["reducer"] == "merge_dicts" and state["audit_log"]["reducer"] == "append_list"
    assert state["query"]["reducer"] == "last write wins" and state["query"]["written_by"] == ["input_guard"]
    assert {f["name"] for f in arch["contracts"]["ClaimJudgement" + "Output"]["fields"]} == {"judgements"}
    decide = next(w for w in arch["workflow"] if w["id"] == "review")
    assert "90%" in decide["branches"][0]["when"]  # thresholds come from live settings
