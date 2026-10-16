"""API tests: health, UI, research lifecycle, SSE replay, PDF export, uploads, follow-ups."""

from __future__ import annotations

import io
import json

import pytest
from fastapi.testclient import TestClient
from pypdf import PdfReader

from eval.simulated_llm import HeuristicMeshSimulator
from src.api.main import ServiceContainer, create_app

RBI_Q = "How does the Reserve Bank of India use the repo rate to control inflation?"


@pytest.fixture
def container(make_settings):
    return ServiceContainer.build(make_settings(), mesh_client=HeuristicMeshSimulator())


@pytest.fixture
def client(container):
    with TestClient(create_app(services=container)) as test_client:
        yield test_client


def _run(client: TestClient, container: ServiceContainer, **body) -> str:
    response = client.post("/api/research", json={"query": RBI_Q, **body})
    assert response.status_code == 202, response.text
    run_id = response.json()["run_id"]
    container.runs.wait(run_id, timeout=60)
    return run_id


def _sse_events(raw: str) -> list[tuple[str, dict]]:
    events = []
    for block in raw.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines() if not line.startswith(":"))
        if "event" in lines:
            events.append((lines["event"], json.loads(lines["data"])))
    return events


def test_health_exposes_config_without_secrets(make_settings):
    settings = make_settings(mesh_api_key="rsk_secret_value", tavily_api_key="tvly-secret")
    container = ServiceContainer.build(settings, mesh_client=HeuristicMeshSimulator())
    with TestClient(create_app(services=container)) as c:
        body = c.get("/api/health").json()
    assert body["status"] == "ok" and body["mesh_configured"] is True
    assert body["config"]["mesh_base_url"] == "https://api.meshapi.ai/v1"
    assert "rsk_secret_value" not in json.dumps(body) and "tvly-secret" not in json.dumps(body)


def test_dashboard_renders(client):
    html = client.get("/").text
    for marker in ("LangGraph execution", "Claim Matrix", "Contradiction Radar", "Audit Trail", "EventSource"):
        assert marker in html


def test_graph_endpoint_returns_mermaid(client):
    assert "verifier -.-> researcher" in client.get("/api/graph").json()["mermaid"]


def test_research_request_validation(client):
    assert client.post("/api/research", json={"query": "short"}).status_code == 422
    assert client.post("/api/research", json={"query": RBI_Q, "max_iterations": 99}).status_code == 422
    missing = client.post("/api/research", json={"query": RBI_Q, "document_ids": ["DOC-nope"]})
    assert missing.status_code == 404


def test_research_rejected_without_mesh_key(make_settings):
    container = ServiceContainer.build(make_settings())
    with TestClient(create_app(services=container)) as c:
        response = c.post("/api/research", json={"query": RBI_Q})
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "mesh_not_configured"


def test_full_research_lifecycle(client, container):
    run_id = _run(client, container, max_iterations=2)

    run = client.get(f"/api/runs/{run_id}").json()
    assert run["status"] == "completed"
    report = run["report"]
    assert report["sections"] and report["metrics"]["llm_calls"] > 0

    claims = client.get(f"/api/runs/{run_id}/claims").json()
    assert claims and {c["verdict"] for c in claims} <= {"SUPPORTED", "CONTRADICTED", "UNSUPPORTED"}
    sources = client.get(f"/api/runs/{run_id}/sources").json()
    assert sources and "content" not in sources[0]
    detail = client.get(f"/api/runs/{run_id}/sources/{sources[0]['source_id']}").json()
    assert detail["content"] and detail["evidence_spans"]
    traces = client.get(f"/api/runs/{run_id}/traces").json()
    assert traces["summary"]["llm_calls"] == len([t for t in traces["traces"] if t["operation"] == "chat"])
    audit = client.get(f"/api/runs/{run_id}/audit").json()
    assert audit[0]["event_type"] == "run_started" and audit[-1]["event_type"] == "run_completed"
    assert run_id in [r["id"] for r in client.get("/api/runs").json()]


def test_sse_stream_replays_audit_and_traces_then_done(client, container):
    run_id = _run(client, container)
    with client.stream("GET", f"/api/runs/{run_id}/stream") as response:
        assert response.headers["content-type"].startswith("text/event-stream")
        raw = "".join(response.iter_text())
    events = _sse_events(raw)
    kinds = [kind for kind, _ in events]
    assert kinds[-1] == "done" and events[-1][1]["status"] == "completed"
    assert "audit" in kinds and "trace" in kinds
    audit = [data for kind, data in events if kind == "audit"]
    assert audit[0]["event_type"] == "run_started"
    assert any(e["event_type"] == "route_decision" for e in audit)


def test_sse_replays_from_sqlite_for_runs_from_previous_process(client, container):
    run_id = _run(client, container)
    container.bus._events.pop(run_id)  # simulate a restart: the in-memory bus forgot the run
    with client.stream("GET", f"/api/runs/{run_id}/stream") as response:
        events = _sse_events("".join(response.iter_text()))
    assert events[-1] == ("done", {"status": "completed", "replayed": True})
    assert len([e for e in events if e[0] == "audit"]) == len(client.get(f"/api/runs/{run_id}/audit").json())


def test_pdf_export_contains_report_and_audit_appendix(client, container):
    run_id = _run(client, container)
    response = client.get(f"/api/runs/{run_id}/report.pdf")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert response.content.startswith(b"%PDF")
    text = "\n".join(page.extract_text() for page in PdfReader(io.BytesIO(response.content)).pages)
    for marker in ("Executive summary", "Contradiction radar", "Claim verification audit", "Mesh API telemetry"):
        assert marker in text


def test_pdf_and_followup_require_completed_run(client, container):
    container.store.create_run("run_pending01", RBI_Q, max_iterations=1)
    assert client.get("/api/runs/run_pending01/report.pdf").status_code == 409
    assert client.post("/api/runs/run_pending01/followup", json={"question": "What is it?"}).status_code == 409
    assert client.get("/api/runs/run_missing/report.pdf").status_code == 404


def test_followup_chat_is_grounded(client, container):
    run_id = _run(client, container)
    answer = client.post(
        f"/api/runs/{run_id}/followup", json={"question": "What is the RBI inflation target for CPI inflation?"}
    ).json()
    assert answer["answerable"] is True and answer["cited_source_ids"]
    history = client.get(f"/api/runs/{run_id}/followups").json()
    assert history[0]["question"].startswith("What is the RBI")


def test_document_upload_dedupe_and_use_in_research(client, container):
    payload = b"Treasury note: the bank's internal estimate is that repo rate changes reach new loan pricing within two months for external benchmark linked loans."
    first = client.post("/api/documents", files=[("files", ("note.txt", payload, "text/plain"))])
    assert first.status_code == 201
    doc = first.json()[0]
    assert doc["duplicate"] is False and doc["id"].startswith("DOC-")
    again = client.post("/api/documents", files=[("files", ("copy.txt", payload, "text/plain"))]).json()[0]
    assert again["duplicate"] is True and again["id"] == doc["id"]
    bad = client.post("/api/documents", files=[("files", ("x.exe", b"MZ", "application/octet-stream"))])
    assert bad.status_code == 415
    assert [d["id"] for d in client.get("/api/documents").json()] == [doc["id"]]

    run_id = _run(client, container, document_ids=[doc["id"]])
    sources = client.get(f"/api/runs/{run_id}/sources").json()
    assert any(s["retrieval_tier"] == "uploaded_document" and s["authority_rule"] == "uploaded_document" for s in sources)


def test_unknown_run_is_404(client):
    for path in ("", "/claims", "/sources", "/traces", "/audit", "/stream"):
        assert client.get(f"/api/runs/run_doesnotexist{path}").status_code == 404
