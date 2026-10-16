"""REST + Server-Sent Events routes.

    POST /api/research                      start a run (202) -> run_id
    GET  /api/runs                          recent runs
    GET  /api/runs/{id}                     run status + final report
    GET  /api/runs/{id}/stream              SSE: live audit events + Mesh traces, then `done`
    GET  /api/runs/{id}/claims              claims_audit (every iteration)
    GET  /api/runs/{id}/sources[/{sid}]     scored sources (content on the detail route)
    GET  /api/runs/{id}/traces              immutable Mesh call traces + per-agent summary
    GET  /api/runs/{id}/audit               persisted audit trail
    GET  /api/runs/{id}/report.pdf          executive PDF with audit appendix
    POST /api/runs/{id}/followup            grounded follow-up Q&A
    GET  /api/runs/{id}/followups           follow-up history
    POST /api/documents                     upload PDF/TXT/MD (multipart)
    GET  /api/documents                     uploaded document library
    GET  /api/health, /api/graph            config/status, LangGraph Mermaid diagram
    GET  /api/governance/policy             active guardrail / governance policy (no secrets)
    GET  /api/runs/{id}/governance          per-run guardrail findings, budgets, models
    GET  /api/runs/{id}/audit/verify        recompute the tamper-evident audit hash chain
    POST /api/admin/retention/purge         delete runs older than N days (admin key)

Security: when API_KEYS is set every /api route except /api/health requires a key
(X-API-Key / Bearer; `api_key` query param for GET streams and downloads). Research,
follow-up and upload routes are rate-limited per client; uploads pass the content shield.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from src import __version__
from src.agents.catalog import describe_architecture
from src.api.security import (
    client_identity,
    content_security_policy,
    new_nonce,
    presented_api_key,
    verify_api_key,
)
from src.core.exceptions import (
    AuthenticationRequiredError,
    MeshNotConfiguredError,
    RunNotFoundError,
)
from src.guardrails.shields import ContentGuard
from src.schemas.domain import FollowUpAnswer, RunStatus
from src.tools.document_ingestor import content_hash, ingest_bytes
from src.tools.pdf_exporter import build_report_pdf, export_filename

if TYPE_CHECKING:
    from src.api.main import ServiceContainer



async def require_api_key(request: Request) -> None:
    """Router-wide auth: every /api route except /api/health needs a key when API_KEYS is set."""
    if not request.url.path.startswith("/api/") or request.url.path == "/api/health":
        return
    verify_api_key(presented_api_key(request), services(request).settings.api_key_set)


def rate_limited(bucket: str) -> Any:
    async def dependency(request: Request) -> None:
        svc = services(request)
        per_minute = {
            "research": svc.settings.rate_limit_research_per_minute,
            "followup": svc.settings.rate_limit_followup_per_minute,
            "upload": svc.settings.rate_limit_upload_per_minute,
        }[bucket]
        svc.rate_limiter.check(bucket, client_identity(request), per_minute)

    return Depends(dependency)


router = APIRouter(dependencies=[Depends(require_api_key)])
SSE_POLL_SECONDS = 0.25
SSE_HEARTBEAT_SECONDS = 15.0
MAX_UPLOAD_FILES = 10


class ResearchRequest(BaseModel):
    query: str = Field(min_length=8, max_length=1000)
    max_iterations: int | None = Field(default=None, ge=1, le=6)
    document_ids: list[str] = Field(default_factory=list, max_length=20)


class ResearchAccepted(BaseModel):
    run_id: str
    status: str
    stream_url: str
    run_url: str


class FollowUpRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    history: list[dict[str, str]] = Field(default_factory=list, max_length=20)


def services(request: Request) -> ServiceContainer:
    return request.app.state.services


def _run_or_404(svc: ServiceContainer, run_id: str) -> dict[str, Any]:
    try:
        return svc.store.get_run(run_id)
    except RunNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found") from exc


# ---------------------------------------------------------------------------
# UI + meta
# ---------------------------------------------------------------------------
@router.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index(request: Request) -> HTMLResponse:
    svc = services(request)
    nonce = new_nonce()
    response: HTMLResponse = request.app.state.templates.TemplateResponse(
        request,
        "index.html",
        {"config": svc.settings.public_summary(), "version": __version__, "csp_nonce": nonce},
    )
    response.headers["Content-Security-Policy"] = content_security_policy(nonce)
    response.headers["Cache-Control"] = "no-store"
    return response


@router.get("/api/health")
async def health(request: Request) -> dict[str, Any]:
    svc = services(request)
    return {
        "status": "ok",
        "version": __version__,
        "mesh_configured": svc.gateway.configured,
        "config": svc.settings.public_summary(),
        "search_providers": svc.retriever.providers(),
        "vector_store": svc.vector_store.status(),
        "auth_required": bool(svc.settings.api_key_set),
        "active_runs": svc.runs.active_runs(),
    }


@router.get("/api/architecture")
async def architecture(request: Request) -> dict[str, Any]:
    """Machine-readable agent workflow: graph, steps, agents + prompts, state channels, contracts, policy."""
    svc = services(request)
    cached = svc.extra.get("architecture")
    if cached is None:
        cached = describe_architecture(svc.orchestrator, svc.settings)
        svc.extra["architecture"] = cached
    return cached


@router.get("/api/architecture/stats")
async def architecture_stats(request: Request) -> dict[str, Any]:
    """Behaviour of every agent across all runs: latency, tokens, success, verdicts, guardrails."""
    return await asyncio.to_thread(services(request).store.fleet_stats)


@router.get("/api/governance/policy")
async def governance_policy(request: Request) -> dict[str, Any]:
    return services(request).settings.governance_policy()


@router.get("/api/graph")
async def graph(request: Request) -> dict[str, Any]:
    return {"mermaid": services(request).orchestrator.mermaid()}


# ---------------------------------------------------------------------------
# Research runs
# ---------------------------------------------------------------------------
@router.post("/api/research", status_code=202, response_model=ResearchAccepted, dependencies=[rate_limited("research")])
async def start_research(body: ResearchRequest, request: Request) -> ResearchAccepted:
    svc = services(request)
    if not svc.gateway.configured:
        raise MeshNotConfiguredError(
            "MESH_API_KEY is not configured. DeepDive routes every LLM call through Mesh API."
        )
    known = {d["id"] for d in svc.store.get_documents(body.document_ids)}
    missing = [d for d in body.document_ids if d not in known]
    if missing:
        raise HTTPException(status_code=404, detail=f"Unknown document id(s): {', '.join(missing)}")
    # The run row stores only the redacted question; the input_guard node screens (and may block) it.
    run_id = svc.runs.start(
        body.query.strip(),
        max_iterations=body.max_iterations or svc.settings.max_iterations,
        document_ids=body.document_ids,
    )
    return ResearchAccepted(
        run_id=run_id,
        status=RunStatus.PENDING.value,
        stream_url=f"/api/runs/{run_id}/stream",
        run_url=f"/api/runs/{run_id}",
    )


@router.get("/api/runs")
async def list_runs(request: Request, limit: int = 25) -> list[dict[str, Any]]:
    return services(request).store.list_runs(limit=max(1, min(limit, 100)))


@router.get("/api/runs/{run_id}")
async def get_run(run_id: str, request: Request) -> dict[str, Any]:
    return _run_or_404(services(request), run_id)


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


@router.get("/api/runs/{run_id}/stream")
async def stream_run(run_id: str, request: Request) -> StreamingResponse:
    svc = services(request)
    _run_or_404(svc, run_id)

    async def events() -> AsyncIterator[str]:
        if not svc.bus.known(run_id):
            # Run from a previous process: replay the persisted audit trail, then close.
            for ev in svc.store.list_audit_events(run_id):
                yield _sse("audit", {"kind": "audit", **ev})
            yield _sse("done", {"status": svc.store.get_run(run_id)["status"], "replayed": True})
            return
        cursor = 0
        last_beat = time.monotonic()
        while True:
            if await request.is_disconnected():
                return
            batch, done = svc.bus.snapshot(run_id, cursor)
            for item in batch:
                yield _sse(item["kind"], item)
            cursor += len(batch)
            if done and not batch:
                yield _sse("done", {"status": svc.store.get_run(run_id)["status"], "replayed": False})
                return
            if time.monotonic() - last_beat > SSE_HEARTBEAT_SECONDS:
                last_beat = time.monotonic()
                yield ": keep-alive\n\n"
            await asyncio.sleep(SSE_POLL_SECONDS)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


@router.get("/api/runs/{run_id}/claims")
async def run_claims(run_id: str, request: Request) -> list[dict[str, Any]]:
    svc = services(request)
    _run_or_404(svc, run_id)
    return svc.store.list_claims(run_id)


@router.get("/api/runs/{run_id}/sources")
async def run_sources(run_id: str, request: Request) -> list[dict[str, Any]]:
    svc = services(request)
    _run_or_404(svc, run_id)
    return svc.store.list_sources(run_id)


@router.get("/api/runs/{run_id}/sources/{source_id}")
async def run_source_detail(run_id: str, source_id: str, request: Request) -> dict[str, Any]:
    svc = services(request)
    _run_or_404(svc, run_id)
    for src in svc.store.list_sources(run_id, include_content=True):
        if src["source_id"] == source_id:
            src["evidence_spans"] = [
                {"chunk_id": c.chunk_id, "start": c.char_start, "end": c.char_end}
                for c in svc.store.list_evidence(run_id)
                if c.source_id == source_id
            ]
            src["verified_quotes"] = [
                {"claim_id": c["claim_id"], "start": c["evidence_char_start"], "end": c["evidence_char_end"],
                 "verdict": c["verdict"]}
                for c in svc.store.list_claims(run_id)
                if c["evidence_source_id"] == source_id and c["quote_verified"]
            ]
            return src
    raise HTTPException(status_code=404, detail=f"Source {source_id} not found in run {run_id}")


@router.get("/api/runs/{run_id}/traces")
async def run_traces(run_id: str, request: Request) -> dict[str, Any]:
    svc = services(request)
    _run_or_404(svc, run_id)
    return {"summary": svc.store.trace_summary(run_id), "traces": svc.store.list_traces(run_id)}


@router.get("/api/runs/{run_id}/audit")
async def run_audit(run_id: str, request: Request) -> list[dict[str, Any]]:
    svc = services(request)
    _run_or_404(svc, run_id)
    return svc.store.list_audit_events(run_id)


@router.get("/api/runs/{run_id}/timeline")
async def run_timeline(run_id: str, request: Request) -> dict[str, Any]:
    svc = services(request)
    _run_or_404(svc, run_id)
    return svc.store.run_timeline(run_id)


@router.get("/api/runs/{run_id}/audit/verify")
async def run_audit_verify(run_id: str, request: Request) -> dict[str, Any]:
    svc = services(request)
    _run_or_404(svc, run_id)
    return svc.store.verify_audit_chain(run_id)


@router.get("/api/runs/{run_id}/bundle")
async def run_bundle(run_id: str, request: Request) -> Response:
    """Self-contained audit bundle: run, report, claims, sources, audit trail + chain verification."""
    svc = services(request)
    run = _run_or_404(svc, run_id)
    bundle = {
        "bundle_version": "deepdive-audit-bundle/1",
        "generated_at": datetime.now(UTC).isoformat(),
        "deepdive_version": __version__,
        "run": run,
        "claims": svc.store.list_claims(run_id),
        "sources": svc.store.list_sources(run_id),
        "audit_events": svc.store.list_audit_events(run_id),
        "audit_chain": svc.store.verify_audit_chain(run_id),
        "mesh_usage": svc.store.trace_summary(run_id),
        "followups": svc.store.list_followups(run_id),
        "policy": svc.settings.governance_policy(),
    }
    body = json.dumps(bundle, indent=2, default=str).encode("utf-8")
    digest = hashlib.sha256(body).hexdigest()
    return Response(
        content=body,
        media_type="application/json",
        headers={
            "Content-Disposition": f'attachment; filename="deepdive-{run_id}-audit-bundle.json"',
            "X-Bundle-SHA256": digest,
        },
    )


@router.get("/api/runs/{run_id}/governance")
async def run_governance(run_id: str, request: Request) -> dict[str, Any]:
    svc = services(request)
    run = _run_or_404(svc, run_id)
    events = svc.store.list_audit_events(run_id)
    findings = [e["payload"] for e in events if e["event_type"] == "guardrail_finding"]
    blocked = next((e["message"] for e in events if e["event_type"] == "guardrail_blocked"), None)
    report = run.get("report") or {}
    return {
        "run_id": run_id,
        "status": run["status"],
        "blocked_reason": blocked,
        "summary": report.get("governance"),
        "findings": findings,
        "budget_events": [e for e in events if e["event_type"] == "budget_guard"],
        "audit_chain": svc.store.verify_audit_chain(run_id),
        "policy": svc.settings.governance_policy(),
    }


@router.post("/api/admin/retention/purge")
async def retention_purge(request: Request, days: int) -> dict[str, Any]:
    svc = services(request)
    admin = svc.settings.admin_api_key
    presented = request.headers.get("x-admin-key")
    if admin is None or presented is None or not _const_eq(presented, admin.get_secret_value()):
        raise AuthenticationRequiredError("Admin key required (X-Admin-Key header).")
    if days < 1:
        raise HTTPException(status_code=422, detail="days must be >= 1")
    return {"purged": svc.store.purge_runs_older_than(days), "older_than_days": days}


def _const_eq(a: str, b: str) -> bool:
    import hmac

    return hmac.compare_digest(a.encode(), b.encode())


@router.get("/api/runs/{run_id}/report.pdf")
async def run_report_pdf(run_id: str, request: Request) -> Response:
    svc = services(request)
    _run_or_404(svc, run_id)
    report = svc.store.get_report(run_id)
    if report is None:
        raise HTTPException(status_code=409, detail="Report not available until the run completes")
    pdf = await asyncio.to_thread(
        build_report_pdf,
        report,
        claims=svc.store.list_claims(run_id),
        trace_summary=svc.store.trace_summary(run_id),
    )
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{export_filename(report)}"'},
    )


@router.post("/api/runs/{run_id}/followup", response_model=FollowUpAnswer, dependencies=[rate_limited("followup")])
async def run_followup(run_id: str, body: FollowUpRequest, request: Request) -> FollowUpAnswer:
    svc = services(request)
    run = _run_or_404(svc, run_id)
    if run["status"] != RunStatus.COMPLETED.value:
        raise HTTPException(status_code=409, detail="Follow-up questions require a completed run")
    return await asyncio.to_thread(svc.orchestrator.followup, run_id, body.question, history=body.history)


@router.get("/api/runs/{run_id}/followups")
async def run_followups(run_id: str, request: Request) -> list[dict[str, Any]]:
    svc = services(request)
    _run_or_404(svc, run_id)
    return svc.store.list_followups(run_id)


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------
@router.post("/api/documents", status_code=201, dependencies=[rate_limited("upload")])
async def upload_documents(request: Request, files: list[UploadFile] = File(...)) -> list[dict[str, Any]]:
    svc = services(request)
    if len(files) > MAX_UPLOAD_FILES:
        raise HTTPException(status_code=413, detail=f"At most {MAX_UPLOAD_FILES} files per upload")
    shield = ContentGuard(svc.settings)
    out: list[dict[str, Any]] = []
    for upload in files:
        data = await upload.read()
        doc = await asyncio.to_thread(ingest_bytes, upload.filename or "upload.txt", data)
        # Content shield before storage: indirect injection neutralised, secrets (+ PII by policy) redacted.
        content, findings = shield.sanitize(
            doc.content, location=f"upload:{doc.filename}", redact_pii=svc.settings.redact_pii_in_documents
        )
        digest = content_hash(content)
        guardrails = [f.model_dump(mode="json") for f in findings]
        existing = svc.store.find_document_by_hash(digest)
        if existing is not None:
            out.append({**existing, "duplicate": True, "guardrails": guardrails})
            continue
        svc.store.save_document(
            doc.doc_id,
            filename=doc.filename,
            content_type=doc.content_type,
            content=content,
            page_count=doc.page_count,
            content_hash=digest,
            guardrails=guardrails,
        )
        out.append(
            {
                "id": doc.doc_id,
                "filename": doc.filename,
                "content_type": doc.content_type,
                "char_count": len(content),
                "page_count": doc.page_count,
                "duplicate": False,
                "guardrails": guardrails,
            }
        )
    return out


@router.get("/api/documents")
async def list_documents(request: Request) -> list[dict[str, Any]]:
    return services(request).store.list_documents()
