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
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from src import __version__
from src.core.exceptions import MeshNotConfiguredError, RunNotFoundError
from src.schemas.domain import FollowUpAnswer, RunStatus
from src.tools.document_ingestor import ingest_bytes
from src.tools.pdf_exporter import build_report_pdf, export_filename

if TYPE_CHECKING:
    from src.api.main import ServiceContainer

router = APIRouter()
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
    return request.app.state.templates.TemplateResponse(
        request,
        "index.html",
        {"config": svc.settings.public_summary(), "version": __version__},
    )


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
    }


@router.get("/api/graph")
async def graph(request: Request) -> dict[str, Any]:
    return {"mermaid": services(request).orchestrator.mermaid()}


# ---------------------------------------------------------------------------
# Research runs
# ---------------------------------------------------------------------------
@router.post("/api/research", status_code=202, response_model=ResearchAccepted)
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


@router.post("/api/runs/{run_id}/followup", response_model=FollowUpAnswer)
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
@router.post("/api/documents", status_code=201)
async def upload_documents(request: Request, files: list[UploadFile] = File(...)) -> list[dict[str, Any]]:
    svc = services(request)
    if len(files) > MAX_UPLOAD_FILES:
        raise HTTPException(status_code=413, detail=f"At most {MAX_UPLOAD_FILES} files per upload")
    out: list[dict[str, Any]] = []
    for upload in files:
        data = await upload.read()
        doc = await asyncio.to_thread(ingest_bytes, upload.filename or "upload.txt", data)
        existing = svc.store.find_document_by_hash(doc.content_hash)
        if existing is not None:
            out.append({**existing, "duplicate": True})
            continue
        svc.store.save_document(
            doc.doc_id,
            filename=doc.filename,
            content_type=doc.content_type,
            content=doc.content,
            page_count=doc.page_count,
            content_hash=doc.content_hash,
        )
        out.append(
            {
                "id": doc.doc_id,
                "filename": doc.filename,
                "content_type": doc.content_type,
                "char_count": len(doc.content),
                "page_count": doc.page_count,
                "duplicate": False,
            }
        )
    return out


@router.get("/api/documents")
async def list_documents(request: Request) -> list[dict[str, Any]]:
    return services(request).store.list_documents()
