"""FastAPI application: composition root, CORS, lifespan, templates, error mapping.

Run locally:  uvicorn src.api.main:app --reload --port 8000
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from collections.abc import AsyncIterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates

from src import __version__
from src.agents.orchestrator import ResearchOrchestrator
from src.core.config import Settings, get_settings
from src.core.exceptions import (
    ConfigurationError,
    DeepDiveError,
    DocumentIngestionError,
    MeshGatewayError,
    MeshNotConfiguredError,
    RunNotFoundError,
    UnsupportedDocumentTypeError,
)
from src.core.mesh_gateway import MeshGateway
from src.schemas.domain import AuditEvent, new_run_id
from src.storage.database import AuditStore
from src.storage.vector_store import HybridVectorStore
from src.tools.web_search import WebRetriever

logger = logging.getLogger("deepdive")
TEMPLATES_DIR = Path(__file__).parent / "templates"
MAX_TRACKED_RUNS = 64


class RunEventBus:
    """In-process fan-out of live run events for SSE subscribers (replayable per run)."""

    def __init__(self) -> None:
        self._events: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
        self._done: dict[str, bool] = {}
        self._lock = threading.Lock()

    def open(self, run_id: str) -> None:
        with self._lock:
            self._events[run_id] = []
            self._done[run_id] = False
            while len(self._events) > MAX_TRACKED_RUNS:
                old, _ = self._events.popitem(last=False)
                self._done.pop(old, None)

    def _publish(self, run_id: str | None, item: dict[str, Any]) -> None:
        if not run_id:
            return
        with self._lock:
            if run_id in self._events:
                self._events[run_id].append(item)

    def publish_audit(self, event: AuditEvent) -> None:
        self._publish(event.run_id, {"kind": "audit", **event.model_dump(mode="json")})

    def publish_trace(self, trace: dict[str, Any]) -> None:
        keys = ("trace_id", "agent_name", "operation", "model", "status", "attempts", "repair_attempts",
                "latency_ms", "prompt_tokens", "completion_tokens", "total_tokens")
        self._publish(trace.get("run_id"), {"kind": "trace", **{k: trace.get(k) for k in keys}})

    def mark_done(self, run_id: str) -> None:
        with self._lock:
            if run_id in self._done:
                self._done[run_id] = True

    def known(self, run_id: str) -> bool:
        with self._lock:
            return run_id in self._events

    def snapshot(self, run_id: str, cursor: int) -> tuple[list[dict[str, Any]], bool]:
        with self._lock:
            return list(self._events.get(run_id, [])[cursor:]), self._done.get(run_id, True)


class RunManager:
    """Executes research runs on a background thread pool."""

    def __init__(self, orchestrator: ResearchOrchestrator, store: AuditStore, bus: RunEventBus, workers: int = 2) -> None:
        self.orchestrator = orchestrator
        self.store = store
        self.bus = bus
        self._executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="deepdive-run")
        self._futures: dict[str, Future[Any]] = {}

    def start(self, query: str, *, max_iterations: int, document_ids: list[str]) -> str:
        run_id = new_run_id()
        self.store.create_run(run_id, query, max_iterations=max_iterations, document_ids=document_ids)
        self.bus.open(run_id)
        self._futures[run_id] = self._executor.submit(self._run, run_id, query, max_iterations, document_ids)
        return run_id

    def _run(self, run_id: str, query: str, max_iterations: int, document_ids: list[str]) -> None:
        try:
            self.orchestrator.run(query, run_id=run_id, max_iterations=max_iterations, document_ids=document_ids)
        except Exception:
            logger.exception("Research run %s failed", run_id)  # status + RUN_FAILED already persisted
        finally:
            self.bus.mark_done(run_id)

    def wait(self, run_id: str, timeout: float | None = None) -> None:
        future = self._futures.get(run_id)
        if future is not None:
            future.result(timeout=timeout)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


@dataclass
class ServiceContainer:
    settings: Settings
    store: AuditStore
    gateway: MeshGateway
    retriever: WebRetriever
    vector_store: HybridVectorStore
    orchestrator: ResearchOrchestrator
    bus: RunEventBus
    runs: RunManager
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def build(cls, settings: Settings, *, mesh_client: Any | None = None) -> ServiceContainer:
        store = AuditStore(settings.database_url)
        bus = RunEventBus()
        gateway = MeshGateway(settings, client=mesh_client, audit_store=store, trace_sinks=[bus.publish_trace])
        vector_store = HybridVectorStore(settings, gateway)
        retriever = WebRetriever(settings)
        orchestrator = ResearchOrchestrator(
            settings, gateway, retriever, vector_store, store=store, event_sink=bus.publish_audit
        )
        runs = RunManager(orchestrator, store, bus)
        return cls(settings, store, gateway, retriever, vector_store, orchestrator, bus, runs)

    def shutdown(self) -> None:
        self.runs.shutdown()


_STATUS_FOR_ERROR: list[tuple[type[DeepDiveError], int]] = [
    (RunNotFoundError, 404),
    (UnsupportedDocumentTypeError, 415),
    (DocumentIngestionError, 422),
    (MeshNotConfiguredError, 503),
    (ConfigurationError, 503),
    (MeshGatewayError, 502),
]


def create_app(services: ServiceContainer | None = None, settings: Settings | None = None) -> FastAPI:
    """App factory. Tests inject a prebuilt `ServiceContainer`; production builds one at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        container = services or ServiceContainer.build(settings or get_settings())
        app.state.services = container
        logging.basicConfig(level=container.settings.log_level)
        if not container.gateway.configured:
            logger.warning("MESH_API_KEY is not set: research runs will be rejected until it is configured.")
        yield
        container.shutdown()

    app = FastAPI(
        title="DeepDive",
        description="Self-verifying multi-agent research & claim-entailment system (Agent Arena 2026, Track 01).",
        version=__version__,
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    app.state.templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

    @app.exception_handler(DeepDiveError)
    async def _deepdive_error(_request: Request, exc: DeepDiveError) -> JSONResponse:
        status = next((code for kind, code in _STATUS_FOR_ERROR if isinstance(exc, kind)), 500)
        return JSONResponse(status_code=status, content={"error": exc.to_dict()})

    from src.api.routes import router

    app.include_router(router)
    return app


app = create_app()
