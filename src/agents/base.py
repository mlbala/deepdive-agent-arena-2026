"""Shared agent plumbing: Mesh-backed LLM access, audit-event collection, prompt rendering."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, ClassVar, TypeVar

from pydantic import BaseModel

from src.core.config import Settings
from src.core.mesh_gateway import MeshGateway
from src.schemas.domain import AuditEvent, AuditEventType, EvidenceChunk, SourceDocument
from src.tools.source_scorer import tier_label

T = TypeVar("T", bound=BaseModel)
R = TypeVar("R")
EventSink = Callable[[AuditEvent], None]

INPUT_MARKER = "INPUT (JSON):"


class EventCollector:
    """Collects audit events for one node execution and streams them live."""

    def __init__(self, run_id: str, node: str, iteration: int = 0, sink: EventSink | None = None) -> None:
        self.run_id = run_id
        self.node = node
        self.iteration = iteration
        self.sink = sink
        self.events: list[AuditEvent] = []
        self._lock = threading.Lock()

    def emit(self, event_type: AuditEventType, message: str, **payload: Any) -> AuditEvent:
        event = AuditEvent(
            run_id=self.run_id,
            node=self.node,
            event_type=event_type,
            message=message,
            iteration=self.iteration,
            payload=payload,
        )
        with self._lock:
            self.events.append(event)
        if self.sink is not None:
            self.sink(event)
        return event


def render_input(task: str, payload: dict[str, Any]) -> str:
    """User prompt = natural-language task + a JSON block (unambiguous for any model)."""
    return f"{task.strip()}\n\n{INPUT_MARKER}\n{json.dumps(payload, ensure_ascii=False, indent=1, default=str)}"


def evidence_payload(chunks: list[EvidenceChunk], sources: dict[str, SourceDocument]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for chunk in chunks:
        src = sources.get(chunk.source_id)
        out.append(
            {
                "source_id": chunk.source_id,
                "title": src.title if src else chunk.title,
                "domain": src.domain if src else "",
                "authority": round(src.authority.score if src else chunk.authority_score, 2),
                "tier": tier_label(src.authority.tier) if src else "unknown",
                "published": src.published_date if src else None,
                "text": chunk.text,
            }
        )
    return out


class BaseAgent:
    """Every agent talks to LLMs exclusively through the shared MeshGateway."""

    name: ClassVar[str] = "agent"

    def __init__(self, gateway: MeshGateway, settings: Settings) -> None:
        self.gateway = gateway
        self.settings = settings

    def _llm(
        self,
        run_id: str,
        system_prompt: str,
        user_prompt: str,
        response_model: type[T],
        **kwargs: Any,
    ) -> T:
        return self.gateway.complete_structured(
            self.name, system_prompt, user_prompt, response_model, run_id=run_id, **kwargs
        )

    def _parallel(self, fn: Callable[[Any], R], items: list[Any]) -> list[R]:
        if len(items) <= 1:
            return [fn(item) for item in items]
        with ThreadPoolExecutor(max_workers=min(self.settings.agent_max_workers, len(items))) as pool:
            return list(pool.map(fn, items))
