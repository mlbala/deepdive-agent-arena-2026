"""Shared pytest fixtures. The suite is hermetic: no network, no `.env` leakage."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import httpx
import openai
import pytest

from src.core.config import Settings
from src.core.mesh_gateway import MeshGateway
from src.storage.database import AuditStore

Responder = Callable[[dict[str, Any]], "str | Exception"]


@pytest.fixture(autouse=True)
def _isolate_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip real credentials from the environment so tests never hit live endpoints."""
    for var in ("MESH_API_KEY", "TAVILY_API_KEY", "MESH_BASE_URL", "DATABASE_URL"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def make_settings() -> Callable[..., Settings]:
    """Build a Settings object that ignores `.env` and applies explicit overrides."""

    def _factory(**overrides: Any) -> Settings:
        base: dict[str, Any] = {
            "environment": "test",
            "database_url": "sqlite:///:memory:",
            "search_mode": "offline",
            "vector_backend": "tfidf",
            "mesh_backoff_base_seconds": 0.01,
            "mesh_backoff_max_seconds": 0.05,
        }
        base.update(overrides)
        return Settings(_env_file=None, **base)  # type: ignore[call-arg]

    return _factory


# ---------------------------------------------------------------------------
# Fake Mesh API client (mirrors the openai.OpenAI surface the gateway uses)
# ---------------------------------------------------------------------------
_MESH_URL = "https://api.meshapi.ai/v1/chat/completions"


def api_error(kind: type[openai.APIStatusError], status: int, message: str = "error") -> Exception:
    request = httpx.Request("POST", _MESH_URL)
    response = httpx.Response(status, request=request, headers={})
    return kind(message, response=response, body=None)


def connection_error() -> Exception:
    return openai.APIConnectionError(request=httpx.Request("POST", _MESH_URL))


def hashed_embedding(text: str, dims: int = 96) -> list[float]:
    """Deterministic bag-of-words embedding so dense retrieval is testable offline."""
    vec = [0.0] * dims
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        h = int(hashlib.md5(token.encode()).hexdigest(), 16)
        vec[h % dims] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


class FakeMeshClient:
    """Scripted stand-in for `openai.OpenAI` pointed at Mesh.

    `script` items are consumed in order (str → completion content, Exception → raised);
    once exhausted, `responder(kwargs)` is consulted.
    """

    def __init__(
        self,
        script: list[str | Exception] | None = None,
        responder: Responder | None = None,
        *,
        report_usage: bool = True,
    ) -> None:
        self.script = list(script or [])
        self.responder = responder
        self.report_usage = report_usage
        self.calls: list[dict[str, Any]] = []
        self.embedding_calls: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._chat_create))
        self.embeddings = SimpleNamespace(create=self._embed_create)

    def _chat_create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.script:
            out: str | Exception = self.script.pop(0)
        elif self.responder is not None:
            out = self.responder(kwargs)
        else:
            raise AssertionError("FakeMeshClient script exhausted")
        if isinstance(out, Exception):
            raise out
        prompt_chars = sum(len(m["content"]) for m in kwargs["messages"])
        usage = (
            SimpleNamespace(
                prompt_tokens=prompt_chars // 4,
                completion_tokens=len(out) // 4,
                total_tokens=prompt_chars // 4 + len(out) // 4,
            )
            if self.report_usage
            else None
        )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=out))], usage=usage
        )

    def _embed_create(self, *, model: str, input: list[str]) -> Any:
        self.embedding_calls.append({"model": model, "input": input})
        data = [
            SimpleNamespace(index=i, embedding=hashed_embedding(text)) for i, text in enumerate(input)
        ]
        tokens = sum(len(t) for t in input) // 4
        return SimpleNamespace(data=data, usage=SimpleNamespace(prompt_tokens=tokens, total_tokens=tokens))


@pytest.fixture
def audit_store() -> AuditStore:
    return AuditStore("sqlite:///:memory:")


@pytest.fixture
def make_gateway(make_settings, audit_store):
    """Gateway wired to a FakeMeshClient, the in-memory audit store, and a no-op sleep."""

    def _factory(
        script: list[str | Exception] | None = None,
        responder: Responder | None = None,
        *,
        sleeps: list[float] | None = None,
        **settings_overrides: Any,
    ) -> tuple[MeshGateway, FakeMeshClient]:
        client = FakeMeshClient(script, responder)
        recorded = sleeps if sleeps is not None else []
        gateway = MeshGateway(
            make_settings(**settings_overrides),
            client=client,
            audit_store=audit_store,
            sleep=recorded.append,
        )
        return gateway, client

    return _factory


@pytest.fixture
def make_orchestrator(make_settings, audit_store):
    """Full orchestrator on the offline corpus + deterministic simulator + in-memory SQLite.

    Returns (orchestrator, store, live_events). `client` defaults to a
    HeuristicMeshSimulator; pass `client=None` to build an unconfigured gateway.
    """
    from eval.simulated_llm import HeuristicMeshSimulator
    from src.agents.orchestrator import ResearchOrchestrator
    from src.storage.vector_store import HybridVectorStore
    from src.tools.web_search import WebRetriever

    _default = object()

    def _factory(client: Any = _default, **settings_overrides: Any):
        settings = make_settings(**settings_overrides)
        if client is _default:
            client = HeuristicMeshSimulator()
        gateway = MeshGateway(settings, client=client, audit_store=audit_store, sleep=lambda _s: None)
        live: list[Any] = []
        orchestrator = ResearchOrchestrator(
            settings,
            gateway,
            WebRetriever(settings),
            HybridVectorStore(settings, gateway),
            store=audit_store,
            event_sink=live.append,
        )
        return orchestrator, audit_store, live

    return _factory
