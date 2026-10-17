"""Mesh API gateway tests: config contract, structured output, self-healing,
backoff, capability fallback, token/latency tracing, and Mesh-only compliance."""

from __future__ import annotations

import json
import re
from pathlib import Path

import openai
import pytest
from pydantic import BaseModel, Field, ValidationError
from sqlalchemy import select

from src.core.config import MESH_DEFAULT_BASE_URL, Settings
from src.core.exceptions import (
    ImmutableRecordError,
    MeshGatewayError,
    MeshNotConfiguredError,
    MeshStructuredOutputError,
    MeshTransientError,
)
from src.core.mesh_gateway import (
    MeshGateway,
    backoff_delay,
    coerce_payload,
    extract_json_payload,
)
from src.schemas.domain import ClaimExtractionOutput
from src.storage.database import AgentTrace
from tests.conftest import api_error, connection_error

REPO_ROOT = Path(__file__).resolve().parents[1]


class Answer(BaseModel):
    answer: str
    confidence: float = Field(ge=0, le=1)


# ---------------------------------------------------------------------------
# Configuration contract
# ---------------------------------------------------------------------------
def test_mesh_base_url_defaults_to_mesh_gateway(make_settings):
    settings = make_settings()
    assert settings.mesh_base_url == MESH_DEFAULT_BASE_URL == "https://api.meshapi.ai/v1"
    assert settings.mesh_configured is False


def test_mesh_settings_load_from_environment(monkeypatch):
    monkeypatch.setenv("MESH_API_KEY", "rsk_test_key")
    monkeypatch.setenv("MESH_BASE_URL", "https://mesh.example.test/v1/")
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.mesh_configured
    assert settings.mesh_api_key is not None
    assert settings.mesh_api_key.get_secret_value() == "rsk_test_key"
    assert settings.mesh_base_url == "https://mesh.example.test/v1"


def test_secrets_never_leak_into_public_summary(make_settings):
    settings = make_settings(mesh_api_key="rsk_super_secret", tavily_api_key="tvly-secret")
    summary = repr(settings.public_summary()) + repr(settings)
    assert "rsk_super_secret" not in summary
    assert "tvly-secret" not in summary
    assert settings.public_summary()["tavily_configured"] is True


def test_blank_keys_are_treated_as_missing(make_settings):
    settings = make_settings(mesh_api_key="   ", tavily_api_key="")
    assert settings.mesh_configured is False
    assert settings.tavily_configured is False


def test_per_agent_model_override(make_settings):
    settings = make_settings(verifier_model="openai/gpt-4o", planner_model="")
    assert settings.model_for("verifier") == "openai/gpt-4o"
    assert settings.model_for("planner") == settings.mesh_default_model


def test_chunk_overlap_must_be_smaller_than_chunk(make_settings):
    with pytest.raises(ValidationError):
        make_settings(chunk_size=300, chunk_overlap=300)


def test_real_client_is_pointed_at_mesh(make_settings):
    gateway = MeshGateway(make_settings(mesh_api_key="rsk_test"))
    client = gateway.client
    assert isinstance(client, openai.OpenAI)
    assert str(client.base_url).rstrip("/") == "https://api.meshapi.ai/v1"
    assert client.api_key == "rsk_test"
    assert client.max_retries == 0  # backoff is owned by the gateway


def test_unconfigured_gateway_fails_fast(make_settings):
    gateway = MeshGateway(make_settings())
    assert gateway.configured is False
    with pytest.raises(MeshNotConfiguredError):
        gateway.complete_structured("planner", "sys", "user", Answer)


# ---------------------------------------------------------------------------
# Structured output + self-healing
# ---------------------------------------------------------------------------
def test_complete_structured_validates_and_injects_schema(make_gateway):
    gateway, client = make_gateway([json.dumps({"answer": "42", "confidence": 0.9})])
    result = gateway.complete_structured("planner", "You plan.", "Q?", Answer, run_id="run_a")
    assert result == Answer(answer="42", confidence=0.9)
    call = client.calls[0]
    assert call["model"] == "openai/gpt-5.1"
    assert call["response_format"] == {"type": "json_object"}
    assert "OUTPUT CONTRACT" in call["messages"][0]["content"]
    assert '"confidence"' in call["messages"][0]["content"]


def test_model_override_wins(make_gateway):
    gateway, client = make_gateway(
        [json.dumps({"answer": "x", "confidence": 0.1})], allowed_models="anthropic/claude-sonnet-4.5"
    )
    gateway.complete_structured("verifier", "s", "u", Answer, model_override="anthropic/claude-sonnet-4.5")
    assert client.calls[0]["model"] == "anthropic/claude-sonnet-4.5"


def test_self_healing_repairs_invalid_json_then_invalid_schema(make_gateway, audit_store):
    gateway, client = make_gateway(
        [
            "Sure! Here is the answer: {answer: 42",               # not JSON
            json.dumps({"answer": "42", "confidence": 7}),          # schema violation
            "```json\n" + json.dumps({"answer": "42", "confidence": 0.7}) + "\n```",
        ]
    )
    result = gateway.complete_structured("drafter", "s", "u", Answer, run_id="run_heal")
    assert result.confidence == 0.7
    assert len(client.calls) == 3
    repair_msg = client.calls[2]["messages"][-1]["content"]
    assert "confidence" in repair_msg and "rejected" in repair_msg
    traces = audit_store.list_traces("run_heal")
    assert len(traces) == 1
    assert traces[0]["status"] == "repaired"
    assert traces[0]["repair_attempts"] == 2


def test_self_healing_gives_up_with_typed_error(make_gateway, audit_store):
    gateway, _ = make_gateway(["nope", "still nope", "never json"], mesh_json_repair_attempts=2)
    with pytest.raises(MeshStructuredOutputError) as exc_info:
        gateway.complete_structured("verifier", "s", "u", Answer, run_id="run_bad")
    assert exc_info.value.details["agent"] == "verifier"
    assert audit_store.list_traces("run_bad")[0]["status"] == "invalid_output"


def test_bare_array_is_coerced_into_single_list_field(make_gateway):
    payload = [{"text": "Water boils at 100 C at sea level", "cited_source_ids": ["src-1"]}]
    gateway, _ = make_gateway([json.dumps(payload)])
    out = gateway.complete_structured("verifier", "s", "u", ClaimExtractionOutput)
    assert out.claims[0].cited_source_ids == ["SRC-001"]


def test_extract_json_payload_variants():
    assert extract_json_payload('{"a": 1}') == {"a": 1}
    assert extract_json_payload('Result:\n```json\n{"a": [1, 2]}\n```\nThanks') == {"a": [1, 2]}
    assert extract_json_payload('preamble {"a": {"b": 2}} trailing') == {"a": {"b": 2}}
    with pytest.raises(ValueError):
        extract_json_payload("no json here")
    assert coerce_payload([1], Answer) == [1]  # no single list field -> untouched


# ---------------------------------------------------------------------------
# Resilience: backoff + capability fallback
# ---------------------------------------------------------------------------
def test_exponential_backoff_on_transient_errors(make_gateway, audit_store):
    sleeps: list[float] = []
    gateway, client = make_gateway(
        [
            api_error(openai.RateLimitError, 429, "slow down"),
            connection_error(),
            api_error(openai.InternalServerError, 503, "upstream"),
            json.dumps({"answer": "ok", "confidence": 1}),
        ],
        sleeps=sleeps,
        mesh_backoff_base_seconds=1.0,
        mesh_backoff_max_seconds=30.0,
    )
    result = gateway.complete_structured("planner", "s", "u", Answer, run_id="run_retry")
    assert result.answer == "ok"
    assert len(client.calls) == 4
    assert len(sleeps) == 3
    assert sleeps == sorted(sleeps), "backoff delays must be non-decreasing"
    assert 0.5 <= sleeps[0] <= 1.0 and 2.0 <= sleeps[2] <= 4.0
    assert audit_store.list_traces("run_retry")[0]["attempts"] == 4


def test_transient_errors_exhaust_into_typed_error(make_gateway):
    gateway, client = make_gateway(
        [api_error(openai.RateLimitError, 429)] * 3, mesh_max_retries=2
    )
    with pytest.raises(MeshTransientError):
        gateway.complete_structured("planner", "s", "u", Answer)
    assert len(client.calls) == 3


def test_authentication_errors_are_not_retried(make_gateway):
    gateway, client = make_gateway([api_error(openai.AuthenticationError, 401, "bad key")])
    with pytest.raises(MeshGatewayError, match="authentication"):
        gateway.complete_structured("planner", "s", "u", Answer)
    assert len(client.calls) == 1


def test_json_mode_fallback_when_model_rejects_response_format(make_gateway):
    gateway, client = make_gateway(
        [
            api_error(openai.BadRequestError, 400, "response_format json_object not supported"),
            json.dumps({"answer": "a", "confidence": 0.5}),
            json.dumps({"answer": "b", "confidence": 0.5}),
        ]
    )
    gateway.complete_structured("planner", "s", "u", Answer)
    gateway.complete_structured("planner", "s", "u", Answer)
    assert "response_format" in client.calls[0]
    assert "response_format" not in client.calls[1]
    assert "response_format" not in client.calls[2]


def test_temperature_fallback_for_reasoning_models(make_gateway):
    gateway, client = make_gateway(
        [
            api_error(openai.BadRequestError, 400, "Unsupported value: 'temperature'"),
            json.dumps({"answer": "a", "confidence": 0.5}),
        ],
        mesh_json_mode=False,
        allowed_models="openai/o4-mini",
    )
    gateway.complete_structured("planner", "s", "u", Answer, model_override="openai/o4-mini")
    assert "temperature" in client.calls[0] and "temperature" not in client.calls[1]


def test_backoff_delay_respects_cap_and_retry_after():
    assert backoff_delay(10, base=1.0, cap=5.0) <= 5.0
    assert backoff_delay(1, base=1.0, cap=30.0, retry_after=7.0) == 7.0


# ---------------------------------------------------------------------------
# Token / latency tracing + immutability
# ---------------------------------------------------------------------------
def test_trace_rows_capture_tokens_latency_and_are_immutable(make_gateway, audit_store):
    gateway, _ = make_gateway(
        [json.dumps({"answer": "a", "confidence": 0.5}), json.dumps({"answer": "b", "confidence": 0.5})]
    )
    gateway.complete_structured("planner", "s", "u" * 400, Answer, run_id="run_t")
    gateway.complete_structured("verifier", "s", "u", Answer, run_id="run_t")

    traces = audit_store.list_traces("run_t")
    assert [t["agent_name"] for t in traces] == ["planner", "verifier"]
    for t in traces:
        assert t["prompt_tokens"] > 0 and t["completion_tokens"] > 0
        assert t["total_tokens"] == t["prompt_tokens"] + t["completion_tokens"]
        assert t["latency_ms"] >= 0
        assert t["model"] == "openai/gpt-5.1"

    usage = gateway.usage_for("run_t")
    assert usage["llm_calls"] == 2
    assert usage["total_tokens"] == sum(t["total_tokens"] for t in traces)

    summary = audit_store.trace_summary("run_t")
    assert summary["llm_calls"] == 2 and set(summary["per_agent"]) == {"planner", "verifier"}

    with pytest.raises(ImmutableRecordError), audit_store.session() as s:
        row = s.scalars(select(AgentTrace)).first()
        row.total_tokens = 0


def test_missing_usage_is_estimated_and_flagged(make_settings, audit_store):
    from tests.conftest import FakeMeshClient

    client = FakeMeshClient([json.dumps({"answer": "a", "confidence": 0.5})], report_usage=False)
    gateway = MeshGateway(make_settings(), client=client, audit_store=audit_store)
    gateway.complete_structured("planner", "s", "u", Answer, run_id="run_est")
    trace = audit_store.list_traces("run_est")[0]
    assert trace["usage_estimated"] is True and trace["total_tokens"] > 0


def test_trace_sink_failures_never_break_calls(make_gateway):
    gateway, _ = make_gateway([json.dumps({"answer": "a", "confidence": 0.5})])
    seen: list[dict] = []
    gateway.add_trace_sink(lambda t: (_ for _ in ()).throw(RuntimeError("sink down")))
    gateway.add_trace_sink(seen.append)
    assert gateway.complete_structured("planner", "s", "u", Answer).answer == "a"
    assert seen and seen[0]["agent_name"] == "planner"


def test_embeddings_route_through_mesh_and_are_traced(make_gateway, audit_store):
    gateway, client = make_gateway()
    vectors = gateway.embed(["solar capacity", "repo rate"], run_id="run_e")
    assert len(vectors) == 2 and len(vectors[0]) == 96
    assert client.embedding_calls[0]["model"] == "openai/text-embedding-3-small"
    assert audit_store.list_traces("run_e")[0]["operation"] == "embedding"


# ---------------------------------------------------------------------------
# Hackathon compliance: Mesh API is the only LLM egress
# ---------------------------------------------------------------------------
FORBIDDEN_ENDPOINTS = (
    "api.openai.com",
    "api.anthropic.com",
    "generativelanguage.googleapis.com",
    "aiplatform.googleapis.com",
)
FORBIDDEN_IMPORTS = re.compile(r"^\s*(import|from)\s+(anthropic|google\.generativeai|google\.genai)\b", re.M)


def _python_sources() -> list[Path]:
    return [p for d in ("src", "eval") for p in (REPO_ROOT / d).rglob("*.py")]


def test_no_direct_provider_endpoints_or_sdks():
    for path in _python_sources():
        text = path.read_text(encoding="utf-8")
        for endpoint in FORBIDDEN_ENDPOINTS:
            assert endpoint not in text, f"{path} references {endpoint}"
        assert not FORBIDDEN_IMPORTS.search(text), f"{path} imports a non-Mesh provider SDK"


def test_openai_client_is_only_constructed_in_mesh_gateway():
    offenders = [
        p.relative_to(REPO_ROOT).as_posix()
        for p in _python_sources()
        if re.search(r"\bOpenAI\(", p.read_text(encoding="utf-8"))
    ]
    assert offenders == ["src/core/mesh_gateway.py"]
