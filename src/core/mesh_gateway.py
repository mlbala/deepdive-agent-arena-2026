"""Centralized Mesh API gateway — the ONLY place DeepDive talks to an LLM.

Every chat completion and embedding in the codebase goes through
`MeshGateway`, which wraps the official `openai` SDK pointed at Mesh API:

    OpenAI(base_url=settings.mesh_base_url,   # MESH_BASE_URL, default https://api.meshapi.ai/v1
           api_key=settings.mesh_api_key)     # MESH_API_KEY

Responsibilities
----------------
* `complete_structured()` — JSON-contract completions validated against a
  Pydantic model, with automatic self-healing: if the output is not valid JSON
  or fails schema validation, the validation error is fed back to the model and
  the call is retried (up to `MESH_JSON_REPAIR_ATTEMPTS`).
* Exponential backoff with jitter on transient failures (429 / 5xx / timeouts /
  connection resets), honouring `Retry-After` headers.
* Graceful capability fallback: models that reject `response_format=json_object`
  or a non-default `temperature` are retried without them.
* Exact wall-clock `latency_ms` plus prompt/completion/total token accounting
  for every call, persisted as an immutable `agent_traces` row in SQLite and
  fanned out to live trace sinks (SSE dashboard).
"""

from __future__ import annotations

import json
import logging
import random
import re
import threading
import time
import uuid
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, TypeVar

import openai
from openai import OpenAI
from pydantic import BaseModel, ValidationError

from src.core.config import Settings, get_settings
from src.core.exceptions import (
    BudgetExceededError,
    MeshGatewayError,
    MeshNotConfiguredError,
    MeshStructuredOutputError,
    MeshTransientError,
    ModelNotAllowedError,
)

if TYPE_CHECKING:
    from src.storage.database import AuditStore

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)
R = TypeVar("R")
TraceSink = Callable[[dict[str, Any]], None]

_TRANSIENT_ERRORS: tuple[type[Exception], ...] = (
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.InternalServerError,
)
_PREVIEW_CHARS = 600
_EMBED_BATCH = 96
_EMBED_CONCURRENCY = 4


class JSONExtractionError(ValueError):
    """Model output did not contain a decodable JSON value."""


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested directly)
# ---------------------------------------------------------------------------
def extract_json_payload(text: str) -> Any:
    """Extract the first JSON value from a model response.

    Accepts raw JSON, JSON wrapped in ```json fences, or JSON preceded/followed
    by stray prose. Raises `JSONExtractionError` if nothing decodes.
    """
    if not text or not text.strip():
        raise JSONExtractionError("empty response")
    cleaned = text.strip()
    candidates: list[str] = []
    fence = re.search(r"```(?:json|JSON)?\s*(.*?)```", cleaned, re.DOTALL)
    if fence:
        candidates.append(fence.group(1).strip())
    candidates.append(cleaned)

    decoder = json.JSONDecoder()
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
        starts = [i for i, ch in enumerate(candidate) if ch in "{["][:25]
        for start in starts:
            try:
                value, _ = decoder.raw_decode(candidate[start:])
                return value
            except json.JSONDecodeError:
                continue
    raise JSONExtractionError("no valid JSON object found in model output")


def coerce_payload(payload: Any, response_model: type[BaseModel]) -> Any:
    """Wrap a bare JSON array when the contract has exactly one list field.

    Models frequently answer `[{...}, {...}]` when asked for `{"claims": [...]}`.
    """
    if isinstance(payload, list):
        list_fields = [
            name
            for name, field in response_model.model_fields.items()
            if getattr(field.annotation, "__origin__", None) is list
        ]
        if len(list_fields) == 1:
            return {list_fields[0]: payload}
    return payload


def schema_instructions(response_model: type[BaseModel]) -> str:
    schema = json.dumps(response_model.model_json_schema(), separators=(",", ":"))
    return (
        "\n\n## OUTPUT CONTRACT\n"
        "Respond with ONE JSON object and nothing else: no prose, no markdown fences, no comments. "
        "It MUST validate against this JSON Schema:\n"
        f"{schema}"
    )


def repair_prompt(error: Exception) -> str:
    detail = str(error)
    if isinstance(error, ValidationError):
        detail = "; ".join(
            f"{'.'.join(str(p) for p in e['loc']) or '<root>'}: {e['msg']}" for e in error.errors()[:12]
        )
    return (
        "Your previous response was rejected by the schema validator.\n"
        f"Errors: {detail[:1500]}\n"
        "Return ONLY the corrected JSON object that satisfies the OUTPUT CONTRACT. "
        "Do not apologise or add any text outside the JSON."
    )


def backoff_delay(attempt: int, base: float, cap: float, retry_after: float | None = None) -> float:
    """Exponential backoff with bounded jitter: base * 2^(attempt-1) * U(0.5, 1.0), capped."""
    exp = min(cap, base * (2 ** max(0, attempt - 1)))
    delay = exp * random.uniform(0.5, 1.0)
    if retry_after is not None:
        delay = max(delay, retry_after)
    return min(cap, delay)


def _retry_after_seconds(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    value = headers.get("retry-after") or headers.get("Retry-After")
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _message_content(response: Any) -> str:
    try:
        return response.choices[0].message.content or ""
    except (AttributeError, IndexError):
        return ""


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


@dataclass
class UsageTotals:
    llm_calls: int = 0
    embedding_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: int = 0
    errors: int = 0
    repaired_calls: int = 0

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass
class _CallUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated: bool = False

    def add(self, response: Any, prompt_text: str, completion_text: str, *, embedding: bool = False) -> None:
        usage = getattr(response, "usage", None)
        prompt = getattr(usage, "prompt_tokens", None) if usage is not None else None
        completion = getattr(usage, "completion_tokens", None) if usage is not None else None
        if embedding and prompt is not None:
            completion = 0  # embeddings have no completion tokens; reported usage is exact
        if prompt is None or completion is None:
            self.estimated = True
            prompt = _estimate_tokens(prompt_text)
            completion = _estimate_tokens(completion_text) if completion_text else 0
        total = getattr(usage, "total_tokens", None) if usage is not None else None
        self.prompt_tokens += int(prompt)
        self.completion_tokens += int(completion)
        self.total_tokens += int(total) if total is not None else int(prompt) + int(completion)


# ---------------------------------------------------------------------------
# Gateway
# ---------------------------------------------------------------------------
class MeshGateway:
    """Thread-safe Mesh API client shared by every agent."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: Any | None = None,
        audit_store: AuditStore | None = None,
        trace_sinks: Sequence[TraceSink] = (),
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.settings = settings or get_settings()
        self._client = client
        self._client_lock = threading.Lock()
        self._audit_store = audit_store
        self._trace_sinks: list[TraceSink] = list(trace_sinks)
        self._sleep = sleep
        self._json_mode_enabled = self.settings.mesh_json_mode
        self._temperature_unsupported: set[str] = set()
        self._usage: dict[str, UsageTotals] = {}
        self._usage_lock = threading.Lock()

    # -- client --------------------------------------------------------------
    @property
    def configured(self) -> bool:
        return self._client is not None or self.settings.mesh_configured

    @property
    def client(self) -> Any:
        if self._client is None:
            with self._client_lock:
                if self._client is None:
                    if self.settings.mesh_api_key is None:
                        raise MeshNotConfiguredError(
                            "MESH_API_KEY is not set. DeepDive routes every LLM call through Mesh API; "
                            "add MESH_API_KEY to your environment or .env file."
                        )
                    self._client = OpenAI(
                        base_url=self.settings.mesh_base_url,
                        api_key=self.settings.mesh_api_key.get_secret_value(),
                        timeout=self.settings.mesh_timeout_seconds,
                        max_retries=0,  # retries are owned by the gateway's backoff policy
                    )
        return self._client

    def add_trace_sink(self, sink: TraceSink) -> None:
        self._trace_sinks.append(sink)

    def usage_for(self, run_id: str) -> dict[str, int]:
        with self._usage_lock:
            return (self._usage.get(run_id) or UsageTotals()).as_dict()

    # -- governance ----------------------------------------------------------
    def budget_status(self, run_id: str) -> dict[str, Any]:
        usage = self.usage_for(run_id)
        caps = (self.settings.max_llm_calls_per_run, self.settings.max_tokens_per_run)
        fraction = max(usage["llm_calls"] / caps[0], usage["total_tokens"] / caps[1])
        return {
            "llm_calls": usage["llm_calls"],
            "max_llm_calls": caps[0],
            "tokens": usage["total_tokens"],
            "max_tokens": caps[1],
            "fraction_used": round(fraction, 4),
            "soft_limit_reached": fraction >= self.settings.budget_soft_fraction,
        }

    def _enforce_governance(self, model: str, run_id: str | None, *, operation: str) -> None:
        """Model allowlist + hard per-run budgets, checked before any request leaves the process."""
        if model not in self.settings.allowed_model_set:
            raise ModelNotAllowedError(
                f"Model '{model}' is not on the governance allowlist",
                details={"model": model, "allowed": sorted(self.settings.allowed_model_set)},
            )
        if not run_id:
            return
        usage = self.usage_for(run_id)
        if operation == "chat" and usage["llm_calls"] >= self.settings.max_llm_calls_per_run:
            raise BudgetExceededError(
                f"Run {run_id} reached its Mesh call budget ({self.settings.max_llm_calls_per_run})",
                details=usage,
            )
        if usage["total_tokens"] >= self.settings.max_tokens_per_run:
            raise BudgetExceededError(
                f"Run {run_id} reached its token budget ({self.settings.max_tokens_per_run:,})", details=usage
            )

    # -- structured chat -----------------------------------------------------
    def complete_structured(
        self,
        agent_name: str,
        system_prompt: str,
        user_prompt: str,
        response_model: type[T],
        model_override: str | None = None,
        run_id: str | None = None,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> T:
        """Run a Mesh chat completion and return a validated `response_model` instance."""
        model = model_override or self.settings.model_for(agent_name)
        self._enforce_governance(model, run_id, operation="chat")
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt + schema_instructions(response_model)},
            {"role": "user", "content": user_prompt},
        ]
        usage = _CallUsage()
        attempts = 0
        content = ""
        last_error: Exception | None = None
        started = time.perf_counter()

        for repair_round in range(self.settings.mesh_json_repair_attempts + 1):
            try:
                response, used_attempts = self._chat(model, messages, temperature, max_tokens)
            except MeshGatewayError as exc:
                self._emit_trace(
                    run_id=run_id, agent_name=agent_name, operation="chat", model=model,
                    status="error", attempts=attempts + 1, repair_attempts=repair_round,
                    started=started, usage=usage, error=str(exc),
                    request_preview=user_prompt, response_preview=content,
                )
                raise
            attempts += used_attempts
            content = _message_content(response)
            usage.add(response, "".join(m["content"] for m in messages), content)
            try:
                payload = coerce_payload(extract_json_payload(content), response_model)
                result = response_model.model_validate(payload)
            except (JSONExtractionError, ValidationError) as exc:
                last_error = exc
                logger.warning(
                    "Mesh %s output failed validation (round %d): %s", agent_name, repair_round, exc
                )
                messages = [
                    *messages,
                    {"role": "assistant", "content": content[:8000] or "<empty>"},
                    {"role": "user", "content": repair_prompt(exc)},
                ]
                continue

            self._emit_trace(
                run_id=run_id, agent_name=agent_name, operation="chat", model=model,
                status="success" if repair_round == 0 else "repaired", attempts=attempts,
                repair_attempts=repair_round, started=started, usage=usage, error=None,
                request_preview=user_prompt, response_preview=content,
            )
            return result

        self._emit_trace(
            run_id=run_id, agent_name=agent_name, operation="chat", model=model,
            status="invalid_output", attempts=attempts,
            repair_attempts=self.settings.mesh_json_repair_attempts, started=started,
            usage=usage, error=str(last_error), request_preview=user_prompt, response_preview=content,
        )
        raise MeshStructuredOutputError(
            f"{agent_name}: model output failed {response_model.__name__} validation after "
            f"{self.settings.mesh_json_repair_attempts} repair attempt(s)",
            details={"agent": agent_name, "model": model, "error": str(last_error)[:500]},
        )

    def _chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float | None,
        max_tokens: int | None,
    ) -> tuple[Any, int]:
        """One logical chat call with capability fallbacks + transient-error backoff."""
        total_attempts = 0
        for _ in range(3):  # at most: json-mode fallback, temperature fallback, real call
            kwargs: dict[str, Any] = {"model": model, "messages": messages}
            if model not in self._temperature_unsupported:
                kwargs["temperature"] = (
                    self.settings.mesh_temperature if temperature is None else temperature
                )
            if max_tokens:
                kwargs["max_tokens"] = max_tokens
            if self._json_mode_enabled:
                kwargs["response_format"] = {"type": "json_object"}
            try:
                response, attempts = self._call_with_backoff(
                    partial(self.client.chat.completions.create, **kwargs)
                )
                return response, total_attempts + attempts
            except openai.BadRequestError as exc:
                total_attempts += 1
                message = str(exc).lower()
                if "response_format" in kwargs and ("response_format" in message or "json" in message):
                    logger.info("Mesh model %s rejected response_format; using prompt-enforced JSON", model)
                    self._json_mode_enabled = False
                    continue
                if "temperature" in kwargs and "temperature" in message:
                    logger.info("Mesh model %s rejected temperature; using model default", model)
                    self._temperature_unsupported.add(model)
                    continue
                raise MeshGatewayError(
                    f"Mesh API rejected the request for model '{model}': {exc}",
                    details={"model": model, "status": 400},
                ) from exc
        raise MeshGatewayError(f"Mesh API request for model '{model}' could not be satisfied")

    def _call_with_backoff(self, fn: Callable[[], R]) -> tuple[R, int]:
        attempt = 0
        while True:
            attempt += 1
            try:
                return fn(), attempt
            except openai.BadRequestError:
                raise
            except (openai.AuthenticationError, openai.PermissionDeniedError) as exc:
                raise MeshGatewayError(
                    "Mesh API authentication failed - check MESH_API_KEY.", details={"status": 401}
                ) from exc
            except openai.NotFoundError as exc:
                raise MeshGatewayError(
                    f"Mesh API resource not found (unknown model id?): {exc}", details={"status": 404}
                ) from exc
            except _TRANSIENT_ERRORS as exc:
                self._backoff_or_raise(attempt, exc)
            except openai.APIStatusError as exc:
                if exc.status_code >= 500:
                    self._backoff_or_raise(attempt, exc)
                else:
                    raise MeshGatewayError(
                        f"Mesh API error {exc.status_code}: {exc}", details={"status": exc.status_code}
                    ) from exc

    def _backoff_or_raise(self, attempt: int, exc: Exception) -> None:
        if attempt > self.settings.mesh_max_retries:
            raise MeshTransientError(
                f"Mesh API transient failure persisted after {attempt} attempt(s): {exc}",
                details={"attempts": attempt},
            ) from exc
        delay = backoff_delay(
            attempt,
            self.settings.mesh_backoff_base_seconds,
            self.settings.mesh_backoff_max_seconds,
            _retry_after_seconds(exc),
        )
        logger.warning("Mesh transient error (%s); retry %d in %.2fs", type(exc).__name__, attempt, delay)
        self._sleep(delay)

    # -- embeddings ----------------------------------------------------------
    def embed(
        self,
        texts: Sequence[str],
        *,
        run_id: str | None = None,
        agent_name: str = "embedder",
        model_override: str | None = None,
    ) -> list[list[float]]:
        """Embed texts through Mesh API in batches; returns one vector per input."""
        model = model_override or self.settings.mesh_embedding_model
        self._enforce_governance(model, run_id, operation="embedding")
        batches = [
            [t if t.strip() else " " for t in texts[offset : offset + _EMBED_BATCH]]
            for offset in range(0, len(texts), _EMBED_BATCH)
        ]
        if len(batches) <= 1:
            results = [self._embed_batch(b, model=model, run_id=run_id, agent_name=agent_name) for b in batches]
        else:  # independent batches run concurrently; order is preserved by map()
            with ThreadPoolExecutor(max_workers=min(_EMBED_CONCURRENCY, len(batches))) as pool:
                results = list(pool.map(lambda b: self._embed_batch(b, model=model, run_id=run_id, agent_name=agent_name), batches))
        vectors: list[list[float]] = [v for batch_vectors in results for v in batch_vectors]
        return vectors

    def _embed_batch(self, batch: list[str], *, model: str, run_id: str | None, agent_name: str) -> list[list[float]]:
        started = time.perf_counter()
        usage = _CallUsage()
        try:
            response, attempts = self._call_with_backoff(
                partial(self.client.embeddings.create, model=model, input=batch)
            )
        except (MeshGatewayError, openai.BadRequestError) as exc:
            self._emit_trace(
                run_id=run_id, agent_name=agent_name, operation="embedding", model=model,
                status="error", attempts=1, repair_attempts=0, started=started, usage=usage,
                error=str(exc), request_preview=f"{len(batch)} texts", response_preview="",
            )
            if isinstance(exc, MeshGatewayError):
                raise
            raise MeshGatewayError(f"Mesh embeddings request rejected: {exc}") from exc
        data = sorted(response.data, key=lambda d: getattr(d, "index", 0))
        usage.add(response, "".join(batch), "", embedding=True)
        self._emit_trace(
            run_id=run_id, agent_name=agent_name, operation="embedding", model=model,
            status="success", attempts=attempts, repair_attempts=0, started=started,
            usage=usage, error=None, request_preview=f"{len(batch)} texts",
            response_preview=f"{len(data)} vectors x {len(data[0].embedding) if data else 0} dims",
        )
        return [list(d.embedding) for d in data]

    # -- tracing -------------------------------------------------------------
    def _emit_trace(
        self,
        *,
        run_id: str | None,
        agent_name: str,
        operation: str,
        model: str,
        status: str,
        attempts: int,
        repair_attempts: int,
        started: float,
        usage: _CallUsage,
        error: str | None,
        request_preview: str,
        response_preview: str,
    ) -> None:
        latency_ms = round((time.perf_counter() - started) * 1000)
        trace: dict[str, Any] = {
            "trace_id": uuid.uuid4().hex,
            "run_id": run_id,
            "agent_name": agent_name,
            "operation": operation,
            "model": model,
            "status": status,
            "attempts": attempts,
            "repair_attempts": repair_attempts,
            "latency_ms": latency_ms,
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
            "usage_estimated": usage.estimated,
            "error": error[:2000] if error else None,
            "request_preview": (request_preview or "")[:_PREVIEW_CHARS],
            "response_preview": (response_preview or "")[:_PREVIEW_CHARS],
        }
        if run_id:
            with self._usage_lock:
                totals = self._usage.setdefault(run_id, UsageTotals())
                if operation == "embedding":
                    totals.embedding_calls += 1
                else:
                    totals.llm_calls += 1
                totals.prompt_tokens += usage.prompt_tokens
                totals.completion_tokens += usage.completion_tokens
                totals.total_tokens += usage.total_tokens
                totals.latency_ms += latency_ms
                totals.errors += status in ("error", "invalid_output")
                totals.repaired_calls += status == "repaired"
        if self._audit_store is not None:
            try:
                self._audit_store.record_trace(trace)
            except Exception:  # tracing must never break the reasoning path
                logger.exception("Failed to persist agent trace")
        for sink in self._trace_sinks:
            try:
                sink(trace)
            except Exception:
                logger.exception("Trace sink failed")
