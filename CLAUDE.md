# CLAUDE.md — DeepDive engineering standards

DeepDive (Agent Arena 2026, Track 01) is a self-verifying multi-agent research system:
Planner ⇄ Master Approver (plan review) → Researcher → Drafter ⇄ Verifier (quick fixes) → Master Approver (round review:
another round | Synthesizer) → Synthesizer ⇄ Report Approver (sentence-level report check) → output shield, plus a
FollowUp agent and an on-demand plain-AI Baseline for comparison, wired as a LangGraph `StateGraph` (supervisor pattern) and
served by FastAPI. Every node posts its real hand-off to the agents' group chat (`AGENT_MESSAGE` audit events).

## Commands

```bash
python3 -m venv .venv && source .venv/bin/activate   # Python 3.11 or 3.12
pip install -r requirements.txt
cp .env.example .env                                  # then set MESH_API_KEY

uvicorn src.api.main:app --reload --port 8000         # dashboard at http://localhost:8000
python3 -m compileall src tests eval                  # syntax gate (run before every commit)
pytest                                                # hermetic test suite (no network)
python3 -m eval.run_eval                              # benchmark → eval/latest_results.json
ruff check . && mypy src                              # lint + type check (pip install ruff mypy)
docker compose up --build                             # full stack in a container
```

## Non-negotiable rules

1. **Mesh API only.** Every LLM and embedding call goes through `src/core/mesh_gateway.py`
   (`openai.OpenAI(base_url=MESH_BASE_URL, api_key=MESH_API_KEY)`). Never import a provider SDK
   or hardcode an OpenAI/Anthropic/Gemini URL anywhere else.
2. **Tavily is a scraper, not a reasoner.** `include_answer=False` always.
3. **No hardcoded outputs.** Sub-questions, sources, verdicts, contradictions, and audit events
   are computed at runtime. The offline corpus in `src/tools/offline_corpus.json` is *input*
   evidence for hermetic tests/eval, never a canned answer.
4. **Every claim is auditable.** Verifier verdicts must quote a span that exists verbatim in the
   cited source (`quote_verified`); unsupported claims are stripped from the report but kept in
   `claims_audit` and the audit trail.
5. **Guardrails are graph nodes, not afterthoughts.** New inputs pass `InputGuard`, new evidence passes
   `ContentGuard`, new fetches pass `UrlSafetyPolicy`, anything published passes `OutputGuard`; every
   decision is emitted as a `GUARDRAIL_FINDING` audit event. Never persist an unredacted question.
6. **Never commit** `.env`, `venv/`, `.venv/`, `data/`, `*.db`, `__pycache__/`, or Chroma files.

## Code conventions

- Python ≥ 3.11, `from __future__ import annotations`, full type hints, Pydantic v2 models for
  every agent input/output contract (`src/schemas/domain.py`).
- Agents receive their dependencies (gateway, retriever, vector store) via the constructor so
  tests inject fakes; no module-level network clients.
- Raise typed errors from `src/core/exceptions.py`; never bare `Exception`.
- Tests are hermetic: use the `FakeMeshClient` from `tests/conftest.py` and the offline corpus.
  Tests that need real endpoints are marked `@pytest.mark.live`.
- Conventional Commits: `feat(scope): …`, `fix(scope): …`, `docs: …`, `build: …`, `chore: …`.

## Layout

| Path | Responsibility |
| --- | --- |
| `src/core/` | Settings, Mesh gateway (retries, JSON repair, token/latency tracing), exceptions |
| `src/schemas/` | Pydantic domain models + LangGraph `ResearchState` with reducers |
| `src/storage/` | SQLAlchemy audit store (runs, traces, claims, sources) + hybrid evidence index |
| `src/tools/` | Source authority scorer, 3-tier web retriever, document ingestor, PDF exporter |
| `src/agents/` | Guardian + Master Approver (`master.py`) + Report Approver (`report_approver.py`) + six agents + LangGraph orchestrator (`input_guard`, `route_master`, `route_verification` as the gate floor, `route_report`, `output_guard`), group-chat messages, architecture catalog |
| `src/guardrails/` | Input/content/output shields, injection & harm patterns, PII/secret redaction, SSRF policy |
| `src/api/` | FastAPI app, security middleware, REST + SSE routes, dashboard (ES modules in `static/`) |
| `eval/` | Benchmark questions, evaluation harness, committed metrics |
