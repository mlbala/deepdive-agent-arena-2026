"""Architecture catalog: a machine-readable description of DeepDive's agent workflow.

Everything that can be introspected is introspected — the compiled LangGraph (nodes and
edges), each agent's real system prompt, the Pydantic contracts (JSON Schema), the
LangGraph state channels and their reducers, the configured models and policies — so the
"How it works" views can never drift from the code. The small curated parts (plain-English
descriptions and which state channels each node writes) are verified by tests against a
real graph execution.
"""

from __future__ import annotations

import typing
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from src.agents import (
    approver,
    drafter,
    followup,
    guardian,
    planner,
    researcher,
    synthesizer,
    verifier,
)
from src.core.config import Settings
from src.schemas import domain
from src.schemas.state import ResearchState

if TYPE_CHECKING:
    from src.agents.orchestrator import ResearchOrchestrator

# Which state channels each graph node returns (asserted against real node updates in tests).
NODE_WRITES: dict[str, list[str]] = {
    "input_guard": ["query", "blocked", "block_reason", "llm_screen", "audit_log"],
    "planner": [
        "sub_questions", "research_strategy", "target_sub_question_ids", "iteration_count", "plan_round",
        "group_chat", "audit_log",
    ],
    "approver": ["plan_review", "plan_reviews", "group_chat", "audit_log"],
    "researcher": [
        "iteration_count", "sources", "evidence", "evidence_by_sq", "refinement_queries", "group_chat", "audit_log",
    ],
    "drafter": ["drafts", "group_chat", "audit_log"],
    "verifier": [
        "budget_exhausted", "claims", "verifications", "contradictions", "verification_gaps",
        "target_sub_question_ids", "coverage_score", "grounding_rate", "route", "iteration_metrics",
        "claim_history", "verification_history", "group_chat", "audit_log",
    ],
    "synthesizer": ["final_report", "group_chat", "audit_log"],
    "output_guard": ["final_report", "group_chat", "audit_log"],
}

STATE_NOTES: dict[str, str] = {
    "run_id": "Identifier of the run.", "query": "The (redacted) research question.",
    "document_ids": "Uploaded documents to include as private evidence.", "max_iterations": "Research-pass budget.",
    "blocked": "Set by the input shield when the question is refused.", "block_reason": "Why it was refused.",
    "llm_screen": "Guardian verdict: passed / flagged / blocked / skipped / error.",
    "budget_exhausted": "Set when the run reaches its soft Mesh budget: no more loop-backs.",
    "sub_questions": "MECE sub-questions with their search queries.", "research_strategy": "Planner's decomposition logic.",
    "plan_round": "Which version of the plan this is (1 = first draft).",
    "plan_review": "The Approver's latest verdict: rubric, issues, score, passed / forced.",
    "plan_reviews": "Every plan review, in order (append-only).",
    "group_chat": "Agent-to-agent hand-offs: what each agent passed to the next (append-only).",
    "sources": "Scored, shielded source documents keyed by SRC id.", "evidence": "Exact character spans keyed by chunk id.",
    "evidence_by_sq": "Evidence selected for each sub-question.", "refinement_queries": "Every targeted refinement query (history).",
    "drafts": "Latest cited draft per sub-question.", "claims": "Latest atomic claims per sub-question.",
    "verifications": "Latest verdicts per sub-question.", "contradictions": "Quote-validated conflicts (recomputed each pass).",
    "verification_gaps": "Why each failing sub-question failed.", "target_sub_question_ids": "What the next pass re-researches.",
    "iteration_count": "Research passes so far.", "coverage_score": "Share of sub-questions with a supported claim.",
    "grounding_rate": "Share of claims supported by their cited source.", "route": "Last decision of route_verification.",
    "iteration_metrics": "Metrics of every pass (append-only).", "claim_history": "Every claim ever drafted (append-only).",
    "verification_history": "Every verdict ever given (append-only).", "audit_log": "Every event, in order (append-only).",
    "errors": "Non-fatal errors (append-only).", "final_report": "The verified, redacted, governed report.",
}

WORKFLOW: list[dict[str, Any]] = [
    {"id": "ask", "title": "Ask", "node": None, "actor": "You → FastAPI",
     "what": "A question arrives over the API. It is length-checked and rate-limited, and only a PII-redacted copy is ever written to disk.",
     "inputs": ["ResearchRequest"], "outputs": ["run row (status: pending)"],
     "checks": ["API key (optional)", "per-client rate limit", "concurrent-run cap", "PII redaction before persistence"], "branches": []},
    {"id": "screen", "title": "Screen", "node": "input_guard", "actor": "Input shield + Guardian agent",
     "what": "Deterministic patterns catch prompt injection, credential probing and operational harm. The Guardian model (via Mesh) catches paraphrased attacks. Personal data and secrets are redacted.",
     "inputs": ["question"], "outputs": ["screened question", "GuardrailFinding[]"],
     "checks": ["injection & jailbreak patterns", "harmful-intent patterns", "PII + secret redaction", "Guardian ≥ 0.80 to block"],
     "branches": [{"to": "end_blocked", "when": "a guardrail blocks: the run ends before any research agent runs"}]},
    {"id": "plan", "title": "Plan", "node": "planner", "actor": "Planner agent",
     "what": "Splits the question into 3–5 non-overlapping sub-questions, including one that tests a doubtful premise, each with targeted search queries. When the Approver sends the plan back, the Planner revises it against the listed issues and notes each change.",
     "inputs": ["question", "PlanReview (on revision)"], "outputs": ["SubQuestion[]", "AgentMessage"],
     "checks": ["near-duplicate removal", "schema self-repair"], "branches": []},
    {"id": "approve", "title": "Approve", "node": "approver", "actor": "Approver agent + deterministic plan checks",
     "what": "Scores the plan on coverage, distinctness, searchability, premise checking and specificity, and runs deterministic checks for dropped entities, overlapping sub-questions and malformed queries. Planner and Approver go back and forth in the plan-review room until the plan clears the bar.",
     "inputs": ["SubQuestion[]"], "outputs": ["PlanReview", "AgentMessage"],
     "checks": ["rubric score ≥ {plan_threshold}", "no blocking issue (a model-raised high issue must be corroborated by its rubric)", "at most {plan_revisions} revision(s)", "fails open on the model, never on the checks"],
     "branches": [{"to": "plan", "when": "score < {plan_threshold} or a blocking issue, with revisions left"},
                  {"to": "retrieve", "when": "approved, or the revision budget is spent (open issues stay on record)"}]},
    {"id": "retrieve", "title": "Retrieve", "node": "researcher", "actor": "Researcher agent + tools · evidence room",
     "what": "Opens the evidence room, where Researcher, Drafter and Verifier pass work to each other until the quality gate passes. Runs searches in parallel (Tavily → DuckDuckGo → offline corpus). Each page is checked by the URL shield, cleaned by the content shield, scored for authority and indexed as exact spans. Evidence is then selected for each sub-question.",
     "inputs": ["SubQuestion[]", "VerificationGap[] (on loop-back)"], "outputs": ["SourceDocument[]", "EvidenceChunk[]"],
     "checks": ["URL shield (SSRF)", "content shield (indirect injection)", "authority tiers", "dedupe by URL + content hash"], "branches": []},
    {"id": "draft", "title": "Draft", "node": "drafter", "actor": "Drafter agent",
     "what": "Writes each sub-question's answer from evidence only. Every factual sentence carries [SRC-XXX] citations. On a re-draft it receives the Verifier's reason for every rejected claim.",
     "inputs": ["EvidenceChunk[]", "VerificationGap[] with per-claim feedback (on loop-back)"], "outputs": ["DraftSection[]"], "checks": ["citations limited to the evidence pool"], "branches": []},
    {"id": "verify", "title": "Verify", "node": "verifier", "actor": "Verifier agent + deterministic guards",
     "what": "Splits drafts into atomic claims and checks each against its cited spans. The quote has to exist verbatim, the figures have to match, and the source has to clear the authority floor. The contradiction radar compares sources.",
     "inputs": ["DraftSection[]"], "outputs": ["AtomicClaim[]", "VerificationResult[]", "ContradictionItem[]", "IterationMetrics"],
     "checks": ["verbatim quote localisation", "numeric grounding", "citation scope", "admissibility floor", "contradiction quote validation"], "branches": []},
    {"id": "decide", "title": "Decide", "node": "router", "actor": "Quality gate · route_verification (conditional edge)",
     "what": "Compares grounding and coverage against their thresholds. Weak sub-questions are sent back to retrieval with refinement queries; otherwise the run moves on to synthesis.",
     "inputs": ["IterationMetrics"], "outputs": ["route"], "checks": ["iteration budget", "budget soft limit"],
     "branches": [{"to": "retrieve", "when": "coverage < {coverage} or grounding < {grounding}, with iterations and budget left"},
                  {"to": "synthesize", "when": "thresholds met, max iterations reached, or budget soft limit hit"}]},
    {"id": "synthesize", "title": "Synthesize", "node": "synthesizer", "actor": "Synthesizer agent",
     "what": "Writes the report from SUPPORTED claims only. Claims outranked by a contradicting source are withheld, and a citation guard drops any sentence citing unverified sources.",
     "inputs": ["verified claims", "contradictions"], "outputs": ["FinalReport"], "checks": ["authority-weighted resolution", "synthesis citation guard"], "branches": []},
    {"id": "publish", "title": "Publish", "node": "output_guard", "actor": "Output shield",
     "what": "Redacts secrets (always) and personal data (by policy), then attaches the governance summary and the disclosure.",
     "inputs": ["FinalReport"], "outputs": ["published FinalReport", "GovernanceSummary"], "checks": ["secret + PII redaction"], "branches": []},
    {"id": "audit", "title": "Audit", "node": None, "actor": "Hash-chained audit store",
     "what": "Every event above is SHA-256-chained per run. Anyone can verify the chain, export a signed-digest bundle, or replay the run event by event.",
     "inputs": ["AuditEvent[]"], "outputs": ["verification", "audit bundle"], "checks": ["append-only rows", "hash-chain verification"], "branches": []},
    {"id": "followup", "title": "Follow up", "node": "followup", "actor": "FollowUp agent",
     "what": "Answers follow-up questions strictly from this run's evidence. Questions pass the input shield and answers pass the output shield.",
     "inputs": ["question"], "outputs": ["FollowUpAnswer"], "checks": ["input + output shields", "citation guard"], "branches": []},
]


def _agents(settings: Settings) -> list[dict[str, Any]]:
    m = settings.model_for
    return [
        {"id": "guardian", "name": "Guardian", "kind": "guardrail", "node": "input_guard", "llm": True, "model": m("guardian"),
         "role": "Safety classifier for incoming questions (defence in depth after the deterministic shield).",
         "prompts": [{"name": "system", "text": guardian.SYSTEM_PROMPT}], "llm_contract": "GuardianOutput", "produces": ["GuardrailFinding"],
         "tools": ["InputGuard (deterministic patterns)"], "guardrails": ["blocks only at confidence ≥ 0.80", "fails open"],
         "code": "src/agents/guardian.py"},
        {"id": "planner", "name": "Planner", "kind": "agent", "node": "planner", "llm": True, "model": m("planner"),
         "role": "Splits the question into non-overlapping sub-questions with targeted search queries.",
         "prompts": [{"name": "system", "text": planner.SYSTEM_PROMPT}], "llm_contract": "PlannerOutput", "produces": ["SubQuestion"],
         "tools": [], "guardrails": ["3–5 sub-questions enforced by schema", "near-duplicate removal"], "code": "src/agents/planner.py"},
        {"id": "approver", "name": "Approver", "kind": "agent", "node": "approver", "llm": True, "model": m("approver"),
         "role": "Quality gate for the plan: scores it on a five-part rubric and sends it back with concrete fixes until it clears the bar.",
         "prompts": [{"name": "system", "text": approver.SYSTEM_PROMPT}, {"name": "planner revision", "text": planner.REVISION_PROMPT}],
         "llm_contract": "PlanReviewOutput", "produces": ["PlanReview", "AgentMessage"],
         "tools": ["plan_checks (dropped entities, overlapping sub-questions, query shape)"],
         "guardrails": [f"approve at score ≥ {settings.plan_approval_threshold:.2f} with no blocking issue",
                        "model-raised high issues block only when its rubric agrees (a criterion < 0.70)",
                        f"at most {settings.max_plan_revisions} revision(s), then proceed with issues on record",
                        "fails open on the model, never on the deterministic checks"],
         "code": "src/agents/approver.py"},
        {"id": "researcher", "name": "Researcher", "kind": "agent", "node": "researcher", "llm": True, "model": m("researcher"),
         "role": "Retrieves, shields, scores and indexes evidence; plans targeted refinement queries on loop-back.",
         "prompts": [{"name": "refinement", "text": researcher.REFINEMENT_PROMPT}], "llm_contract": "RefinementOutput",
         "produces": ["SourceDocument", "EvidenceChunk"],
         "tools": ["WebRetriever: Tavily → DuckDuckGo + BeautifulSoup → offline corpus", "UrlSafetyPolicy", "ContentGuard",
                   "score_source (Tier 1/2/3)", "HybridVectorStore (BM25 + Chroma via Mesh embeddings)"],
         "guardrails": ["URL shield", "content shield", "authority penalty for injected pages"], "code": "src/agents/researcher.py"},
        {"id": "drafter", "name": "Drafter", "kind": "agent", "node": "drafter", "llm": True, "model": m("drafter"),
         "role": "Writes evidence-only answers with an [SRC-XXX] citation on every factual sentence.",
         "prompts": [{"name": "system", "text": drafter.SYSTEM_PROMPT}], "llm_contract": "DraftOutput", "produces": ["DraftSection"],
         "tools": [], "guardrails": ["invalid citations flagged"], "code": "src/agents/drafter.py"},
        {"id": "verifier", "name": "Verifier", "kind": "agent", "node": "verifier", "llm": True, "model": m("verifier"),
         "role": "Decomposes drafts into atomic claims, checks each one against its cited spans, and finds contradictions.",
         "prompts": [{"name": "claim decomposition", "text": verifier.CLAIM_PROMPT},
                     {"name": "entailment", "text": verifier.ENTAILMENT_PROMPT},
                     {"name": "contradiction radar", "text": verifier.CONTRADICTION_PROMPT}],
         "llm_contract": "ClaimJudgementOutput", "produces": ["AtomicClaim", "VerificationResult", "ContradictionItem", "IterationMetrics"],
         "tools": ["locate_quote (exact → normalised → elliptical → fuzzy, numbers locked)", "ungrounded_numbers", "HybridVectorStore"],
         "guardrails": ["hallucinated-quote downgrade", "miscitation downgrade", "confidence floor", "numeric grounding",
                        f"admissibility floor {settings.min_source_authority:.2f}"], "code": "src/agents/verifier.py"},
        {"id": "router", "name": "Quality gate", "kind": "decision", "node": "router", "llm": False, "model": None,
         "role": "route_verification: the evidence room's exit. Loops back to the Researcher or moves on to synthesis, and posts the decision to the group chat.",
         "prompts": [], "llm_contract": None, "produces": ["route"], "tools": [],
         "guardrails": [f"loop if coverage < {settings.coverage_threshold:.0%} or grounding < {settings.grounding_threshold:.0%}",
                        f"at most {settings.max_iterations} research passes", f"budget soft limit {settings.budget_soft_fraction:.0%}"],
         "code": "src/agents/orchestrator.py"},
        {"id": "synthesizer", "name": "Synthesizer", "kind": "agent", "node": "synthesizer", "llm": True, "model": m("synthesizer"),
         "role": "Writes the report from verified claims only and surfaces conflicts openly.",
         "prompts": [{"name": "system", "text": synthesizer.SYSTEM_PROMPT}], "llm_contract": "SynthesisOutput", "produces": ["FinalReport"],
         "tools": [], "guardrails": ["authority-weighted contradiction resolution", "synthesis citation guard"], "code": "src/agents/synthesizer.py"},
        {"id": "output_guard", "name": "Output shield", "kind": "guardrail", "node": "output_guard", "llm": False, "model": None,
         "role": "Redacts what gets published and attaches the governance summary.",
         "prompts": [], "llm_contract": None, "produces": ["GovernanceSummary"], "tools": ["OutputGuard"],
         "guardrails": ["secrets always redacted", "PII redacted by policy"], "code": "src/guardrails/shields.py"},
        {"id": "followup", "name": "FollowUp", "kind": "agent", "node": "followup", "llm": True, "model": m("followup"),
         "role": "Grounded follow-up Q&A over the run's evidence pool.",
         "prompts": [{"name": "system", "text": followup.SYSTEM_PROMPT}], "llm_contract": "FollowUpOutput", "produces": ["FollowUpAnswer"],
         "tools": ["HybridVectorStore (rehydrated from SQLite)"], "guardrails": ["input + output shields", "citation guard"],
         "code": "src/agents/followup.py"},
    ]


def _type_name(tp: Any) -> str:
    origin = typing.get_origin(tp)
    if origin is typing.Annotated:
        return _type_name(typing.get_args(tp)[0])
    if origin in (list, dict, tuple, set):
        return f"{origin.__name__}[{', '.join(_type_name(a) for a in typing.get_args(tp))}]"
    if origin in (typing.Union, getattr(__import__("types"), "UnionType", None)):
        return " | ".join(_type_name(a) for a in typing.get_args(tp))
    name = getattr(tp, "__name__", str(tp).replace("typing.", ""))
    return "None" if name == "NoneType" else name


def _state_channels() -> list[dict[str, Any]]:
    hints = typing.get_type_hints(ResearchState, include_extras=True)
    channels = []
    for name, tp in hints.items():
        reducer = "last write wins"
        if typing.get_origin(tp) is typing.Annotated:
            meta = typing.get_args(tp)[1:]
            reducer = next((getattr(fn, "__name__", str(fn)) for fn in meta if callable(fn)), reducer)
        channels.append({
            "name": name, "type": _type_name(tp), "reducer": reducer,
            "written_by": [node for node, keys in NODE_WRITES.items() if name in keys],
            "description": STATE_NOTES.get(name, ""),
        })
    return channels


CONTRACT_MODELS: tuple[type[BaseModel], ...] = (
    domain.SubQuestion, domain.SourceDocument, domain.EvidenceChunk, domain.DraftSection, domain.AtomicClaim,
    domain.VerificationResult, domain.ContradictionItem, domain.IterationMetrics, domain.FinalReport,
    domain.GuardrailFinding, domain.GovernanceSummary, domain.FollowUpAnswer, domain.PlanReview, domain.AgentMessage,
    domain.GuardianOutput, domain.PlannerOutput, domain.PlanReviewOutput, domain.RefinementOutput, domain.DraftOutput,
    domain.ClaimExtractionOutput, domain.ClaimJudgementOutput, domain.ContradictionScanOutput,
    domain.SynthesisOutput, domain.FollowUpOutput,
)


def describe_contracts() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for model in CONTRACT_MODELS:
        schema = model.model_json_schema()
        required = set(schema.get("required", []))
        fields = []
        for name, prop in schema.get("properties", {}).items():
            kind = prop.get("type") or ("ref" if "$ref" in prop else "any")
            if "anyOf" in prop:
                kind = " | ".join(str(p.get("type") or p.get("$ref", "").split("/")[-1]) for p in prop["anyOf"])
            elif kind == "array":
                item = prop.get("items", {})
                kind = f"array<{item.get('type') or item.get('$ref', '').split('/')[-1] or 'any'}>"
            elif kind == "ref":
                kind = prop["$ref"].split("/")[-1]
            fields.append({"name": name, "type": kind, "required": name in required, "description": prop.get("description", "")})
        out[model.__name__] = {
            "kind": "llm_output" if model.__name__.endswith("Output") else "domain",
            "doc": (model.__doc__ or "").strip().split("\n")[0],
            "fields": fields,
        }
    return out


def describe_architecture(orchestrator: ResearchOrchestrator, settings: Settings) -> dict[str, Any]:
    compiled = orchestrator.graph.get_graph()
    nodes = [{"id": node_id, "terminal": node_id in ("__start__", "__end__")} for node_id in compiled.nodes]
    edges = [{"source": e.source, "target": e.target, "conditional": bool(e.conditional), "label": str(e.data or "")}
             for e in compiled.edges]
    values = {
        "coverage": f"{settings.coverage_threshold:.0%}", "grounding": f"{settings.grounding_threshold:.0%}",
        "plan_threshold": f"{settings.plan_approval_threshold:.2f}", "plan_revisions": settings.max_plan_revisions,
    }
    workflow = []
    for step in WORKFLOW:
        branches = [{**b, "when": b["when"].format(**values)} for b in step["branches"]]
        workflow.append({**step, "checks": [c.format(**values) for c in step["checks"]], "branches": branches})
    return {
        "graph": {"nodes": nodes, "edges": edges, "mermaid": orchestrator.mermaid()},
        "workflow": workflow,
        "agents": _agents(settings),
        "state": _state_channels(),
        "contracts": describe_contracts(),
        "policy": settings.governance_policy(),
        "gateway": {
            "base_url": settings.mesh_base_url, "default_model": settings.mesh_default_model,
            "embedding_model": settings.mesh_embedding_model, "json_repair_attempts": settings.mesh_json_repair_attempts,
            "max_retries": settings.mesh_max_retries, "allowed_models": sorted(settings.allowed_model_set),
        },
    }
