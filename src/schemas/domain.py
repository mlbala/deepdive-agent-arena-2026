"""Pydantic v2 domain contracts for DeepDive.

Two families of models live here:

1. **Domain records** — the durable artefacts of a research run (sub-questions,
   sources, evidence spans, claims, verdicts, contradictions, audit events,
   the final report). These flow through the LangGraph state and the API.
2. **LLM I/O contracts** — the exact JSON shapes each agent must return from a
   Mesh API completion. `MeshGateway.complete_structured` validates every model
   response against one of these and self-heals when validation fails.
"""

from __future__ import annotations

import re
import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Identifiers & citation helpers
# ---------------------------------------------------------------------------
_SOURCE_ID_RE = re.compile(r"SRC[-_ ]?(\d{1,4})", re.IGNORECASE)
_CITATION_GROUP_RE = re.compile(r"\[([^\[\]]*?SRC[-_ ]?\d{1,4}[^\[\]]*?)\]", re.IGNORECASE)


def utcnow() -> datetime:
    return datetime.now(UTC)


def new_run_id() -> str:
    return f"run_{uuid.uuid4().hex[:12]}"


def new_event_id() -> str:
    return uuid.uuid4().hex


def format_source_id(index: int) -> str:
    return f"SRC-{index:03d}"


def normalize_source_id(raw: str) -> str | None:
    """Canonicalise `src-1`, `SRC_01`, `SRC-001` → `SRC-001`. Returns None if not an id."""
    match = _SOURCE_ID_RE.search(raw or "")
    return format_source_id(int(match.group(1))) if match else None


def extract_citations(text: str) -> list[str]:
    """Return the ordered, de-duplicated `[SRC-XXX]` ids cited in `text`.

    Handles `[SRC-001]`, `[SRC-001, SRC-004]` and `[SRC-1; SRC-2]` forms.
    """
    seen: dict[str, None] = {}
    for group in _CITATION_GROUP_RE.findall(text or ""):
        for match in _SOURCE_ID_RE.finditer(group):
            seen.setdefault(format_source_id(int(match.group(1))), None)
    return list(seen)


def _normalize_id_list(values: Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        values = re.split(r"[,\s;]+", values)
    out: dict[str, None] = {}
    for value in values:
        normalized = normalize_source_id(str(value))
        if normalized:
            out.setdefault(normalized, None)
    return list(out)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------
class Verdict(StrEnum):
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    UNSUPPORTED = "UNSUPPORTED"


class SourceTier(StrEnum):
    TIER_1 = "TIER_1"
    TIER_2 = "TIER_2"
    TIER_3 = "TIER_3"


class RetrievalTier(StrEnum):
    TAVILY = "tavily"
    DUCKDUCKGO = "duckduckgo"
    OFFLINE = "offline_corpus"
    UPLOAD = "uploaded_document"


class RunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


class Severity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class AuditEventType(StrEnum):
    RUN_STARTED = "run_started"
    RUN_COMPLETED = "run_completed"
    RUN_FAILED = "run_failed"
    NODE_STARTED = "node_started"
    NODE_COMPLETED = "node_completed"
    LLM_CALL = "llm_call"
    PLAN_CREATED = "plan_created"
    PLAN_REVIEWED = "plan_reviewed"
    ROUND_REVIEWED = "round_reviewed"
    AGENT_MESSAGE = "agent_message"
    SEARCH_EXECUTED = "search_executed"
    SOURCE_INDEXED = "source_indexed"
    REFINEMENT_PLANNED = "refinement_planned"
    DRAFT_CREATED = "draft_created"
    CITATION_INVALID = "citation_invalid"
    CLAIMS_EXTRACTED = "claims_extracted"
    CLAIM_VERIFIED = "claim_verified"
    CLAIM_DOWNGRADED = "claim_downgraded"
    CONTRADICTION_DETECTED = "contradiction_detected"
    METRICS_COMPUTED = "metrics_computed"
    ROUTE_DECISION = "route_decision"
    LOOP_BACK = "loop_back"
    CLAIM_REMOVED = "claim_removed"
    REPORT_COMPILED = "report_compiled"
    FOLLOWUP_ANSWERED = "followup_answered"
    RUN_BLOCKED = "run_blocked"
    GUARDRAIL_CHECK = "guardrail_check"
    GUARDRAIL_FINDING = "guardrail_finding"
    GUARDRAIL_BLOCKED = "guardrail_blocked"
    BUDGET_GUARD = "budget_guard"
    WARNING = "warning"
    ERROR = "error"


class GuardrailAction(StrEnum):
    ALLOW = "allow"
    FLAG = "flag"
    REDACT = "redact"
    NEUTRALIZE = "neutralize"
    BLOCK = "block"


class GuardrailSeverity(StrEnum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class _Model(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True, use_enum_values=False)


# ---------------------------------------------------------------------------
# Guardrails & governance
# ---------------------------------------------------------------------------
class GuardrailFinding(_Model):
    """One decision taken by a guardrail. Excerpts are always redacted and truncated."""

    guard: str  # input_shield | content_shield | url_shield | output_shield | llm_screen | budget | model_policy
    kind: str  # prompt_injection | harmful_request | pii | secret | hidden_text | ssrf | ...
    severity: GuardrailSeverity
    action: GuardrailAction
    detail: str
    location: str = ""
    excerpt: str = ""


class GovernanceSummary(_Model):
    policy_version: str
    input_screen: str = "passed"  # passed | passed_with_redactions | blocked
    llm_screen: str = "skipped"  # passed | flagged | blocked | skipped | error
    findings: list[GuardrailFinding] = Field(default_factory=list)
    findings_by_guard: dict[str, int] = Field(default_factory=dict)
    injections_neutralized: int = 0
    urls_blocked: int = 0
    pii_redactions: int = 0
    secret_redactions: int = 0
    claims_downgraded: int = 0
    budget: dict[str, Any] = Field(default_factory=dict)
    models_used: list[str] = Field(default_factory=list)
    audit_chain_head: str | None = None
    audit_events: int = 0
    disclaimer: str = ""


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------
class SubQuestion(_Model):
    id: str = Field(pattern=r"^SQ-\d+$")
    question: str = Field(min_length=5)
    rationale: str = ""
    search_queries: list[str] = Field(min_length=1, max_length=4)
    priority: int = Field(default=1, ge=1, le=5)


# ---------------------------------------------------------------------------
# Sources & evidence
# ---------------------------------------------------------------------------
class SourceAuthority(_Model):
    score: float = Field(ge=0.0, le=1.0)
    tier: SourceTier
    rule: str
    signals: list[str] = Field(default_factory=list)


class SourceDocument(_Model):
    """A retrieved document, scored for authority, with its full cleaned text."""

    source_id: str = Field(pattern=r"^SRC-\d{3,4}$")
    url: str
    title: str
    domain: str
    content: str
    retrieval_tier: RetrievalTier
    authority: SourceAuthority
    query: str = ""
    sub_question_ids: list[str] = Field(default_factory=list)
    published_date: str | None = None
    retrieved_at: datetime = Field(default_factory=utcnow)
    content_hash: str = ""
    guardrail_flags: list[str] = Field(default_factory=list)


class EvidenceChunk(_Model):
    """An exact character span `[char_start, char_end)` of a SourceDocument's content."""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=False)

    chunk_id: str
    source_id: str
    sub_question_id: str | None = None
    text: str
    char_start: int = Field(ge=0)
    char_end: int = Field(gt=0)
    authority_score: float = Field(default=0.5, ge=0.0, le=1.0)
    retrieval_score: float = 0.0
    url: str = ""
    title: str = ""

    @model_validator(mode="after")
    def _span_is_consistent(self) -> EvidenceChunk:
        if self.char_end <= self.char_start:
            raise ValueError("char_end must be greater than char_start")
        if len(self.text) != self.char_end - self.char_start:
            raise ValueError("chunk text length must equal its character span")
        return self


class RawSearchResult(_Model):
    """Retriever output before authority scoring / id assignment."""

    url: str
    title: str
    content: str
    retrieval_tier: RetrievalTier
    query: str = ""
    published_date: str | None = None
    provider_score: float | None = None


# ---------------------------------------------------------------------------
# Drafting & verification
# ---------------------------------------------------------------------------
class DraftSection(_Model):
    sub_question_id: str
    iteration: int = Field(ge=1)
    text: str
    cited_source_ids: list[str] = Field(default_factory=list)
    invalid_citations: list[str] = Field(default_factory=list)
    evidence_chunk_ids: list[str] = Field(default_factory=list)


class AtomicClaim(_Model):
    claim_id: str
    sub_question_id: str
    iteration: int = Field(ge=1)
    text: str = Field(min_length=3)
    cited_source_ids: list[str] = Field(default_factory=list)

    @field_validator("cited_source_ids", mode="before")
    @classmethod
    def _normalize_cited(cls, value: Any) -> list[str]:
        return _normalize_id_list(value)


class VerificationResult(_Model):
    claim_id: str
    sub_question_id: str = ""
    iteration: int = 1
    verdict: Verdict
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_source_id: str | None = None
    evidence_quote: str | None = None
    evidence_char_start: int | None = None
    evidence_char_end: int | None = None
    quote_verified: bool = False
    rationale: str = ""
    downgraded: bool = False
    downgrade_reason: str | None = None


class ContradictionItem(_Model):
    contradiction_id: str
    topic: str
    sub_question_id: str | None = None
    statement_a: str
    source_a: str
    quote_a_verified: bool = False
    statement_b: str
    source_b: str
    quote_b_verified: bool = False
    severity: Severity = Severity.MEDIUM
    explanation: str = ""
    origin: Literal["cross_source", "claim_verification"] = "cross_source"
    resolution_hint: str = ""


class VerificationGap(_Model):
    sub_question_id: str
    reason: str
    unsupported_claims: list[str] = Field(default_factory=list)
    feedback: list[str] = Field(default_factory=list)  # "claim — why the Verifier rejected it", for the re-draft


class IterationMetrics(_Model):
    iteration: int
    total_claims: int
    supported: int
    contradicted: int
    unsupported: int
    grounding_rate: float
    coverage_score: float
    citation_precision: float
    quote_verification_rate: float
    sources_indexed: int
    target_sub_question_ids: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Plan review & agent group chat
# ---------------------------------------------------------------------------
class PlanIssue(_Model):
    target: str = Field(default="plan", description="'plan' or the sub-question id (SQ-n) the issue is about.")
    severity: Severity = Severity.MEDIUM
    problem: str = Field(min_length=3)
    fix: str = Field(default="", description="A concrete change the Planner can make.")
    origin: Literal["master", "check"] = "master"  # Master Approver model review or deterministic plan check
    blocking: bool = False  # set by the Master Approver: this issue alone keeps the plan from passing

    @field_validator("origin", mode="before")
    @classmethod
    def _legacy_origin(cls, value: Any) -> Any:
        return "master" if value in ("lead", "approver") else value  # runs recorded under the earlier names

    @field_validator("severity", mode="before")
    @classmethod
    def _lower_severity(cls, value: Any) -> Any:
        return value.strip().lower() if isinstance(value, str) else value


class PlanReview(_Model):
    """The Master Approver's verdict on one version of the plan."""

    round: int = Field(ge=1)
    score: float = Field(ge=0.0, le=1.0)
    threshold: float = Field(ge=0.0, le=1.0)
    passed: bool
    forced: bool = False  # revision budget spent: the plan proceeds with its issues on record
    rubric: dict[str, float] = Field(default_factory=dict)
    issues: list[PlanIssue] = Field(default_factory=list)
    summary: str = ""
    llm_reviewed: bool = True  # False when the Master Approver model was unavailable and only the checks ran


class AgentInstruction(_Model):
    """A Master Approver instruction to one agent for the next evidence round."""

    agent: Literal["researcher", "drafter"]
    sub_question_id: str
    instruction: str
    origin: Literal["master", "gate"] = "master"  # written by the Master Approver model, or derived from the Verifier's gap

    @field_validator("origin", mode="before")
    @classmethod
    def _legacy_origin(cls, value: Any) -> Any:
        return "master" if value in ("lead", "approver") else value  # runs recorded under the earlier names


class RoundReview(_Model):
    """The Master Approver's decision after one evidence round (researcher -> drafter -> verifier)."""

    round: int = Field(ge=1)
    decision: Literal["approve", "another_round"]
    gate_passed: bool  # the deterministic grounding/coverage thresholds were met
    floor: Literal["researcher", "synthesizer"]  # what the deterministic quality gate alone would do
    reason: str
    rationale: str = ""
    target_sub_question_ids: list[str] = Field(default_factory=list)
    instructions: list[AgentInstruction] = Field(default_factory=list)
    grounding_rate: float = 0.0
    coverage_score: float = 0.0
    grounding_threshold: float = 0.0
    coverage_threshold: float = 0.0
    extra_round: bool = False  # the gate passed but the Master Approver asked for one more targeted pass
    overruled: bool = False  # the Master Approver wanted to approve below the bar; the gate kept the loop going
    llm_reviewed: bool = False


class AgentMessage(_Model):
    """One hand-off in the agents' group chat: what one agent actually passes to the next."""

    message_id: str = Field(default_factory=new_event_id)
    run_id: str
    channel: Literal["plan_review", "evidence_loop", "publish"]
    round: int = Field(default=1, ge=1)
    sender: str
    to: list[str] = Field(default_factory=list)
    kind: Literal["proposal", "critique", "revision", "approval", "evidence", "draft", "verdict", "decision", "report"]
    text: str
    items: list[str] = Field(default_factory=list)
    score: float | None = None
    threshold: float | None = None
    passed: bool | None = None
    refs: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utcnow)


# ---------------------------------------------------------------------------
# Audit trail
# ---------------------------------------------------------------------------
class AuditEvent(_Model):
    event_id: str = Field(default_factory=new_event_id)
    run_id: str
    timestamp: datetime = Field(default_factory=utcnow)
    node: str
    event_type: AuditEventType
    message: str
    iteration: int = 0
    payload: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Final report
# ---------------------------------------------------------------------------
class ReportSection(_Model):
    sub_question_id: str
    heading: str
    body: str
    claim_ids: list[str] = Field(default_factory=list)
    cited_source_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class RemovedClaim(_Model):
    claim_id: str
    sub_question_id: str
    text: str
    verdict: Verdict
    reason: str


class SourceReference(_Model):
    source_id: str
    title: str
    url: str
    domain: str
    authority_score: float
    tier: SourceTier
    retrieval_tier: RetrievalTier
    cited_in_report: bool = False
    flags: list[str] = Field(default_factory=list)


class ReportMetrics(_Model):
    grounding_rate_pre: float = 0.0
    grounding_rate_final: float = 0.0
    coverage_score: float = 0.0
    citation_precision: float = 0.0
    quote_verification_rate: float = 0.0
    report_citation_coverage: float = 0.0
    iterations: int = 0
    loop_backs: int = 0
    total_claims_evaluated: int = 0
    claims_in_report: int = 0
    claims_removed: int = 0
    contradictions: int = 0
    sources_indexed: int = 0
    sources_cited: int = 0
    llm_calls: int = 0
    total_tokens: int = 0
    llm_latency_ms: int = 0
    plan_rounds: int = 0
    plan_score: float = 0.0


class FinalReport(_Model):
    run_id: str
    query: str
    title: str
    executive_summary: str
    key_findings: list[str] = Field(default_factory=list)
    sections: list[ReportSection] = Field(default_factory=list)
    contradictions: list[ContradictionItem] = Field(default_factory=list)
    removed_claims: list[RemovedClaim] = Field(default_factory=list)
    sources: list[SourceReference] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    sub_questions: list[SubQuestion] = Field(default_factory=list)
    metrics: ReportMetrics = Field(default_factory=ReportMetrics)
    guard_removed_sentences: list[str] = Field(default_factory=list)
    governance: GovernanceSummary | None = None
    plan_reviews: list[PlanReview] = Field(default_factory=list)
    round_reviews: list[RoundReview] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=utcnow)


class FollowUpAnswer(_Model):
    run_id: str
    question: str
    answer: str
    answerable: bool
    cited_source_ids: list[str] = Field(default_factory=list)
    evidence_chunk_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    removed_citations: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utcnow)


# ---------------------------------------------------------------------------
# LLM I/O contracts (validated by MeshGateway.complete_structured)
# ---------------------------------------------------------------------------
class PlannedSubQuestion(_Model):
    question: str = Field(min_length=5, description="A focused, self-contained research sub-question.")
    rationale: str = Field(default="", description="Why this sub-question is needed and how it differs from the others.")
    search_queries: list[str] = Field(
        min_length=1, max_length=3, description="1-3 concise web search queries targeting this sub-question."
    )
    priority: int = Field(default=1, ge=1, le=5, description="1 = most important.")


class PlannerOutput(_Model):
    research_strategy: str = Field(default="", description="One or two sentences describing the decomposition logic.")
    sub_questions: list[PlannedSubQuestion] = Field(min_length=3, max_length=5)
    revision_notes: list[str] = Field(
        default_factory=list, description="Only when revising: each change made and the review issue it addresses."
    )


def _unit_score(value: Any) -> Any:
    """Accept 0-1, 1-10 or 1-100 scales from the model and normalise to 0-1."""
    if isinstance(value, int | float) and not isinstance(value, bool):
        if 1.0 < value <= 10.0:
            return value / 10.0
        if 10.0 < value <= 100.0:
            return value / 100.0
    return value


class PlanRubric(_Model):
    coverage: float = Field(ge=0.0, le=1.0, description="Together the sub-questions fully answer the research question.")
    distinctness: float = Field(ge=0.0, le=1.0, description="No two sub-questions ask for the same facts.")
    searchability: float = Field(ge=0.0, le=1.0, description="Search queries are concise and will surface primary sources.")
    premise_check: float = Field(
        ge=0.0, le=1.0, description="A doubtful premise in the question is tested directly (1.0 if there is none)."
    )
    specificity: float = Field(ge=0.0, le=1.0, description="Entities, metrics, jurisdictions and periods are precise.")

    @field_validator("coverage", "distinctness", "searchability", "premise_check", "specificity", mode="before")
    @classmethod
    def _scale(cls, value: Any) -> Any:
        return _unit_score(value)


class PlanIssueOut(_Model):
    target: str = Field(default="plan", description="'plan' or a sub-question id such as SQ-2.")
    severity: Severity = Severity.MEDIUM
    problem: str = Field(min_length=3)
    fix: str = ""

    @field_validator("severity", mode="before")
    @classmethod
    def _lower_severity(cls, value: Any) -> Any:
        return value.strip().lower() if isinstance(value, str) else value


class PlanReviewOutput(_Model):
    rubric: PlanRubric
    issues: list[PlanIssueOut] = Field(default_factory=list)
    summary: str = Field(default="", description="1-3 sentences addressed to the Planner.")


class AgentInstructionOut(_Model):
    agent: Literal["researcher", "drafter"]
    sub_question_id: str
    instruction: str = Field(min_length=3, description="One concrete, actionable instruction.")

    @field_validator("agent", mode="before")
    @classmethod
    def _lower_agent(cls, value: Any) -> Any:
        return value.strip().lower() if isinstance(value, str) else value


class RoundReviewOutput(_Model):
    decision: Literal["approve", "another_round"]
    rationale: str = Field(default="", description="1-2 sentences addressed to the team.")
    focus_sub_question_ids: list[str] = Field(default_factory=list)
    instructions: list[AgentInstructionOut] = Field(default_factory=list)

    @field_validator("decision", mode="before")
    @classmethod
    def _normalise_decision(cls, value: Any) -> Any:
        if isinstance(value, str):
            v = value.strip().lower().replace(" ", "_").replace("-", "_")
            aliases = {"approved": "approve", "loop": "another_round", "retry": "another_round",
                       "revise": "another_round", "loop_back": "another_round", "another": "another_round"}
            return aliases.get(v, v)
        return value


class RefinementQuery(_Model):
    sub_question_id: str
    query: str = Field(min_length=3)
    rationale: str = ""


class RefinementOutput(_Model):
    queries: list[RefinementQuery] = Field(min_length=1, max_length=10)


class DraftOutput(_Model):
    answer: str = Field(
        min_length=1,
        description="Markdown-free prose answer. Every factual sentence ends with [SRC-XXX] citation(s).",
    )
    evidence_sufficient: bool = Field(default=True, description="False if the evidence cannot answer the sub-question.")


class ExtractedClaim(_Model):
    text: str = Field(min_length=3, description="One atomic, independently checkable factual claim.")
    cited_source_ids: list[str] = Field(default_factory=list, description="Source ids cited for this claim in the draft.")

    @field_validator("cited_source_ids", mode="before")
    @classmethod
    def _normalize_cited(cls, value: Any) -> list[str]:
        return _normalize_id_list(value)


class ClaimExtractionOutput(_Model):
    claims: list[ExtractedClaim] = Field(default_factory=list, max_length=25)


class ClaimJudgement(_Model):
    claim_id: str
    verdict: Verdict
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_source_id: str | None = Field(default=None, description="The SRC id whose text decides the verdict.")
    evidence_quote: str | None = Field(
        default=None, description="Verbatim span copied exactly from the evidence that entails or contradicts the claim."
    )
    rationale: str = ""

    @field_validator("verdict", mode="before")
    @classmethod
    def _upper_verdict(cls, value: Any) -> Any:
        if isinstance(value, str):
            cleaned = value.strip().upper().replace(" ", "_")
            aliases = {"SUPPORT": "SUPPORTED", "ENTAILED": "SUPPORTED", "REFUTED": "CONTRADICTED",
                       "NOT_SUPPORTED": "UNSUPPORTED", "NOT_ENOUGH_INFO": "UNSUPPORTED", "NEI": "UNSUPPORTED"}
            return aliases.get(cleaned, cleaned)
        return value

    @field_validator("evidence_source_id", mode="before")
    @classmethod
    def _normalize_source(cls, value: Any) -> Any:
        if value in (None, "", "null", "none", "None"):
            return None
        return normalize_source_id(str(value))


class ClaimJudgementOutput(_Model):
    judgements: list[ClaimJudgement]


class DetectedContradiction(_Model):
    topic: str
    source_a: str
    quote_a: str = Field(description="Verbatim span from source_a.")
    source_b: str
    quote_b: str = Field(description="Verbatim span from source_b that conflicts with quote_a.")
    severity: Severity = Severity.MEDIUM
    explanation: str = ""

    @field_validator("severity", mode="before")
    @classmethod
    def _lower_severity(cls, value: Any) -> Any:
        return value.strip().lower() if isinstance(value, str) else value

    @field_validator("source_a", "source_b", mode="before")
    @classmethod
    def _normalize_sources(cls, value: Any) -> Any:
        return normalize_source_id(str(value)) or str(value)


class ContradictionScanOutput(_Model):
    contradictions: list[DetectedContradiction] = Field(default_factory=list, max_length=10)


class SynthesizedSection(_Model):
    sub_question_id: str
    heading: str
    body: str = Field(description="Prose built ONLY from the verified claims; keep their [SRC-XXX] citations.")


class SynthesisOutput(_Model):
    title: str
    executive_summary: str = Field(description="3-5 sentences, every sentence cited.")
    key_findings: list[str] = Field(default_factory=list, max_length=8)
    sections: list[SynthesizedSection] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list, max_length=6)


class GuardianOutput(_Model):
    verdict: Literal["allow", "flag", "block"] = Field(description="block only for clear policy violations")
    categories: list[str] = Field(default_factory=list, description="e.g. prompt_injection, harmful_request, pii")
    prompt_injection: bool = False
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    rationale: str = ""

    @field_validator("verdict", mode="before")
    @classmethod
    def _lower_verdict(cls, value: Any) -> Any:
        return value.strip().lower() if isinstance(value, str) else value


class FollowUpOutput(_Model):
    answer: str
    answerable: bool = Field(description="False when the evidence pool does not contain the answer.")
    cited_source_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    @field_validator("cited_source_ids", mode="before")
    @classmethod
    def _normalize_cited(cls, value: Any) -> list[str]:
        return _normalize_id_list(value)
