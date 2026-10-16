"""LangGraph `ResearchState` with annotated reducers.

Each node returns a *partial* update. Reducers decide how that update merges
into the running state:

* `merge_dicts`  — keyed collections (sources, evidence, per-sub-question drafts,
  claims, verdicts). A loop-back iteration that re-drafts SQ-2 only overwrites
  SQ-2's entries and leaves already-verified sub-questions untouched.
* `append_list`  — append-only histories (audit trail, iteration metrics, every
  claim/verdict ever produced). Nothing is ever deleted from these, which is what
  makes the run auditable after unsupported claims are stripped from the report.
* Plain keys (no reducer) are last-write-wins.
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict, TypeVar

from src.schemas.domain import (
    AgentMessage,
    AtomicClaim,
    AuditEvent,
    ContradictionItem,
    DraftSection,
    EvidenceChunk,
    FinalReport,
    IterationMetrics,
    PlanReview,
    RefinementQuery,
    SourceDocument,
    SubQuestion,
    VerificationGap,
    VerificationResult,
)

K = TypeVar("K")
V = TypeVar("V")
T = TypeVar("T")


def merge_dicts(left: dict[K, V] | None, right: dict[K, V] | None) -> dict[K, V]:
    """Shallow merge; keys in `right` overwrite keys in `left`."""
    merged: dict[K, V] = dict(left or {})
    merged.update(right or {})
    return merged


def append_list(left: list[T] | None, right: list[T] | None) -> list[T]:
    """Concatenate append-only histories."""
    return [*(left or []), *(right or [])]


class ResearchState(TypedDict, total=False):
    # --- Run inputs ---------------------------------------------------------
    run_id: str
    query: str
    document_ids: list[str]
    max_iterations: int

    # --- Guardrails & governance -------------------------------------------------
    blocked: bool
    block_reason: str
    llm_screen: str
    budget_exhausted: bool

    # --- Planning & plan review (Planner <-> Approver) ------------------------
    sub_questions: list[SubQuestion]
    research_strategy: str
    plan_round: int
    plan_review: PlanReview | None
    plan_reviews: Annotated[list[PlanReview], append_list]

    # --- Retrieval (keyed, merged across iterations) --------------------------
    sources: Annotated[dict[str, SourceDocument], merge_dicts]          # source_id -> doc
    evidence: Annotated[dict[str, EvidenceChunk], merge_dicts]          # chunk_id -> span
    evidence_by_sq: Annotated[dict[str, list[str]], merge_dicts]        # sq_id -> chunk_ids
    refinement_queries: Annotated[list[RefinementQuery], append_list]  # full history

    # --- Drafting & verification (keyed by sub-question id) -----------------
    drafts: Annotated[dict[str, DraftSection], merge_dicts]
    claims: Annotated[dict[str, list[AtomicClaim]], merge_dicts]
    verifications: Annotated[dict[str, list[VerificationResult]], merge_dicts]
    contradictions: list[ContradictionItem]
    verification_gaps: list[VerificationGap]
    target_sub_question_ids: list[str]

    # --- Loop control ---------------------------------------------------------
    iteration_count: int
    coverage_score: float
    grounding_rate: float
    route: str

    # --- Append-only audit histories ------------------------------------------
    iteration_metrics: Annotated[list[IterationMetrics], append_list]
    claim_history: Annotated[list[AtomicClaim], append_list]
    verification_history: Annotated[list[VerificationResult], append_list]
    audit_log: Annotated[list[AuditEvent], append_list]
    group_chat: Annotated[list[AgentMessage], append_list]  # agent-to-agent hand-offs, in order
    errors: Annotated[list[str], append_list]

    # --- Output -------------------------------------------------------------
    final_report: FinalReport | None


def initial_state(
    run_id: str,
    query: str,
    *,
    max_iterations: int,
    document_ids: list[str] | None = None,
) -> ResearchState:
    """Seed state for a new run."""
    return ResearchState(
        run_id=run_id,
        query=query,
        document_ids=list(document_ids or []),
        max_iterations=max_iterations,
        blocked=False,
        block_reason="",
        llm_screen="skipped",
        budget_exhausted=False,
        sub_questions=[],
        research_strategy="",
        plan_round=0,
        plan_review=None,
        plan_reviews=[],
        sources={},
        evidence={},
        evidence_by_sq={},
        refinement_queries=[],
        drafts={},
        claims={},
        verifications={},
        contradictions=[],
        verification_gaps=[],
        target_sub_question_ids=[],
        iteration_count=0,
        coverage_score=0.0,
        grounding_rate=0.0,
        route="",
        iteration_metrics=[],
        claim_history=[],
        verification_history=[],
        audit_log=[],
        group_chat=[],
        errors=[],
        final_report=None,
    )


def all_verifications(state: dict[str, Any]) -> list[VerificationResult]:
    """Flatten the current (latest-per-sub-question) verdicts."""
    return [v for results in (state.get("verifications") or {}).values() for v in results]


def all_claims(state: dict[str, Any]) -> list[AtomicClaim]:
    """Flatten the current (latest-per-sub-question) claims."""
    return [c for claims in (state.get("claims") or {}).values() for c in claims]
