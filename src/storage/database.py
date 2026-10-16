"""SQLite + SQLAlchemy 2.0 audit store.

Tables
------
runs            one row per research run (status, headline metrics, final report JSON)
agent_traces    one immutable row per Mesh API call (model, latency, tokens, status)
claims_audit    every atomic claim ever verified, its verdict, quote, and report inclusion
sources         every scored source document retrieved for a run
evidence_chunks exact character spans indexed for a run (powers follow-up chat)
audit_events    the full node-by-node audit trail
documents       user-uploaded PDF/TXT/MD documents
followups       grounded follow-up Q&A turns

`agent_traces`, `claims_audit`, and `audit_events` are append-only: an ORM-level
guard raises `ImmutableRecordError` on any UPDATE or DELETE attempt.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    event,
    func,
    select,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker
from sqlalchemy.pool import StaticPool

from src.core.exceptions import ImmutableRecordError, RunNotFoundError
from src.schemas.domain import (
    AtomicClaim,
    AuditEvent,
    EvidenceChunk,
    FinalReport,
    FollowUpAnswer,
    RunStatus,
    SourceDocument,
    VerificationResult,
)


def _now() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class ResearchRun(Base):
    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    query: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(16), default=RunStatus.PENDING.value, index=True)
    max_iterations: Mapped[int] = mapped_column(Integer, default=3)
    iterations: Mapped[int] = mapped_column(Integer, default=0)
    grounding_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    coverage_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    report_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    document_ids_json: Mapped[str] = mapped_column(Text, default="[]")
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class AgentTrace(Base):
    __tablename__ = "agent_traces"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    trace_id: Mapped[str] = mapped_column(String(32), unique=True)
    run_id: Mapped[str | None] = mapped_column(String(32), index=True, nullable=True)
    agent_name: Mapped[str] = mapped_column(String(32), index=True)
    operation: Mapped[str] = mapped_column(String(16), default="chat")
    model: Mapped[str] = mapped_column(String(128))
    status: Mapped[str] = mapped_column(String(16))
    attempts: Mapped[int] = mapped_column(Integer, default=1)
    repair_attempts: Mapped[int] = mapped_column(Integer, default=0)
    latency_ms: Mapped[int] = mapped_column(Integer)
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0)
    usage_estimated: Mapped[bool] = mapped_column(Boolean, default=False)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    request_preview: Mapped[str] = mapped_column(Text, default="")
    response_preview: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class ClaimAudit(Base):
    __tablename__ = "claims_audit"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(32), ForeignKey("runs.id"), index=True)
    claim_id: Mapped[str] = mapped_column(String(48))
    sub_question_id: Mapped[str] = mapped_column(String(16))
    iteration: Mapped[int] = mapped_column(Integer)
    claim_text: Mapped[str] = mapped_column(Text)
    cited_source_ids_json: Mapped[str] = mapped_column(Text, default="[]")
    verdict: Mapped[str] = mapped_column(String(16), index=True)
    confidence: Mapped[float] = mapped_column(Float)
    evidence_source_id: Mapped[str | None] = mapped_column(String(16), nullable=True)
    evidence_quote: Mapped[str | None] = mapped_column(Text, nullable=True)
    evidence_char_start: Mapped[int | None] = mapped_column(Integer, nullable=True)
    evidence_char_end: Mapped[int | None] = mapped_column(Integer, nullable=True)
    quote_verified: Mapped[bool] = mapped_column(Boolean, default=False)
    downgraded: Mapped[bool] = mapped_column(Boolean, default=False)
    downgrade_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    rationale: Mapped[str] = mapped_column(Text, default="")
    included_in_report: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class SourceRecord(Base):
    __tablename__ = "sources"
    __table_args__ = (UniqueConstraint("run_id", "source_id", name="uq_run_source"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(32), ForeignKey("runs.id"), index=True)
    source_id: Mapped[str] = mapped_column(String(16))
    url: Mapped[str] = mapped_column(Text)
    title: Mapped[str] = mapped_column(Text)
    domain: Mapped[str] = mapped_column(String(255))
    authority_score: Mapped[float] = mapped_column(Float)
    authority_tier: Mapped[str] = mapped_column(String(8))
    authority_rule: Mapped[str] = mapped_column(Text)
    retrieval_tier: Mapped[str] = mapped_column(String(24))
    query: Mapped[str] = mapped_column(Text, default="")
    content: Mapped[str] = mapped_column(Text)
    content_hash: Mapped[str] = mapped_column(String(64), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class EvidenceRecord(Base):
    __tablename__ = "evidence_chunks"
    __table_args__ = (UniqueConstraint("run_id", "chunk_id", name="uq_run_chunk"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(32), ForeignKey("runs.id"), index=True)
    chunk_id: Mapped[str] = mapped_column(String(48))
    source_id: Mapped[str] = mapped_column(String(16))
    sub_question_id: Mapped[str | None] = mapped_column(String(16), nullable=True)
    char_start: Mapped[int] = mapped_column(Integer)
    char_end: Mapped[int] = mapped_column(Integer)
    authority_score: Mapped[float] = mapped_column(Float)
    text: Mapped[str] = mapped_column(Text)


class AuditEventRecord(Base):
    __tablename__ = "audit_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_id: Mapped[str] = mapped_column(String(32), unique=True)
    run_id: Mapped[str] = mapped_column(String(32), index=True)
    node: Mapped[str] = mapped_column(String(32))
    event_type: Mapped[str] = mapped_column(String(32), index=True)
    iteration: Mapped[int] = mapped_column(Integer, default=0)
    message: Mapped[str] = mapped_column(Text)
    payload_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class UploadedDocument(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    filename: Mapped[str] = mapped_column(String(255))
    content_type: Mapped[str] = mapped_column(String(64))
    content: Mapped[str] = mapped_column(Text)
    char_count: Mapped[int] = mapped_column(Integer)
    page_count: Mapped[int] = mapped_column(Integer, default=1)
    content_hash: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class FollowUpRecord(Base):
    __tablename__ = "followups"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(32), ForeignKey("runs.id"), index=True)
    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text)
    answerable: Mapped[bool] = mapped_column(Boolean)
    cited_source_ids_json: Mapped[str] = mapped_column(Text, default="[]")
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


# --- Append-only enforcement --------------------------------------------------
_IMMUTABLE_MODELS: tuple[type[Base], ...] = (AgentTrace, ClaimAudit, AuditEventRecord)


def _forbid_mutation(mapper: Any, connection: Any, target: Base) -> None:
    raise ImmutableRecordError(
        f"{type(target).__name__} rows are append-only and cannot be modified or deleted"
    )


for _model in _IMMUTABLE_MODELS:
    event.listen(_model, "before_update", _forbid_mutation)
    event.listen(_model, "before_delete", _forbid_mutation)


# --- Serialisation helpers ------------------------------------------------------
def _iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.isoformat()


def _row_to_dict(row: Base, *, exclude: Sequence[str] = ()) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for column in row.__table__.columns:
        if column.key in exclude:
            continue
        value = getattr(row, column.key)
        if isinstance(value, datetime):
            value = _iso(value)
        if column.key.endswith("_json") and isinstance(value, str):
            out[column.key.removesuffix("_json")] = json.loads(value)
            continue
        out[column.key] = value
    return out


class AuditStore:
    """Thread-safe facade over the SQLAlchemy session for the whole application."""

    def __init__(self, database_url: str) -> None:
        self.database_url = database_url
        self.engine = self._build_engine(database_url)
        Base.metadata.create_all(self.engine)
        self._session_factory = sessionmaker(self.engine, expire_on_commit=False)
        self._write_lock = threading.RLock()

    # -- engine ------------------------------------------------------------
    @staticmethod
    def _build_engine(database_url: str) -> Engine:
        if not database_url.startswith("sqlite"):
            return create_engine(database_url, pool_pre_ping=True)
        connect_args = {"check_same_thread": False, "timeout": 30}
        if database_url in ("sqlite://", "sqlite:///:memory:"):
            return create_engine(database_url, connect_args=connect_args, poolclass=StaticPool)
        db_path = database_url.split("///", 1)[-1]
        Path(db_path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        engine = create_engine(database_url, connect_args=connect_args)

        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(dbapi_conn: Any, _record: Any) -> None:
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

        return engine

    @contextmanager
    def session(self) -> Iterator[Session]:
        with self._write_lock, self._session_factory() as session:
            try:
                yield session
                session.commit()
            except Exception:
                session.rollback()
                raise

    # -- runs --------------------------------------------------------------
    def create_run(
        self, run_id: str, query: str, *, max_iterations: int, document_ids: list[str] | None = None
    ) -> None:
        with self.session() as s:
            s.add(
                ResearchRun(
                    id=run_id,
                    query=query,
                    status=RunStatus.PENDING.value,
                    max_iterations=max_iterations,
                    document_ids_json=json.dumps(document_ids or []),
                )
            )

    def update_run_status(self, run_id: str, status: RunStatus, *, error: str | None = None) -> None:
        with self.session() as s:
            run = s.get(ResearchRun, run_id)
            if run is None:
                raise RunNotFoundError(f"Run {run_id} not found")
            run.status = status.value
            if error is not None:
                run.error = error[:4000]
            if status in (RunStatus.COMPLETED, RunStatus.FAILED):
                run.completed_at = _now()

    def save_final_report(self, report: FinalReport) -> None:
        with self.session() as s:
            run = s.get(ResearchRun, report.run_id)
            if run is None:
                raise RunNotFoundError(f"Run {report.run_id} not found")
            run.report_json = report.model_dump_json()
            run.iterations = report.metrics.iterations
            run.grounding_rate = report.metrics.grounding_rate_final
            run.coverage_score = report.metrics.coverage_score

    def get_run(self, run_id: str) -> dict[str, Any]:
        with self.session() as s:
            run = s.get(ResearchRun, run_id)
            if run is None:
                raise RunNotFoundError(f"Run {run_id} not found")
            data = _row_to_dict(run, exclude=("report_json",))
            data["report"] = json.loads(run.report_json) if run.report_json else None
            return data

    def get_report(self, run_id: str) -> FinalReport | None:
        with self.session() as s:
            run = s.get(ResearchRun, run_id)
            if run is None:
                raise RunNotFoundError(f"Run {run_id} not found")
            return FinalReport.model_validate_json(run.report_json) if run.report_json else None

    def list_runs(self, limit: int = 25) -> list[dict[str, Any]]:
        with self.session() as s:
            rows = s.scalars(select(ResearchRun).order_by(ResearchRun.created_at.desc()).limit(limit))
            return [_row_to_dict(r, exclude=("report_json",)) for r in rows]

    # -- traces --------------------------------------------------------------
    def record_trace(self, trace: dict[str, Any]) -> None:
        with self.session() as s:
            s.add(AgentTrace(**trace))

    def list_traces(self, run_id: str) -> list[dict[str, Any]]:
        with self.session() as s:
            rows = s.scalars(
                select(AgentTrace).where(AgentTrace.run_id == run_id).order_by(AgentTrace.id)
            )
            return [_row_to_dict(r) for r in rows]

    def trace_summary(self, run_id: str) -> dict[str, Any]:
        with self.session() as s:
            row = s.execute(
                select(
                    func.count(AgentTrace.id),
                    func.coalesce(func.sum(AgentTrace.total_tokens), 0),
                    func.coalesce(func.sum(AgentTrace.prompt_tokens), 0),
                    func.coalesce(func.sum(AgentTrace.completion_tokens), 0),
                    func.coalesce(func.sum(AgentTrace.latency_ms), 0),
                ).where(AgentTrace.run_id == run_id)
            ).one()
            per_agent = s.execute(
                select(
                    AgentTrace.agent_name,
                    func.count(AgentTrace.id),
                    func.coalesce(func.sum(AgentTrace.total_tokens), 0),
                    func.coalesce(func.avg(AgentTrace.latency_ms), 0),
                )
                .where(AgentTrace.run_id == run_id)
                .group_by(AgentTrace.agent_name)
            ).all()
        return {
            "llm_calls": int(row[0]),
            "total_tokens": int(row[1]),
            "prompt_tokens": int(row[2]),
            "completion_tokens": int(row[3]),
            "total_latency_ms": int(row[4]),
            "per_agent": {
                name: {"calls": int(calls), "tokens": int(tokens), "avg_latency_ms": round(float(lat))}
                for name, calls, tokens, lat in per_agent
            },
        }

    # -- claims ----------------------------------------------------------------
    def record_claims(
        self,
        run_id: str,
        claims: Sequence[AtomicClaim],
        verifications: Sequence[VerificationResult],
        included_claim_ids: set[str],
    ) -> None:
        by_id = {v.claim_id: v for v in verifications}
        with self.session() as s:
            for claim in claims:
                verdict = by_id.get(claim.claim_id)
                if verdict is None:
                    continue
                s.add(
                    ClaimAudit(
                        run_id=run_id,
                        claim_id=claim.claim_id,
                        sub_question_id=claim.sub_question_id,
                        iteration=claim.iteration,
                        claim_text=claim.text,
                        cited_source_ids_json=json.dumps(claim.cited_source_ids),
                        verdict=verdict.verdict.value,
                        confidence=verdict.confidence,
                        evidence_source_id=verdict.evidence_source_id,
                        evidence_quote=verdict.evidence_quote,
                        evidence_char_start=verdict.evidence_char_start,
                        evidence_char_end=verdict.evidence_char_end,
                        quote_verified=verdict.quote_verified,
                        downgraded=verdict.downgraded,
                        downgrade_reason=verdict.downgrade_reason,
                        rationale=verdict.rationale,
                        included_in_report=claim.claim_id in included_claim_ids,
                    )
                )

    def list_claims(self, run_id: str) -> list[dict[str, Any]]:
        with self.session() as s:
            rows = s.scalars(
                select(ClaimAudit).where(ClaimAudit.run_id == run_id).order_by(ClaimAudit.id)
            )
            return [_row_to_dict(r) for r in rows]

    # -- sources & evidence ------------------------------------------------------
    def record_sources(self, run_id: str, sources: Sequence[SourceDocument]) -> None:
        with self.session() as s:
            existing = set(
                s.scalars(select(SourceRecord.source_id).where(SourceRecord.run_id == run_id))
            )
            for src in sources:
                if src.source_id in existing:
                    continue
                s.add(
                    SourceRecord(
                        run_id=run_id,
                        source_id=src.source_id,
                        url=src.url,
                        title=src.title,
                        domain=src.domain,
                        authority_score=src.authority.score,
                        authority_tier=src.authority.tier.value,
                        authority_rule=src.authority.rule,
                        retrieval_tier=src.retrieval_tier.value,
                        query=src.query,
                        content=src.content,
                        content_hash=src.content_hash,
                    )
                )

    def list_sources(self, run_id: str, *, include_content: bool = False) -> list[dict[str, Any]]:
        exclude = () if include_content else ("content",)
        with self.session() as s:
            rows = s.scalars(
                select(SourceRecord).where(SourceRecord.run_id == run_id).order_by(SourceRecord.source_id)
            )
            return [_row_to_dict(r, exclude=exclude) for r in rows]

    def record_evidence(self, run_id: str, chunks: Sequence[EvidenceChunk]) -> None:
        with self.session() as s:
            existing = set(
                s.scalars(select(EvidenceRecord.chunk_id).where(EvidenceRecord.run_id == run_id))
            )
            for chunk in chunks:
                if chunk.chunk_id in existing:
                    continue
                existing.add(chunk.chunk_id)
                s.add(
                    EvidenceRecord(
                        run_id=run_id,
                        chunk_id=chunk.chunk_id,
                        source_id=chunk.source_id,
                        sub_question_id=chunk.sub_question_id,
                        char_start=chunk.char_start,
                        char_end=chunk.char_end,
                        authority_score=chunk.authority_score,
                        text=chunk.text,
                    )
                )

    def list_evidence(self, run_id: str) -> list[EvidenceChunk]:
        with self.session() as s:
            sources = {
                r.source_id: r
                for r in s.scalars(select(SourceRecord).where(SourceRecord.run_id == run_id))
            }
            rows = s.scalars(
                select(EvidenceRecord).where(EvidenceRecord.run_id == run_id).order_by(EvidenceRecord.id)
            )
            out: list[EvidenceChunk] = []
            for r in rows:
                src = sources.get(r.source_id)
                out.append(
                    EvidenceChunk(
                        chunk_id=r.chunk_id,
                        source_id=r.source_id,
                        sub_question_id=r.sub_question_id,
                        text=r.text,
                        char_start=r.char_start,
                        char_end=r.char_end,
                        authority_score=r.authority_score,
                        url=src.url if src else "",
                        title=src.title if src else "",
                    )
                )
            return out

    # -- audit events ------------------------------------------------------------
    def record_audit_events(self, events: Sequence[AuditEvent]) -> None:
        if not events:
            return
        with self.session() as s:
            for ev in events:
                s.add(
                    AuditEventRecord(
                        event_id=ev.event_id,
                        run_id=ev.run_id,
                        node=ev.node,
                        event_type=ev.event_type.value,
                        iteration=ev.iteration,
                        message=ev.message,
                        payload_json=json.dumps(ev.payload, default=str),
                        created_at=ev.timestamp,
                    )
                )

    def list_audit_events(self, run_id: str) -> list[dict[str, Any]]:
        with self.session() as s:
            rows = s.scalars(
                select(AuditEventRecord)
                .where(AuditEventRecord.run_id == run_id)
                .order_by(AuditEventRecord.id)
            )
            return [_row_to_dict(r) for r in rows]

    # -- documents ---------------------------------------------------------------
    def save_document(
        self,
        doc_id: str,
        *,
        filename: str,
        content_type: str,
        content: str,
        page_count: int,
        content_hash: str,
    ) -> None:
        with self.session() as s:
            s.add(
                UploadedDocument(
                    id=doc_id,
                    filename=filename,
                    content_type=content_type,
                    content=content,
                    char_count=len(content),
                    page_count=page_count,
                    content_hash=content_hash,
                )
            )

    def find_document_by_hash(self, content_hash: str) -> dict[str, Any] | None:
        with self.session() as s:
            row = s.scalars(
                select(UploadedDocument).where(UploadedDocument.content_hash == content_hash)
            ).first()
            return _row_to_dict(row, exclude=("content",)) if row else None

    def get_documents(self, doc_ids: Sequence[str]) -> list[dict[str, Any]]:
        if not doc_ids:
            return []
        with self.session() as s:
            rows = s.scalars(select(UploadedDocument).where(UploadedDocument.id.in_(list(doc_ids))))
            return [_row_to_dict(r) for r in rows]

    def list_documents(self) -> list[dict[str, Any]]:
        with self.session() as s:
            rows = s.scalars(select(UploadedDocument).order_by(UploadedDocument.created_at.desc()))
            return [_row_to_dict(r, exclude=("content",)) for r in rows]

    # -- follow-ups --------------------------------------------------------------
    def record_followup(self, answer: FollowUpAnswer) -> None:
        with self.session() as s:
            s.add(
                FollowUpRecord(
                    run_id=answer.run_id,
                    question=answer.question,
                    answer=answer.answer,
                    answerable=answer.answerable,
                    cited_source_ids_json=json.dumps(answer.cited_source_ids),
                    confidence=answer.confidence,
                )
            )

    def list_followups(self, run_id: str) -> list[dict[str, Any]]:
        with self.session() as s:
            rows = s.scalars(
                select(FollowUpRecord).where(FollowUpRecord.run_id == run_id).order_by(FollowUpRecord.id)
            )
            return [_row_to_dict(r) for r in rows]
