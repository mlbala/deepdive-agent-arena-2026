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
guard raises `ImmutableRecordError` on any UPDATE or DELETE attempt. On top of that,
`audit_events` is tamper-evident: every row stores `hash = sha256(prev_hash + canonical
event)` per run, so an edit made outside the ORM (raw SQL) breaks the chain and is
reported by `verify_audit_chain`. The only sanctioned deletion path is the
retention purge (`purge_runs_older_than`), which removes whole runs.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
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
    delete,
    event,
    func,
    inspect,
    select,
    text,
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
    created_at_iso: Mapped[str | None] = mapped_column(String(40), nullable=True)
    prev_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    hash: Mapped[str | None] = mapped_column(String(64), nullable=True)


GENESIS_HASH = "0" * 64


def chain_hash(prev_hash: str, row: dict[str, Any]) -> str:
    """sha256 over the previous hash and the canonical, exactly-stored event fields."""
    canonical = json.dumps(
        [row["event_id"], row["run_id"], row["node"], row["event_type"], row["iteration"],
         row["message"], row["payload_json"], row["created_at_iso"]],
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256((prev_hash + canonical).encode("utf-8")).hexdigest()


class UploadedDocument(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    filename: Mapped[str] = mapped_column(String(255))
    content_type: Mapped[str] = mapped_column(String(64))
    content: Mapped[str] = mapped_column(Text)
    char_count: Mapped[int] = mapped_column(Integer)
    page_count: Mapped[int] = mapped_column(Integer, default=1)
    content_hash: Mapped[str] = mapped_column(String(64))
    guardrails_json: Mapped[str] = mapped_column(Text, default="[]")
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
        self._migrate()
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

    def _migrate(self) -> None:
        """Additive, idempotent column migrations for databases created by older versions."""
        additions = {
            "audit_events": {"created_at_iso": "VARCHAR(40)", "prev_hash": "VARCHAR(64)", "hash": "VARCHAR(64)"},
            "documents": {"guardrails_json": "TEXT DEFAULT '[]'"},
        }
        inspector = inspect(self.engine)
        with self.engine.begin() as conn:
            for table, columns in additions.items():
                existing = {c["name"] for c in inspector.get_columns(table)}
                for name, ddl in columns.items():
                    if name not in existing:
                        conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}"))

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

    def ensure_run(
        self, run_id: str, query: str, *, max_iterations: int, document_ids: list[str] | None = None
    ) -> None:
        """Create the run row if it does not exist yet (idempotent)."""
        with self.session() as s:
            if s.get(ResearchRun, run_id) is not None:
                return
        self.create_run(run_id, query, max_iterations=max_iterations, document_ids=document_ids)

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
            heads: dict[str, str] = {}
            for ev in events:
                if ev.run_id not in heads:
                    heads[ev.run_id] = s.scalar(
                        select(AuditEventRecord.hash)
                        .where(AuditEventRecord.run_id == ev.run_id, AuditEventRecord.hash.is_not(None))
                        .order_by(AuditEventRecord.id.desc())
                        .limit(1)
                    ) or GENESIS_HASH
                row = {
                    "event_id": ev.event_id,
                    "run_id": ev.run_id,
                    "node": ev.node,
                    "event_type": ev.event_type.value,
                    "iteration": ev.iteration,
                    "message": ev.message,
                    "payload_json": json.dumps(ev.payload, default=str, sort_keys=True),
                    "created_at_iso": _iso(ev.timestamp),
                }
                digest = chain_hash(heads[ev.run_id], row)
                s.add(
                    AuditEventRecord(
                        **row, created_at=ev.timestamp, prev_hash=heads[ev.run_id], hash=digest
                    )
                )
                heads[ev.run_id] = digest

    def audit_chain_head(self, run_id: str) -> str | None:
        with self.session() as s:
            return s.scalar(
                select(AuditEventRecord.hash)
                .where(AuditEventRecord.run_id == run_id, AuditEventRecord.hash.is_not(None))
                .order_by(AuditEventRecord.id.desc())
                .limit(1)
            )

    def verify_audit_chain(self, run_id: str) -> dict[str, Any]:
        """Recompute the run's hash chain; any edited, inserted or removed row breaks it."""
        with self.session() as s:
            rows = list(
                s.scalars(select(AuditEventRecord).where(AuditEventRecord.run_id == run_id).order_by(AuditEventRecord.id))
            )
        prev = GENESIS_HASH
        legacy = 0
        for index, r in enumerate(rows):
            if r.hash is None:
                legacy += 1
                continue
            row = {
                "event_id": r.event_id, "run_id": r.run_id, "node": r.node, "event_type": r.event_type,
                "iteration": r.iteration, "message": r.message, "payload_json": r.payload_json,
                "created_at_iso": r.created_at_iso,
            }
            if r.prev_hash != prev or chain_hash(prev, row) != r.hash:
                return {
                    "run_id": run_id, "valid": False, "events": len(rows), "verified": index,
                    "broken_at": r.event_id, "head": rows[-1].hash, "legacy_unchained": legacy,
                }
            prev = r.hash
        return {
            "run_id": run_id, "valid": True, "events": len(rows), "verified": len(rows) - legacy,
            "broken_at": None, "head": prev if rows else None, "legacy_unchained": legacy,
        }

    # -- retention (the only sanctioned deletion path) ------------------------------
    def fleet_stats(self) -> dict[str, Any]:
        return fleet_stats(self)

    def run_timeline(self, run_id: str) -> dict[str, Any]:
        self.get_run(run_id)  # 404 semantics
        return run_timeline(self, run_id)

    def purge_runs_older_than(self, days: int) -> dict[str, int]:
        """Delete whole runs older than `days` across every table (bypasses row guards by design)."""
        cutoff = datetime.now(UTC) - timedelta(days=days)
        with self._write_lock, self.engine.begin() as conn:
            run_ids = [r[0] for r in conn.execute(select(ResearchRun.id).where(ResearchRun.created_at < cutoff))]
            if not run_ids:
                return {"runs": 0}
            counts: dict[str, int] = {}
            for model in (FollowUpRecord, EvidenceRecord, SourceRecord, ClaimAudit, AuditEventRecord, AgentTrace):
                table = Base.metadata.tables[model.__tablename__]
                result = conn.execute(delete(table).where(table.c.run_id.in_(run_ids)))
                counts[model.__tablename__] = result.rowcount or 0
            runs = Base.metadata.tables[ResearchRun.__tablename__]
            conn.execute(delete(runs).where(runs.c.id.in_(run_ids)))
            counts["runs"] = len(run_ids)
            return counts

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
        guardrails: Sequence[dict[str, Any]] = (),
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
                    guardrails_json=json.dumps(list(guardrails)),
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


# ---------------------------------------------------------------------------
# Architecture analytics (fleet stats + per-run timeline)
# ---------------------------------------------------------------------------
def _parse_ts(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def _reason_key(reason: str | None) -> str:
    text = (reason or "").strip()
    for sep in (" (", ":", ";"):
        if sep in text:
            text = text.split(sep, 1)[0]
    words = text.split()
    return " ".join(w for w in words if not w.startswith("SRC-"))[:80] or "unspecified"


def _node_spans(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    open_nodes: dict[str, dict[str, Any]] = {}
    spans: list[dict[str, Any]] = []
    for ev in events:
        ts = _parse_ts(ev.get("created_at_iso") or ev.get("created_at"))
        if ts is None:
            continue
        if ev["event_type"] == "node_started":
            open_nodes[ev["node"]] = {"node": ev["node"], "iteration": ev.get("iteration", 0), "start": ts, "message": ev["message"]}
        elif ev["event_type"] == "node_completed" and ev["node"] in open_nodes:
            span = open_nodes.pop(ev["node"])
            span["end"] = ts
            spans.append(span)
    return spans


def fleet_stats(store: AuditStore) -> dict[str, Any]:
    """Aggregate agent behaviour across every run in the store."""
    with store.session() as s:
        runs = s.execute(select(ResearchRun.status, ResearchRun.iterations, ResearchRun.grounding_rate, ResearchRun.coverage_score)).all()
        traces = s.execute(select(AgentTrace.agent_name, AgentTrace.operation, AgentTrace.latency_ms, AgentTrace.total_tokens, AgentTrace.status)).all()
        claims = s.execute(select(ClaimAudit.verdict, ClaimAudit.downgraded, ClaimAudit.downgrade_reason, ClaimAudit.included_in_report)).all()
        events: list[dict[str, Any]] = [
            {"run_id": r.run_id, "node": r.node, "event_type": r.event_type, "iteration": r.iteration,
             "message": r.message, "created_at_iso": r.created_at_iso or _iso(r.created_at), "payload_json": r.payload_json}
            for r in s.scalars(
                select(AuditEventRecord)
                .where(AuditEventRecord.event_type.in_(["node_started", "node_completed", "guardrail_finding"]))
                .order_by(AuditEventRecord.id)
            )
        ]

    by_status: dict[str, int] = {}
    for status, *_ in runs:
        by_status[status] = by_status.get(status, 0) + 1
    completed = [r for r in runs if r[0] == RunStatus.COMPLETED.value]
    grounding = [r[2] for r in completed if r[2] is not None]
    coverage = [r[3] for r in completed if r[3] is not None]

    agents: dict[str, dict[str, Any]] = {}
    for name, operation, latency, tokens, status in traces:
        a = agents.setdefault(name, {"agent": name, "operation": operation, "calls": 0, "latencies": [], "tokens": 0, "ok": 0, "repaired": 0, "errors": 0})
        a["calls"] += 1
        a["latencies"].append(float(latency or 0))
        a["tokens"] += int(tokens or 0)
        a["ok"] += status in ("success", "repaired")
        a["repaired"] += status == "repaired"
        a["errors"] += status in ("error", "invalid_output")
    agent_rows = [
        {
            "agent": a["agent"], "operation": a["operation"], "calls": a["calls"],
            "avg_latency_ms": round(sum(a["latencies"]) / a["calls"]),
            "p95_latency_ms": round(_percentile(a["latencies"], 0.95)),
            "tokens": a["tokens"], "avg_tokens": round(a["tokens"] / a["calls"]),
            "success_rate": round(a["ok"] / a["calls"], 4), "repaired": a["repaired"], "errors": a["errors"],
        }
        for a in agents.values()
    ]
    agent_rows.sort(key=lambda r: -r["tokens"])

    verdicts: dict[str, int] = {}
    reasons: dict[str, int] = {}
    published = 0
    for verdict, downgraded, reason, included in claims:
        verdicts[verdict] = verdicts.get(verdict, 0) + 1
        published += bool(included)
        if downgraded:
            key = _reason_key(reason)
            reasons[key] = reasons.get(key, 0) + 1

    guardrails: dict[str, int] = {}
    by_run: dict[str, list[dict[str, Any]]] = {}
    for ev in events:
        if ev["event_type"] == "guardrail_finding":
            payload = json.loads(str(ev["payload_json"] or "{}"))
            key = f"{payload.get('guard', '?')}: {payload.get('kind', '?')}"
            guardrails[key] = guardrails.get(key, 0) + 1
        else:
            by_run.setdefault(str(ev["run_id"]), []).append(ev)
    durations: dict[str, list[float]] = {}
    for run_events in by_run.values():
        for span in _node_spans(run_events):
            durations.setdefault(span["node"], []).append((span["end"] - span["start"]).total_seconds() * 1000)
    node_rows = {
        node: {"runs": len(vals), "avg_ms": round(sum(vals) / len(vals)), "p95_ms": round(_percentile(vals, 0.95))}
        for node, vals in durations.items()
    }
    return {
        "runs": {
            "total": len(runs), "by_status": by_status,
            "loop_back_rate": round(sum(1 for r in completed if (r[1] or 0) > 1) / len(completed), 4) if completed else None,
            "block_rate": round(by_status.get("blocked", 0) / len(runs), 4) if runs else None,
            "avg_grounding": round(sum(grounding) / len(grounding), 4) if grounding else None,
            "avg_coverage": round(sum(coverage) / len(coverage), 4) if coverage else None,
        },
        "agents": agent_rows,
        "nodes": node_rows,
        "claims": {"total": len(claims), "published": published, "by_verdict": verdicts,
                   "top_downgrade_reasons": sorted(reasons.items(), key=lambda kv: -kv[1])[:6]},
        "guardrails": dict(sorted(guardrails.items(), key=lambda kv: -kv[1])),
    }


def run_timeline(store: AuditStore, run_id: str) -> dict[str, Any]:
    """Node spans, Mesh call spans and decision markers on one time axis (ms since run start)."""
    events = store.list_audit_events(run_id)
    traces = store.list_traces(run_id)
    if not events:
        return {"run_id": run_id, "duration_ms": 0, "spans": [], "llm_calls": [], "markers": [], "lanes": []}
    t0 = _parse_ts(events[0].get("created_at_iso") or events[0].get("created_at")) or datetime.now(UTC)

    def rel(ts: datetime | None) -> int:
        return max(0, round(((ts or t0) - t0).total_seconds() * 1000))

    spans = [
        {"node": sp["node"], "iteration": sp["iteration"], "start_ms": rel(sp["start"]), "end_ms": rel(sp["end"]),
         "duration_ms": rel(sp["end"]) - rel(sp["start"]), "message": sp["message"]}
        for sp in _node_spans(events)
    ]
    calls = []
    for tr in traces:
        end = _parse_ts(tr.get("created_at"))
        if end is None:
            continue
        end_ms = rel(end)
        calls.append({"agent": tr["agent_name"], "operation": tr["operation"], "model": tr["model"], "status": tr["status"],
                      "start_ms": max(0, end_ms - int(tr["latency_ms"] or 0)), "end_ms": end_ms,
                      "latency_ms": tr["latency_ms"], "tokens": tr["total_tokens"]})
    marker_types = {"route_decision", "loop_back", "guardrail_blocked", "budget_guard", "run_completed", "run_blocked", "run_failed"}
    markers = [
        {"type": ev["event_type"], "t_ms": rel(_parse_ts(ev.get("created_at_iso") or ev.get("created_at"))),
         "node": ev["node"], "message": ev["message"], "route": (ev.get("payload") or {}).get("route")}
        for ev in events if ev["event_type"] in marker_types
    ]
    last = max([s["end_ms"] for s in spans] + [c["end_ms"] for c in calls] + [m["t_ms"] for m in markers] + [0])
    lanes = list(dict.fromkeys(s["node"] for s in spans))
    return {"run_id": run_id, "duration_ms": last, "spans": spans, "llm_calls": calls, "markers": markers, "lanes": lanes}
