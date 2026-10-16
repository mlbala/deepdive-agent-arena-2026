"""ReportLab executive PDF report with a full verification audit appendix."""

from __future__ import annotations

import io
from datetime import UTC, datetime
from typing import Any
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from src.core.exceptions import ReportExportError
from src.schemas.domain import FinalReport

INK = colors.HexColor("#0f172a")
MUTED = colors.HexColor("#475569")
ACCENT = colors.HexColor("#4f46e5")
RULE = colors.HexColor("#cbd5e1")
HEADER_BG = colors.HexColor("#e0e7ff")
VERDICT_COLORS = {
    "SUPPORTED": colors.HexColor("#047857"),
    "CONTRADICTED": colors.HexColor("#b91c1c"),
    "UNSUPPORTED": colors.HexColor("#b45309"),
}


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("t", parent=base["Title"], fontSize=20, leading=24, textColor=INK, alignment=TA_LEFT),
        "kicker": ParagraphStyle("k", parent=base["Normal"], fontSize=8, textColor=ACCENT, leading=10, spaceAfter=2),
        "h1": ParagraphStyle("h1", parent=base["Heading1"], fontSize=14, leading=18, textColor=INK, spaceBefore=10, spaceAfter=6),
        "h2": ParagraphStyle("h2", parent=base["Heading2"], fontSize=11, leading=14, textColor=INK, spaceBefore=8, spaceAfter=4),
        "body": ParagraphStyle("b", parent=base["BodyText"], fontSize=9.5, leading=13.5, textColor=INK),
        "muted": ParagraphStyle("m", parent=base["BodyText"], fontSize=8, leading=11, textColor=MUTED),
        "cell": ParagraphStyle("c", parent=base["BodyText"], fontSize=7.5, leading=9.5, textColor=INK),
        "cellb": ParagraphStyle("cb", parent=base["BodyText"], fontSize=7.5, leading=9.5, textColor=INK, fontName="Helvetica-Bold"),
        "bullet": ParagraphStyle("bl", parent=base["BodyText"], fontSize=9.5, leading=13.5, leftIndent=10, bulletIndent=0, textColor=INK),
    }


def _p(text: Any, style: ParagraphStyle) -> Paragraph:
    return Paragraph(escape(str(text if text is not None else "")), style)


def _table(rows: list[list[Any]], widths: list[float], *, zebra: bool = True) -> Table:
    table = Table(rows, colWidths=widths, repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), HEADER_BG),
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, ACCENT),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LINEBELOW", (0, 1), (-1, -1), 0.25, RULE),
    ]
    if zebra:
        for r in range(2, len(rows), 2):
            style.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#f8fafc")))
    table.setStyle(TableStyle(style))
    return table


def _pct(value: float) -> str:
    return f"{value * 100:.0f}%"


def build_report_pdf(
    report: FinalReport,
    *,
    claims: list[dict[str, Any]] | None = None,
    trace_summary: dict[str, Any] | None = None,
) -> bytes:
    """Render the executive report + audit appendix to PDF bytes."""
    try:
        return _render(report, claims or [], trace_summary or {})
    except Exception as exc:  # reportlab raises assorted layout errors
        raise ReportExportError(f"PDF export failed: {exc}") from exc


def _render(report: FinalReport, claims: list[dict[str, Any]], trace_summary: dict[str, Any]) -> bytes:
    s = _styles()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=16 * mm,
        title=report.title,
        author="DeepDive (Agent Arena 2026)",
        subject=report.query,
    )
    width = doc.width
    m = report.metrics
    story: list[Any] = []

    # -- cover -------------------------------------------------------------
    story += [
        _p("DEEPDIVE · SELF-VERIFYING RESEARCH REPORT", s["kicker"]),
        _p(report.title, s["title"]),
        _p(f"Research question: {report.query}", s["body"]),
        Spacer(1, 4),
        _p(
            f"Run {report.run_id} · generated {report.generated_at.astimezone(UTC):%Y-%m-%d %H:%M UTC} · "
            f"{m.iterations} verification iteration(s), {m.loop_backs} loop-back(s)",
            s["muted"],
        ),
        Spacer(1, 8),
    ]
    metric_rows = [
        [_p("Metric", s["cellb"]), _p("Value", s["cellb"]), _p("Metric", s["cellb"]), _p("Value", s["cellb"])],
        [_p("Grounding rate (first draft)", s["cell"]), _p(_pct(m.grounding_rate_pre), s["cell"]),
         _p("Grounding rate (final iteration)", s["cell"]), _p(_pct(m.grounding_rate_final), s["cell"])],
        [_p("Sub-question coverage", s["cell"]), _p(_pct(m.coverage_score), s["cell"]),
         _p("Citation precision", s["cell"]), _p(_pct(m.citation_precision), s["cell"])],
        [_p("Evidence quotes located verbatim", s["cell"]), _p(_pct(m.quote_verification_rate), s["cell"]),
         _p("Report sentences with citations", s["cell"]), _p(_pct(m.report_citation_coverage), s["cell"])],
        [_p("Claims kept / removed", s["cell"]), _p(f"{m.claims_in_report} / {m.claims_removed}", s["cell"]),
         _p("Contradictions surfaced", s["cell"]), _p(m.contradictions, s["cell"])],
        [_p("Sources indexed / cited", s["cell"]), _p(f"{m.sources_indexed} / {m.sources_cited}", s["cell"]),
         _p("Mesh LLM calls / tokens", s["cell"]), _p(f"{m.llm_calls} / {m.total_tokens:,}", s["cell"])],
    ]
    story += [_table(metric_rows, [width * 0.32, width * 0.18, width * 0.32, width * 0.18], zebra=False), Spacer(1, 10)]

    # -- body --------------------------------------------------------------
    story += [_p("Executive summary", s["h1"]), _p(report.executive_summary, s["body"])]
    if report.key_findings:
        story.append(_p("Key findings", s["h1"]))
        story += [Paragraph(escape(f), s["bullet"], bulletText="•") for f in report.key_findings]
    story.append(_p("Detailed findings", s["h1"]))
    for section in report.sections:
        story.append(
            KeepTogether(
                [
                    _p(f"{section.sub_question_id} · {section.heading}", s["h2"]),
                    _p(section.body, s["body"]),
                    _p(
                        f"{len(section.claim_ids)} verified claim(s) · mean verifier confidence "
                        f"{section.confidence:.2f} · sources {', '.join(section.cited_source_ids) or 'none'}",
                        s["muted"],
                    ),
                ]
            )
        )

    story.append(_p("Contradiction radar", s["h1"]))
    if report.contradictions:
        rows = [[_p(h, s["cellb"]) for h in ("ID", "Topic", "Position A", "Position B", "Severity / resolution")]]
        for c in report.contradictions:
            rows.append(
                [
                    _p(c.contradiction_id, s["cell"]),
                    _p(c.topic, s["cell"]),
                    _p(f"[{c.source_a}] {c.statement_a}", s["cell"]),
                    _p(f"[{c.source_b}] {c.statement_b}", s["cell"]),
                    _p(f"{c.severity.value.upper()} · {c.resolution_hint}", s["cell"]),
                ]
            )
        story.append(_table(rows, [width * 0.08, width * 0.17, width * 0.27, width * 0.27, width * 0.21]))
    else:
        story.append(_p("No cross-source contradictions survived quote validation.", s["body"]))

    if report.limitations:
        story.append(_p("Limitations", s["h1"]))
        story += [Paragraph(escape(item), s["bullet"], bulletText="•") for item in report.limitations]

    story.append(_p("Sources", s["h1"]))
    rows = [[_p(h, s["cellb"]) for h in ("ID", "Title / URL", "Tier", "Authority", "Retrieval", "Cited")]]
    for src in report.sources:
        rows.append(
            [
                _p(src.source_id, s["cell"]),
                _p(f"{src.title} · {src.url}", s["cell"]),
                _p(src.tier.value.replace("_", " "), s["cell"]),
                _p(f"{src.authority_score:.2f}", s["cell"]),
                _p(src.retrieval_tier.value, s["cell"]),
                _p("yes" if src.cited_in_report else "no", s["cell"]),
            ]
        )
    story.append(_table(rows, [width * 0.09, width * 0.51, width * 0.1, width * 0.1, width * 0.12, width * 0.08]))

    # -- appendix ----------------------------------------------------------
    story += [PageBreak(), _p("Appendix A · Claim verification audit", s["h1"])]
    story.append(
        _p(
            "Every atomic claim drafted in every iteration, its entailment verdict against its cited source, "
            "and whether the verbatim evidence quote was located in that source.",
            s["muted"],
        )
    )
    rows = [[_p(h, s["cellb"]) for h in ("Claim", "It.", "Verdict", "Conf.", "Evidence", "Quote", "In report")]]
    for c in claims:
        verdict = str(c.get("verdict", ""))
        verdict_style = ParagraphStyle("v", parent=s["cellb"], textColor=VERDICT_COLORS.get(verdict, INK))
        evidence = c.get("evidence_quote") or c.get("downgrade_reason") or c.get("rationale") or ""
        rows.append(
            [
                _p(f"{c.get('claim_id')}: {c.get('claim_text')}", s["cell"]),
                _p(c.get("iteration"), s["cell"]),
                _p(verdict, verdict_style),
                _p(f"{float(c.get('confidence') or 0):.2f}", s["cell"]),
                _p(f"[{c.get('evidence_source_id') or '-'}] {evidence}", s["cell"]),
                _p("verified" if c.get("quote_verified") else "not found", s["cell"]),
                _p("yes" if c.get("included_in_report") else "no", s["cell"]),
            ]
        )
    story.append(
        _table(rows, [width * 0.3, width * 0.05, width * 0.12, width * 0.07, width * 0.3, width * 0.08, width * 0.08])
    )

    if report.removed_claims:
        story.append(_p("Appendix B · Claims stripped from the report", s["h1"]))
        rows = [[_p(h, s["cellb"]) for h in ("Claim", "Verdict", "Reason")]]
        for r in report.removed_claims:
            rows.append([_p(f"{r.claim_id}: {r.text}", s["cell"]), _p(r.verdict.value, s["cell"]), _p(r.reason, s["cell"])])
        story.append(_table(rows, [width * 0.45, width * 0.15, width * 0.4]))

    per_agent = trace_summary.get("per_agent", {})
    if per_agent:
        story.append(_p("Appendix C · Mesh API telemetry", s["h1"]))
        rows = [[_p(h, s["cellb"]) for h in ("Agent", "Calls", "Tokens", "Avg latency (ms)")]]
        for agent, stats in sorted(per_agent.items()):
            rows.append(
                [_p(agent, s["cell"]), _p(stats["calls"], s["cell"]), _p(f"{stats['tokens']:,}", s["cell"]),
                 _p(stats["avg_latency_ms"], s["cell"])]
            )
        story.append(_table(rows, [width * 0.4, width * 0.2, width * 0.2, width * 0.2]))

    def _footer(canvas: Any, _doc: Any) -> None:
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(MUTED)
        canvas.drawString(18 * mm, 9 * mm, f"DeepDive · {report.run_id} · all LLM reasoning via Mesh API")
        canvas.drawRightString(A4[0] - 18 * mm, 9 * mm, f"Page {canvas.getPageNumber()}")
        canvas.restoreState()

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return buf.getvalue()


def export_filename(report: FinalReport) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%d")
    return f"deepdive-{report.run_id}-{stamp}.pdf"
