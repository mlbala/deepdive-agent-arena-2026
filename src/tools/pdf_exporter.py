"""ReportLab executive PDF: a designed research report with a full verification and audit appendix.

Layout: a cover with the verification seal and headline metrics; the executive summary; how the
report was produced (agent workflow diagram, the Master Approver's plan and round decisions); a
verification dashboard (charts); detailed findings with confidence bars; the contradiction radar;
sources with authority bars; methodology; and appendices (claim audit, stripped claims, the agents'
group chat, governance and the hash-chained audit trail, Mesh API telemetry).

Only the built-in Helvetica family is used (WinAnsi encoding), so text is normalised to cp1252 glyphs.
"""

from __future__ import annotations

import io
from collections import Counter
from datetime import UTC, datetime
from typing import Any
from xml.sax.saxutils import escape

from reportlab.graphics.shapes import Circle, Drawing, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    CondPageBreak,
    Flowable,
    Frame,
    KeepTogether,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

from src.core.exceptions import ReportExportError
from src.schemas.domain import FinalReport

# ------------------------------------------------------------------ design tokens
INK = colors.HexColor("#0f172a")
BODY = colors.HexColor("#1e293b")
MUTED = colors.HexColor("#64748b")
SUBTLE = colors.HexColor("#94a3b8")
RULE = colors.HexColor("#e2e8f0")
PAPER = colors.HexColor("#f8fafc")
NAVY = colors.HexColor("#0b1530")
NAVY_2 = colors.HexColor("#1e2a5a")
ACCENT = colors.HexColor("#4f46e5")  # Master Approver indigo
ACCENT_SOFT = colors.HexColor("#eef2ff")
BLUE = colors.HexColor("#2a78d6")  # series 1
ORANGE = colors.HexColor("#eb6834")  # series 2
GOOD = colors.HexColor("#15803d")
GOOD_SOFT = colors.HexColor("#ecfdf5")
WARN = colors.HexColor("#b45309")
WARN_SOFT = colors.HexColor("#fff7ed")
BAD = colors.HexColor("#b91c1c")
BAD_SOFT = colors.HexColor("#fef2f2")
VERDICT_COLORS = {"SUPPORTED": GOOD, "CONTRADICTED": BAD, "UNSUPPORTED": WARN}
TIER_COLORS = {"TIER_1": GOOD, "TIER_2": BLUE, "TIER_3": WARN}
AGENT_COLORS = {
    "input_guard": colors.HexColor("#f59e0b"), "planner": colors.HexColor("#a855f7"), "master": ACCENT,
    "researcher": colors.HexColor("#0891b2"), "drafter": colors.HexColor("#db2777"), "verifier": colors.HexColor("#16a34a"),
    "synthesizer": colors.HexColor("#2563eb"), "output_guard": colors.HexColor("#f59e0b"), "quality_gate": MUTED,
    "approver": ACCENT, "lead": ACCENT,
}
AGENT_NAMES = {
    "input_guard": "Input shield", "planner": "Planner", "master": "Master Approver", "researcher": "Researcher",
    "drafter": "Drafter", "verifier": "Verifier", "synthesizer": "Synthesizer", "output_guard": "Output shield",
    "quality_gate": "Quality gate", "approver": "Approver", "lead": "Master Approver",
}
PAGE_W, PAGE_H = A4
MARGIN = 18 * mm
COVER_BAND = 108 * mm
_GLYPHS = {  # glyphs outside WinAnsi that commonly appear in model output
    "\u2265": ">=", "\u2264": "<=", "\u2192": "->", "\u2190": "<-", "\u21c4": "<->", "\u2713": "OK", "\u2717": "x",
    "\u2715": "x", "\u2248": "~", "\u2212": "-", "\u2009": " ", "\u202f": " ",
}


def _clean(text: Any) -> str:
    out = str(text if text is not None else "")
    for k, v in _GLYPHS.items():
        out = out.replace(k, v)
    return out.encode("cp1252", "replace").decode("cp1252")


def _x(text: Any) -> str:
    return escape(_clean(text))


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    def ps(name: str, **kw: Any) -> ParagraphStyle:
        return ParagraphStyle(name, parent=base["BodyText"], **({"fontName": "Helvetica", "textColor": BODY} | kw))
    return {
        "cover_kicker": ps("ck", fontSize=8.5, leading=11, textColor=colors.HexColor("#a5b4fc"), fontName="Helvetica-Bold"),
        "cover_title": ps("ct", fontSize=25, leading=29, textColor=colors.white, fontName="Helvetica-Bold", alignment=TA_LEFT),
        "cover_q": ps("cq", fontSize=11.5, leading=15.5, textColor=colors.HexColor("#cbd5e1")),
        "cover_meta": ps("cm", fontSize=8.5, leading=11.5, textColor=colors.HexColor("#94a3b8")),
        "h2": ps("h2", fontSize=11.5, leading=15, textColor=INK, fontName="Helvetica-Bold", spaceBefore=8, spaceAfter=4),
        "body": ps("b", fontSize=9.6, leading=14.2),
        "master": ps("l", fontSize=10.6, leading=15.8, textColor=INK),
        "muted": ps("m", fontSize=8, leading=11, textColor=MUTED),
        "small": ps("s", fontSize=7.6, leading=10, textColor=MUTED),
        "cell": ps("c", fontSize=7.5, leading=9.6),
        "cellb": ps("cb", fontSize=7.5, leading=9.6, fontName="Helvetica-Bold", textColor=INK),
        "cellh": ps("chd", fontSize=7, leading=9, fontName="Helvetica-Bold", textColor=MUTED),
        "tile_k": ps("tk", fontSize=7, leading=9, fontName="Helvetica-Bold", textColor=MUTED),
        "tile_v": ps("tv", fontSize=19, leading=22, fontName="Helvetica-Bold", textColor=INK),
        "tile_s": ps("ts", fontSize=7, leading=9, textColor=MUTED),
        "finding": ps("f", fontSize=9.8, leading=14.2, textColor=INK),
        "quote": ps("q", fontSize=9.6, leading=14, textColor=INK, leftIndent=10),
        "chat_who": ps("cw", fontSize=8, leading=10, fontName="Helvetica-Bold", textColor=INK),
        "chat": ps("cht", fontSize=8, leading=11),
    }


def _p(text: Any, style: ParagraphStyle) -> Paragraph:
    return Paragraph(_x(text), style)


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.0f}%"


# ------------------------------------------------------------------ custom flowables
class SectionHeader(Flowable):
    """Numbered section heading with an accent block and a hairline rule."""

    def __init__(self, number: str, title: str, subtitle: str = "") -> None:
        super().__init__()
        self.number, self.title, self.subtitle = number, title, subtitle
        self.height = 30 if subtitle else 22

    def wrap(self, avail_w: float, avail_h: float) -> tuple[float, float]:
        self.width = avail_w
        return avail_w, self.height + 8

    def draw(self) -> None:
        c = self.canv
        top = self.height + 4
        c.setFillColor(ACCENT)
        c.roundRect(0, top - 17, 22, 17, 4, stroke=0, fill=1)
        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 8.5)
        c.drawCentredString(11, top - 12, self.number)
        c.setFillColor(INK)
        c.setFont("Helvetica-Bold", 14.5)
        c.drawString(30, top - 13.5, _clean(self.title))
        if self.subtitle:
            c.setFillColor(MUTED)
            c.setFont("Helvetica", 8.3)
            c.drawString(30, top - 26, _clean(self.subtitle))
        c.setStrokeColor(RULE)
        c.setLineWidth(0.6)
        c.line(0, 2, self.width, 2)


class Bar(Flowable):
    """Inline horizontal gauge: value 0..1 with an optional threshold marker."""

    def __init__(self, value: float, *, width: float = 60, height: float = 5, color: Any = BLUE, target: float | None = None) -> None:
        super().__init__()
        self.value, self.w, self.h, self.color, self.target = max(0.0, min(1.0, value)), width, height, color, target

    def wrap(self, avail_w: float, avail_h: float) -> tuple[float, float]:
        return self.w, self.h + 2

    def draw(self) -> None:
        c = self.canv
        c.setFillColor(RULE)
        c.roundRect(0, 1, self.w, self.h, self.h / 2, stroke=0, fill=1)
        if self.value > 0:
            c.setFillColor(self.color)
            c.roundRect(0, 1, max(self.h, self.w * self.value), self.h, self.h / 2, stroke=0, fill=1)
        if self.target is not None:
            c.setStrokeColor(INK)
            c.setLineWidth(1.1)
            x = self.w * self.target
            c.line(x, -1, x, self.h + 3)


def _table(rows: list[list[Any]], widths: list[float], *, zebra: bool = True, header: bool = True) -> Table:
    table = Table(rows, colWidths=widths, repeatRows=1 if header else 0)
    style: list[Any] = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LINEBELOW", (0, 1 if header else 0), (-1, -1), 0.3, RULE),
    ]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), PAPER), ("LINEBELOW", (0, 0), (-1, 0), 0.8, ACCENT)]
    if zebra:
        for r in range(2 if header else 1, len(rows), 2):
            style.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#fbfcfe")))
    table.setStyle(TableStyle(style))
    return table


def _tiles(items: list[tuple[str, str, str, Any]], width: float, s: dict[str, ParagraphStyle], per_row: int = 4) -> Table:
    """Metric tiles: (label, value, sub, background)."""
    cells = [[_p(k.upper(), s["tile_k"]), _p(v, s["tile_v"]), _p(sub, s["tile_s"])] for k, v, sub, _ in items]
    rows = [cells[i:i + per_row] for i in range(0, len(cells), per_row)]
    while len(rows[-1]) < per_row:
        rows[-1].append([""])
    gap = 6
    col_w = (width - gap * (per_row - 1)) / per_row
    data = [[_tile_cell(cell, col_w) for cell in row] for row in rows]
    t = Table(data, colWidths=[col_w + (gap if i < per_row - 1 else 0) for i in range(per_row)])
    style = [("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), gap), ("TOPPADDING", (0, 0), (-1, -1), 0),
             ("BOTTOMPADDING", (0, 0), (-1, -1), gap), ("RIGHTPADDING", (per_row - 1, 0), (per_row - 1, -1), 0)]
    t.setStyle(TableStyle(style))
    backgrounds = [bg for *_, bg in items]
    for idx, cell in enumerate([c for row in data for c in row]):
        if isinstance(cell, Table) and idx < len(backgrounds):
            cell.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), backgrounds[idx])]))
    return t


def _tile_cell(cell: list[Any], width: float) -> Any:
    if cell == [""]:
        return ""
    inner = Table([[c] for c in cell], colWidths=[width])
    inner.setStyle(TableStyle([
        ("LEFTPADDING", (0, 0), (-1, -1), 9), ("RIGHTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (0, 0), 8), ("BOTTOMPADDING", (0, -1), (-1, -1), 8),
        ("TOPPADDING", (0, 1), (-1, -1), 1), ("BOTTOMPADDING", (0, 0), (-1, -2), 1),
        ("ROUNDEDCORNERS", [6, 6, 6, 6]),
    ]))
    return inner


def _callout(flowables: list[Any], width: float, *, bg: Any = ACCENT_SOFT, bar: Any = ACCENT) -> Table:
    t = Table([[flowables]], colWidths=[width])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg), ("LINEBEFORE", (0, 0), (0, -1), 3, bar),
        ("LEFTPADDING", (0, 0), (-1, -1), 12), ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 9), ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
    ]))
    return t


# ------------------------------------------------------------------ drawings
def _box(d: Drawing, x: float, y: float, w: float, h: float, title: str, sub: str, color: Any, *, dashed: bool = False,
         fill: Any = colors.white, bold_border: bool = False) -> None:
    d.add(Rect(x, y, w, h, rx=6, ry=6, fillColor=fill, strokeColor=color, strokeWidth=1.6 if bold_border else 1,
               strokeDashArray=[3, 2] if dashed else None))
    if title:
        d.add(Rect(x, y + h - 3, w, 3, fillColor=color, strokeColor=None))
        d.add(String(x + w / 2, y + h - 15, _clean(title), fontName="Helvetica-Bold", fontSize=8, fillColor=INK, textAnchor="middle"))
    if sub:
        d.add(String(x + w / 2, y + h - 26, _clean(sub)[:34], fontName="Helvetica", fontSize=6.4, fillColor=MUTED, textAnchor="middle"))


def _arrow(d: Drawing, x1: float, y1: float, x2: float, y2: float, color: Any = SUBTLE, *, dashed: bool = False, width: float = 1.1) -> None:
    d.add(Line(x1, y1, x2, y2, strokeColor=color, strokeWidth=width, strokeDashArray=[3, 2] if dashed else None))
    import math
    ang = math.atan2(y2 - y1, x2 - x1)
    size = 4.2
    p1 = (x2 - size * math.cos(ang - 0.45), y2 - size * math.sin(ang - 0.45))
    p2 = (x2 - size * math.cos(ang + 0.45), y2 - size * math.sin(ang + 0.45))
    d.add(Polygon([x2, y2, p1[0], p1[1], p2[0], p2[1]], fillColor=color, strokeColor=color, strokeWidth=0.5))


def _label(d: Drawing, x: float, y: float, text: str, color: Any = MUTED, anchor: str = "middle") -> None:
    txt = _clean(text)
    w = len(txt) * 3.2 + 8
    left = x - w / 2 if anchor == "middle" else (x if anchor == "start" else x - w)
    d.add(Rect(left, y - 3.5, w, 10, rx=5, ry=5, fillColor=colors.white, strokeColor=RULE, strokeWidth=0.6))
    d.add(String(left + w / 2, y - 0.2, txt, fontName="Helvetica", fontSize=6.2, fillColor=color, textAnchor="middle"))


def workflow_diagram(report: FinalReport, width: float) -> Drawing:
    """Shield -> Planner <-> Master Approver -> (Researcher -> Drafter -> Verifier) -> Master Approver -> Synthesizer -> Publish."""
    h = 190
    d = Drawing(width, h)
    m = report.metrics
    rounds = len(report.round_reviews) or m.iterations
    plan_v = len(report.plan_reviews) or 1
    sent_back = sum(1 for r in report.plan_reviews if not r.passed and not r.forced)
    extra = sum(1 for r in report.round_reviews if r.decision == "another_round")
    unit = (width - 4 * 14) / 7.3  # five columns, the group column is 3.3 units wide
    gap = 14
    xs = [0.0]
    for w_units in (1, 1, 3.3, 1):
        xs.append(xs[-1] + w_units * unit + gap)
    col_w = [unit, unit, 3.3 * unit, unit, unit]
    row_y, row_h = 18, 48
    # top: Master Approver over the group
    lx, lw, ly, lh = xs[2], col_w[2], 128, 50
    d.add(Rect(lx, ly, lw, lh, rx=8, ry=8, fillColor=ACCENT_SOFT, strokeColor=ACCENT, strokeWidth=1.6))
    d.add(Circle(lx + 18, ly + lh / 2, 9, fillColor=ACCENT, strokeColor=None))
    d.add(String(lx + 18, ly + lh / 2 - 3, "TL", fontName="Helvetica-Bold", fontSize=7, fillColor=colors.white, textAnchor="middle"))
    d.add(String(lx + 34, ly + lh - 17, "Master Approver", fontName="Helvetica-Bold", fontSize=9, fillColor=INK))
    final = report.round_reviews[-1] if report.round_reviews else None
    verdict = ("approved for synthesis" if final and final.gate_passed else "approved, gaps disclosed") if final else "approved"
    d.add(String(lx + 34, ly + lh - 29, _clean(f"plan v{plan_v} approved ({m.plan_score:.2f})  .  {rounds} round(s) reviewed"), fontName="Helvetica", fontSize=6.8, fillColor=MUTED))
    d.add(String(lx + 34, ly + lh - 40, _clean(f"final decision: {verdict}"), fontName="Helvetica-Bold", fontSize=6.8,
                 fillColor=GOOD if not final or final.gate_passed else WARN))
    # bottom row
    boxes = [("input_guard", "Input shield", (report.governance.input_screen if report.governance else "passed").replace("_", " ")),
             ("planner", "Planner", f"{len(report.sub_questions)} sub-questions")]
    for i, (key, title, sub) in enumerate(boxes):
        _box(d, xs[i], row_y, col_w[i], row_h, title, sub, AGENT_COLORS[key])
    gx, gw = xs[2], col_w[2]
    d.add(Rect(gx - 4, row_y - 12, gw + 8, row_h + 16, rx=9, ry=9, fillColor=colors.HexColor("#f0fdf4"),
               strokeColor=colors.HexColor("#86efac"), strokeWidth=0.9, strokeDashArray=[3, 2]))
    d.add(String(gx + 2, row_y - 8, "GROUP CHAT: RESEARCHER + DRAFTER + VERIFIER", fontName="Helvetica-Bold", fontSize=5.8, fillColor=GOOD))
    d.add(String(gx + gw - 2, row_y - 8, _clean(f"{rounds} round(s)"), fontName="Helvetica", fontSize=5.8, fillColor=GOOD, textAnchor="end"))
    sub_w = (gw - 2 * 10) / 3
    group = [("researcher", "Researcher", f"{m.sources_indexed} sources"), ("drafter", "Drafter", "cited drafts"),
             ("verifier", "Verifier", f"{m.total_claims_evaluated} claims checked")]
    centers = []
    for j, (key, title, sub) in enumerate(group):
        bx = gx + j * (sub_w + 10)
        _box(d, bx, row_y, sub_w, row_h, title, sub, AGENT_COLORS[key])
        centers.append(bx + sub_w / 2)
        if j:
            _arrow(d, bx - 10 + 1, row_y + row_h / 2, bx - 1, row_y + row_h / 2)
    _box(d, xs[3], row_y, col_w[3], row_h, "Synthesizer", f"{m.claims_in_report} claims kept", AGENT_COLORS["synthesizer"])
    _box(d, xs[4], row_y, col_w[4], row_h, "Publish", "redact . seal", AGENT_COLORS["output_guard"])
    mid = row_y + row_h / 2
    _arrow(d, xs[0] + col_w[0] + 1, mid, xs[1] - 1, mid)
    _arrow(d, xs[3] + col_w[3] + 1, mid, xs[4] - 1, mid)
    # planner <-> master
    pcx = xs[1] + col_w[1] / 2
    _arrow(d, pcx - 8, row_y + row_h, pcx - 8, ly + lh * 0.65, SUBTLE)
    d.add(Line(pcx - 8, ly + lh * 0.65, lx, ly + lh * 0.65, strokeColor=SUBTLE, strokeWidth=1.1))
    _arrow(d, lx - 0.5, ly + lh * 0.65, lx, ly + lh * 0.65, SUBTLE)
    d.add(Line(lx, ly + lh * 0.3, pcx + 8, ly + lh * 0.3, strokeColor=ORANGE if sent_back else SUBTLE, strokeWidth=1.1, strokeDashArray=[3, 2]))
    _arrow(d, pcx + 8, ly + lh * 0.3, pcx + 8, row_y + row_h + 1, ORANGE if sent_back else SUBTLE, dashed=True)
    _label(d, (pcx + lx) / 2, ly + lh * 0.65 + 6, f"plan v{plan_v}")
    _label(d, (pcx + lx) / 2, ly + lh * 0.3 - 9, f"revise x{sent_back}" if sent_back else "revise", ORANGE if sent_back else MUTED)
    # master <-> group
    _arrow(d, centers[0], ly, centers[0], row_y + row_h + 1, ORANGE if extra else SUBTLE, dashed=bool(extra))
    _label(d, centers[0], (ly + row_y + row_h) / 2, f"another round x{extra}" if extra else "dispatch", ORANGE if extra else MUTED)
    _arrow(d, centers[1], ly, centers[1], row_y + row_h + 1)
    _label(d, centers[1], (ly + row_y + row_h) / 2, "instructions")
    _arrow(d, centers[2], row_y + row_h, centers[2], ly - 1)
    _label(d, centers[2], (ly + row_y + row_h) / 2, "round report")
    # master -> synthesizer
    sx = xs[3] + col_w[3] / 2
    d.add(Line(lx + lw, ly + lh / 2, sx, ly + lh / 2, strokeColor=GOOD, strokeWidth=1.4))
    _arrow(d, sx, ly + lh / 2, sx, row_y + row_h + 1, GOOD, width=1.4)
    _label(d, (lx + lw + sx) / 2, ly + lh / 2 + 6, "approved", GOOD)
    return d


def rounds_chart(report: FinalReport, iterations: list[dict[str, Any]], width: float) -> Drawing | None:
    data = [(int(r.get("iteration", i + 1)), float(r.get("grounding_rate", 0)), float(r.get("coverage_score", 0)))
            for i, r in enumerate(iterations)]
    if not data and report.round_reviews:
        data = [(r.round, r.grounding_rate, r.coverage_score) for r in report.round_reviews]
    if not data:
        return None
    gt = report.round_reviews[-1].grounding_threshold if report.round_reviews else 0.75
    ct = report.round_reviews[-1].coverage_threshold if report.round_reviews else 0.70
    h, left, bottom, top = 160, 30, 26, 30
    d = Drawing(width, h)
    plot_w, plot_h = width - left - 92, h - bottom - top
    for frac in (0, 0.25, 0.5, 0.75, 1.0):
        y = bottom + plot_h * frac
        d.add(Line(left, y, left + plot_w, y, strokeColor=RULE, strokeWidth=0.5))
        d.add(String(left - 4, y - 2.5, f"{frac * 100:.0f}%", fontName="Helvetica", fontSize=6, fillColor=MUTED, textAnchor="end"))
    group_w = plot_w / len(data)
    bar_w = min(26, group_w * 0.28)
    for i, (it, g, c) in enumerate(data):
        cx = left + group_w * (i + 0.5)
        for j, (val, color) in enumerate(((g, BLUE), (c, ORANGE))):
            x = cx - bar_w - 1 + j * (bar_w + 2)
            d.add(Rect(x, bottom, bar_w, max(0.5, plot_h * val), fillColor=color, strokeColor=None, rx=1.5, ry=1.5))
            d.add(String(x + bar_w / 2, bottom + plot_h * val + 3, f"{val * 100:.0f}%", fontName="Helvetica-Bold", fontSize=6.2, fillColor=INK, textAnchor="middle"))
        d.add(String(cx, bottom - 11, f"Round {it}", fontName="Helvetica", fontSize=6.8, fillColor=BODY, textAnchor="middle"))
    ys = [bottom + plot_h * gt, bottom + plot_h * ct]
    nudge = [0.0, 0.0]
    if abs(ys[0] - ys[1]) < 9:  # keep the two threshold labels from overlapping
        hi = 0 if ys[0] >= ys[1] else 1
        nudge[hi], nudge[1 - hi] = 4.5, -4.5
    for (val, color, name), y, dy in zip(((gt, BLUE, "grounding bar"), (ct, ORANGE, "coverage bar")), ys, nudge, strict=True):
        d.add(Line(left, y, left + plot_w, y, strokeColor=color, strokeWidth=0.9, strokeDashArray=[3, 2]))
        d.add(String(left + plot_w + 4, y - 2.2 + dy, _clean(f"{name} {val * 100:.0f}%"), fontName="Helvetica", fontSize=6.2, fillColor=color))
    for k, (color, name) in enumerate(((BLUE, "Grounding"), (ORANGE, "Coverage"), (MUTED, "dashed lines = the quality bar"))):
        x = left + k * 70
        if k < 2:
            d.add(Rect(x, h - 10, 7, 7, fillColor=color, strokeColor=None, rx=1.5, ry=1.5))
            d.add(String(x + 10, h - 9, name, fontName="Helvetica", fontSize=7, fillColor=BODY))
        else:
            d.add(String(x, h - 9, name, fontName="Helvetica", fontSize=6.6, fillColor=MUTED))
    return d


def stacked_bar(parts: list[tuple[str, int, Any]], width: float, title: str) -> Drawing:
    total = sum(n for _, n, _ in parts) or 1
    d = Drawing(width, 42)
    d.add(String(0, 32, _clean(title), fontName="Helvetica-Bold", fontSize=7.5, fillColor=INK))
    x = 0.0
    for _name, n, color in parts:
        w = width * n / total
        if w > 0:
            d.add(Rect(x, 14, max(0.8, w - 1.5), 12, fillColor=color, strokeColor=None, rx=2, ry=2))
            if w > 34:
                d.add(String(x + 4, 17.5, _clean(f"{n}"), fontName="Helvetica-Bold", fontSize=6.8, fillColor=colors.white))
        x += w
    lx = 0.0
    for name, n, color in parts:
        d.add(Rect(lx, 2, 6, 6, fillColor=color, strokeColor=None, rx=1.2, ry=1.2))
        label = _clean(f"{name} {n} ({n * 100 / total:.0f}%)")
        d.add(String(lx + 9, 2.6, label, fontName="Helvetica", fontSize=6.5, fillColor=BODY))
        lx += len(label) * 3.4 + 22
    return d


def hbars(rows: list[tuple[str, float, str]], width: float, title: str, color: Any = BLUE) -> Drawing:
    rows = rows[:9]
    h = 18 + 14 * len(rows)
    d = Drawing(width, h)
    d.add(String(0, h - 9, _clean(title), fontName="Helvetica-Bold", fontSize=7.5, fillColor=INK))
    top = max((v for _, v, _ in rows), default=1) or 1
    label_w, val_w = 74, 104
    bar_max = width - label_w - val_w
    for i, (name, value, shown) in enumerate(rows):
        y = h - 26 - i * 14
        d.add(String(label_w - 6, y + 1.5, _clean(name)[:18], fontName="Helvetica", fontSize=6.8, fillColor=BODY, textAnchor="end"))
        d.add(Rect(label_w, y, bar_max, 8, fillColor=PAPER, strokeColor=None, rx=2, ry=2))
        d.add(Rect(label_w, y, max(1.2, bar_max * value / top), 8, fillColor=color, strokeColor=None, rx=2, ry=2))
        d.add(String(label_w + bar_max + 5, y + 1.5, _clean(shown), fontName="Helvetica", fontSize=6.6, fillColor=MUTED))
    return d


# ------------------------------------------------------------------ public API
def build_report_pdf(
    report: FinalReport,
    *,
    claims: list[dict[str, Any]] | None = None,
    trace_summary: dict[str, Any] | None = None,
    chat: list[dict[str, Any]] | None = None,
    iterations: list[dict[str, Any]] | None = None,
    chain: dict[str, Any] | None = None,
) -> bytes:
    """Render the designed executive report + audit appendices to PDF bytes."""
    try:
        return _render(report, claims or [], trace_summary or {}, chat or [], iterations or [], chain)
    except Exception as exc:  # reportlab raises assorted layout errors
        raise ReportExportError(f"PDF export failed: {exc}") from exc


def _render(
    report: FinalReport,
    claims: list[dict[str, Any]],
    trace_summary: dict[str, Any],
    chat: list[dict[str, Any]],
    iterations: list[dict[str, Any]],
    chain: dict[str, Any] | None,
) -> bytes:
    s = _styles()
    buf = io.BytesIO()
    doc = BaseDocTemplate(
        buf, pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN, topMargin=20 * mm, bottomMargin=17 * mm,
        title=_clean(report.title), author="DeepDive (Agent Arena 2026)", subject=_clean(report.query),
        creator="DeepDive self-verifying research",
    )
    width = doc.width
    m = report.metrics
    g = report.governance
    story: list[Any] = []

    # -- cover ---------------------------------------------------------------------
    generated = report.generated_at.astimezone(UTC)
    title_style = s["cover_title"] if len(report.title) <= 70 else ParagraphStyle("ct2", parent=s["cover_title"], fontSize=19, leading=23)
    cover_block = [
        _p("DEEPDIVE  .  SELF-VERIFYING RESEARCH REPORT", s["cover_kicker"]), Spacer(1, 3 * mm),
        _p(report.title, title_style), Spacer(1, 3 * mm),
        _p(report.query, s["cover_q"]), Spacer(1, 3 * mm),
        _p(f"Run {report.run_id}  .  {generated:%d %B %Y, %H:%M UTC}  .  {m.iterations} research round(s)  .  "
           f"plan approved after {max(1, m.plan_rounds)} version(s)", s["cover_meta"]),
    ]
    block = Table([[cover_block]], colWidths=[width - 38 * mm], rowHeights=[COVER_BAND - 20 * mm - 6 * mm], hAlign="LEFT")
    block.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0)]))
    story += [block, Spacer(1, 12 * mm)]
    stopped_by = Counter("fabricated quote" if "quote not found" in r.reason or "hallucinated" in r.reason else
                         "outweighed / below authority floor" if "outweighed" in r.reason or "admissibility" in r.reason else
                         "no citation" if "no citation" in r.reason else "unsupported or contradicted" for r in report.removed_claims)
    final_review = report.round_reviews[-1] if report.round_reviews else None
    g_bar = final_review.grounding_threshold if final_review else 0.75
    c_bar = final_review.coverage_threshold if final_review else 0.70

    def status_bg(value: float, bar: float) -> Any:  # tile colour follows the measured status, never decoration
        return GOOD_SOFT if value >= bar else WARN_SOFT if value >= bar / 2 else BAD_SOFT

    tiles = [
        ("Grounding", _pct(m.grounding_rate_final), f"first draft {_pct(m.grounding_rate_pre)} . bar {_pct(g_bar)}", status_bg(m.grounding_rate_final, g_bar)),
        ("Coverage", _pct(m.coverage_score), f"sub-questions answered . bar {_pct(c_bar)}", status_bg(m.coverage_score, c_bar)),
        ("Citation precision", _pct(m.citation_precision), "cited claims entailed", status_bg(m.citation_precision, 0.8)),
        ("Quotes verified", _pct(m.quote_verification_rate), "located verbatim in source", status_bg(m.quote_verification_rate, 0.9)),
        ("Claims published", str(m.claims_in_report), f"of {m.total_claims_evaluated} checked", GOOD_SOFT if m.claims_in_report else WARN_SOFT),
        ("Claims stopped", str(m.claims_removed), ", ".join(f"{v} {k}" for k, v in stopped_by.most_common(2)) or "none", WARN_SOFT),
        ("Conflicts surfaced", str(m.contradictions), "shown, not averaged away", BAD_SOFT if m.contradictions else PAPER),
        ("Sources", f"{m.sources_cited}/{m.sources_indexed}", "cited / indexed", PAPER),
    ]
    story.append(_tiles(tiles, width, s))
    chain_info = chain or {}
    if chain_info.get("valid"):
        chain_state = f"Hash chain intact: {chain_info.get('verified', 0)} audit events verified"
    elif chain:
        chain_state = "Hash chain BROKEN"
    else:
        chain_state = "Hash chain not verified in this export"
    head = chain_info.get("head") or (g.audit_chain_head if g else None) or "-"
    published = ("Every published sentence passed entailment against a verbatim span of its cited source"
                 if m.claims_in_report else "No claim cleared verification, so nothing is stated as fact; the gaps are disclosed")
    seal = [
        Paragraph(f"<b>Verification seal.</b> {_x(chain_state)}. {published}, and every decision is recorded in a SHA-256 "
                  "hash-chained audit trail.", s["body"]),
        Spacer(1, 3),
        _p(f"Chain head {head}", s["small"]),
    ]
    story += [Spacer(1, 2), _callout(seal, width, bg=GOOD_SOFT, bar=GOOD), Spacer(1, 7 * mm),
              _p("How this report was produced", s["h2"]), workflow_diagram(report, width),
              NextPageTemplate("body"), PageBreak()]

    # -- 1 executive summary ------------------------------------------------------------
    story += [SectionHeader("1", "Executive summary", "What the verified evidence says"), Spacer(1, 4)]
    story.append(_callout([_p(report.executive_summary, s["master"])], width))
    if report.key_findings:
        story += [Spacer(1, 10), _p("Key findings", s["h2"])]
        rows = [[Paragraph(f'<font color="#4f46e5"><b>{i:02d}</b></font>', s["finding"]), _p(f, s["finding"])]
                for i, f in enumerate(report.key_findings, 1)]
        t = Table(rows, colWidths=[22, width - 22])
        t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEBELOW", (0, 0), (-1, -1), 0.3, RULE),
                               ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                               ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
        story.append(t)
    if report.limitations:
        story += [Spacer(1, 8), _p("Limitations", s["h2"])]
        story.append(_callout([_p(f"- {x}", s["body"]) for x in report.limitations], width, bg=WARN_SOFT, bar=WARN))

    # -- 2 how it was produced ------------------------------------------------------------
    story += [CondPageBreak(80 * mm), Spacer(1, 10),
              SectionHeader("2", "How this report was produced", "A Master Approver supervising a group of verifying agents"),
              Spacer(1, 4)]
    story.append(_p(
        "The workflow on the cover shows the team. The input shield screens the question. The Planner splits it into sub-questions, and the Master Approver scores the plan and sends it "
        "back until it clears the bar. The Master Approver then dispatches the Researcher, Drafter and Verifier, who work as a group chat. After "
        "every round the Verifier reports to the Master Approver, which either orders another round, with an instruction for each agent, or "
        "approves the work for the Synthesizer. A deterministic quality gate is a floor the Master Approver cannot lower.", s["body"]))
    if report.plan_reviews:
        story += [Spacer(1, 6), _p("Plan reviews by the Master Approver", s["h2"])]
        rows = [[_p(h, s["cellh"]) for h in ("VER.", "SCORE", "DECISION", "SIGNIFICANT ISSUES", "SUMMARY")]]
        for r in report.plan_reviews:
            decision = "approved" if r.passed else "proceeded with open issues" if r.forced else "sent back"
            major = [i for i in r.issues if i.blocking or i.severity.value != "low"]
            minor = len(r.issues) - len(major)
            issues = "; ".join(f"{i.target} ({i.severity.value}{', blocking' if i.blocking else ''}): {i.problem}" for i in major[:4]) or "none"
            if minor:
                issues += f" (+{minor} low-severity note{'s' if minor > 1 else ''})"
            rows.append([_p(f"v{r.round}", s["cellb"]), _p(f"{r.score:.2f} / {r.threshold:.2f}", s["cell"]),
                         Paragraph(f'<font color="{"#15803d" if r.passed else "#b45309"}"><b>{_x(decision)}</b></font>', s["cell"]),
                         _p(issues, s["cell"]), _p(r.summary, s["cell"])])
        story.append(_table(rows, [width * 0.08, width * 0.11, width * 0.14, width * 0.37, width * 0.30]))
    if report.round_reviews:
        story += [CondPageBreak(45 * mm), Spacer(1, 8), _p("Round reviews by the Master Approver", s["h2"])]
        rows = [[_p(h, s["cellh"]) for h in ("RND", "GROUNDING", "COVERAGE", "DECISION", "RATIONALE AND INSTRUCTIONS")]]
        for rr in report.round_reviews:
            decision = ("extra pass" if rr.extra_round else "another round") if rr.decision == "another_round" else ("approved" if rr.gate_passed else "approved, gaps disclosed")
            ins = "".join(f"<br/>- <b>{_x(i.agent.title())} {_x(i.sub_question_id)}</b>: {_x(i.instruction)}" for i in rr.instructions[:4])
            rows.append([_p(str(rr.round), s["cellb"]),
                         [_p(f"{rr.grounding_rate * 100:.0f}% (bar {rr.grounding_threshold * 100:.0f}%)", s["cell"]), Bar(rr.grounding_rate, width=width * 0.11, color=BLUE, target=rr.grounding_threshold)],
                         [_p(f"{rr.coverage_score * 100:.0f}% (bar {rr.coverage_threshold * 100:.0f}%)", s["cell"]), Bar(rr.coverage_score, width=width * 0.11, color=ORANGE, target=rr.coverage_threshold)],
                         Paragraph(f'<font color="{"#b45309" if rr.decision == "another_round" else "#15803d"}"><b>{_x(decision)}</b></font>', s["cell"]),
                         Paragraph(_x(rr.rationale) + ins, s["cell"])])
        story.append(_table(rows, [width * 0.07, width * 0.15, width * 0.15, width * 0.13, width * 0.50]))

    # -- 3 verification dashboard ------------------------------------------------------------
    story += [CondPageBreak(80 * mm), Spacer(1, 10),
              SectionHeader("3", "Verification dashboard", "Measured on this run, from the audit trail"), Spacer(1, 6)]
    chart = rounds_chart(report, iterations, width)
    if chart is not None:
        story += [_p("Grounding and coverage by research round", s["h2"]), chart, Spacer(1, 6)]
    verdicts = Counter(str(c.get("verdict")) for c in claims)
    if verdicts:
        story += [stacked_bar([("Supported", verdicts.get("SUPPORTED", 0), GOOD), ("Unsupported", verdicts.get("UNSUPPORTED", 0), WARN),
                               ("Contradicted", verdicts.get("CONTRADICTED", 0), BAD)], width, "Verdicts on every claim checked, all rounds"),
                  Spacer(1, 8)]
    tiers = Counter(src.tier.value for src in report.sources)
    cited = Counter(src.tier.value for src in report.sources if src.cited_in_report)
    if tiers:
        story += [hbars([(t.replace("_", " ").title(), float(tiers[t]), f"{cited.get(t, 0)} cited of {tiers[t]}") for t in sorted(tiers)],
                        width, "Sources by authority tier (Tier 1 = official / peer reviewed)", color=BLUE), Spacer(1, 6)]
    per_agent = trace_summary.get("per_agent", {})
    if per_agent:
        rows_t = sorted(((AGENT_NAMES.get(a, a.title()), float(v.get("tokens", 0)), f"{v.get('tokens', 0):,} tok . {v.get('calls', 0)} calls")
                         for a, v in per_agent.items()), key=lambda r: -r[1])
        story.append(hbars(rows_t, width, "Mesh API tokens by agent", color=ACCENT))

    # -- 4 detailed findings ------------------------------------------------------------------
    story += [CondPageBreak(60 * mm), Spacer(1, 10),
              SectionHeader("4", "Detailed findings", "One section per sub-question, built only from verified claims"), Spacer(1, 4)]
    for section in report.sections:
        meta = Table([[_p(f"{len(section.claim_ids)} verified claim(s)  .  verifier confidence {section.confidence:.2f}", s["small"]),
                       Bar(section.confidence, width=70, color=GOOD),
                       _p(f"sources: {', '.join(section.cited_source_ids) or 'none'}", s["small"])]],
                     colWidths=[width * 0.36, 80, width * 0.64 - 80])
        meta.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
        story.append(KeepTogether([
            Paragraph(f'<font color="#4f46e5"><b>{_x(section.sub_question_id)}</b></font>&nbsp;&nbsp;{_x(section.heading)}', s["h2"]),
            _p(section.body, s["body"]), Spacer(1, 3), meta, Spacer(1, 6),
        ]))

    # -- 5 contradiction radar --------------------------------------------------------------------
    story += [CondPageBreak(50 * mm), Spacer(1, 8),
              SectionHeader("5", "Contradiction radar", "Conflicts between sources, shown rather than averaged away"), Spacer(1, 4)]
    if report.contradictions:
        rows = [[_p(h, s["cellh"]) for h in ("TOPIC", "POSITION A", "POSITION B", "SEVERITY / RESOLUTION")]]
        sev_color = {"high": "#b91c1c", "medium": "#b45309", "low": "#64748b"}
        for c in report.contradictions:
            rows.append([_p(c.topic, s["cellb"]), _p(f"[{c.source_a}] {c.statement_a}", s["cell"]), _p(f"[{c.source_b}] {c.statement_b}", s["cell"]),
                         Paragraph(f'<font color="{sev_color.get(c.severity.value, "#64748b")}"><b>{c.severity.value.upper()}</b></font><br/>{_x(c.resolution_hint)}', s["cell"])])
        story.append(_table(rows, [width * 0.18, width * 0.29, width * 0.29, width * 0.24]))
    else:
        story.append(_callout([_p("No cross-source contradictions survived quote validation.", s["body"])], width, bg=PAPER, bar=SUBTLE))

    # -- 6 sources -----------------------------------------------------------------------------------
    story += [CondPageBreak(50 * mm), Spacer(1, 8),
              SectionHeader("6", "Sources", "Every source the agents weighed, scored for authority"), Spacer(1, 4)]
    rows = [[_p(h, s["cellh"]) for h in ("ID", "TITLE AND URL", "TIER", "AUTHORITY", "RETRIEVAL", "CITED")]]
    for src in sorted(report.sources, key=lambda x: (not x.cited_in_report, -x.authority_score)):
        tier_hex = {"TIER_1": "#15803d", "TIER_2": "#2a78d6", "TIER_3": "#b45309"}.get(src.tier.value, "#64748b")
        rows.append([_p(src.source_id, s["cellb"]), _p(f"{src.title}  .  {src.url}", s["cell"]),
                     Paragraph(f'<font color="{tier_hex}"><b>{src.tier.value.replace("_", " ")}</b></font>', s["cell"]),
                     [_p(f"{src.authority_score:.2f}", s["cell"]), Bar(src.authority_score, width=width * 0.1, color=TIER_COLORS.get(src.tier.value, BLUE))],
                     _p(src.retrieval_tier.value.replace("_", " "), s["cell"]),
                     Paragraph('<font color="#15803d"><b>yes</b></font>' if src.cited_in_report else "no", s["cell"])])
    story.append(_table(rows, [width * 0.09, width * 0.47, width * 0.09, width * 0.13, width * 0.13, width * 0.09]))

    # -- 7 methodology --------------------------------------------------------------------------------
    story += [CondPageBreak(60 * mm), Spacer(1, 8),
              SectionHeader("7", "Methodology", "How every claim in this report was checked"), Spacer(1, 4)]
    method = [
        ("Planning under review", "The Master Approver scores each plan on coverage, distinctness, searchability, premise testing and specificity, plus "
         "deterministic checks for dropped entities and overlapping sub-questions. Weak plans are revised before any retrieval runs."),
        ("Shielded retrieval", "Pages are fetched through a URL shield (no private networks), cleaned of injected instructions by a content shield, "
         "scored for authority (Tier 1 official and peer reviewed, Tier 2 reputable, Tier 3 other) and indexed as exact character spans."),
        ("Claim-level verification", "Drafts are split into atomic claims. Each claim is judged only against the spans of the sources it cites. The "
         "supporting quote must exist verbatim in the source, figures must match, and the source must clear the admissibility floor."),
        ("Master Approver rounds", "The Verifier reports each round to the Master Approver. Below the grounding and coverage bar the group must go again, with "
         "instructions for each agent; the Master Approver approves only work that clears the bar, or discloses the gaps when rounds run out."),
        ("Publication", "The Synthesizer sees only supported claims; a citation guard drops any sentence citing unverified sources. The output shield "
         "redacts personal data and secrets, and every step is sealed in a SHA-256 hash-chained audit trail."),
    ]
    rows = [[Paragraph(f'<font color="#4f46e5"><b>{i}</b></font>', s["body"]), Paragraph(f"<b>{_x(k)}.</b> {_x(v)}", s["body"])]
            for i, (k, v) in enumerate(method, 1)]
    t = Table(rows, colWidths=[16, width - 16])
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 5), ("LINEBELOW", (0, 0), (-1, -2), 0.3, RULE)]))
    story.append(t)
    if g and g.disclaimer:
        story += [Spacer(1, 6), _callout([_p(g.disclaimer, s["muted"])], width, bg=PAPER, bar=SUBTLE)]

    # -- appendices -----------------------------------------------------------------------------------
    story += [PageBreak(), SectionHeader("A", "Appendix A . Claim verification audit",
                                         "Every atomic claim from every round, its verdict and the evidence it rests on"), Spacer(1, 4)]
    rows = [[_p(h, s["cellh"]) for h in ("CLAIM", "RND", "VERDICT", "CONF.", "EVIDENCE", "QUOTE", "IN REPORT")]]
    for row in claims:
        verdict = str(row.get("verdict", ""))
        hexc = {"SUPPORTED": "#15803d", "CONTRADICTED": "#b91c1c", "UNSUPPORTED": "#b45309"}.get(verdict, "#0f172a")
        evidence = row.get("evidence_quote") or row.get("downgrade_reason") or row.get("rationale") or ""
        rows.append([_p(f"{row.get('claim_id')}: {row.get('claim_text')}", s["cell"]), _p(row.get("iteration"), s["cell"]),
                     Paragraph(f'<font color="{hexc}"><b>{_x(verdict)}</b></font>', s["cell"]),
                     _p(f"{float(row.get('confidence') or 0):.2f}", s["cell"]),
                     _p(f"[{row.get('evidence_source_id') or '-'}] {evidence}", s["cell"]),
                     _p("verified" if row.get("quote_verified") else "not found", s["cell"]),
                     _p("yes" if row.get("included_in_report") else "no", s["cell"])])
    story.append(_table(rows, [width * 0.30, width * 0.05, width * 0.12, width * 0.07, width * 0.30, width * 0.08, width * 0.08]))

    if report.removed_claims:
        story += [Spacer(1, 10), SectionHeader("B", "Appendix B . Claims stripped from the report", "Kept here so nothing disappears silently"), Spacer(1, 4)]
        rows = [[_p(h, s["cellh"]) for h in ("CLAIM", "VERDICT", "REASON")]]
        for rc in report.removed_claims:
            rows.append([_p(f"{rc.claim_id}: {rc.text}", s["cell"]), _p(rc.verdict.value, s["cellb"]), _p(rc.reason, s["cell"])])
        story.append(_table(rows, [width * 0.45, width * 0.13, width * 0.42]))

    if chat:
        story += [CondPageBreak(50 * mm), Spacer(1, 10),
                  SectionHeader("C", "Appendix C . The agents' group chat", "Every hand-off between agents, in order, as recorded in the audit trail"),
                  Spacer(1, 4)]
        rows = []
        room = None
        for msg in chat:
            label = {"plan_review": f"Plan review . version {msg.get('round', 1)}", "evidence_loop": f"Group chat . round {msg.get('round', 1)}",
                     "publish": "Publish"}.get(str(msg.get("channel")), str(msg.get("channel")))
            if label != room:
                rows.append([Paragraph(f'<font color="#64748b"><b>{_x(label.upper())}</b></font>', s["cellh"]), ""])
                room = label
            sender = str(msg.get("sender", ""))
            hexc = AGENT_COLORS.get(sender, MUTED).hexval().replace("0x", "#")
            to = ", ".join(AGENT_NAMES.get(t, t) for t in msg.get("to", [])) or "team"
            items = "".join(f"<br/>- {_x(i.strip())}" for i in (msg.get("items") or [])[:6])
            rows.append([Paragraph(f'<font color="{hexc}"><b>{_x(AGENT_NAMES.get(sender, sender))}</b></font><br/><font color="#94a3b8">to {_x(to)}</font>', s["chat"]),
                         Paragraph(_x(msg.get("text", "")) + f'<font color="#64748b">{items}</font>', s["chat"])])
        t = Table(rows, colWidths=[width * 0.2, width * 0.8])
        style = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEBELOW", (0, 0), (-1, -1), 0.3, RULE),
                 ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]
        for idx, trow in enumerate(rows):
            if trow[1] == "":
                style += [("SPAN", (0, idx), (1, idx)), ("BACKGROUND", (0, idx), (1, idx), PAPER)]
        t.setStyle(TableStyle(style))
        story.append(t)

    story += [CondPageBreak(60 * mm), Spacer(1, 10),
              SectionHeader("D", "Appendix D . Governance and audit trail", "Guardrail decisions, budgets, models and the hash chain"), Spacer(1, 4)]
    if g:
        gov_tiles = [
            ("Input screen", g.input_screen.replace("_", " "), f"Guardian: {g.llm_screen}", GOOD_SOFT if g.input_screen.startswith("passed") else BAD_SOFT),
            ("Injections neutralised", str(g.injections_neutralized), "in retrieved pages", PAPER),
            ("PII / secrets redacted", f"{g.pii_redactions} / {g.secret_redactions}", "before publication", PAPER),
            ("Unsafe URLs refused", str(g.urls_blocked), "SSRF shield", PAPER),
        ]
        story += [_tiles(gov_tiles, width, s), Spacer(1, 4)]
        budget = g.budget or {}
        story.append(_p(f"Policy {g.policy_version}  .  models {', '.join(g.models_used) or '-'}  .  budget used "
                        f"{_pct(budget.get('fraction_used'))} ({budget.get('calls', budget.get('llm_calls', '-'))} calls)  .  "
                        f"{g.audit_events} audit events", s["small"]))
        if g.findings:
            rows = [[_p(h, s["cellh"]) for h in ("GUARD", "KIND", "ACTION", "DETAIL")]]
            for f in g.findings[:40]:
                rows.append([_p(f.guard, s["cell"]), _p(f.kind, s["cell"]), _p(f.action.value, s["cellb"]), _p(f"{f.detail} ({f.location})", s["cell"])])
            story += [Spacer(1, 4), _table(rows, [width * 0.18, width * 0.2, width * 0.12, width * 0.5])]
    chain_text = ("intact: every event's SHA-256 hash links to the previous one" if chain and chain.get("valid")
                  else f"BROKEN at {chain.get('broken_at')}" if chain else "not checked in this export")
    story += [Spacer(1, 6), _callout([_p(f"Audit chain {chain_text}.", s["body"]),
                                      _p(f"Head: {head}", s["small"])],
                                     width, bg=GOOD_SOFT if chain and chain.get("valid") else PAPER, bar=GOOD if chain and chain.get("valid") else SUBTLE)]

    if per_agent:
        story += [CondPageBreak(40 * mm), Spacer(1, 10),
                  SectionHeader("E", "Appendix E . Mesh API telemetry", "Every model call went through the Mesh API gateway"), Spacer(1, 4)]
        rows = [[_p(h, s["cellh"]) for h in ("AGENT", "CALLS", "TOKENS", "AVG LATENCY (MS)", "SHARE OF TOKENS")]]
        total_tokens = sum(v.get("tokens", 0) for v in per_agent.values()) or 1
        for agent, stats in sorted(per_agent.items(), key=lambda kv: -kv[1].get("tokens", 0)):
            share = stats.get("tokens", 0) / total_tokens
            rows.append([_p(AGENT_NAMES.get(agent, agent), s["cellb"]), _p(stats.get("calls", 0), s["cell"]), _p(f"{stats.get('tokens', 0):,}", s["cell"]),
                         _p(stats.get("avg_latency_ms", "-"), s["cell"]), [_p(f"{share * 100:.0f}%", s["cell"]), Bar(share, width=width * 0.22, color=ACCENT)]])
        story.append(_table(rows, [width * 0.22, width * 0.12, width * 0.16, width * 0.2, width * 0.3]))

    def _cover(canvas: Any, _doc: Any) -> None:
        canvas.saveState()
        band = COVER_BAND
        canvas.setFillColor(NAVY)
        canvas.rect(0, PAGE_H - band, PAGE_W, band, stroke=0, fill=1)
        canvas.setFillColor(NAVY_2)
        canvas.circle(PAGE_W - 24 * mm, PAGE_H - 30 * mm, 46 * mm, stroke=0, fill=1)
        canvas.setFillColor(ACCENT)
        canvas.rect(0, PAGE_H - band, PAGE_W, 2.2 * mm, stroke=0, fill=1)
        # seal
        cx, cy = PAGE_W - MARGIN - 15 * mm, PAGE_H - 30 * mm
        verified = bool(chain and chain.get("valid"))
        canvas.setStrokeColor(colors.HexColor("#86efac") if verified else colors.HexColor("#cbd5e1"))
        canvas.setLineWidth(1.4)
        canvas.circle(cx, cy, 13 * mm, stroke=1, fill=0)
        canvas.circle(cx, cy, 11.3 * mm, stroke=1, fill=0)
        canvas.setFillColor(colors.white)
        canvas.setFont("Helvetica-Bold", 7.5)
        canvas.drawCentredString(cx, cy + 3, "VERIFIED" if verified else "AUDITED")
        canvas.setFont("Helvetica", 5.6)
        canvas.drawCentredString(cx, cy - 4, "hash-chained trail")
        canvas.drawCentredString(cx, cy - 10, f"{chain_info.get('verified', g.audit_events if g else 0)} events")
        _footer(canvas)
        canvas.restoreState()

    def _later(canvas: Any, _doc: Any) -> None:
        canvas.saveState()
        canvas.setFillColor(ACCENT)
        canvas.rect(MARGIN, PAGE_H - 11 * mm, 3, 4.2 * mm, stroke=0, fill=1)
        canvas.setFillColor(INK)
        canvas.setFont("Helvetica-Bold", 7.5)
        canvas.drawString(MARGIN + 7, PAGE_H - 10 * mm, "DeepDive")
        canvas.setFillColor(MUTED)
        canvas.setFont("Helvetica", 7.5)
        canvas.drawString(MARGIN + 44, PAGE_H - 10 * mm, "Self-verifying research report")
        title = _clean(report.title)
        canvas.drawRightString(PAGE_W - MARGIN, PAGE_H - 10 * mm, title if len(title) < 70 else title[:67] + "...")
        canvas.setStrokeColor(RULE)
        canvas.setLineWidth(0.5)
        canvas.line(MARGIN, PAGE_H - 13 * mm, PAGE_W - MARGIN, PAGE_H - 13 * mm)
        _footer(canvas)
        canvas.restoreState()

    def _footer(canvas: Any) -> None:
        canvas.setStrokeColor(RULE)
        canvas.setLineWidth(0.5)
        canvas.line(MARGIN, 12 * mm, PAGE_W - MARGIN, 12 * mm)
        canvas.setFillColor(MUTED)
        canvas.setFont("Helvetica", 7)
        canvas.drawString(MARGIN, 8 * mm, f"Run {report.run_id}  .  every LLM call via Mesh API  .  {datetime.now(UTC):%Y-%m-%d}")
        canvas.drawRightString(PAGE_W - MARGIN, 8 * mm, f"Page {canvas.getPageNumber()}")

    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="content",
                  leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    doc.addPageTemplates([PageTemplate(id="cover", frames=[frame], onPage=_cover),
                          PageTemplate(id="body", frames=[frame], onPage=_later)])
    doc.build(story)
    return buf.getvalue()


def export_filename(report: FinalReport) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%d")
    return f"deepdive-{report.run_id}-{stamp}.pdf"
