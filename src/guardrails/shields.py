"""Input, content, and output shields.

    InputGuard    user question / follow-up -> block (injection, harmful request, credential probing)
                  or allow with PII + secrets redacted before anything leaves the process
    ContentGuard  retrieved page / uploaded document -> injected sentences neutralised, hidden
                  characters stripped, secrets (and optionally PII) redacted, before indexing
    OutputGuard   report / follow-up answer -> secrets (always) and PII (policy) redacted before
                  publication
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from src.core.config import Settings
from src.guardrails.patterns import HARM_MESSAGES, PatternHit, detect_harm, detect_injection
from src.guardrails.pii import mask_excerpt, redact
from src.schemas.domain import (
    BaselineComparison,
    FinalReport,
    FollowUpAnswer,
    GuardrailAction,
    GuardrailFinding,
    GuardrailSeverity,
)
from src.tools.document_ingestor import clean_text

NEUTRALIZED_MARKER = "[removed by DeepDive content shield: suspected prompt-injection text]"
_INVISIBLE = re.compile(r"[​-‏‪-‮⁠-⁤﻿]")
_SENTENCE = re.compile(r"[^.!?\n]+[.!?]?")


@dataclass
class ScreenResult:
    allowed: bool
    text: str
    findings: list[GuardrailFinding] = field(default_factory=list)
    blocked_reason: str | None = None

    @property
    def redacted(self) -> bool:
        return any(f.action is GuardrailAction.REDACT for f in self.findings)


def _hidden_char_count(text: str) -> int:
    return len(_INVISIBLE.findall(text)) + sum(
        1 for ch in text if unicodedata.category(ch) == "Cf" and not _INVISIBLE.match(ch)
    )


def _redaction_findings(guard: str, counts: dict[str, int], location: str, *, pii_action: str) -> list[GuardrailFinding]:
    findings: list[GuardrailFinding] = []
    for kind, n in sorted(counts.items()):
        is_secret = kind.endswith(("_KEY", "_TOKEN")) or kind == "PRIVATE_KEY"
        findings.append(
            GuardrailFinding(
                guard=guard,
                kind="secret" if is_secret else "pii",
                severity=GuardrailSeverity.HIGH if is_secret else GuardrailSeverity.LOW,
                action=GuardrailAction.REDACT,
                detail=f"{n} {kind} value(s) redacted ({pii_action})",
                location=location,
            )
        )
    return findings


class InputGuard:
    """Deterministic shield for user-supplied questions."""

    guard_name = "input_shield"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def redact_only(self, text: str) -> str:
        """Data minimisation before persistence: secrets always, PII per policy."""
        return redact(clean_text(text), pii=self.settings.redact_pii_in_queries, secrets=True)[0]

    def screen(self, text: str, *, location: str = "query") -> ScreenResult:
        if not self.settings.guardrails_enabled:
            return ScreenResult(True, text)
        findings: list[GuardrailFinding] = []
        hidden = _hidden_char_count(text)
        normalized = clean_text(text)
        if hidden:
            findings.append(
                GuardrailFinding(
                    guard=self.guard_name, kind="hidden_text", severity=GuardrailSeverity.MEDIUM,
                    action=GuardrailAction.NEUTRALIZE, detail=f"{hidden} invisible/format character(s) stripped",
                    location=location,
                )
            )

        blocked: str | None = None
        for hit in detect_harm(normalized):
            findings.append(self._hit_finding(hit, normalized, location, GuardrailSeverity.CRITICAL))
            blocked = blocked or HARM_MESSAGES.get(
                hit.name, f"Request blocked by content policy ({hit.name.replace('_', ' ')})."
            )
        for hit in detect_injection(normalized):
            action_block = self.settings.block_prompt_injection
            findings.append(
                self._hit_finding(
                    hit, normalized, location, GuardrailSeverity.HIGH,
                    action=GuardrailAction.BLOCK if action_block else GuardrailAction.FLAG,
                )
            )
            if action_block:
                blocked = blocked or (
                    f"Request blocked: it contains a prompt-injection pattern ({hit.name.replace('_', ' ')}). "
                    "DeepDive only accepts research questions."
                )

        sanitized, counts = redact(normalized, pii=self.settings.redact_pii_in_queries, secrets=True)
        findings += _redaction_findings(self.guard_name, counts, location, pii_action="before any egress")
        return ScreenResult(blocked is None, sanitized, findings, blocked)

    def _hit_finding(
        self,
        hit: PatternHit,
        text: str,
        location: str,
        severity: GuardrailSeverity,
        action: GuardrailAction = GuardrailAction.BLOCK,
    ) -> GuardrailFinding:
        return GuardrailFinding(
            guard=self.guard_name,
            kind=hit.category,
            severity=severity,
            action=action,
            detail=f"pattern '{hit.name}' matched",
            location=location,
            excerpt=mask_excerpt(text[max(0, hit.start - 20) : hit.end + 20]),
        )


class ContentGuard:
    """Shield for untrusted evidence (web pages and uploads) before it is indexed."""

    guard_name = "content_shield"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def sanitize(self, text: str, *, location: str, redact_pii: bool = False) -> tuple[str, list[GuardrailFinding]]:
        if not self.settings.guardrails_enabled:
            return text, []
        findings: list[GuardrailFinding] = []
        hidden = _hidden_char_count(text)
        if hidden:
            findings.append(
                GuardrailFinding(
                    guard=self.guard_name, kind="hidden_text", severity=GuardrailSeverity.MEDIUM,
                    action=GuardrailAction.NEUTRALIZE, detail=f"{hidden} invisible/format character(s) stripped",
                    location=location,
                )
            )
            text = _INVISIBLE.sub("", text)

        hits = detect_injection(text, indirect=True)
        if hits:
            text, neutralized = self._neutralize(text, hits)
            for hit, excerpt in neutralized:
                findings.append(
                    GuardrailFinding(
                        guard=self.guard_name, kind="indirect_prompt_injection", severity=GuardrailSeverity.HIGH,
                        action=GuardrailAction.NEUTRALIZE, detail=f"pattern '{hit.name}' neutralised before indexing",
                        location=location, excerpt=excerpt,
                    )
                )

        text, counts = redact(text, pii=redact_pii, secrets=True)
        findings += _redaction_findings(self.guard_name, counts, location, pii_action="before indexing")
        return text, findings

    @staticmethod
    def _neutralize(text: str, hits: list[PatternHit]) -> tuple[str, list[tuple[PatternHit, str]]]:
        """Replace every sentence that contains an injection hit with a visible marker."""
        spans: list[tuple[int, int, PatternHit]] = []
        for hit in hits:
            start = max(text.rfind(".", 0, hit.start), text.rfind("\n", 0, hit.start), text.rfind("!", 0, hit.start),
                        text.rfind("?", 0, hit.start)) + 1
            ends = [i for i in (text.find(c, hit.end) for c in ".!?\n") if i != -1]
            end = (min(ends) + 1) if ends else len(text)
            spans.append((start, end, hit))
        spans.sort(key=lambda span: (span[0], span[1]))
        merged: list[tuple[int, int, PatternHit]] = []
        for start, end, hit in spans:
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(end, merged[-1][1]), merged[-1][2])
            else:
                merged.append((start, end, hit))
        out: list[str] = []
        cursor = 0
        neutralized: list[tuple[PatternHit, str]] = []
        for start, end, hit in merged:
            out.append(text[cursor:start])
            lead = " " if start > 0 and not text[start - 1].isspace() else ""
            out.append(f"{lead}{NEUTRALIZED_MARKER}")
            neutralized.append((hit, mask_excerpt(text[start:end].strip())))
            cursor = end
        out.append(text[cursor:])
        return "".join(out), neutralized


class OutputGuard:
    """Shield applied to everything DeepDive publishes."""

    guard_name = "output_shield"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def _clean(self, text: str, counts: dict[str, int]) -> str:
        cleaned, found = redact(text, pii=self.settings.redact_pii_in_output, secrets=True)
        for kind, n in found.items():
            counts[kind] = counts.get(kind, 0) + n
        return cleaned

    def apply_to_report(self, report: FinalReport) -> tuple[FinalReport, list[GuardrailFinding]]:
        counts: dict[str, int] = {}
        clean = self._clean
        updated = report.model_copy(
            update={
                "title": clean(report.title, counts),
                "executive_summary": clean(report.executive_summary, counts),
                "key_findings": [clean(f, counts) for f in report.key_findings],
                "limitations": [clean(item, counts) for item in report.limitations],
                "sections": [
                    s.model_copy(update={"body": clean(s.body, counts), "heading": clean(s.heading, counts)})
                    for s in report.sections
                ],
                "contradictions": [
                    c.model_copy(update={"statement_a": clean(c.statement_a, counts), "statement_b": clean(c.statement_b, counts)})
                    for c in report.contradictions
                ],
                "removed_claims": [r.model_copy(update={"text": clean(r.text, counts)}) for r in report.removed_claims],
            }
        )
        return updated, _redaction_findings(self.guard_name, counts, "final_report", pii_action="before publication")

    def apply_to_comparison(self, comparison: BaselineComparison) -> tuple[BaselineComparison, list[GuardrailFinding]]:
        counts: dict[str, int] = {}
        cleaned = comparison.model_copy(update={
            "answer": self._clean(comparison.answer, counts),
            "claims": [c.model_copy(update={"text": self._clean(c.text, counts)}) for c in comparison.claims],
        })
        return cleaned, _redaction_findings(self.guard_name, counts, "baseline_answer", pii_action="before publication")

    def apply_to_answer(self, answer: FollowUpAnswer) -> tuple[FollowUpAnswer, list[GuardrailFinding]]:
        counts: dict[str, int] = {}
        cleaned = answer.model_copy(update={"answer": self._clean(answer.answer, counts)})
        return cleaned, _redaction_findings(self.guard_name, counts, "followup_answer", pii_action="before publication")
