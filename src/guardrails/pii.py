"""PII and secret detection / redaction (deterministic, checksum-validated where possible)."""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass

# --- Secrets: always redacted, everywhere ------------------------------------
SECRET_PATTERNS: dict[str, re.Pattern[str]] = {
    "MESH_API_KEY": re.compile(r"\brsk_[A-Za-z0-9_\-]{16,}"),
    "OPENAI_API_KEY": re.compile(r"\bsk-(?:proj-|ant-)?[A-Za-z0-9_\-]{20,}"),
    "TAVILY_API_KEY": re.compile(r"\btvly-[A-Za-z0-9_\-]{16,}"),
    "AWS_ACCESS_KEY": re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    "GITHUB_TOKEN": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    "PRIVATE_KEY": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?(?:-----END [A-Z ]*PRIVATE KEY-----|\Z)"),
    "BEARER_TOKEN": re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{24,}"),
}

# --- PII ----------------------------------------------------------------------
EMAIL = re.compile(r"(?<![\w.+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,24}\b")
PHONE = re.compile(r"(?<![\w.])\+?\d[\d \t().\-]{8,18}\d(?![\w])")
CARD = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")
US_SSN = re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")
AADHAAR = re.compile(r"(?<!\d)[2-9]\d{3}[ -]?\d{4}[ -]?\d{4}(?!\d)")
PAN_IN = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")

_VERHOEFF_D = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), (1, 2, 3, 4, 0, 6, 7, 8, 9, 5), (2, 3, 4, 0, 1, 7, 8, 9, 5, 6),
    (3, 4, 0, 1, 2, 8, 9, 5, 6, 7), (4, 0, 1, 2, 3, 9, 5, 6, 7, 8), (5, 9, 8, 7, 6, 0, 4, 3, 2, 1),
    (6, 5, 9, 8, 7, 1, 0, 4, 3, 2), (7, 6, 5, 9, 8, 2, 1, 0, 4, 3), (8, 7, 6, 5, 9, 3, 2, 1, 0, 4),
    (9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
)
_VERHOEFF_P = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), (1, 5, 7, 6, 2, 8, 3, 0, 9, 4), (5, 8, 0, 3, 7, 9, 6, 1, 4, 2),
    (8, 9, 1, 6, 0, 4, 3, 5, 2, 7), (9, 4, 5, 3, 1, 2, 6, 8, 7, 0), (4, 2, 8, 6, 5, 7, 3, 9, 0, 1),
    (2, 7, 9, 3, 8, 0, 6, 4, 1, 5), (7, 0, 4, 6, 9, 1, 3, 2, 5, 8),
)
_VERHOEFF_INV = (0, 4, 3, 2, 1, 5, 6, 7, 8, 9)


def luhn_valid(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        n = int(ch)
        if i % 2 == 1:
            n = n * 2 - 9 if n > 4 else n * 2
        total += n
    return total % 10 == 0


def verhoeff_valid(digits: str) -> bool:
    c = 0
    for i, ch in enumerate(reversed(digits)):
        c = _VERHOEFF_D[c][_VERHOEFF_P[i % 8][int(ch)]]
    return c == 0


def verhoeff_check_digit(digits: str) -> str:
    c = 0
    for i, ch in enumerate(reversed(digits)):
        c = _VERHOEFF_D[c][_VERHOEFF_P[(i + 1) % 8][int(ch)]]
    return str(_VERHOEFF_INV[c])


def _only_digits(text: str) -> str:
    return re.sub(r"\D", "", text)


def _plausible_phone(match: str) -> bool:
    digits = _only_digits(match)
    if not 10 <= len(digits) <= 15:
        return False
    if match.count(".") == 1 and not re.search(r"[ ()\-]", match):
        return False  # a decimal number such as 3.14159265358
    groups = re.findall(r"\d+", match)
    # a run of years in a statistics table is not a phone number
    return not all(len(g) == 4 and 1900 <= int(g) <= 2100 for g in groups)


@dataclass(frozen=True)
class PiiMatch:
    kind: str
    start: int
    end: int


def find_sensitive(text: str, *, pii: bool = True, secrets: bool = True) -> list[PiiMatch]:
    """Non-overlapping sensitive spans, longest match wins on overlap."""
    found: list[PiiMatch] = []
    if secrets:
        for kind, pattern in SECRET_PATTERNS.items():
            found += [PiiMatch(kind, m.start(), m.end()) for m in pattern.finditer(text)]
    if pii:
        found += [PiiMatch("EMAIL", m.start(), m.end()) for m in EMAIL.finditer(text)]
        found += [PiiMatch("US_SSN", m.start(), m.end()) for m in US_SSN.finditer(text)]
        found += [PiiMatch("PAN_IN", m.start(), m.end()) for m in PAN_IN.finditer(text)]
        found += [
            PiiMatch("CREDIT_CARD", m.start(), m.end())
            for m in CARD.finditer(text)
            if 13 <= len(_only_digits(m.group())) <= 19 and luhn_valid(_only_digits(m.group()))
        ]
        found += [
            PiiMatch("AADHAAR", m.start(), m.end())
            for m in AADHAAR.finditer(text)
            if verhoeff_valid(_only_digits(m.group()))
        ]
        found += [PiiMatch("PHONE", m.start(), m.end()) for m in PHONE.finditer(text) if _plausible_phone(m.group())]

    found.sort(key=lambda m: (m.start, -(m.end - m.start)))
    resolved: list[PiiMatch] = []
    for match in found:
        if resolved and match.start < resolved[-1].end:
            continue
        resolved.append(match)
    return resolved


def redact(text: str, *, pii: bool = True, secrets: bool = True) -> tuple[str, Counter[str]]:
    """Replace sensitive spans with `[REDACTED:KIND]`; returns (text, counts by kind)."""
    matches = find_sensitive(text, pii=pii, secrets=secrets)
    if not matches:
        return text, Counter()
    out: list[str] = []
    cursor = 0
    for m in matches:
        out.append(text[cursor : m.start])
        out.append(f"[REDACTED:{m.kind}]")
        cursor = m.end
    out.append(text[cursor:])
    return "".join(out), Counter(m.kind for m in matches)


def redact_secrets(text: str) -> str:
    return redact(text, pii=False, secrets=True)[0]


def mask_excerpt(text: str, limit: int = 120) -> str:
    """Safe excerpt for audit logs: secrets + PII redacted, truncated."""
    cleaned = redact(text)[0].replace("\n", " ")
    return cleaned if len(cleaned) <= limit else cleaned[: limit - 1] + "…"


class SecretRedactingFilter(logging.Filter):
    """Logging filter that masks credential-shaped strings before any handler writes them."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:
            return True
        redacted = redact_secrets(message)
        if redacted != message:
            record.msg = redacted
            record.args = None
        return True


def install_log_redaction(*logger_names: str) -> None:
    flt = SecretRedactingFilter()
    for name in ("", *logger_names):
        logger = logging.getLogger(name)
        if not any(isinstance(f, SecretRedactingFilter) for f in logger.filters):
            logger.addFilter(flt)
        for handler in logger.handlers:
            if not any(isinstance(f, SecretRedactingFilter) for f in handler.filters):
                handler.addFilter(flt)
