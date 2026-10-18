"""Deterministic Source Authority Weighting Engine.

Every retrieved document receives an authority score in [0, 1] and a tier.
The rules are explicit and order-sensitive so a judge can audit exactly why a
source was trusted:

    Tier 1 (0.85-1.00)  primary / official / peer-reviewed
                        .gov  .edu  .gouv.fr  .go.jp  .bund.de  arxiv.org
                        nature.com  ieee.org  sec.gov  rbi.org.in  iea.org
                        iea.blob.core.windows.net  destatis.de  who.int ...
    Tier 2 (0.65-0.84)  established editorial outlets & reference works,
                        independent analysts, first-party AI labs
                        reuters.com  bloomberg.com  ft.com  ember-energy.org
                        deepmind.google  openai.com  anthropic.com ...
    Tier 3 (0.30-0.64)  everything else: blogs, Medium, forums, social media,
                        market-research teaser pages (kept below the
                        admissibility floor)

Scoring = base score of the first matching rule, then small bounded
adjustments for content signals (thin content, blog/opinion paths, primary
PDFs, dated publications). Adjustments can never move a source out of its
tier's band, so the tier stays a stable, explainable label.
"""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse

from src.schemas.domain import SourceAuthority, SourceTier

TIER_BANDS: dict[SourceTier, tuple[float, float]] = {
    SourceTier.TIER_1: (0.85, 1.00),
    SourceTier.TIER_2: (0.65, 0.84),
    SourceTier.TIER_3: (0.30, 0.64),
}

# --- Tier 1: primary, official, peer-reviewed --------------------------------
TIER_1_DOMAINS: dict[str, float] = {
    "sec.gov": 0.97,
    "rbi.org.in": 0.97,
    "nih.gov": 0.96,
    "nasa.gov": 0.96,
    "nature.com": 0.95,
    "science.org": 0.95,
    "who.int": 0.95,
    "nejm.org": 0.95,
    "thelancet.com": 0.94,
    "iea.org": 0.94,
    "bis.org": 0.93,
    "imf.org": 0.93,
    "worldbank.org": 0.93,
    "arxiv.org": 0.92,
    "ieee.org": 0.92,
    "oecd.org": 0.92,
    "acm.org": 0.91,
    "irena.org": 0.91,
    "un.org": 0.90,
    "europa.eu": 0.90,
    "sciencedirect.com": 0.89,
    "springer.com": 0.88,
    "pnas.org": 0.93,
    "cell.com": 0.92,
    "bmj.com": 0.93,
    # IEA's own file host. Exact host only: never add `blob.core.windows.net`,
    # because anyone can host files on Azure Blob Storage.
    "iea.blob.core.windows.net": 0.92,
    "iea-pvps.org": 0.88,
    "bundeswirtschaftsministerium.de": 0.90,
    "bundesregierung.de": 0.90,
    "destatis.de": 0.91,
}
TIER_1_SUFFIXES: dict[str, float] = {
    ".gov": 0.92,
    ".gov.in": 0.92,
    ".gov.uk": 0.92,
    ".gc.ca": 0.91,
    ".gov.au": 0.91,
    ".mil": 0.88,
    ".int": 0.89,
    ".edu": 0.88,
    ".ac.uk": 0.88,
    ".ac.in": 0.87,
    ".edu.au": 0.87,
    ".nic.in": 0.87,
    ".gouv.fr": 0.91,
    ".go.jp": 0.91,
    ".gov.sg": 0.91,
    ".govt.nz": 0.91,
    ".gov.br": 0.90,
    ".gob.mx": 0.90,
    ".gob.es": 0.90,
    ".bund.de": 0.90,
    ".admin.ch": 0.90,
}

# --- Tier 2: established editorial / reference -------------------------------
TIER_2_DOMAINS: dict[str, float] = {
    "reuters.com": 0.82,
    "bloomberg.com": 0.82,
    "apnews.com": 0.81,
    "ft.com": 0.81,
    "economist.com": 0.80,
    "ourworldindata.org": 0.80,
    "pewresearch.org": 0.80,
    "wsj.com": 0.79,
    "scientificamerican.com": 0.78,
    "nytimes.com": 0.77,
    "bbc.com": 0.77,
    "bbc.co.uk": 0.77,
    "britannica.com": 0.76,
    "carbonbrief.org": 0.76,
    "newscientist.com": 0.75,
    "washingtonpost.com": 0.75,
    "theguardian.com": 0.74,
    "nationalgeographic.com": 0.74,
    "thehindu.com": 0.72,
    "cnbc.com": 0.72,
    "livemint.com": 0.71,
    "economictimes.indiatimes.com": 0.70,
    "arstechnica.com": 0.70,
    "statista.com": 0.70,
    "mckinsey.com": 0.70,
    "wired.com": 0.69,
    "techcrunch.com": 0.68,
    "wikipedia.org": 0.66,
    # First-party AI labs: primary for their own releases but not independent,
    # so Tier 2 rather than Tier 1.
    "deepmind.google": 0.76,
    "research.google": 0.76,
    "blog.google": 0.72,
    "openai.com": 0.74,
    "anthropic.com": 0.74,
    "ai.meta.com": 0.74,
    # Independent energy analysts.
    "ember-energy.org": 0.76,
    "ren21.net": 0.76,
}

# --- Tier 3: explicitly low-authority platforms ------------------------------
TIER_3_DOMAINS: dict[str, float] = {
    "github.com": 0.55,
    "stackoverflow.com": 0.52,
    "stackexchange.com": 0.50,
    "substack.com": 0.42,
    "medium.com": 0.40,
    "linkedin.com": 0.40,
    "wordpress.com": 0.36,
    "blogspot.com": 0.35,
    "youtube.com": 0.35,
    "tumblr.com": 0.33,
    "reddit.com": 0.32,
    "facebook.com": 0.30,
    "quora.com": 0.30,
    "twitter.com": 0.30,
    "x.com": 0.30,
    # Market-research teaser pages: estimates with no published methodology.
    # Scored so that even every positive adjustment (+0.02 at most) keeps them
    # below the admissibility floor, so they can never support a claim.
    "gminsights.com": 0.38,
    "maximizemarketresearch.com": 0.38,
    "fortunebusinessinsights.com": 0.38,
    "marketresearchfuture.com": 0.38,
    "sphericalinsights.com": 0.38,
    "technavio.com": 0.40,
    "grandviewresearch.com": 0.40,
}
TIER_3_DEFAULT = 0.48
UPLOADED_DOCUMENT_SCORE = 0.75  # user-supplied: trusted context, unverifiable provenance

LOW_QUALITY_PATH_MARKERS = ("/blog", "/forum", "/opinion", "/sponsored", "/thread", "/community", "/comments")
PRIMARY_DOCUMENT_MARKERS = (".pdf", "/publications/", "/reports/", "/abs/", "/doi/")


@dataclass(frozen=True)
class _Match:
    tier: SourceTier
    base: float
    rule: str


def normalize_domain(url: str) -> str:
    """`https://www.Reuters.com/x` → `reuters.com`; `upload://...` → `upload`."""
    parsed = urlparse(url if "://" in url else f"https://{url}")
    host = (parsed.hostname or "").lower().rstrip(".")
    if parsed.scheme == "upload":
        return "upload"
    return host.removeprefix("www.")


def _domain_matches(host: str, domain: str) -> bool:
    return host == domain or host.endswith("." + domain)


def _match_rule(host: str) -> _Match:
    for domain, score in TIER_1_DOMAINS.items():
        if _domain_matches(host, domain):
            return _Match(SourceTier.TIER_1, score, f"tier1_domain:{domain}")
    for suffix, score in TIER_1_SUFFIXES.items():
        if host.endswith(suffix):
            return _Match(SourceTier.TIER_1, score, f"tier1_suffix:{suffix}")
    for domain, score in TIER_2_DOMAINS.items():
        if _domain_matches(host, domain):
            return _Match(SourceTier.TIER_2, score, f"tier2_domain:{domain}")
    for domain, score in TIER_3_DOMAINS.items():
        if _domain_matches(host, domain):
            return _Match(SourceTier.TIER_3, score, f"tier3_platform:{domain}")
    return _Match(SourceTier.TIER_3, TIER_3_DEFAULT, "tier3_default:unrecognised_domain")


def score_source(
    url: str,
    *,
    content: str = "",
    published_date: str | None = None,
    is_upload: bool = False,
    injection_detected: bool = False,
) -> SourceAuthority:
    """Score a source deterministically. Same inputs always yield the same output."""
    if is_upload or url.startswith("upload://"):
        return SourceAuthority(
            score=UPLOADED_DOCUMENT_SCORE,
            tier=SourceTier.TIER_2,
            rule="uploaded_document",
            signals=["user-supplied document: trusted context, provenance not independently verified"],
        )

    host = normalize_domain(url)
    match = _match_rule(host)
    score = match.base
    signals: list[str] = [f"base {match.base:.2f} via {match.rule}"]
    path = (urlparse(url).path or "").lower()

    if any(marker in path for marker in LOW_QUALITY_PATH_MARKERS):
        score -= 0.04
        signals.append("-0.04 blog/forum/opinion path")
    if match.tier is SourceTier.TIER_1 and any(m in path for m in PRIMARY_DOCUMENT_MARKERS):
        score += 0.02
        signals.append("+0.02 primary publication path")

    length = len(content.strip())
    if content and length < 400:
        score -= 0.05
        signals.append(f"-0.05 thin content ({length} chars)")
    elif length > 3000:
        score += 0.01
        signals.append("+0.01 substantive content")

    if published_date:
        score += 0.01
        signals.append("+0.01 dated publication")

    if injection_detected:
        score -= 0.10
        signals.append("-0.10 contained prompt-injection text (neutralised)")

    if not url.lower().startswith("https://") and "://" in url:
        score -= 0.02
        signals.append("-0.02 non-HTTPS")

    low, high = TIER_BANDS[match.tier]
    clamped = round(min(high, max(low, score)), 3)
    return SourceAuthority(score=clamped, tier=match.tier, rule=match.rule, signals=signals)


def tier_label(tier: SourceTier) -> str:
    return {
        SourceTier.TIER_1: "Tier 1 · primary/official",
        SourceTier.TIER_2: "Tier 2 · established editorial",
        SourceTier.TIER_3: "Tier 3 · general web",
    }[tier]
