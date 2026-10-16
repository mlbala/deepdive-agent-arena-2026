"""3-Tier Resilient Web Retriever.

    Tier 1  Tavily Search API      (TAVILY_API_KEY present)
            search_depth="advanced", include_raw_content=True, include_answer=False
            -> Tavily is used purely as a document scraper; 100% of reasoning stays on Mesh.
    Tier 2  DuckDuckGo + httpx + BeautifulSoup full-page extraction
            (no key required; automatic fallback if Tavily is missing or errors)
    Tier 3  Curated offline corpus (`offline_corpus.json`) with a BM25 relevance gate
            (hermetic CI / no-network fallback; irrelevant queries return nothing)

`SEARCH_MODE` selects the cascade: auto (1→2→3), tavily (1→3), ddg (2→3), offline (3).
Every provider attempt is recorded in `SearchOutcome.attempts` for the audit trail.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urldefrag

import httpx
from bs4 import BeautifulSoup

from src.core.config import Settings
from src.core.exceptions import SearchProviderError
from src.schemas.domain import RawSearchResult, RetrievalTier
from src.storage.vector_store import BM25Index, tokenize
from src.tools.document_ingestor import clean_text, extract_pdf_text

logger = logging.getLogger(__name__)

OFFLINE_CORPUS_PATH = Path(__file__).with_name("offline_corpus.json")
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36 DeepDiveResearchBot/1.0"
)
_BOILERPLATE_TAGS = ("script", "style", "noscript", "nav", "footer", "header", "aside", "form", "svg", "iframe")
MIN_PAGE_CHARS = 300


@dataclass
class ProviderAttempt:
    provider: str
    ok: bool
    results: int
    latency_ms: int
    error: str | None = None


@dataclass
class SearchOutcome:
    query: str
    results: list[RawSearchResult] = field(default_factory=list)
    tier: RetrievalTier | None = None
    attempts: list[ProviderAttempt] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "tier": self.tier.value if self.tier else None,
            "results": len(self.results),
            "attempts": [a.__dict__ for a in self.attempts],
        }


def normalize_url(url: str) -> str:
    return urldefrag(url.strip())[0].rstrip("/")


# ---------------------------------------------------------------------------
# Tier 3: offline corpus
# ---------------------------------------------------------------------------
class OfflineCorpus:
    """BM25 search over the curated corpus with a relevance gate."""

    def __init__(self, path: Path = OFFLINE_CORPUS_PATH) -> None:
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.documents: list[dict[str, Any]] = payload["documents"]
        self._index = BM25Index()
        for i, doc in enumerate(self.documents):
            self._index.add(str(i), f"{doc['title']}\n{doc['title']}\n{doc['content']}")

    def search(self, query: str, max_results: int = 5) -> list[RawSearchResult]:
        hits = self._index.search(query, k=max_results * 2)
        if not hits:
            return []
        q_terms = set(tokenize(query))
        min_overlap = 1 if len(q_terms) <= 2 else 2
        top = hits[0][1]
        results: list[RawSearchResult] = []
        for doc_idx, score in hits:
            doc = self.documents[int(doc_idx)]
            overlap = q_terms & set(tokenize(f"{doc['title']} {doc['content']}"))
            if score < 0.3 * top or len(overlap) < min_overlap:
                continue
            results.append(
                RawSearchResult(
                    url=doc["url"],
                    title=doc["title"],
                    content=clean_text(doc["content"]),
                    retrieval_tier=RetrievalTier.OFFLINE,
                    query=query,
                    published_date=doc.get("published_date"),
                    provider_score=round(score, 4),
                )
            )
            if len(results) >= max_results:
                break
        return results


@lru_cache(maxsize=1)
def default_corpus() -> OfflineCorpus:
    return OfflineCorpus()


# ---------------------------------------------------------------------------
# HTML extraction
# ---------------------------------------------------------------------------
def extract_main_text(html: str) -> tuple[str, str]:
    """Return (title, main_text) from an HTML page using BeautifulSoup heuristics."""
    soup = BeautifulSoup(html, "html.parser")
    title = (soup.title.get_text(" ", strip=True) if soup.title else "").strip()
    for tag in soup(_BOILERPLATE_TAGS):
        tag.decompose()
    container = soup.find("article") or soup.find("main") or soup.body or soup
    blocks: list[str] = []
    for node in container.find_all(["h1", "h2", "h3", "p", "li", "blockquote", "td"]):
        text = node.get_text(" ", strip=True)
        is_heading = node.name in ("h1", "h2", "h3")
        if (is_heading and len(text) >= 8) or len(text) >= 40:
            blocks.append(text)
    body = "\n\n".join(dict.fromkeys(blocks))  # de-duplicate repeated blocks, keep order
    if len(body) < MIN_PAGE_CHARS:
        body = container.get_text("\n", strip=True)
    return title, clean_text(body)


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------
class WebRetriever:
    """Cascading retriever. Inject `tavily_client`, `ddgs_factory`, `http_client` in tests."""

    def __init__(
        self,
        settings: Settings,
        *,
        tavily_client: Any | None = None,
        ddgs_factory: Callable[[], Any] | None = None,
        http_client: httpx.Client | None = None,
        corpus: OfflineCorpus | None = None,
    ) -> None:
        self.settings = settings
        self._tavily_client = tavily_client
        self._ddgs_factory = ddgs_factory
        self._http_client = http_client
        self._corpus = corpus

    # -- cascade -------------------------------------------------------------
    def providers(self) -> list[str]:
        tavily_available = self._tavily_client is not None or self.settings.tavily_configured
        mode = self.settings.search_mode
        if mode == "offline":
            return ["offline"]
        if mode == "tavily":
            return ["tavily", "offline"]
        if mode == "ddg":
            return ["duckduckgo", "offline"]
        return (["tavily"] if tavily_available else []) + ["duckduckgo", "offline"]

    def search(self, query: str, max_results: int | None = None) -> SearchOutcome:
        n = max_results or self.settings.search_max_results
        outcome = SearchOutcome(query=query)
        handlers: dict[str, tuple[Callable[[str, int], list[RawSearchResult]], RetrievalTier]] = {
            "tavily": (self._search_tavily, RetrievalTier.TAVILY),
            "duckduckgo": (self._search_ddg, RetrievalTier.DUCKDUCKGO),
            "offline": (self._search_offline, RetrievalTier.OFFLINE),
        }
        for provider in self.providers():
            handler, tier = handlers[provider]
            started = time.perf_counter()
            try:
                results = self._dedupe(handler(query, n))
            except Exception as exc:  # any provider failure → next tier, never crash the run
                outcome.attempts.append(
                    ProviderAttempt(provider, False, 0, _ms(started), f"{type(exc).__name__}: {exc}"[:300])
                )
                logger.warning("Search provider %s failed for %r: %s", provider, query, exc)
                continue
            outcome.attempts.append(ProviderAttempt(provider, bool(results), len(results), _ms(started)))
            if results:
                outcome.results = results[:n]
                outcome.tier = tier
                return outcome
        return outcome

    @staticmethod
    def _dedupe(results: list[RawSearchResult]) -> list[RawSearchResult]:
        seen: set[str] = set()
        out: list[RawSearchResult] = []
        for r in results:
            key = normalize_url(r.url)
            if key in seen or not r.content.strip():
                continue
            seen.add(key)
            out.append(r)
        return out

    # -- tier 1: Tavily ------------------------------------------------------
    def _tavily(self) -> Any:
        if self._tavily_client is None:
            if self.settings.tavily_api_key is None:
                raise SearchProviderError("tavily", "TAVILY_API_KEY not configured")
            from tavily import TavilyClient

            self._tavily_client = TavilyClient(api_key=self.settings.tavily_api_key.get_secret_value())
        return self._tavily_client

    def _search_tavily(self, query: str, n: int) -> list[RawSearchResult]:
        response = self._tavily().search(
            query=query,
            search_depth="advanced",
            max_results=n,
            include_raw_content=True,
            include_answer=False,  # Tavily scrapes; Mesh reasons.
        )
        results: list[RawSearchResult] = []
        for item in response.get("results", []):
            raw = item.get("raw_content") or item.get("content") or ""
            content = clean_text(raw)[: self.settings.max_document_chars]
            if not content:
                continue
            results.append(
                RawSearchResult(
                    url=item.get("url", ""),
                    title=item.get("title") or item.get("url", ""),
                    content=content,
                    retrieval_tier=RetrievalTier.TAVILY,
                    query=query,
                    published_date=item.get("published_date"),
                    provider_score=item.get("score"),
                )
            )
        return results

    # -- tier 2: DuckDuckGo + BeautifulSoup ----------------------------------
    def _ddgs(self) -> Any:
        if self._ddgs_factory is not None:
            return self._ddgs_factory()
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from duckduckgo_search import DDGS

            return DDGS(timeout=int(self.settings.fetch_timeout_seconds))

    def _search_ddg(self, query: str, n: int) -> list[RawSearchResult]:
        hits = list(self._ddgs().text(query, max_results=n) or [])
        if not hits:
            return []
        with ThreadPoolExecutor(max_workers=min(5, len(hits))) as pool:
            pages = list(pool.map(lambda h: self.fetch_page(h.get("href", "")), hits))
        results: list[RawSearchResult] = []
        for hit, page in zip(hits, pages, strict=True):
            url = hit.get("href", "")
            snippet = clean_text(hit.get("body", ""))
            title, text = page if page else ("", "")
            content = text if len(text) >= MIN_PAGE_CHARS else snippet
            if not url or not content:
                continue
            results.append(
                RawSearchResult(
                    url=url,
                    title=hit.get("title") or title or url,
                    content=content[: self.settings.max_document_chars],
                    retrieval_tier=RetrievalTier.DUCKDUCKGO,
                    query=query,
                )
            )
        return results

    def fetch_page(self, url: str) -> tuple[str, str] | None:
        """Download a page (HTML or PDF) and return (title, text); None on failure."""
        if not url.startswith(("http://", "https://")):
            return None
        client = self._http_client or httpx.Client(
            timeout=self.settings.fetch_timeout_seconds,
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT, "Accept": "text/html,application/pdf;q=0.9,*/*;q=0.5"},
        )
        try:
            response = client.get(url)
            response.raise_for_status()
            ctype = response.headers.get("content-type", "").lower()
            if "pdf" in ctype or url.lower().endswith(".pdf"):
                text, _pages = extract_pdf_text(response.content)
                return "", clean_text(text)
            if "html" not in ctype and "text" not in ctype:
                return None
            return extract_main_text(response.text)
        except Exception as exc:
            logger.debug("Fetch failed for %s: %s", url, exc)
            return None
        finally:
            if self._http_client is None:
                client.close()

    # -- tier 3: offline -----------------------------------------------------
    def _search_offline(self, query: str, n: int) -> list[RawSearchResult]:
        corpus = self._corpus or default_corpus()
        return corpus.search(query, max_results=n)


def _ms(started: float) -> int:
    return round((time.perf_counter() - started) * 1000)
