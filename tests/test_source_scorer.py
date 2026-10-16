"""Tools-layer tests: authority scoring, 3-tier retrieval cascade, ingestion,
span-exact chunking, and the hybrid evidence index."""

from __future__ import annotations

import io
import itertools
import json

import httpx
import pytest

from src.core.exceptions import UnsupportedDocumentTypeError
from src.core.mesh_gateway import MeshGateway
from src.schemas.domain import RetrievalTier, SourceTier
from src.storage.vector_store import (
    BM25Index,
    HybridVectorStore,
    build_chunks,
    reciprocal_rank_fusion,
)
from src.tools.document_ingestor import chunk_text, clean_text, ingest_bytes
from src.tools.source_scorer import TIER_BANDS, normalize_domain, score_source
from src.tools.web_search import OfflineCorpus, WebRetriever, extract_main_text
from tests.conftest import FakeMeshClient

LONG = "Substantive analysis paragraph. " * 120  # > 3000 chars


# ---------------------------------------------------------------------------
# Source authority weighting
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("url", "tier"),
    [
        ("https://www.sec.gov/edgar", SourceTier.TIER_1),
        ("https://www.rbi.org.in/Scripts/FS_Overview.aspx", SourceTier.TIER_1),
        ("https://www.iea.org/reports/renewables-2023", SourceTier.TIER_1),
        ("https://arxiv.org/abs/1706.03762", SourceTier.TIER_1),
        ("https://www.nature.com/articles/x", SourceTier.TIER_1),
        ("https://ieeexplore.ieee.org/document/1", SourceTier.TIER_1),
        ("https://cs.stanford.edu/paper", SourceTier.TIER_1),
        ("https://www.energy.gov/eere/solar", SourceTier.TIER_1),
        ("https://pib.gov.in/release", SourceTier.TIER_1),
        ("https://www.reuters.com/markets/x", SourceTier.TIER_2),
        ("https://www.bloomberg.com/news/x", SourceTier.TIER_2),
        ("https://www.ft.com/content/x", SourceTier.TIER_2),
        ("https://www.economist.com/finance", SourceTier.TIER_2),
        ("https://someone.medium.com/post", SourceTier.TIER_3),
        ("https://www.reddit.com/r/investing", SourceTier.TIER_3),
        ("https://random-blog.example/blog/post", SourceTier.TIER_3),
    ],
)
def test_domain_tiers(url, tier):
    authority = score_source(url, content=LONG)
    assert authority.tier is tier
    low, high = TIER_BANDS[tier]
    assert low <= authority.score <= high


def test_tier_ordering_is_strict():
    gov = score_source("https://www.sec.gov/x", content=LONG).score
    news = score_source("https://www.reuters.com/x", content=LONG).score
    blog = score_source("https://x.medium.com/y", content=LONG).score
    assert gov > news > blog


def test_scoring_is_deterministic_and_explained():
    a = score_source("https://www.iea.org/reports/x", content=LONG, published_date="2024-01-01")
    b = score_source("https://www.iea.org/reports/x", content=LONG, published_date="2024-01-01")
    assert a == b
    assert a.rule == "tier1_domain:iea.org"
    assert any("primary publication" in s for s in a.signals)


def test_adjustments_never_leave_tier_band():
    thin_blog_gov = score_source("http://data.example.gov/blog/x", content="short")
    assert thin_blog_gov.tier is SourceTier.TIER_1
    assert thin_blog_gov.score == TIER_BANDS[SourceTier.TIER_1][0]
    assert any("thin content" in s for s in thin_blog_gov.signals)
    forum = score_source("https://www.quora.com/forum/q", content="tiny")
    assert forum.score == TIER_BANDS[SourceTier.TIER_3][0]


def test_lookalike_domains_do_not_inherit_authority():
    assert score_source("https://notreuters.com/x").tier is SourceTier.TIER_3
    assert score_source("https://sec.gov.evil.example/x").tier is SourceTier.TIER_3
    assert normalize_domain("https://WWW.Reuters.com/path") == "reuters.com"


def test_uploaded_documents_get_explicit_rule():
    authority = score_source("upload://DOC-1/report.pdf", is_upload=True)
    assert authority.rule == "uploaded_document" and authority.tier is SourceTier.TIER_2


# ---------------------------------------------------------------------------
# 3-tier retrieval cascade
# ---------------------------------------------------------------------------
class FakeTavily:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[dict] = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("tavily quota exceeded")
        return {
            "results": [
                {"url": "https://www.iea.org/a", "title": "IEA", "content": "snippet", "raw_content": LONG},
                {"url": "https://www.iea.org/a#frag", "title": "dup", "content": "dup", "raw_content": LONG},
            ]
        }


class FakeDDGS:
    def __init__(self, hits):
        self.hits = hits

    def text(self, query, max_results=5):
        return self.hits[:max_results]


def _html_client() -> httpx.Client:
    article = "<p>" + ("Solar module prices fell sharply in 2023 according to analysts. " * 12) + "</p>"
    html = f"<html><head><title>T</title></head><body><nav>menu</nav><article>{article}</article></body></html>"

    def handler(request: httpx.Request) -> httpx.Response:
        if "broken" in str(request.url):
            return httpx.Response(500)
        return httpx.Response(200, text=html, headers={"content-type": "text/html"})

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_tavily_is_tier1_and_never_asks_for_an_answer(make_settings):
    tavily = FakeTavily()
    retriever = WebRetriever(make_settings(search_mode="auto"), tavily_client=tavily)
    outcome = retriever.search("global solar additions 2023")
    assert outcome.tier is RetrievalTier.TAVILY
    call = tavily.calls[0]
    assert call["include_answer"] is False
    assert call["include_raw_content"] is True
    assert call["search_depth"] == "advanced"
    assert len(outcome.results) == 1  # fragment duplicate removed


def test_tavily_failure_falls_back_to_duckduckgo(make_settings):
    hits = [
        {"href": "https://www.reuters.com/solar", "title": "Reuters", "body": "snippet"},
        {"href": "https://broken.example/x", "title": "Broken", "body": "Fallback snippet text"},
    ]
    retriever = WebRetriever(
        make_settings(search_mode="auto"),
        tavily_client=FakeTavily(fail=True),
        ddgs_factory=lambda: FakeDDGS(hits),
        http_client=_html_client(),
    )
    outcome = retriever.search("solar module prices 2023")
    assert outcome.tier is RetrievalTier.DUCKDUCKGO
    assert [a.provider for a in outcome.attempts] == ["tavily", "duckduckgo"]
    assert outcome.attempts[0].ok is False and "quota" in (outcome.attempts[0].error or "")
    full_page = outcome.results[0]
    assert "module prices fell" in full_page.content and "menu" not in full_page.content
    assert outcome.results[1].content == "Fallback snippet text"  # fetch failed -> snippet


def test_missing_tavily_key_skips_tier1(make_settings):
    retriever = WebRetriever(make_settings(search_mode="auto"), ddgs_factory=lambda: FakeDDGS([]))
    assert retriever.providers() == ["duckduckgo", "offline"]


def test_everything_down_falls_back_to_offline_corpus(make_settings):
    def broken():
        raise RuntimeError("network unreachable")

    retriever = WebRetriever(make_settings(search_mode="auto"), ddgs_factory=broken)
    outcome = retriever.search("RBI repo rate inflation target")
    assert outcome.tier is RetrievalTier.OFFLINE
    assert any("rbi.org.in" in r.url for r in outcome.results)


def test_offline_corpus_relevance_gate():
    corpus = OfflineCorpus()
    assert corpus.search("best pizza dough hydration ratio naples") == []
    hits = corpus.search("Great Wall of China visible from the Moon")
    assert hits, "on-topic query must pass the relevance gate"
    assert {"nasa.gov", "scientificamerican.com"} & {h.url.split("/")[2].removeprefix("www.").removeprefix("earthobservatory.") for h in hits[:3]}


def test_extract_main_text_strips_boilerplate():
    html = "<html><body><header>Site</header><main><h2>Key findings here</h2><p>" + "x " * 50 + "</p></main><footer>(c)</footer></body></html>"
    _title, text = extract_main_text(html)
    assert "Key findings here" in text and "(c)" not in text and "Site" not in text


# ---------------------------------------------------------------------------
# Ingestion + span-exact chunking
# ---------------------------------------------------------------------------
def test_chunks_are_exact_spans_with_overlap():
    corpus = OfflineCorpus()
    text = clean_text(corpus.documents[0]["content"] * 3)
    spans = chunk_text(text, chunk_size=400, overlap=80)
    assert len(spans) > 3
    for span in spans:
        assert text[span.start : span.end] == span.text
        assert span.text == span.text.strip()
    for prev, nxt in itertools.pairwise(spans):
        assert nxt.start < prev.end, "consecutive chunks must overlap"
        assert nxt.start > prev.start
    assert spans[-1].end == len(text)


def test_chunk_text_handles_unbroken_text():
    text = "a" * 2000
    spans = chunk_text(text, chunk_size=500, overlap=100)
    assert spans[0].text == "a" * 500 and spans[-1].end == 2000


def test_ingest_txt_and_markdown():
    doc = ingest_bytes("notes.md", b"# Heading\n\nThe **repo rate** is set by the [MPC](https://x).\n\n\n\nEnd.")
    assert doc.content_type == "text/markdown"
    assert "#" not in doc.content and "**" not in doc.content and "MPC" in doc.content
    assert doc.doc_id.startswith("DOC-") and len(doc.content_hash) == 64


def test_ingest_pdf_roundtrip():
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    buf = io.BytesIO()
    pdf = canvas.Canvas(buf, pagesize=A4)
    pdf.drawString(72, 760, "Quarterly report: revenue grew 12 percent year on year.")
    pdf.showPage()
    pdf.drawString(72, 760, "Second page discusses operating margins in detail.")
    pdf.save()
    doc = ingest_bytes("q3.pdf", buf.getvalue())
    assert doc.page_count == 2
    assert "revenue grew 12 percent" in doc.content and "operating margins" in doc.content


def test_ingest_rejects_unsupported_types():
    with pytest.raises(UnsupportedDocumentTypeError):
        ingest_bytes("malware.exe", b"MZ....")


# ---------------------------------------------------------------------------
# Hybrid evidence index
# ---------------------------------------------------------------------------
def _chunks(source_id: str, text: str):
    text = clean_text(text)
    return build_chunks(
        source_id=source_id,
        content=text,
        spans=chunk_text(text, chunk_size=300, overlap=50),
        authority_score=0.9,
        url=f"https://{source_id.lower()}.example",
        title=source_id,
    )


def _corpus_chunks():
    corpus = OfflineCorpus()
    out = []
    for i, doc in enumerate(corpus.documents[:8], start=1):
        out.extend(_chunks(f"SRC-{i:03d}", doc["content"]))
    return out


def test_bm25_ranks_relevant_documents_first():
    index = BM25Index()
    index.add("a", "repo rate inflation target monetary policy committee")
    index.add("b", "solar photovoltaic module prices fell")
    index.add("c", "transformer attention heads encoder decoder")
    assert index.search("what is the repo rate target")[0][0] == "a"
    assert index.search("attention heads", allowed={"a", "b"}) == []


def test_rrf_rewards_agreement():
    fused = dict(reciprocal_rank_fusion([["x", "y", "z"], ["y", "x", "w"]]))
    assert fused["x"] == fused["y"] > fused["z"]


def test_tfidf_store_search_and_source_filter(make_settings):
    store = HybridVectorStore(make_settings(vector_backend="tfidf"))
    chunks = _corpus_chunks()
    assert store.add_chunks("run_test", chunks) == len(chunks)
    assert store.add_chunks("run_test", chunks) == 0  # idempotent
    hits = store.search("run_test", "Transformer BLEU English-to-German translation", k=3)
    assert hits and "BLEU" in hits[0].text
    assert hits[0].retrieval_score > 0
    filtered = store.search("run_test", "solar module prices", k=5, source_ids={"SRC-001"})
    assert filtered and {h.source_id for h in filtered} == {"SRC-001"}
    assert store.status()["dense"] is None


def test_hybrid_store_uses_mesh_embeddings_with_chroma(make_settings):
    settings = make_settings(vector_backend="hybrid")
    client = FakeMeshClient()
    gateway = MeshGateway(settings, client=client)
    store = HybridVectorStore(settings, gateway)
    assert store.dense_enabled
    store.add_chunks("run_hybrid", _corpus_chunks())
    hits = store.search("run_hybrid", "microplastics detected in human blood donors", k=3)
    assert store.dense_enabled, store.dense_disabled_reason
    assert client.embedding_calls, "dense path must embed via Mesh"
    assert all(call["model"] == "openai/text-embedding-3-small" for call in client.embedding_calls)
    assert hits and any("blood" in h.text for h in hits)


def test_hybrid_store_degrades_to_sparse_on_embedding_failure(make_settings):
    settings = make_settings(vector_backend="hybrid")

    class ExplodingEmbeddings(FakeMeshClient):
        def _embed_create(self, *, model, input):
            raise httpx.ConnectError("mesh unreachable")

    gateway = MeshGateway(settings, client=ExplodingEmbeddings(), sleep=lambda _s: None)
    store = HybridVectorStore(settings, gateway)
    store.add_chunks("run_x", _corpus_chunks())
    assert not store.dense_enabled and "embedding" in (store.dense_disabled_reason or "")
    assert store.search("run_x", "repo rate", k=2) is not None


def test_offline_corpus_is_valid_json_and_has_traps():
    from src.tools.web_search import OFFLINE_CORPUS_PATH

    payload = json.loads(OFFLINE_CORPUS_PATH.read_text())
    urls = [d["url"] for d in payload["documents"]]
    assert len(urls) == len(set(urls)) >= 20
    assert sum(".example/" in u for u in urls) >= 5  # adversarial low-authority sources
