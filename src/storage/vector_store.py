"""Hybrid evidence index: BM25/TF-IDF sparse retrieval + ChromaDB dense retrieval.

* **Sparse** (always on): a pure-Python Okapi BM25 index. Deterministic, zero
  dependencies, works offline and in CI.
* **Dense** (`VECTOR_BACKEND=hybrid` + Mesh configured): ChromaDB collection
  whose vectors come from Mesh API embeddings (`MeshGateway.embed`). Chroma's
  built-in embedding functions are deliberately disabled so no model download
  or non-Mesh inference ever happens.
* **Fusion**: Reciprocal Rank Fusion (k=60) over both rankings.

Every indexed item is an `EvidenceChunk` that carries its exact
`[char_start, char_end)` span in the parent `SourceDocument.content`.
Namespaces isolate runs (`run_xxx`) from the shared upload library.
If the dense path fails (network, quota), the index degrades to sparse-only
and records why — retrieval never crashes a run.
"""

from __future__ import annotations

import logging
import math
import re
import threading
import uuid
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from src.core.config import Settings
from src.schemas.domain import EvidenceChunk

if TYPE_CHECKING:
    from src.core.mesh_gateway import MeshGateway

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.'][a-z0-9]+)*")
STOPWORDS = frozenset(
    ["a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "can", "could", "did", "do", "does", "for", "from", "had", "has", "have", "how", "i", "if", "in", "into", "is", "it", "its", "may", "might", "more", "most", "no", "not", "of", "on", "or", "our", "over", "so", "such", "than", "that", "the", "their", "them", "then", "there", "these", "they", "this", "those", "through", "to", "was", "we", "were", "what", "when", "where", "which", "while", "who", "why", "will", "with", "would", "you", "your", "about", "after", "also", "any", "between", "both", "during", "each", "few", "further", "here", "just", "only", "other", "own", "same", "should", "some", "very"]
)
RRF_K = 60


def tokenize(text: str) -> list[str]:
    """Lower-case alphanumeric tokens without stopwords; keeps numbers like `6.5` and `2023`."""
    return [t for t in _TOKEN_RE.findall((text or "").lower()) if t not in STOPWORDS and len(t) > 1]


class BM25Index:
    """Okapi BM25 over a growing set of documents."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._ids: list[str] = []
        self._tfs: list[Counter[str]] = []
        self._lengths: list[int] = []
        self._df: Counter[str] = Counter()
        self._positions: dict[str, int] = {}

    def __len__(self) -> int:
        return len(self._ids)

    def add(self, doc_id: str, text: str) -> None:
        if doc_id in self._positions:
            return
        tokens = tokenize(text)
        tf = Counter(tokens)
        self._positions[doc_id] = len(self._ids)
        self._ids.append(doc_id)
        self._tfs.append(tf)
        self._lengths.append(len(tokens))
        self._df.update(tf.keys())

    def search(self, query: str, k: int = 10, allowed: set[str] | None = None) -> list[tuple[str, float]]:
        q_terms = tokenize(query)
        if not q_terms or not self._ids:
            return []
        n = len(self._ids)
        avgdl = (sum(self._lengths) / n) or 1.0
        scores: list[tuple[str, float]] = []
        for idx, doc_id in enumerate(self._ids):
            if allowed is not None and doc_id not in allowed:
                continue
            tf = self._tfs[idx]
            dl = self._lengths[idx]
            score = 0.0
            for term in set(q_terms):
                freq = tf.get(term)
                if not freq:
                    continue
                df = self._df[term]
                idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
                score += idf * (freq * (self.k1 + 1)) / (freq + self.k1 * (1 - self.b + self.b * dl / avgdl))
            if score > 0:
                scores.append((doc_id, score))
        scores.sort(key=lambda pair: pair[1], reverse=True)
        return scores[:k]


def reciprocal_rank_fusion(rankings: Iterable[Sequence[str]], k: int = RRF_K) -> list[tuple[str, float]]:
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda pair: pair[1], reverse=True)


@dataclass
class _Namespace:
    chunks: dict[str, EvidenceChunk] = field(default_factory=dict)
    bm25: BM25Index = field(default_factory=BM25Index)
    collection: Any | None = None


class HybridVectorStore:
    """Thread-safe, namespaced evidence index with sparse + dense retrieval."""

    def __init__(self, settings: Settings, gateway: MeshGateway | None = None) -> None:
        self.settings = settings
        self.gateway = gateway
        self._namespaces: dict[str, _Namespace] = {}
        self._lock = threading.RLock()
        self._instance = uuid.uuid4().hex[:8]
        self._chroma_client: Any | None = None
        self.dense_disabled_reason: str | None = None
        if settings.vector_backend != "hybrid":
            self.dense_disabled_reason = "VECTOR_BACKEND=tfidf"
        elif gateway is None or not gateway.configured:
            self.dense_disabled_reason = "Mesh API not configured (embeddings unavailable)"

    # -- status --------------------------------------------------------------
    @property
    def dense_enabled(self) -> bool:
        return self.dense_disabled_reason is None

    def status(self) -> dict[str, Any]:
        return {
            "sparse": "bm25",
            "dense": "chromadb+mesh-embeddings" if self.dense_enabled else None,
            "dense_disabled_reason": self.dense_disabled_reason,
            "namespaces": len(self._namespaces),
        }

    def _disable_dense(self, reason: str) -> None:
        if self.dense_disabled_reason is None:
            logger.warning("Dense retrieval disabled, falling back to BM25 only: %s", reason)
        self.dense_disabled_reason = reason

    # -- chroma --------------------------------------------------------------
    def _chroma(self) -> Any:
        if self._chroma_client is None:
            import chromadb
            from chromadb.config import Settings as ChromaSettings

            chroma_settings = ChromaSettings(anonymized_telemetry=False, allow_reset=False)
            if self.settings.chroma_persist_dir:
                self._chroma_client = chromadb.PersistentClient(
                    path=self.settings.chroma_persist_dir, settings=chroma_settings
                )
            else:
                self._chroma_client = chromadb.EphemeralClient(settings=chroma_settings)
        return self._chroma_client

    def _collection_name(self, namespace: str) -> str:
        safe = re.sub(r"[^a-zA-Z0-9_-]", "_", namespace)[:40]
        return f"dd_{self._instance}_{safe}"

    def _ensure_collection(self, ns_name: str, ns: _Namespace) -> Any | None:
        if not self.dense_enabled:
            return None
        if ns.collection is None:
            try:
                ns.collection = self._chroma().get_or_create_collection(
                    self._collection_name(ns_name),
                    embedding_function=None,  # vectors are supplied by Mesh API embeddings
                    metadata={"hnsw:space": "cosine"},
                )
            except Exception as exc:  # chromadb raises a variety of internal errors
                self._disable_dense(f"ChromaDB unavailable: {exc}")
                return None
        return ns.collection

    # -- write ---------------------------------------------------------------
    def add_chunks(self, namespace: str, chunks: Sequence[EvidenceChunk]) -> int:
        """Index chunks (idempotent on chunk_id). Returns the number of new chunks."""
        with self._lock:
            ns = self._namespaces.setdefault(namespace, _Namespace())
            fresh = [c for c in chunks if c.chunk_id not in ns.chunks]
            for chunk in fresh:
                ns.chunks[chunk.chunk_id] = chunk
                ns.bm25.add(chunk.chunk_id, f"{chunk.title}\n{chunk.text}")
            collection = self._ensure_collection(namespace, ns) if fresh else None

        if collection is not None and fresh and self.gateway is not None:
            try:
                vectors = self.gateway.embed(
                    [c.text for c in fresh], run_id=namespace if namespace.startswith("run_") else None
                )
                collection.add(
                    ids=[c.chunk_id for c in fresh],
                    embeddings=vectors,
                    documents=[c.text for c in fresh],
                    metadatas=[
                        {"source_id": c.source_id, "sub_question_id": c.sub_question_id or ""}
                        for c in fresh
                    ],
                )
            except Exception as exc:
                self._disable_dense(f"embedding/indexing failed: {exc}")
        return len(fresh)

    def drop(self, namespace: str) -> None:
        with self._lock:
            ns = self._namespaces.pop(namespace, None)
        if ns is not None and ns.collection is not None:
            try:
                self._chroma().delete_collection(self._collection_name(namespace))
            except Exception:
                logger.debug("Could not delete chroma collection for %s", namespace)

    # -- read ----------------------------------------------------------------
    def has_namespace(self, namespace: str) -> bool:
        return namespace in self._namespaces

    def count(self, namespace: str) -> int:
        ns = self._namespaces.get(namespace)
        return len(ns.chunks) if ns else 0

    def get(self, namespace: str, chunk_ids: Iterable[str]) -> list[EvidenceChunk]:
        ns = self._namespaces.get(namespace)
        if ns is None:
            return []
        return [ns.chunks[c] for c in chunk_ids if c in ns.chunks]

    def all_chunks(self, namespace: str) -> list[EvidenceChunk]:
        ns = self._namespaces.get(namespace)
        return list(ns.chunks.values()) if ns else []

    def search(
        self,
        namespace: str,
        query: str,
        k: int = 6,
        *,
        source_ids: Iterable[str] | None = None,
    ) -> list[EvidenceChunk]:
        """Hybrid search. Returned chunks carry their fused `retrieval_score`."""
        with self._lock:
            ns = self._namespaces.get(namespace)
            if ns is None or not ns.chunks:
                return []
            allowed: set[str] | None = None
            if source_ids is not None:
                wanted = set(source_ids)
                allowed = {cid for cid, c in ns.chunks.items() if c.source_id in wanted}
                if not allowed:
                    return []
            pool = max(k * 4, 20)
            sparse = [cid for cid, _ in ns.bm25.search(query, k=pool, allowed=allowed)]
            collection = ns.collection

        rankings: list[list[str]] = [sparse]
        if collection is not None and self.dense_enabled and self.gateway is not None:
            dense = self._dense_search(collection, namespace, query, pool, allowed)
            if dense:
                rankings.append(dense)

        fused = reciprocal_rank_fusion(rankings)[:k]
        out: list[EvidenceChunk] = []
        for chunk_id, score in fused:
            chunk = ns.chunks.get(chunk_id)
            if chunk is not None:
                out.append(chunk.model_copy(update={"retrieval_score": round(score, 6)}))
        return out

    def _dense_search(
        self, collection: Any, namespace: str, query: str, pool: int, allowed: set[str] | None
    ) -> list[str]:
        assert self.gateway is not None
        try:
            total = collection.count()
            if total == 0:
                return []
            vector = self.gateway.embed(
                [query], run_id=namespace if namespace.startswith("run_") else None
            )[0]
            kwargs: dict[str, Any] = {"query_embeddings": [vector], "n_results": min(pool, total)}
            result = collection.query(**kwargs)
            ids = result.get("ids", [[]])[0]
            return [i for i in ids if allowed is None or i in allowed]
        except Exception as exc:
            self._disable_dense(f"dense query failed: {exc}")
            return []


def build_chunks(
    *,
    source_id: str,
    content: str,
    spans: Iterable[Any],
    authority_score: float,
    url: str,
    title: str,
    sub_question_id: str | None = None,
) -> list[EvidenceChunk]:
    """Turn `TextSpan`s of a document into `EvidenceChunk`s with exact offsets."""
    chunks: list[EvidenceChunk] = []
    for idx, span in enumerate(spans, start=1):
        assert content[span.start : span.end] == span.text, "span/text mismatch"
        chunks.append(
            EvidenceChunk(
                chunk_id=f"{source_id}#c{idx}",
                source_id=source_id,
                sub_question_id=sub_question_id,
                text=span.text,
                char_start=span.start,
                char_end=span.end,
                authority_score=authority_score,
                url=url,
                title=title,
            )
        )
    return chunks
