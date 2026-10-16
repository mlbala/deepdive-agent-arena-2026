"""PDF / TXT / MD ingestion and span-exact chunking.

`chunk_text` is the single chunker for every evidence source (web pages,
uploads, offline corpus). It guarantees `content[span.start:span.end] ==
span.text`, which is what lets the Verifier map an LLM-quoted passage back
to an exact character range in the original document.
"""

from __future__ import annotations

import hashlib
import io
import re
import unicodedata
import uuid
from dataclasses import dataclass
from pathlib import PurePath

from src.core.exceptions import DocumentIngestionError, UnsupportedDocumentTypeError

SUPPORTED_TYPES: dict[str, str] = {
    ".pdf": "application/pdf",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
}
MAX_UPLOAD_BYTES = 15 * 1024 * 1024

_SPACES = re.compile(r"[ \t\f\v  -​　]+")
_MANY_NEWLINES = re.compile(r"\n{3,}")
_SENTENCE_END = re.compile(r"[.!?][\"')\]]?\s")


@dataclass(frozen=True)
class TextSpan:
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class IngestedDocument:
    doc_id: str
    filename: str
    content_type: str
    content: str
    page_count: int
    content_hash: str


def clean_text(text: str) -> str:
    """Normalise unicode + whitespace while preserving paragraph breaks."""
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "".join(ch for ch in text if ch == "\n" or ch == "\t" or unicodedata.category(ch)[0] != "C")
    lines = [_SPACES.sub(" ", line).strip() for line in text.split("\n")]
    joined = "\n".join(lines)
    return _MANY_NEWLINES.sub("\n\n", joined).strip()


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _find_boundary(text: str, start: int, end: int, min_end: int) -> int:
    """Best break position in (min_end, end]: paragraph > sentence > whitespace."""
    window = text[min_end:end]
    para = window.rfind("\n\n")
    if para != -1:
        return min_end + para + 2
    sentence_ends = list(_SENTENCE_END.finditer(window))
    if sentence_ends:
        return min_end + sentence_ends[-1].end()
    space = window.rfind(" ")
    if space != -1:
        return min_end + space + 1
    return end


def chunk_text(text: str, *, chunk_size: int = 900, overlap: int = 150) -> list[TextSpan]:
    """Split `text` into overlapping, boundary-aligned spans with exact offsets."""
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    spans: list[TextSpan] = []
    n = len(text)
    start = 0
    while start < n:
        while start < n and text[start].isspace():
            start += 1
        if start >= n:
            break
        end = min(n, start + chunk_size)
        if end < n:
            end = _find_boundary(text, start, end, start + int(chunk_size * 0.6))
        trimmed_end = end
        while trimmed_end > start and text[trimmed_end - 1].isspace():
            trimmed_end -= 1
        if trimmed_end > start:
            spans.append(TextSpan(start, trimmed_end, text[start:trimmed_end]))
        if end >= n:
            break
        next_start = max(end - overlap, start + 1)
        # align the overlap start to a word boundary so chunks never begin mid-word
        while next_start < end and not text[next_start - 1].isspace():
            next_start += 1
        start = next_start
    return spans


def extract_pdf_text(data: bytes) -> tuple[str, int]:
    """Extract text from PDF bytes with pypdf. Returns (text, page_count)."""
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(io.BytesIO(data))
        pages = [page.extract_text() or "" for page in reader.pages]
    except (PdfReadError, ValueError, KeyError, OSError) as exc:
        raise DocumentIngestionError(f"Could not read PDF: {exc}") from exc
    return "\n\n".join(pages), len(pages)


def _decode_text(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-16", "latin-1"):
        try:
            text = data.decode(encoding)
        except UnicodeDecodeError:
            continue
        if encoding == "utf-16" and not data.startswith((b"\xff\xfe", b"\xfe\xff")):
            continue
        return text
    raise DocumentIngestionError("Could not decode text document")


def _strip_markdown(text: str) -> str:
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"(\*\*|__|`)", "", text)
    return text


def ingest_bytes(filename: str, data: bytes) -> IngestedDocument:
    """Validate, extract, and normalise an uploaded document."""
    suffix = PurePath(filename or "").suffix.lower()
    if suffix not in SUPPORTED_TYPES:
        raise UnsupportedDocumentTypeError(
            f"Unsupported file type '{suffix or 'none'}'. Upload PDF, TXT, or MD.",
            details={"supported": sorted(SUPPORTED_TYPES)},
        )
    if not data:
        raise DocumentIngestionError("Uploaded file is empty")
    if len(data) > MAX_UPLOAD_BYTES:
        raise DocumentIngestionError(f"File exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit")

    if suffix == ".pdf":
        raw, pages = extract_pdf_text(data)
    else:
        raw, pages = _decode_text(data), 1
        if suffix in (".md", ".markdown"):
            raw = _strip_markdown(raw)

    content = clean_text(raw)
    if len(content) < 20:
        raise DocumentIngestionError("Document contains no extractable text")
    return IngestedDocument(
        doc_id=f"DOC-{uuid.uuid4().hex[:10]}",
        filename=PurePath(filename).name,
        content_type=SUPPORTED_TYPES[suffix],
        content=content,
        page_count=pages,
        content_hash=content_hash(content),
    )
