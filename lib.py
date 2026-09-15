"""Shared chunking, BM25, and embedding helpers for the local hybrid search index."""
import hashlib
import json
import math
import os
import re
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# Folder to index - override with LOCAL_SEARCH_CONTENT_ROOT to point this at
# any directory of markdown files. This tool has no built-in tie to any
# particular project; the default below is just this author's own setup.
CONTENT_ROOT = Path(
    os.environ.get("LOCAL_SEARCH_CONTENT_ROOT", str(Path.home() / ".knowledge-base-sync" / "mirror"))
)

INDEX_DIR = Path(os.environ.get("LOCAL_SEARCH_INDEX_DIR", str(PROJECT_ROOT / "index")))
INDEX_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_TARGET_CHARS = 800
CHUNK_OVERLAP_CHARS = 100

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def chunk_markdown(text: str) -> list[str]:
    """Split on top-level/second-level headings, then further split long
    sections into overlapping windows so no chunk is too large to embed well."""
    sections: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        if re.match(r"^#{1,2}\s", line) and current:
            sections.append("\n".join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        sections.append("\n".join(current))

    chunks: list[str] = []
    for section in sections:
        section = section.strip()
        if not section:
            continue
        if len(section) <= CHUNK_TARGET_CHARS * 1.5:
            chunks.append(section)
            continue
        start = 0
        while start < len(section):
            end = start + CHUNK_TARGET_CHARS
            chunks.append(section[start:end])
            if end >= len(section):
                break
            start = end - CHUNK_OVERLAP_CHARS
    return chunks


def load_corpus() -> list[dict]:
    """Return one dict per chunk: {path, chunk_index, text, hash}."""
    records = []
    for path in sorted(CONTENT_ROOT.rglob("*.md")):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        rel_path = str(path.relative_to(CONTENT_ROOT))
        for i, chunk in enumerate(chunk_markdown(text)):
            h = hashlib.sha256(chunk.encode("utf-8")).hexdigest()[:16]
            records.append({
                "path": rel_path,
                "chunk_index": i,
                "text": chunk,
                "hash": h,
            })
    return records


class BM25:
    """Standard BM25Okapi, implemented directly (no external dependency)."""

    def __init__(self, docs_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs_tokens = docs_tokens
        self.doc_lens = [len(d) for d in docs_tokens]
        self.avg_doc_len = sum(self.doc_lens) / max(len(docs_tokens), 1)
        self.term_freqs: list[Counter] = [Counter(d) for d in docs_tokens]
        df: Counter = Counter()
        for tf in self.term_freqs:
            for term in tf:
                df[term] += 1
        n = len(docs_tokens)
        self.n_docs = n
        self.idf: dict[str, float] = {
            term: math.log((n - freq + 0.5) / (freq + 0.5) + 1.0)
            for term, freq in df.items()
        }

    def idf_for_term(self, term: str) -> float:
        """IDF for a term, including ones absent from the whole corpus (df=0).
        Such terms are the rarest possible and must score higher than any
        in-corpus term, not fall back to 0.0 (which would sort as the most
        common term instead of the rarest)."""
        if term in self.idf:
            return self.idf[term]
        return math.log((self.n_docs + 0.5) / 0.5 + 1.0)

    def scores_for_query(self, query_tokens: list[str]) -> list[float]:
        scores = [0.0] * len(self.docs_tokens)
        for i, tf in enumerate(self.term_freqs):
            dl = self.doc_lens[i]
            score = 0.0
            for term in query_tokens:
                if term not in tf:
                    continue
                idf = self.idf.get(term, 0.0)
                freq = tf[term]
                denom = freq + self.k1 * (1 - self.b + self.b * dl / self.avg_doc_len)
                score += idf * (freq * (self.k1 + 1)) / denom
            scores[i] = score
        return scores


def save_json(obj, name: str):
    (INDEX_DIR / name).write_text(json.dumps(obj))


def load_json(name: str):
    p = INDEX_DIR / name
    if not p.exists():
        return None
    return json.loads(p.read_text())
