"""BM25 lexical index over the chunk texts — the keyword half of hybrid retrieval.

ImageBind's CLIP text tower is weak on exact terms and noisy OCR
("SQL & DBMS BSc ae a cee ..."). BM25 matches literal words, sub-millisecond,
no GPU. `query_pipeline` fuses its ranking into the same RRF as the vector
search, so a query gets both semantic and keyword hits.

The index is built from ChromaDB (`text_preview` of every chunk) and cached;
it rebuilds automatically when the collection's chunk count changes.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from rag import config

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


@dataclass
class _Index:
    fingerprint: int
    bm25: object
    ids: list[str]
    meta: list[dict]          # per-chunk metadata (already flattened, + text_preview)


_cache: _Index | None = None


def _build(collection) -> _Index | None:
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        logger.warning("rank-bm25 not installed — hybrid retrieval disabled")
        return None

    got = collection.get(include=["documents", "metadatas"])
    ids = got.get("ids", [])
    docs = got.get("documents", []) or [""] * len(ids)
    metas = got.get("metadatas", []) or [{}] * len(ids)
    if not ids:
        return _Index(0, None, [], [])

    corpus = [_tokenize(d) for d in docs]
    bm25 = BM25Okapi(corpus)
    meta = []
    for m, d in zip(metas, docs):
        row = dict(m or {})
        row.setdefault("text_preview", d)
        meta.append(row)
    logger.info("BM25 index: %d chunks", len(ids))
    return _Index(len(ids), bm25, ids, meta)


def _index(collection) -> _Index | None:
    global _cache
    fp = collection.count()
    if _cache is None or _cache.fingerprint != fp:
        _cache = _build(collection)
    return _cache


def invalidate() -> None:
    """Call after ingesting/deleting so the next search rebuilds."""
    global _cache
    _cache = None


def search(collection, query: str, k: int = config.BM25_SEARCH_K) -> list[dict]:
    """Top-k chunks by BM25 score. Returns hit dicts shaped like
    vectorstore.query() output (id, distance, flattened metadata) so
    query_pipeline can treat it as just another ranked list."""
    idx = _index(collection)
    if idx is None or idx.bm25 is None or not idx.ids:
        return []
    toks = _tokenize(query)
    if not toks:
        return []
    scores = idx.bm25.get_scores(toks)
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    out: list[dict] = []
    for i in order[:k]:
        if scores[i] < config.BM25_MIN_SCORE:
            break
        row = dict(idx.meta[i])
        row["id"] = idx.ids[i]
        # RRF is rank-based, but give the UI a monotonic pseudo-distance
        row["distance"] = round(1.0 / (1.0 + float(scores[i])), 4)
        row["lexical_score"] = round(float(scores[i]), 3)
        out.append(row)
    return out
