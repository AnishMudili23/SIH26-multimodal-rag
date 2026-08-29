"""Sentence-boundary chunker for the text side of ingestion.

ARCHITECTURE.md §2: chunks are 40-60 tokens, split on sentence boundaries,
**no overlap**. This is required correctness, not style — ImageBind's CLIP text
tower silently truncates anything past 77 tokens, so a normal 300-500 token RAG
chunk would lose most of its content on embedding with no error.

Instead of overlap, every emitted chunk carries prev_chunk_id / next_chunk_id
pointers so rag/generation/answer.py can pull a neighbour for extra context on demand.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Iterable

from rag import config
from rag.core.embeddings import count_tokens

logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    text: str
    token_count: int
    page_number: int | None = None
    # Filled in by link_chunks() once the full ordered list for a doc exists.
    index: int = 0
    prev_chunk_id: str | None = None
    next_chunk_id: str | None = None
    chunk_id: str | None = None


# --------------------------------------------------------------------------
# Sentence splitting
# --------------------------------------------------------------------------

_FALLBACK_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def _split_sentences(text: str) -> list[str]:
    """NLTK punkt when available, regex fallback otherwise."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    try:
        import nltk

        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            try:
                nltk.download("punkt_tab", quiet=True)
            except Exception:  # noqa: BLE001 - older nltk has no punkt_tab
                pass
        return [s.strip() for s in nltk.sent_tokenize(text) if s.strip()]
    except Exception as e:  # noqa: BLE001
        logger.debug("nltk sentence split unavailable (%s); regex fallback", e)
        parts = _FALLBACK_SENT_RE.split(text)
        return [p.strip() for p in parts if p.strip()]


def _hard_split_long_sentence(sentence: str) -> list[str]:
    """A single sentence longer than CHUNK_MAX_TOKENS — split on clause
    punctuation, then on whitespace as a last resort, so nothing reaches the
    embedder above the token ceiling."""
    if count_tokens(sentence) <= config.CHUNK_MAX_TOKENS:
        return [sentence]

    pieces = re.split(r"(?<=[,;:])\s+", sentence)
    out: list[str] = []
    buf: list[str] = []
    for piece in pieces:
        trial = " ".join(buf + [piece]).strip()
        if buf and count_tokens(trial) > config.CHUNK_MAX_TOKENS:
            out.append(" ".join(buf).strip())
            buf = [piece]
        else:
            buf.append(piece)
    if buf:
        out.append(" ".join(buf).strip())

    # Still too long (no clause punctuation): fall back to word windows.
    final: list[str] = []
    for seg in out:
        if count_tokens(seg) <= config.CHUNK_MAX_TOKENS:
            final.append(seg)
            continue
        words = seg.split()
        cur: list[str] = []
        for w in words:
            cur.append(w)
            if count_tokens(" ".join(cur)) >= config.CHUNK_TARGET_TOKENS:
                final.append(" ".join(cur))
                cur = []
        if cur:
            final.append(" ".join(cur))
    return [s for s in final if s.strip()]


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------

def chunk_text(text: str, page_number: int | None = None) -> list[TextChunk]:
    """Group sentences into ~CHUNK_TARGET_TOKENS chunks without overlap."""
    sentences: list[str] = []
    for s in _split_sentences(text):
        sentences.extend(_hard_split_long_sentence(s))

    chunks: list[TextChunk] = []
    buf: list[str] = []
    buf_tokens = 0

    for sent in sentences:
        st = count_tokens(sent)
        if buf and buf_tokens + st > config.CHUNK_MAX_TOKENS:
            chunks.append(_emit(buf, buf_tokens, page_number))
            buf, buf_tokens = [], 0
        buf.append(sent)
        buf_tokens += st
        if buf_tokens >= config.CHUNK_TARGET_TOKENS:
            chunks.append(_emit(buf, buf_tokens, page_number))
            buf, buf_tokens = [], 0

    if buf:
        # Fold a tiny trailing remainder into the previous chunk if it fits.
        if (
            chunks
            and buf_tokens < config.CHUNK_MIN_TOKENS
            and chunks[-1].token_count + buf_tokens <= config.CHUNK_MAX_TOKENS
            and chunks[-1].page_number == page_number
        ):
            merged = (chunks[-1].text + " " + " ".join(buf)).strip()
            chunks[-1] = TextChunk(merged, count_tokens(merged), page_number)
        else:
            chunks.append(_emit(buf, buf_tokens, page_number))

    return chunks


def _emit(buf: list[str], tokens: int, page_number: int | None) -> TextChunk:
    text = " ".join(buf).strip()
    return TextChunk(text=text, token_count=count_tokens(text), page_number=page_number)


def link_chunks(doc_id: str, chunks: list[TextChunk]) -> list[TextChunk]:
    """Assign chunk_ids and wire prev/next pointers across an ordered list
    (spanning all pages of one source document)."""
    for i, c in enumerate(chunks):
        c.index = i
        c.chunk_id = f"{doc_id}::text::{i:04d}"
    for i, c in enumerate(chunks):
        c.prev_chunk_id = chunks[i - 1].chunk_id if i > 0 else None
        c.next_chunk_id = chunks[i + 1].chunk_id if i < len(chunks) - 1 else None
    return chunks
