"""Hosted-preview retrieval pipeline.

A CPU-only stand-in for the full system's ImageBind path: text units (audio
transcript segments, slide OCR, document chunks) are embedded with a small
sentence encoder, searched by cosine + BM25, and fused with reciprocal rank
fusion — the same retrieval shape as the desktop app, minus the GPU model.
`doc_id` groups every unit of one meeting so cross-modal linking still works.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

DATA = Path(__file__).parent / "data" / "ami"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_PER_LIST = 20
RRF_K = 60
CHUNKS_PER_DOC = 3
# grounding gate (tuned for all-MiniLM-L6-v2 cosine similarity):
MIN_SIM = 0.26          # best semantic hit below this -> likely off-corpus
STRONG_BM25 = 9.0       # ...unless a keyword hit clears this


@dataclass
class Unit:
    id: str
    doc_id: str                 # meeting id — the cross-modal group
    modality: str               # audio | image | document
    source_file: str
    source_path: str
    text: str
    start: float | None = None
    end: float | None = None
    speaker: str | None = None
    page: int | None = None
    # filled by search():
    score: float = 0.0
    sim: float = 0.0
    lex: float = 0.0
    matched_by: list[str] = field(default_factory=list)

    def locator(self) -> str:
        if self.modality == "audio" and self.start is not None:
            return f"{int(self.start)//60:02d}:{int(self.start)%60:02d}"
        if self.modality == "image" and self.start is not None:
            return (f"slide {int(self.start)//60:02d}:{int(self.start)%60:02d}"
                    f"-{int(self.end)//60:02d}:{int(self.end)%60:02d}")
        return ""


_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def _chunk(text: str, target_words: int = 55) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    out, cur, n = [], [], 0
    for sent in _SENT.split(text):
        w = len(sent.split())
        if cur and n + w > target_words:
            out.append(" ".join(cur))
            cur, n = [], 0
        cur.append(sent)
        n += w
    if cur:
        out.append(" ".join(cur))
    return out


def load_units() -> list[Unit]:
    units: list[Unit] = []

    # 1. audio transcript segments
    for seg_json in sorted(DATA.rglob("*.segments.json")):
        wav = seg_json.with_name(seg_json.name.replace(".segments.json", ""))
        mid = seg_json.parent.name
        segs = json.loads(seg_json.read_text(encoding="utf-8"))
        for i, s in enumerate(segs):
            t = (s.get("text") or "").strip()
            if len(t.split()) < 2:
                continue
            units.append(Unit(
                id=f"{mid}::audio::{i:04d}", doc_id=mid, modality="audio",
                source_file=wav.name, source_path=str(wav), text=t,
                start=float(s["start"]), end=float(s["end"]),
                speaker=s.get("speaker"),
            ))

    # 2. slide OCR (precomputed .ocr.txt next to each .jpg)
    for ocr in sorted(DATA.rglob("slides/*.ocr.txt")):
        jpg = ocr.with_name(ocr.name.replace(".ocr.txt", ".jpg"))
        mid = ocr.parent.parent.name
        m = re.search(r"(\d+(?:\.\d+)?)__(\d+(?:\.\d+)?)", ocr.name)
        t = re.sub(r"\s+", " ", ocr.read_text(encoding="utf-8")).strip()
        if len(t.split()) < 3:
            t = f"[slide from meeting {mid}]"
        units.append(Unit(
            id=f"{mid}::image::{ocr.stem}", doc_id=mid, modality="image",
            source_file=jpg.name, source_path=str(jpg), text=t,
            start=float(m.group(1)) if m else None,
            end=float(m.group(2)) if m else None,
        ))

    # 3. documents — minutes / summaries / final report
    for doc in sorted(list(DATA.rglob("docs/*.txt")) + list((DATA / "_shared").glob("*.txt"))):
        mid = doc.parent.parent.name if doc.parent.name == "docs" else doc.stem.split(".")[0]
        raw = doc.read_text(encoding="utf-8", errors="replace")
        for i, ch in enumerate(_chunk(raw)):
            units.append(Unit(
                id=f"{mid}::{doc.stem}::{i:04d}", doc_id=mid, modality="document",
                source_file=doc.name, source_path=str(doc), text=ch, page=1,
            ))
    return units


class Retriever:
    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer
        from rank_bm25 import BM25Okapi

        self.units = load_units()
        texts = [u.text for u in self.units]
        self.model = SentenceTransformer(EMBED_MODEL, device="cpu")
        self.emb = self.model.encode(
            texts, normalize_embeddings=True, batch_size=64,
            show_progress_bar=False,
        ).astype(np.float32)
        self.bm25 = BM25Okapi([re.findall(r"[a-z0-9]+", t.lower()) for t in texts])

    # --- retrieval ----------------------------------------------------------
    def _rrf(self, ranked_lists: list[list[int]]) -> dict[int, float]:
        scores: dict[int, float] = {}
        for lst in ranked_lists:
            seen_doc: set[str] = set()
            for rank, idx in enumerate(lst, 1):
                d = self.units[idx].doc_id
                if d in seen_doc:
                    continue
                seen_doc.add(d)
                scores[idx] = scores.get(idx, 0.0) + 1.0 / (RRF_K + rank)
        return scores

    def search(self, query: str, top_docs: int = 6):
        q = self.model.encode([query], normalize_embeddings=True).astype(np.float32)[0]
        sims = self.emb @ q                                    # cosine (normed)
        sem_order = np.argsort(-sims)[:TOP_PER_LIST].tolist()

        toks = re.findall(r"[a-z0-9]+", query.lower())
        lex_scores = self.bm25.get_scores(toks) if toks else np.zeros(len(self.units))
        lex_order = [i for i in np.argsort(-lex_scores)[:TOP_PER_LIST].tolist()
                     if lex_scores[i] > 0]

        for i in sem_order:
            self.units[i].sim = float(sims[i])
        for i in lex_order:
            self.units[i].lex = float(lex_scores[i])

        fused = self._rrf([sem_order, lex_order])
        by_doc: dict[str, list[int]] = {}
        for idx in sorted(fused, key=lambda i: -fused[i]):
            by_doc.setdefault(self.units[idx].doc_id, []).append(idx)

        best_sim = float(sims[sem_order[0]]) if sem_order else 0.0
        best_lex = float(max(lex_scores)) if len(lex_order) else 0.0
        weak = best_sim < MIN_SIM and best_lex < STRONG_BM25

        out: list[Unit] = []
        for d in list(by_doc)[:top_docs]:
            for idx in by_doc[d][:CHUNKS_PER_DOC]:
                u = self.units[idx]
                u.score = fused[idx]
                out.append(u)
        out.sort(key=lambda u: -u.score)
        return out, weak, best_sim, best_lex
