"""Query pipeline — user input -> unified ImageBind search -> RRF merge -> ranked chunks.

Implements ARCHITECTURE.md §3 (router, spoken-query vs audio-content
disambiguation, composite queries) and §4 (Reciprocal Rank Fusion, then
group/dedupe by doc_id). Shares embeddings.py with the ingestion pipeline so
queries land in exactly the same vector space as the corpus.

The router is driven by *which UI widget produced the input* (app.py passes
that in), never by sniffing content — see ARCHITECTURE.md §3.
"""

from __future__ import annotations

import logging
import re
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from rag import config

logger = logging.getLogger(__name__)


class Modality(str, Enum):
    TEXT = "text"            # typed query, OR transcribed spoken query
    IMAGE = "image"          # uploaded/selected image, matched via vision tower
    AUDIO = "audio"          # uploaded audio clip, matched via audio tower


@dataclass
class QueryInput:
    """One component of a (possibly composite) query.

    `origin` records the widget it came from so results can be tagged with
    which part of a composite query produced them (citation transparency,
    ARCHITECTURE.md §3).
    """

    modality: Modality
    text: str | None = None
    image_path: str | None = None
    audio_path: str | None = None
    origin: str = "text_box"        # text_box | image_upload | audio_upload | mic


@dataclass
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    modality: str
    source_file: str
    text_preview: str
    distance: float
    source_path: str | None = None
    rrf_score: float = 0.0
    page_number: int | None = None
    start_time: float | None = None
    end_time: float | None = None
    bbox: list[float] | None = None
    prev_chunk_id: str | None = None
    next_chunk_id: str | None = None
    theme: str | None = None
    image_timestamp: str | None = None
    linked_passage: str | None = None
    matched_by: list[str] = field(default_factory=list)   # which QueryInput(s)

    @classmethod
    def from_hit(cls, hit: dict[str, Any]) -> "RetrievedChunk":
        return cls(
            chunk_id=hit.get("id", ""),
            doc_id=hit.get("doc_id", ""),
            modality=hit.get("modality", ""),
            source_file=hit.get("source_file", ""),
            text_preview=hit.get("text_preview", "") or "",
            distance=float(hit.get("distance", 0.0)),
            source_path=hit.get("source_path"),
            page_number=hit.get("page_number"),
            start_time=hit.get("start_time"),
            end_time=hit.get("end_time"),
            bbox=hit.get("bbox"),
            prev_chunk_id=hit.get("prev_chunk_id"),
            next_chunk_id=hit.get("next_chunk_id"),
            theme=hit.get("theme"),
            image_timestamp=hit.get("image_timestamp"),
            linked_passage=hit.get("linked_passage"),
        )


# --------------------------------------------------------------------------
# Router  (ARCHITECTURE.md §3)
# --------------------------------------------------------------------------

def route_input(
    text: str | None = None,
    image_path: str | None = None,
    audio_path: str | None = None,
    audio_is_spoken_query: bool = False,
) -> list[QueryInput]:
    """Turn raw widget values into one or more QueryInputs.

    `audio_is_spoken_query` distinguishes the mic affordance ("I'm asking a
    question aloud" -> ASR -> text query) from the file-upload affordance
    ("here's a clip to match" -> embed via audio tower). ARCHITECTURE.md §3.
    """
    inputs: list[QueryInput] = []

    if audio_path and audio_is_spoken_query:
        spoken = transcribe_spoken_query(audio_path)
        logger.info("Spoken query transcribed: %r", spoken)
        text = (text + " " + spoken).strip() if text else spoken
    elif audio_path:
        # Match the clip acoustically only if audio windows are indexed;
        # otherwise the transcript bridge below is the whole audio path.
        if config.AUDIO_EMBED_WINDOWS:
            inputs.append(
                QueryInput(Modality.AUDIO, audio_path=audio_path, origin="audio_upload")
            )
        # text bridge: transcribe the clip and also search as text, so related
        # documents/images surface (ImageBind audio<->text is weak, §4).
        try:
            bridge = transcribe_spoken_query(audio_path)
            if bridge and len(bridge.split()) >= 2:
                logger.info("Audio-clip transcript bridge: %r", bridge)
                inputs.append(QueryInput(Modality.TEXT, text=bridge,
                                         origin="audio_transcript"))
        except Exception as e:  # noqa: BLE001
            logger.warning("transcript bridge failed: %s", e)

    if text and text.strip():
        inputs.append(QueryInput(Modality.TEXT, text=text.strip(), origin="text_box"))

    if image_path:
        inputs.append(
            QueryInput(Modality.IMAGE, image_path=image_path, origin="image_upload")
        )
        # text bridge from OCR of the uploaded image
        try:
            from rag.ingest import extraction

            ocr = extraction.extract_image(Path(image_path)).ocr_text.strip()
            if ocr and len(ocr.split()) >= 3:
                logger.info("Image OCR bridge: %r", ocr[:80])
                inputs.append(QueryInput(Modality.TEXT, text=ocr,
                                         origin="image_ocr"))
        except Exception as e:  # noqa: BLE001
            logger.warning("OCR bridge failed: %s", e)

    if not inputs:
        raise ValueError("Empty query — no text, image, or audio provided.")
    return inputs


_TIME_RE = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\b")


def timestamp_in(text: str | None) -> str | None:
    """'HH:MM' if the query names a capture time ('screenshot taken at 14:32')."""
    if not text:
        return None
    m = _TIME_RE.search(text)
    return f"{int(m.group(1)):02d}:{m.group(2)}" if m else None


# --------------------------------------------------------------------------
# Per-modality preprocessing  (ARCHITECTURE.md §3)
# --------------------------------------------------------------------------

def preprocess_image(image_path: str) -> str:
    """Normalise an uploaded image to RGB on disk; return the path to use."""
    try:
        from PIL import Image

        p = Path(image_path)
        with Image.open(p) as img:
            if img.mode == "RGB":
                return str(p)
            out = Path(tempfile.mkdtemp(prefix="rag_q_img_")) / (p.stem + ".png")
            img.convert("RGB").save(out)
            return str(out)
    except Exception as e:  # noqa: BLE001
        logger.warning("preprocess_image passthrough (%s)", e)
        return image_path


def preprocess_audio_clip(audio_path: str) -> list[Path]:
    """Slice an uploaded audio clip into ImageBind's fixed ~2s windows,
    same as ingestion (extraction.slice_audio_windows)."""
    from rag.ingest import extraction

    tmp = Path(tempfile.mkdtemp(prefix="rag_q_audio_"))
    windows, _ = extraction.slice_audio_windows(Path(audio_path), tmp)
    return [w.path for w in windows]


def transcribe_spoken_query(audio_path: str) -> str:
    """Whisper ASR for the mic affordance -> plain text, then embedded as a
    normal text query (ARCHITECTURE.md §3). Decodes via soundfile first so it
    doesn't depend on an ffmpeg binary for WAV."""
    from rag.ingest import extraction

    model = extraction._load_whisper()
    samples = extraction._load_audio_array(Path(audio_path))
    result = model.transcribe(samples, fp16=extraction._whisper_uses_fp16())
    return (result.get("text") or "").strip()


# --------------------------------------------------------------------------
# Embedding  (shared ImageBind space)
# --------------------------------------------------------------------------

def embed_query(qi: QueryInput) -> np.ndarray:
    """Embed a single QueryInput -> one (or, for audio, mean-pooled) 1024-vec."""
    from rag.core import embeddings

    if qi.modality is Modality.TEXT:
        return embeddings.embed_texts([qi.text or ""])[0]
    if qi.modality is Modality.IMAGE:
        return embeddings.embed_images([preprocess_image(qi.image_path)])[0]
    if qi.modality is Modality.AUDIO:
        windows = preprocess_audio_clip(qi.audio_path)
        if not windows:
            raise ValueError(f"No audio windows from {qi.audio_path}")
        vecs = embeddings.embed_audio_windows(windows)
        pooled = vecs.mean(axis=0)
        n = np.linalg.norm(pooled)
        return pooled / n if n else pooled
    raise ValueError(f"Unhandled modality {qi.modality}")


# --------------------------------------------------------------------------
# Search
# --------------------------------------------------------------------------

MODALITIES = ("document", "image", "audio")


def search_unified_index(
    embedding: np.ndarray,
    n_results: int = config.PER_MODALITY_SEARCH_K,
    modality_filter: str | None = None,
    where_extra: dict | None = None,
) -> list[RetrievedChunk]:
    from rag.core import vectorstore

    col = vectorstore.get_collection()
    hits = vectorstore.query(
        col, embedding.tolist(), n_results=n_results,
        modality=modality_filter, where_extra=where_extra,
    )
    return [RetrievedChunk.from_hit(h) for h in hits]


def search_per_modality(
    embedding: np.ndarray, n_results: int = config.PER_MODALITY_SEARCH_K
) -> dict[str, list[RetrievedChunk]]:
    """One search per modality -> {modality: ranked hits}.

    ARCHITECTURE.md §4: cross-modal cosine scores are not comparable (text↔text
    ~0.2, text↔image ~0.8), so a single unified search buries every image and
    audio hit under the documents. Searching each modality separately and then
    RRF-fusing by *rank* lets each modality's best hits surface — this is the
    scale-mismatch problem RRF exists to fix.
    """
    return {
        m: search_unified_index(embedding, n_results=n_results, modality_filter=m)
        for m in MODALITIES
    }


# --------------------------------------------------------------------------
# Merge / rerank  (ARCHITECTURE.md §4)
# --------------------------------------------------------------------------

def reciprocal_rank_fusion(
    results_by_modality: dict[str, list],
    k: int = config.RRF_K,
    weights: dict[str, float] | None = None,
) -> list:
    """Rank-position fusion — ARCHITECTURE.md §4.

    `results_by_modality` maps a source label (a QueryInput origin, or a
    modality) to that source's rank-ordered hit list. Returns
    [(doc_id, fused_score), ...] sorted high to low. `weights` scales a list's
    contribution — used to give an exact metadata match (a `timestamp=` list)
    more pull than a fuzzy embedding list.

    Within a single list, only a doc's *best-ranked* chunk contributes — the §4
    snippet adds every chunk, which lets a file that happened to be split into
    more chunks (e.g. several audio windows) out-score genuinely more relevant
    single-chunk docs. Collapsing per list keeps §4's intent ("near-identical
    chunks from the same source don't crowd out distinct sources") while a doc
    still gains a separate contribution from each *composite-query* list it
    appears in.
    """
    weights = weights or {}
    scores: dict[str, float] = {}
    for label, ranked_hits in results_by_modality.items():
        w = weights.get(label, 1.0)
        seen_in_list: set[str] = set()
        for rank, hit in enumerate(ranked_hits, start=1):
            doc_id = hit.doc_id if isinstance(hit, RetrievedChunk) else hit["doc_id"]
            if doc_id in seen_in_list:
                continue
            seen_in_list.add(doc_id)
            scores[doc_id] = scores.get(doc_id, 0.0) + w / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def group_and_rank(
    results_by_source: dict[str, list[RetrievedChunk]],
    top_k: int = config.DEFAULT_TOP_K,
    allow_multiple_per_doc: bool = False,
    chunks_per_doc: int = 3,
    min_gap: float = 0.04,
) -> list[RetrievedChunk]:
    """RRF-fuse the per-source result lists, then group by doc_id
    (ARCHITECTURE.md §4). Distinct sources are ranked by RRF; within a source
    we keep up to `chunks_per_doc` chunks that are *materially different* (each
    at least `min_gap` farther than the last kept one), so a dense single
    source — a multi-topic slide, a long PDF page — contributes real coverage
    instead of one fragment, while near-duplicate chunks still collapse.
    `allow_multiple_per_doc` lifts the per-doc cap entirely."""
    weights = {lbl: 3.0 for lbl in results_by_source if "timestamp=" in lbl}
    fused = reciprocal_rank_fusion(results_by_source, weights=weights)
    rank_of_doc = {doc_id: i for i, (doc_id, _) in enumerate(fused)}
    score_of_doc = dict(fused)

    # best (smallest-distance) chunk seen per doc_id, plus which sources hit it
    best: dict[str, RetrievedChunk] = {}
    per_doc: dict[str, list[RetrievedChunk]] = {}
    for label, hits in results_by_source.items():
        for h in hits:
            h.matched_by = sorted(set(h.matched_by) | {label})
            per_doc.setdefault(h.doc_id, []).append(h)
            cur = best.get(h.doc_id)
            if cur is None or h.distance < cur.distance:
                merged_matched = sorted(set((cur.matched_by if cur else [])) | set(h.matched_by))
                best[h.doc_id] = h
                best[h.doc_id].matched_by = merged_matched

    ordered_docs = sorted(
        best.keys(), key=lambda d: rank_of_doc.get(d, len(rank_of_doc))
    )

    n_docs = 0
    out: list[RetrievedChunk] = []
    for doc_id in ordered_docs:
        chunks = sorted(per_doc[doc_id], key=lambda c: c.distance)
        if allow_multiple_per_doc:
            take = chunks
        else:
            take, last = [], None
            for c in chunks:
                if last is None or c.distance - last >= min_gap:
                    take.append(c)
                    last = c.distance
                if len(take) >= chunks_per_doc:
                    break
        for c in take:
            c.rrf_score = score_of_doc.get(doc_id, 0.0)
            out.append(c)
        n_docs += 1
        if n_docs >= top_k and not allow_multiple_per_doc:
            break
    return out


# --------------------------------------------------------------------------
# Orchestrator
# --------------------------------------------------------------------------

@dataclass
class QueryResult:
    inputs: list[QueryInput]
    chunks: list[RetrievedChunk]
    transcribed_query: str | None = None
    weak: bool = False              # best vector match worse than the grounding gate
    best_distance: float = 1.0      # smallest cosine distance seen (text queries)
    best_lexical_score: float = 0.0  # top raw BM25 score


def run_query_pipeline(
    text: str | None = None,
    image_path: str | None = None,
    audio_path: str | None = None,
    audio_is_spoken_query: bool = False,
    modality_filter: str | None = None,
    top_k: int = config.DEFAULT_TOP_K,
    unload_after: bool = True,
    offload_after: bool = False,
) -> QueryResult:
    """Full §3+§4 path: route -> embed each input -> one search per input ->
    RRF merge -> group/dedupe -> top_k RetrievedChunks.

    `unload_after` frees ImageBind entirely (next call reloads from disk, ~40 s)
    — right for one-shot CLI use. `offload_after` instead parks it in system RAM
    and only frees the GPU (next call ~2 s) — right for the long-running app,
    where it takes precedence over `unload_after`."""
    inputs = route_input(
        text=text,
        image_path=image_path,
        audio_path=audio_path,
        audio_is_spoken_query=audio_is_spoken_query,
    )

    transcribed = None
    if audio_is_spoken_query and audio_path:
        transcribed = next((i.text for i in inputs if i.modality is Modality.TEXT), None)

    # A text query is embedded in the text space, where it can reach document
    # chunks, image-OCR chunks AND audio-transcript chunks directly — all
    # comparable — so a single unified search already returns every modality
    # ranked by real relevance. An image/audio *clip* query is embedded in the
    # vision/audio space, so we also fan out per modality and RRF-fuse by rank
    # to pull in related documents (ARCHITECTURE.md §4).
    results_by_source: dict[str, list[RetrievedChunk]] = {}
    text_vec_distances: list[float] = []   # cosine distances from text-space searches only
    best_lexical_score = 0.0              # top raw BM25 score (keyword-match strength)
    for qi in inputs:
        vec = embed_query(qi)
        if modality_filter:
            lists = {modality_filter: search_unified_index(
                vec, n_results=config.PER_MODALITY_SEARCH_K,
                modality_filter=modality_filter)}
        elif qi.modality is Modality.TEXT:
            lists = {"unified": search_unified_index(
                vec, n_results=config.PER_MODALITY_SEARCH_K)}
        else:
            lists = search_per_modality(vec, n_results=config.PER_MODALITY_SEARCH_K)
        for target_modality, hits in lists.items():
            label = f"{qi.origin}:{qi.modality.value}->{target_modality}"
            for h in hits:
                h.matched_by = [label]
            results_by_source[label] = hits
            logger.info("%s -> %d hits", label, len(hits))
            if qi.modality is Modality.TEXT:
                text_vec_distances.extend(h.distance for h in hits)

        # BM25 lexical retrieval for any text input — the keyword half of
        # hybrid search, fused into the same RRF (ARCHITECTURE.md Phase 5).
        if config.USE_BM25 and qi.text:
            try:
                from rag.retrieval import lexical
                from rag.core import vectorstore

                lex = lexical.search(vectorstore.get_collection(), qi.text,
                                     k=config.BM25_SEARCH_K)
                if lex:
                    best_lexical_score = max(
                        best_lexical_score,
                        max(h.get("lexical_score", 0.0) for h in lex),
                    )
                    chunks = [RetrievedChunk.from_hit(h) for h in lex]
                    label = f"{qi.origin}:bm25"
                    for h in chunks:
                        h.matched_by = [label]
                    results_by_source[label] = chunks
                    logger.info("%s -> %d hits", label, len(chunks))
            except Exception as e:  # noqa: BLE001
                logger.warning("BM25 skipped: %s", e)

        # "screenshot taken at 14:32" — exact metadata match on capture time,
        # added as its own list so those images rank at the top of the fusion.
        ts = timestamp_in(qi.text)
        if ts:
            thits = search_unified_index(
                vec, n_results=config.PER_MODALITY_SEARCH_K,
                where_extra={"image_timestamp": ts},
            )
            if thits:
                logger.info("timestamp filter %s -> %d image(s)", ts, len(thits))
                for h in thits:
                    h.matched_by = [f"{qi.origin}:timestamp={ts}"]
                results_by_source[f"{qi.origin}:timestamp={ts}"] = thits

    if offload_after:
        from rag.core import embeddings

        embeddings.offload_to_cpu()
    elif unload_after:
        from rag.core import embeddings

        embeddings.unload()

    chunks = group_and_rank(results_by_source, top_k=top_k)

    # Grounding gate: a text query whose closest text-space chunk is still far
    # away is probably off-corpus — UNLESS BM25 found a strong exact-term match
    # (the noisy-OCR / rare-keyword case hybrid search exists for). Flag it so
    # generation refuses to answer from world knowledge. Non-text-only queries
    # (image/audio clips) skip the gate — their distances are cross-modal.
    best_distance = min(text_vec_distances) if text_vec_distances else 1.0
    strong_lexical = best_lexical_score >= config.BM25_STRONG_SCORE
    weak = (
        bool(text_vec_distances)
        and best_distance > config.RETRIEVAL_MAX_DISTANCE
        and not strong_lexical
    )
    if weak:
        logger.info("weak retrieval: best text distance %.3f > %.2f, bm25 %.1f",
                    best_distance, config.RETRIEVAL_MAX_DISTANCE, best_lexical_score)
    return QueryResult(inputs=inputs, chunks=chunks, transcribed_query=transcribed,
                       weak=weak, best_distance=best_distance,
                       best_lexical_score=best_lexical_score)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import sys

    q = " ".join(sys.argv[1:]) or "international development in 2024"
    res = run_query_pipeline(text=q)
    for c in res.chunks:
        loc = (
            f"p{c.page_number}" if c.page_number
            else f"{c.start_time}-{c.end_time}s" if c.start_time is not None
            else ""
        )
        print(f"[{c.modality}] {c.source_file} {loc}  rrf={c.rrf_score:.4f}  {c.text_preview[:90]!r}")
