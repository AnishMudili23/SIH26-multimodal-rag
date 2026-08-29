"""Local RAG backend — FastAPI on 127.0.0.1.

Runs in its own process so the heavy work (torch import, ImageBind, Ollama)
never shares a GIL with the desktop UI. The UI is a thin HTTP client.
Everything is still fully offline: this server only ever binds to localhost
and talks to the local Ollama / Chroma.

    python -m backend.server                 # or  uvicorn backend.server:app
"""

from __future__ import annotations

import json
import logging
import sys
import threading
import time
from collections import Counter
from contextlib import asynccontextmanager
from pathlib import Path

# repo root importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag import config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("backend")

HOST = config.BACKEND_HOST
PORT = config.BACKEND_PORT

_state = {"warmup": "cold", "warmup_error": ""}
_pipeline_lock = threading.Lock()   # serialise pipeline calls (ImageBind singleton)


# --------------------------------------------------------------------------
# Warm-up (background thread in THIS process)
# --------------------------------------------------------------------------

def _warmup():
    try:
        _state["warmup"] = "loading"
        from rag.core import embeddings
        from rag.retrieval import pipeline
        from rag.generation import answer
        embeddings._load_model()
        embeddings.offload_to_cpu()
        try:
            from rag.retrieval import lexical
            from rag.core import vectorstore
            lexical.search(vectorstore.get_collection(), "warmup")
        except Exception:  # noqa: BLE001
            pass
        _state["warmup"] = "ready"
        logger.info("warm-up complete")
    except Exception as e:  # noqa: BLE001
        _state["warmup"] = "error"
        _state["warmup_error"] = str(e)
        logger.exception("warm-up failed")


@asynccontextmanager
async def lifespan(app):
    threading.Thread(target=_warmup, daemon=True).start()
    yield


from fastapi import Body, FastAPI, HTTPException  # noqa: E402
from fastapi.responses import StreamingResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402

app = FastAPI(title="RAG_OS backend", version="1.0", lifespan=lifespan)


# --------------------------------------------------------------------------
# Response shaping (shared shape the UI renders directly)
# --------------------------------------------------------------------------

def _linked_text(doc_id: str, _cache={}) -> str:
    if not doc_id:
        return ""
    if doc_id not in _cache:
        try:
            from rag.core import vectorstore
            _cache[doc_id] = vectorstore.get_passage_text(vectorstore.get_collection(), doc_id)
        except Exception:  # noqa: BLE001
            _cache[doc_id] = ""
    return _cache[doc_id]


def _sources_payload(ga) -> list[dict]:
    cited = {s.number for s in ga.citations_used}
    by_file: dict[str, dict] = {}
    for s in ga.sources:
        c = by_file.get(s.source_file)
        if c is None:
            c = by_file[s.source_file] = {
                "numbers": [], "cited": False, "modality": s.modality,
                "source_file": s.source_file, "locator": s.locator, "texts": [],
                "source_path": s.source_path, "start_time": s.start_time,
                "end_time": s.end_time, "image_timestamp": s.image_timestamp,
                "linked_text": _linked_text(getattr(s, "linked_passage", "") or ""),
            }
        c["numbers"].append(s.number)
        c["cited"] = c["cited"] or (s.number in cited)
        if s.text and s.text not in c["texts"]:
            c["texts"].append(s.text)
        if c["start_time"] is None and s.start_time is not None:
            c["start_time"], c["end_time"] = s.start_time, s.end_time
        if not c["image_timestamp"] and s.image_timestamp:
            c["image_timestamp"] = s.image_timestamp
    out = []
    for c in by_file.values():
        c["text"] = "   …   ".join(c["texts"])
        out.append(c)
    out.sort(key=lambda x: (not x["cited"], min(x["numbers"])))
    return out


def _is_bm25_only(c) -> bool:
    mb = getattr(c, "matched_by", []) or []
    return bool(mb) and all(":bm25" in m for m in mb)


def _confidence(chunks, ga, weak: bool = False, best_distance: float = 1.0) -> str:
    if not chunks or weak:
        return "LOW"
    # BM25 hits carry a pseudo-distance (1/(1+score)) that isn't comparable to
    # cosine — only score the vector-matched chunks for closeness.
    vec = [c.distance for c in chunks if not _is_bm25_only(c)]
    best = min(vec) if vec else min(c.distance for c in chunks)
    best = min(best, best_distance)
    # No citation markers is a grounding red flag only when retrieval was also
    # loose; on a tight match it's just the 3B model skipping the formatting.
    if ga.cited_nothing and best > 0.33:
        return "LOW"
    closeness = max(0.0, 1.0 - min(best, 1.0))
    cited_docs = {s.doc_id for s in ga.citations_used}
    coverage = len(cited_docs) / max(1, len({c.doc_id for c in chunks}))
    pct = round(100 * (0.65 * closeness + 0.35 * min(1.0, coverage * 1.5)))
    return f"{max(5, min(99, pct))}%"


def _insights(chunks, ga, seconds, transcribed, res=None) -> str:
    mods = Counter(c.modality for c in chunks)
    files = sorted({c.source_file for c in chunks})
    lines = [
        f"latency        {seconds:.1f}s",
        f"model          {config.OLLAMA_MODEL}",
        f"embedding      ImageBind (unified) + BM25 lexical",
        f"retrieved      {len(chunks)} chunks  " + " ".join(f"{k}:{v}" for k, v in mods.items()),
        f"cited          {len(ga.citations_used)} of {len(ga.sources)}",
    ]
    if res is not None:
        lines.append(f"match          best_dist {res.best_distance:.3f}  bm25 {res.best_lexical_score:.1f}"
                     + ("  (weak — off-corpus)" if res.weak else ""))
    if ga.dropped_citation_numbers:
        lines.append(f"hallucinated   dropped {ga.dropped_citation_numbers}")
    if transcribed:
        lines.append(f"transcribed    {transcribed!r}")
    lines += ["", "sources:"] + [f"  · {f}" for f in files]
    return "\n".join(lines)


def _warning(chunks, ga, weak: bool = False, best_distance: float = 1.0) -> str:
    w = ""
    if not chunks:
        w = "nothing in the corpus matched this query"
    elif weak:
        w = "no strong match in the corpus — this question may be outside what's indexed"
    elif ga.cited_nothing and best_distance > 0.33:
        w = "the answer cited no source — the retrieved context may not answer this"
    if ga.dropped_citation_numbers:
        w = (w + "  |  " if w else "") + f"dropped invented citation(s) {ga.dropped_citation_numbers}"
    return w


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------

class QueryBody(BaseModel):
    text: str | None = None
    image_path: str | None = None
    audio_path: str | None = None
    spoken: bool = False
    top_k: int = config.DEFAULT_TOP_K


class IngestBody(BaseModel):
    paths: list[str]


class DeleteBody(BaseModel):
    source_file: str


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------

@app.get("/health")
def health():
    info = {"ok": True, "warmup": _state["warmup"], "warmup_error": _state["warmup_error"]}
    try:
        from rag.generation import answer
        info["ollama"] = answer._client() is not None
    except Exception:  # noqa: BLE001
        info["ollama"] = False
    try:
        from rag.core import vectorstore
        info["collection"] = vectorstore.collection_stats(vectorstore.get_collection())
    except Exception:  # noqa: BLE001
        info["collection"] = None
    return info


def _run_query(body: QueryBody):
    from rag.retrieval import pipeline
    t0 = time.time()
    res = pipeline.run_query_pipeline(
        text=body.text or None, image_path=body.image_path,
        audio_path=body.audio_path, audio_is_spoken_query=body.spoken,
        top_k=body.top_k, unload_after=False, offload_after=True,
    )
    return res, time.time() - t0


@app.post("/query")
def query(body: QueryBody):
    if not (body.text or body.image_path or body.audio_path):
        raise HTTPException(400, "empty query")
    with _pipeline_lock:
        from rag.generation import answer
        res, secs = _run_query(body)
        q_for_llm = res.transcribed_query or body.text or "(non-text query)"
        ga = answer.generate_grounded_answer(q_for_llm, res.chunks, weak=res.weak)
    return {
        "question": body.text or (res.transcribed_query or "(query)"),
        "answer": ga.answer,
        "sources": _sources_payload(ga),
        "confidence": _confidence(res.chunks, ga, res.weak, res.best_distance),
        "insights": _insights(res.chunks, ga, secs, res.transcribed_query, res),
        "warning": _warning(res.chunks, ga, res.weak, res.best_distance),
        "transcribed": res.transcribed_query,
    }


@app.post("/query/stream")
def query_stream(body: QueryBody):
    if not (body.text or body.image_path or body.audio_path):
        raise HTTPException(400, "empty query")

    def gen():
        with _pipeline_lock:
            from rag.generation import answer
            try:
                res, secs = _run_query(body)
            except Exception as e:  # noqa: BLE001
                yield _sse("error", {"message": str(e)})
                return
            q_for_llm = res.transcribed_query or body.text or "(non-text query)"
            question = body.text or (res.transcribed_query or "(query)")
            yield _sse("meta", {"question": question, "transcribed": res.transcribed_query})

            ga = None
            for kind, payload in answer.stream_grounded_answer(
                q_for_llm, res.chunks, weak=res.weak
            ):
                if kind == "sources":
                    continue
                if kind == "token":
                    yield _sse("token", {"t": payload})
                elif kind == "done":
                    ga = payload
            yield _sse("done", {
                "answer": ga.answer,
                "sources": _sources_payload(ga),
                "confidence": _confidence(res.chunks, ga, res.weak, res.best_distance),
                "insights": _insights(res.chunks, ga, secs, res.transcribed_query, res),
                "warning": _warning(res.chunks, ga, res.weak, res.best_distance),
            })

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/library")
def library():
    from rag.core import vectorstore
    return {"files": vectorstore.list_sources(vectorstore.get_collection())}


@app.post("/library/delete")
def library_delete(body: DeleteBody):
    from rag.core import vectorstore
    n = vectorstore.delete_source(vectorstore.get_collection(), body.source_file)
    return {"removed": n}


@app.post("/ingest")
def ingest(body: IngestBody):
    if not body.paths:
        raise HTTPException(400, "no paths")

    def gen():
        with _pipeline_lock:
            from rag.ingest import corpus
            state = {"i": 0, "n": len(body.paths), "name": ""}

            def prog(i, n, name):
                state.update(i=i, n=n, name=name)

            # run in a thread so we can stream progress
            result = {}
            done = threading.Event()

            def work():
                try:
                    result["summary"] = corpus.add_files_to_corpus(body.paths, progress=prog)
                except Exception as e:  # noqa: BLE001
                    result["error"] = str(e)
                done.set()

            threading.Thread(target=work, daemon=True).start()
            while not done.wait(0.4):
                yield _sse("progress", {"i": state["i"], "n": state["n"], "name": state["name"]})
            if "error" in result:
                yield _sse("error", {"message": result["error"]})
            else:
                yield _sse("done", result["summary"])

    return StreamingResponse(gen(), media_type="text/event-stream")


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
