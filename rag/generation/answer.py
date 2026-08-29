"""Generation — citation-aware prompt -> local Ollama (Qwen2.5) -> parsed answer.

ARCHITECTURE.md §5. The model is given a numbered source block (modality +
file + locator) and must end every factual claim with a bracketed citation.
The answer is regex-scanned for [n] markers and resolved back to the source
list; hallucinated numbers (not in the list) are dropped silently; citing
nothing at all is surfaced as a signal.

Runs against a local Ollama server only — no external API (offline constraint).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Iterator, Sequence

from rag import config

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You answer strictly from the numbered Sources provided. Rules:\n"
    "1. Write the answer in complete sentences. Every factual sentence ends "
    "with a bracketed citation: [1], or [1][3] when it draws on multiple "
    "sources. Never reply with only a citation marker and no prose.\n"
    "2. Use only citation numbers that appear in the Sources list. Never "
    "invent a source or a detail that is not in the Sources.\n"
    "3. Answer only from the Sources, never from your own knowledge. If no "
    "Source addresses the question, reply exactly: \"The corpus does not "
    "contain information on this.\" and nothing else.\n"
    "4. Be concise — 1 to 4 sentences."
)

# Prepended to the prompt when query_pipeline flags the retrieval as weak
# (best vector match beyond the grounding gate) — the sources are very likely
# unrelated to the question.
WEAK_RETRIEVAL_NOTE = (
    "NOTE: retrieval found no strong match for this question. The Sources "
    "below are the closest available but are probably unrelated. Unless a "
    "Source genuinely answers the question, reply exactly: \"The corpus does "
    "not contain information on this.\"\n\n"
)

_LOCATOR_LABEL = {
    "document": "page {page}",
    "image": "captured {ts}",
    "audio": "{start}-{end}",
}

_CITATION_RE = re.compile(r"\[(\d+)\]")

REFUSAL = "The corpus does not contain information on this."


@dataclass
class Source:
    """One numbered entry in the prompt's Sources block."""

    number: int
    modality: str
    source_file: str
    locator: str
    text: str
    chunk_id: str = ""
    doc_id: str = ""
    page_number: int | None = None
    start_time: float | None = None
    end_time: float | None = None
    source_path: str | None = None       # for the UI to render/play
    image_timestamp: str | None = None
    linked_passage: str | None = None


@dataclass
class GroundedAnswer:
    answer: str
    sources: list[Source]
    citations_used: list[Source]
    cited_nothing: bool = False
    dropped_citation_numbers: list[int] = field(default_factory=list)
    model: str = config.OLLAMA_MODEL


# --------------------------------------------------------------------------
# Prompt construction  (ARCHITECTURE.md §5)
# --------------------------------------------------------------------------

def _fmt_timestamp(seconds: float | None) -> str:
    if seconds is None:
        return "?"
    m, s = divmod(int(round(seconds)), 60)
    return f"{m:02d}:{s:02d}"


def _locator(chunk) -> str:
    modality = _get(chunk, "modality")
    if modality == "document":
        page = _get(chunk, "page_number")
        return f"page {page}" if page is not None else ""
    if modality == "image":
        # image_timestamp: a "HH:MM" capture time (screenshots) or a
        # "mm:ss–mm:ss" on-screen window (AMI slides)
        ts = _get(chunk, "image_timestamp") or _get(chunk, "capture_time")
        if not ts:
            return "image"
        return f"on screen {ts}" if "-" in str(ts) else f"captured {ts}"
    if modality == "audio":
        return f"{_fmt_timestamp(_get(chunk, 'start_time'))}-{_fmt_timestamp(_get(chunk, 'end_time'))}"
    return ""


def _get(obj, name, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def build_prompt(question: str, chunks: Sequence, weak: bool = False) -> tuple[str, list[Source]]:
    """Build the (user) prompt string and the resolved Source list.

    `chunks` are RetrievedChunk instances (or dicts) from query_pipeline.
    `weak` prepends WEAK_RETRIEVAL_NOTE when the retrieval was flagged off-corpus.
    """
    sources: list[Source] = []
    lines: list[str] = ([WEAK_RETRIEVAL_NOTE.rstrip()] if weak else []) + ["Sources:"]
    for i, ch in enumerate(chunks, start=1):
        modality = _get(ch, "modality", "document")
        fname = _get(ch, "source_file", "unknown")
        loc = _locator(ch)
        text = (_get(ch, "text_preview", "") or "").replace("\n", " ").strip()
        label = f'{modality}, "{fname}"' + (f", {loc}" if loc else "")
        lines.append(f"[{i}] ({label}): {text}")
        sources.append(
            Source(
                number=i,
                modality=modality,
                source_file=fname,
                locator=loc,
                text=text,
                chunk_id=_get(ch, "chunk_id", "") or _get(ch, "id", ""),
                doc_id=_get(ch, "doc_id", ""),
                page_number=_get(ch, "page_number"),
                start_time=_get(ch, "start_time"),
                end_time=_get(ch, "end_time"),
                source_path=_get(ch, "source_path"),
                image_timestamp=_get(ch, "image_timestamp"),
                linked_passage=_get(ch, "linked_passage"),
            )
        )

    lines.append("")
    lines.append(f"Question: {question}")
    lines.append("Answer:")
    return "\n".join(lines), sources


# --------------------------------------------------------------------------
# Ollama calls  (local server only)
# --------------------------------------------------------------------------

def _client():
    try:
        import ollama

        return ollama.Client(host=config.OLLAMA_HOST)
    except ImportError as e:
        raise RuntimeError("ollama python client not installed (`pip install ollama`)") from e


def call_ollama(prompt: str, system: str = SYSTEM_PROMPT, model: str | None = None) -> str:
    client = _client()
    model = model or config.OLLAMA_MODEL
    resp = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        options={"temperature": 0.2},
        keep_alive=config.OLLAMA_KEEP_ALIVE,
    )
    return resp["message"]["content"].strip()


def call_ollama_streaming(
    prompt: str, system: str = SYSTEM_PROMPT, model: str | None = None
) -> Iterator[str]:
    client = _client()
    model = model or config.OLLAMA_MODEL
    stream = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        options={"temperature": 0.2},
        keep_alive=config.OLLAMA_KEEP_ALIVE,
        stream=True,
    )
    for part in stream:
        piece = part.get("message", {}).get("content", "")
        if piece:
            yield piece


# --------------------------------------------------------------------------
# Citation parsing  (ARCHITECTURE.md §5)
# --------------------------------------------------------------------------

def extract_citations_used(
    answer: str, sources: Sequence[Source]
) -> tuple[list[Source], list[int]]:
    """Return (sources actually cited, hallucinated numbers dropped)."""
    by_number = {s.number: s for s in sources}
    used: list[Source] = []
    dropped: list[int] = []
    seen: set[int] = set()
    for m in _CITATION_RE.finditer(answer):
        n = int(m.group(1))
        if n in seen:
            continue
        seen.add(n)
        if n in by_number:
            used.append(by_number[n])
        else:
            dropped.append(n)
    return used, dropped


# --------------------------------------------------------------------------
# Orchestrator
# --------------------------------------------------------------------------

def generate_grounded_answer(
    question: str, chunks: Sequence, model: str | None = None, stream: bool = False,
    weak: bool = False,
) -> GroundedAnswer:
    if not chunks:
        return GroundedAnswer(
            answer="No sources were retrieved for this query, so I can't answer it "
            "from the corpus.",
            sources=[],
            citations_used=[],
            cited_nothing=True,
            model=model or config.OLLAMA_MODEL,
        )

    prompt, sources = build_prompt(question, chunks, weak=weak)
    logger.debug("Prompt:\n%s", prompt)

    if weak:
        # Retrieval was flagged off-corpus. Don't let the LLM answer a
        # well-known fact from its own weights — the sources stay visible so
        # the user sees what came closest, but the answer is a flat refusal.
        _, sources = build_prompt(question, chunks, weak=False)
        return GroundedAnswer(
            answer=REFUSAL, sources=sources, citations_used=[],
            cited_nothing=True, model=model or config.OLLAMA_MODEL,
        )

    if stream:
        answer = "".join(call_ollama_streaming(prompt, model=model))
    else:
        answer = call_ollama(prompt, model=model)

    used, dropped = extract_citations_used(answer, sources)
    if dropped:
        logger.warning("Model emitted citation(s) not in source list: %s", dropped)
    return GroundedAnswer(
        answer=answer,
        sources=sources,
        citations_used=used,
        cited_nothing=not used,
        dropped_citation_numbers=dropped,
        model=model or config.OLLAMA_MODEL,
    )


def stream_grounded_answer(question: str, chunks: Sequence, model: str | None = None,
                           weak: bool = False):
    """Generator: yields ('sources', list[Source]) once, then ('token', str)*,
    then ('done', GroundedAnswer). Lets the server push tokens to the UI as
    they arrive (roadmap R4)."""
    if not chunks or weak:
        ga = generate_grounded_answer(question, chunks, model=model, weak=weak)
        yield ("sources", ga.sources)
        yield ("token", ga.answer)
        yield ("done", ga)
        return

    prompt, sources = build_prompt(question, chunks, weak=weak)
    yield ("sources", sources)

    parts: list[str] = []
    try:
        for piece in call_ollama_streaming(prompt, model=model):
            parts.append(piece)
            yield ("token", piece)
    except Exception as e:  # noqa: BLE001
        logger.warning("stream failed (%s) — falling back to non-stream", e)
        answer = call_ollama(prompt, model=model)
        parts = [answer]
        yield ("token", answer)

    answer = "".join(parts)
    used, dropped = extract_citations_used(answer, sources)
    yield ("done", GroundedAnswer(
        answer=answer, sources=sources, citations_used=used,
        cited_nothing=not used, dropped_citation_numbers=dropped,
        model=model or config.OLLAMA_MODEL,
    ))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import sys

    from rag.retrieval import pipeline

    q = " ".join(sys.argv[1:]) or "What does the 2024 report say about international development?"
    res = pipeline.run_query_pipeline(text=q)
    ga = generate_grounded_answer(q, res.chunks)
    print("\n" + ga.answer + "\n")
    print("Citations used:")
    for s in ga.citations_used:
        print(f"  [{s.number}] {s.source_file} ({s.locator})")
    if ga.cited_nothing:
        print("  (none — retrieved context may not answer the question)")
