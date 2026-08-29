# ARCHITECTURE — Multimodal RAG Pipeline

End-to-end flow: **Ingestion → Query → Generation → UI**, all offline.

```
Ingestion:  raw files → extract/transcribe → chunk → ImageBind embed → Chroma
Query:      user input → route/embed → search Chroma → RRF merge → rank
Generation: ranked chunks → citation prompt → Ollama/Qwen2.5 → parsed answer
UI:         Gradio → dispatches to query pipeline → renders answer + citations
```

---

## 1. Embedding space: ImageBind (unified), not per-modality encoders

**Decision:** use ImageBind as a single joint embedding space across text,
image, and audio, and store everything in one Chroma collection.

**Rejected alternative — Option A, matched dual-encoder per modality pair**
(e5-small for text, open_clip for images, CLAP for audio, each with its own
Chroma collection; query embedded once per target collection with the
matching encoder's query tower). This was the initial recommendation because
it's more accurate per modality and avoids the scale-mismatch problems a
unified space introduces (see RRF section below). It was rejected in favor
of ImageBind for simplicity: one encoder, one collection, one embedding call
per query instead of routing logic per modality pair — a meaningfully
smaller amount of retrieval code to build and debug against a hackathon
timeline, at an accepted cost of somewhat lower per-modality retrieval
accuracy and the cross-modal score-comparability problem RRF exists to fix.

**Known constraints this decision introduces (binding on ingestion and
query design):**
- ImageBind's text tower is CLIP-based → **hard 77-token limit**, silently
  truncates rather than erroring. This is the reason chunking is small (see
  below) — it is not a stylistic choice, it is required correctness.
- Text tower is **English-only** — no multilingual support.
- Audio tower expects **fixed-length windows** (~2s, mel-spectrogram input)
  — longer recordings must be chunked into windows, not embedded whole.
- Model is heavier than the rejected per-modality set (~5GB fp16) — see
  `TECH_STACK.md` VRAM budget.

---

## 2. Ingestion pipeline

`raw file → format-specific extraction → chunk → ImageBind embed → Chroma.add()`

**Extraction by format:**
- DOCX → `python-docx`
- PDF → `PyMuPDF` (text layer); `pytesseract` OCR fallback for scanned PDFs
- Images → `pytesseract` OCR run in parallel with the ImageBind vision
  embedding — OCR text is not used for retrieval directly (ImageBind
  embeds the image itself) but is stored as `text_preview` so citations and
  the generation prompt have readable text to show/cite, and so text-heavy
  screenshots (e.g. "email screenshot") are more reliably matched than pure
  visual-embedding similarity alone would achieve.
- Audio → Whisper (small) transcription for text/citation display, chunked
  into ImageBind's fixed audio windows for embedding.

**Chunking strategy — 40-60 tokens, sentence-boundary, no overlap:**
- Driven directly by the ImageBind/CLIP 77-token limit: normal RAG chunk
  sizes (300-500 tokens) would be silently truncated on embedding, losing
  most of the chunk's content without any error signal.
- Split on sentence boundaries (nltk/spaCy sentence tokenizer), not fixed
  character counts — accumulate sentences into a chunk until ~60 tokens,
  then start a new chunk.
- AMI transcript segments and slide-OCR lines are short; longer document
  passages (the meeting minutes / final report) and anything over ~70 tokens
  get re-split rather than passed through and truncated.
- **No explicit overlap.** Overlap is the normal mitigation for chunk-
  boundary information loss, but doubling already-tiny 40-60 token chunks
  meaningfully increases index size and dilutes precision for real
  savings. Instead: every chunk carries `prev_chunk_id` / `next_chunk_id`
  metadata pointers, so the generation step can pull a neighboring chunk
  for extra context on demand, without paying the storage/embedding cost
  of duplicating text into every chunk up front.
- Audio/image chunking is effectively pre-determined by ImageBind's fixed
  input requirements — no separate decision needed there.

**Chroma unified-collection schema:**

Single persistent collection, one entry per chunk (any modality):

| Field | Type | Notes |
|---|---|---|
| `id` | str | unique chunk id |
| `embedding` | vector | ImageBind output, same space regardless of modality |
| `modality` | str | `"document"` \| `"image"` \| `"audio"` |
| `doc_id` | str | groups chunks back to their parent source file |
| `source_file` | str | original filename |
| `text_preview` | str | passage text, OCR text, or transcript segment — always populated, used for citation display and prompt context |
| `page_number` | int? | for document chunks |
| `start_time` / `end_time` | float? | for audio chunks (seconds) |
| `bbox` | list? | for image sub-regions, if applicable |
| `prev_chunk_id` / `next_chunk_id` | str? | neighbor pointers, replaces overlap |
| `theme` | str? | optional, for demo-corpus filtering/eval |

---

## 3. Query pipeline

**Router:** detect input type from which UI widget was used (text box,
image upload, audio upload, mic) rather than sniffing content.

**Spoken-query vs. audio-content disambiguation (deliberate UI decision):**
a single mic button is ambiguous between "I am asking a question aloud" and
"I am giving you an audio clip to match against." Resolved with **two
separate UI affordances**: a mic control = spoken query → routed through
Whisper ASR to get text, then embedded as a normal text query. A file
upload/attach control = audio-as-content → embedded directly via
ImageBind's audio tower, no ASR step (Whisper transcript may still be
generated in parallel for citation display).

**Composite queries** (e.g. text + an attached image together): run **one
search per input embedding**, tag each hit with which input produced it,
merge/dedupe afterward. Vector-averaging the two embeddings into one query
vector before searching was considered and rejected — it's simpler but
destroys the ability to show which part of the composite query drove which
result, which directly conflicts with the citation-transparency goal.

**Unified vector search:** one embedding → `collection.query()` against the
single Chroma collection, optionally filtered by `modality` if the user
scopes the search.

---

## 4. Merge/rerank: Reciprocal Rank Fusion (RRF), not raw score normalization

**Problem:** even in a single ImageBind space, alignment quality is not
uniform across modality pairs — text↔image tends to be well-aligned,
text↔audio noticeably less so — and raw cosine scores across modalities
are not safely comparable as a result. Composite queries also produce
multiple separate result lists that need merging.

**Decision:** merge by **rank position**, not raw score, via RRF:

```python
def reciprocal_rank_fusion(results_by_modality: dict[str, list], k: int = 60) -> list:
    scores = {}
    for modality, ranked_hits in results_by_modality.items():
        for rank, hit in enumerate(ranked_hits, start=1):
            scores[hit.doc_id] = scores.get(hit.doc_id, 0) + 1 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

Rejected alternative: min-max/z-score normalizing raw distances per query
before merging — fragile with the small result sets this system will
typically have (demo-scale corpus), and doesn't address the underlying
scale-mismatch between modality pairs the way rank-based fusion does.

After RRF, results are grouped by `doc_id` and deduplicated so near-
identical chunks from the same source don't crowd out distinct sources.

---

## 5. Generation: citation-aware contract

**Prompt structure:** numbered sources, each labeled with modality + file +
locator (`page N` or `start-end` timestamp), followed by the question:

```
Sources:
[1] (document, "quarterly_report.pdf", page 4): <passage text>
[2] (image, "screenshot_1432.png", captured 14:32): <OCR text>
[3] (audio, "call_recording.wav", 02:15-02:40): <transcript segment>

Question: <user query>
Answer:
```

**Contract with the model:** every factual claim must end with a bracketed
citation number (`[1]`, or `[1][3]` for multi-source claims); if the
sources don't answer the question, say so rather than guessing; no
inventing sources or details not present in the block above.

**Citation parsing:** the generated answer is regex-scanned for `[n]`
markers, resolved back against the source list built for that prompt.
Citation numbers the model outputs that don't exist in the source list
(hallucinated citations) are silently dropped rather than shown — a real
failure mode to test for, especially against a 3B-parameter model. If the
model cites nothing at all, that's surfaced to the user as a signal the
retrieved context may not have actually answered the question.

**Runtime:** local Ollama server, Qwen2.5-3B-Instruct by default (see
`TECH_STACK.md` for the upgrade path to 7B).

---

## 6. UI

Gradio `Blocks` app: chat history, text box, image upload (drag-drop),
audio input (`sources=["upload", "microphone"]` — this is where the
spoken-query vs. audio-content distinction from §3 has to be made clear in
the UI copy/labeling), and a citations panel rendering each answer's
sources with enough metadata to eventually make them clickable/expandable
per the PS's citation-transparency requirement.
