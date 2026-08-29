# ROADMAP — Multimodal RAG

Phased task list. Work top to bottom — each phase unblocks the next.
Update `PROGRESS.md` at the end of every session regardless of which phase
is active (see root `CLAUDE.md`).

## Phase 0 — Ingestion pipeline (current blocker, do this first)

Nothing downstream can be tested against real data until this exists.

> **Status (2026-08-29):** Pipeline (Phases 0, 1, 3), desktop app + FastAPI
> backend, and a grounding gate are all built and stable. Full stack on
> `D:\SIH26_env\venv` (torch 2.6.0+cu124, ImageBind, Whisper, Chroma,
> Tesseract, Ollama+qwen2.5:3b). **Corpus source is being migrated** from the
> MS MARCO + self-created-assets plan to the **AMI Meeting Corpus** (real
> time-aligned audio / transcript / slides / minutes for the same meetings) —
> see the migration entry in `PROGRESS.md` and the rewritten Phase 2 below.
> Recall@5 = 0.96 was measured on the old MS MARCO corpus; a new number
> against AMI transcript retrieval is pending. **Run `. .\env.ps1` before any
> python.**

- [x] Text extraction: `python-docx` for DOCX, `PyMuPDF` for PDF text layer
      (`extraction.py`; also `.txt`/`.md`)
- [x] OCR fallback: `pytesseract` for scanned PDFs and images (screenshots) —
      PDF pages with `<40` chars trigger a 200-dpi render + OCR; images always
      OCR'd into `text_preview`. Binary auto-located via `config.TESSERACT_CMD`.
- [x] Audio transcription: Whisper (small) — segments w/ timestamps. Decoded
      via `soundfile` (no ffmpeg dependency for WAV).
- [x] Sentence-boundary chunker: `chunking.py`, ~55 tok target / 70 hard cap,
      no overlap, `prev_chunk_id`/`next_chunk_id` pointers, over-long sentence
      re-split. Uses ImageBind's real CLIP tokenizer for counting.
- [x] Audio chunking into ImageBind's fixed ~2s windows (`slice_audio_windows`)
- [x] ImageBind calls for image/audio/text (`embeddings.py`) — shared by
      ingestion and `query_pipeline.embed_query`. fp16 on CUDA, `unload()` for
      the VRAM handoff.
- [x] `Chroma` upsert writing the unified-collection schema (`vectorstore.py`
      `ChunkRecord` — modality, doc_id, source_file, text_preview,
      page_number/start_time/end_time, prev/next pointers, bbox, theme)
- [x] Smoke test (`python ingest.py --smoke`): DOCX/PDF/PNG/WAV → 5 chunks in
      Chroma with correct per-modality metadata; manual query returns them
      ranked text > image > audio as expected.

## Phase 1 — Wire RRF into query_pipeline.py

- [x] Implement `reciprocal_rank_fusion` (spec in `ARCHITECTURE.md` §4) and
      wire it into `query_pipeline.py`'s merge step — done verbatim from §4,
      wired through `run_query_pipeline`, unit-tested (fused ordering correct).
- [x] Implement `group_and_rank` — dedupe by `doc_id`, keep highest-ranked
      chunk per doc, `allow_multiple_per_doc` flag for the multi-passage case;
      tracks `matched_by` per composite input. Unit-tested.
- [x] Implement `transcribe_spoken_query` (Whisper call) — done in
      `query_pipeline.py`. Still TODO: confirm the spoken-query → text →
      text-embedding path end to end (needs a real voice clip or TTS sample).

## Phase 2 — Build the real demo corpus (AMI Meeting Corpus)

**Migrated 2026-08-29** from MS MARCO + self-created assets. The AMI Meeting
Corpus gives genuinely simultaneous, same-content audio / transcript / slides
/ documents for the same meetings — the exact thing the PS's cross-format
linking claim needs, with no manufactured pairs. License **CC BY 4.0**
(verified on the official corpus page — attribution only). GovReport also
dropped. Details: `TECH_STACK.md`, and the migration entry in `PROGRESS.md`.

- [x] Verify AMI license (CC BY 4.0, confirmed 2026-08-29) and that
      audio+transcript+slides all coexist for the target meetings.
- [ ] Pick 5–10 scenario meeting IDs with full audio + transcript + slide +
      `shared-doc` coverage (the **ES2002 / ES2003 / IS10xx** series have
      slides, whiteboard, and real minutes + final report).
- [ ] `rag/ingest/manifest.py` (AMI builder) downloads per meeting into
      `data/raw/ami/<meeting_id>/`:
  - transcript segments (time-aligned) — one document per meeting or per
    speaker turn, chunked per `ARCHITECTURE.md` §2
  - slide JPEGs from the mirror `slidesBackUp/` (filename encodes the
    `start__end` display window); OCR text produced by our own pytesseract
    pass in ingestion
  - the meeting audio, sliced per `ARCHITECTURE.md` §1
  - the real `shared-doc` minutes + final report (`.doc`/`.ppt` → convert
    to `.docx`/`.pdf` via LibreOffice headless, then the normal extractor)
- [ ] Manifest keyed by `meeting_id` = `doc_id` group, so a query can link a
      transcript segment ↔ the slide on screen at that time ↔ the audio
      window ↔ the meeting's minutes.
- [ ] Run Phase 0 ingestion over `data/raw/ami` and record the chunk counts.

### DOC/PDF source — decided
AMI ships **real participant-produced documents** per scenario meeting in
`shared-doc/` (`ProjectDocuments.Final.Report.doc`,
`Minutes.of.*-meeting.doc`, plus `.ppt` design docs). These are genuinely
tied to the same meetings as the audio/slides — **step 1 of the DOC/PDF
decision succeeds, no synthetic content needed.** If a chosen meeting turns
out to lack usable docs, fall back to real independently-existing PDFs found
by search (their own honest `doc_id` group), and only generate transcript-
grounded minutes as a last resort (flag clearly in `PROGRESS.md` if so).

## Phase 3 — Flip app.py off mock mode

- [x] `USE_MOCK_BACKEND` now defaults to False (`RAG_MOCK=1` forces mock).
- [x] `handle_submit` (was an inline closure, now module-level + testable)
      runs `run_query_pipeline` + `generate_grounded_answer` against the real
      Chroma collection + local Ollama. Verified: grounded answers with `[n]`
      citations resolving to real MS MARCO passages. UI builds under gradio 6
      (dropped the removed `Chatbot(type=...)` arg).
- [~] PS example queries: themed queries work ("how is hydropower bad for the
      environment" → right passage, cited). Timestamp/"screenshot at 14:32"
      style needs images in the corpus first.
- [ ] text→image / image→text — needs images ingested. **NEEDS USER** (assets)
- [ ] composite queries (text + image) — code path exists; needs UI + assets
- [ ] mic-vs-upload disambiguation — needs a browser click-through. **NEEDS USER**

## Phase 4 — Evaluation

- [x] (old corpus) Text Recall@5 / MRR@10 (`scripts/evaluate.py`, 25 MS MARCO
      dev queries): **Recall@5 = 0.96, MRR@10 = 0.825, 0 misses.** Retained
      as a historical baseline.
- [ ] Rebuild the dev-query / qrels set against AMI: hand-write ~20–30
      questions whose answer is a specific transcript segment / slide /
      minutes passage from the ingested meetings, then re-run
      `scripts/evaluate.py`.
- [ ] Manual spot-check of cross-modal retrieval on AMI — a transcript query
      should surface the slide that was on screen at that timestamp and the
      relevant line of the minutes.
- [x] Hallucinated-citation check vs Qwen2.5-3B: 8 runs, 0 hallucinated `[n]`,
      0 cited-nothing. One run produced a degenerate answer (just "[3]", no
      prose) — mitigated by a system-prompt line ("never reply with only a
      citation marker"). `extract_citations_used` drops any bad numbers anyway.

## Phase 5 — Stretch, only if time remains

- [ ] Upgrade path to `qwen2.5:7b-instruct` if 3B shows weak multi-source
      synthesis or unreliable citation formatting (one-line model-name
      change in `generation.py`, see `TECH_STACK.md`)
- [ ] Cross-encoder reranker on top of RRF output for a stronger final
      relevance signal
- [ ] Make citations in the Gradio UI actually clickable/expandable to
      open the source file at the right page/timestamp, not just listed
