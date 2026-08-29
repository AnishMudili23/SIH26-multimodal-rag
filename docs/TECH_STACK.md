# TECH_STACK — Multimodal RAG

## Models and libraries

| Component | Choice | Why |
|---|---|---|
| Unified embedding | **ImageBind** | Single joint space across text/image/audio — one encoder, one Chroma collection, simpler retrieval code than per-modality encoders. Tradeoff accepted: somewhat lower per-modality accuracy than dedicated encoders, and the 77-token CLIP text-tower limit, which drives the chunking strategy in `ARCHITECTURE.md`. |
| Speech-to-text | **Whisper (small)** | Transcribes recorded calls and spoken queries offline. Also used for citation display text (transcript segments) independent of the ImageBind audio embedding. |
| Vector store | **ChromaDB** | Persistent, offline, native metadata filtering, single collection with modality/doc_id/locator fields as designed in `ARCHITECTURE.md`. Chosen over raw numpy vectors — no meaningful benefit to numpy at this scale, and Chroma gives filtering/persistence for free. |
| Local LLM | **Qwen2.5-3B-Instruct via Ollama** | Fully offline, no API dependency, runs comfortably within the GPU budget below alongside ImageBind + Whisper. Starting point deliberately small — proven to work first, upgrade path to `qwen2.5:7b-instruct` is a one-line model-name swap in `generation.py` if 3B proves too shallow for multi-source synthesis or citation-formatting reliability. |
| UI | **Gradio** | Native multimodal input components (`gr.Audio` with microphone source, `gr.Image` with drag-drop, `gr.Chatbot`) cover the PS's "type, upload, drag-drop, or speak" requirement with far less hand-built UI code than Streamlit would need for the same feature set. |
| Demo corpus | **AMI Meeting Corpus** (5–10 scenario meetings) | 100 h of meetings with signals synchronised to a common timeline: close-talking audio, room/individual video, **projected slides** (JPEG + auto-OCR text, slide-change timestamps), electronic whiteboard, and time-aligned orthographic transcription. The AMI *scenario* meetings also ship **real participant-produced documents** — meeting minutes and a final project report — in each meeting's `shared-doc/`. This gives genuinely simultaneous, same-content text / image / audio / DOC for cross-modal linking, instead of manufactured pairs. License **CC BY 4.0** (confirmed on the official corpus page, attribution only, commercial use allowed). Replaces the earlier MS MARCO + self-created-assets plan; GovReport also dropped. Built by `rag/ingest/manifest.py`. |
| Doc/PDF extraction | **python-docx, PyMuPDF, pytesseract** (+ LibreOffice headless for legacy `.doc`/`.ppt`) | Standard offline extraction libraries; pytesseract covers OCR fallback for scanned PDFs and text-in-screenshot cases. AMI's `shared-doc` files are Word 97 `.doc` / PowerPoint 97 `.ppt`, which `python-docx`/`python-pptx` cannot read — `soffice --headless --convert-to` converts them to `.docx`/`.pdf` at build time so the normal extractor handles them. |

## VRAM budget (GPU laptop/desktop target)

| Model | Approx VRAM |
|---|---|
| ImageBind (fp16) | ~5 GB |
| Whisper small | ~1 GB |
| Qwen2.5-3B-Instruct, 4-bit (Ollama) | ~2 GB |
| Qwen2.5-7B-Instruct, 4-bit (Ollama, upgrade path) | ~4.5 GB |
| Chroma | negligible, CPU |

Guidance: on 8GB or less, don't keep ImageBind and the LLM resident at the
same time — embed/search first, hand off to Ollama after (Ollama manages
its own model loading separately, so this mostly happens automatically).
On 12GB+, everything can coexist without much thought, and the 7B upgrade
is comfortable too.

## Reference module design

> **Superseded.** The system is built. For the real layout see `README.md`
> and `CLAUDE.md`; for what changed each session see `PROGRESS.md`. Module
> map: `query_pipeline.py` → `rag/retrieval/pipeline.py`, `generation.py` →
> `rag/generation/answer.py`, `app.py` → `web/app.py`,
> `build_corpus_manifest.py` → `rag/ingest/manifest.py`, `ingest.py` →
> `rag/ingest/corpus.py`. The table below is kept only as the original design
> intent.

| File | Intended structure |
|---|---|
| `query_pipeline.py` | `Modality` enum, `QueryInput`/`RetrievedChunk` dataclasses, an input router (`route_input`), per-modality preprocessing (`preprocess_image`, `preprocess_audio_clip`, `transcribe_spoken_query` for the mic-vs-upload disambiguation in `ARCHITECTURE.md` §3), `embed_query` calling ImageBind per modality, `search_unified_index` against Chroma, RRF merge (`ARCHITECTURE.md` §4) plus `group_and_rank` for dedup, and a top-level `run_query_pipeline` orchestrator. |
| `generation.py` | Citation-aware `build_prompt` (numbered sources with modality/file/locator labels, per `ARCHITECTURE.md` §5), `call_ollama` / `call_ollama_streaming` against a local Ollama instance running `qwen2.5:3b-instruct`, `extract_citations_used` to regex-parse `[n]` markers back to source metadata (dropping hallucinated citation numbers), and `generate_grounded_answer` tying it together. |
| `app.py` | Gradio `Blocks` UI — chat history, text box, image upload, audio input with both upload and microphone sources, citations panel. Build with a `USE_MOCK_BACKEND` flag so the UI is testable before ingestion/retrieval are wired up; flip it once real data flows through. |
| `build_corpus_manifest.py` | Filters MS MARCO by keyword into a themed CSV manifest (`doc_id`, `theme`, `passage_text`, empty `image_file`/`audio_file`/`image_timestamp` columns to fill in manually). Note: MS MARCO's `datasets` config name and field names vary by version (v1.1 vs v2.1) — verify against whatever is actually downloaded. |

Ingestion (extraction → chunking → ImageBind embedding → Chroma writes) has
no reference module yet at all — that's the actual current blocker,
covered as Phase 0 in `ROADMAP.md`. Design it fresh directly from
`ARCHITECTURE.md` §2.
