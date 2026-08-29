# PROGRESS

Session log for the offline multimodal RAG build. Newest entries appended at
the bottom. Format defined in the kickoff prompt.

---

## Session — 2026-08-28

### Built

All Phase 0 code plus the four reference modules from `TECH_STACK.md`, written
fresh from `ARCHITECTURE.md`. No code existed before this session.

- **`config.py`** — single source of truth: paths (`data/raw`, `data/chroma`,
  `.model_cache`), `COLLECTION_NAME = "multimodal_rag"`, chunk token bounds
  (`CHUNK_TARGET_TOKENS=55`, `CHUNK_MAX_TOKENS=70`), audio window constants
  (2.0s / 16kHz / no overlap), model names (`qwen2.5:3b-instruct`,
  Whisper `small`), RRF `k=60`, top-k values. All overridable via env vars.

- **`embeddings.py`** — ImageBind wrapper, the shared joint space (ARCH §1).
  Lazy singleton load of `imagebind_huge` (fp16 on CUDA), `unload()` for the
  VRAM handoff to Ollama on the 6 GB GPU. `embed_texts` / `embed_images` /
  `embed_audio_windows` all L2-normalize and batch. `count_tokens()` uses
  ImageBind's own CLIP `SimpleTokenizer` when available, else a pessimistic
  word/punct/digit heuristic. `resolve_device()` never imports torch at
  module load.

- **`chunking.py`** — sentence-boundary chunker (ARCH §2): nltk punkt with a
  stdlib-`re` fallback, accumulate sentences to ~55 tokens, hard ceiling 70,
  over-long single sentences split on clause punctuation then word windows,
  tiny trailing remainder folded back. No overlap; `link_chunks()` assigns
  `chunk_id`s and wires `prev_chunk_id` / `next_chunk_id`.

- **`extraction.py`** — per-format (ARCH §2): `python-docx` (paragraphs +
  tables) for DOCX; PyMuPDF text layer for PDF with a per-page
  `< 40 char → pytesseract` OCR fallback; Pillow + pytesseract OCR for images
  (stored as `text_preview` only); Whisper `small` transcription for audio
  (segments w/ timestamps) **plus** `slice_audio_windows()` cutting the file
  into consecutive 2s WAVs on disk for ImageBind. Every external dep degrades
  gracefully with a logged warning (OCR especially — Tesseract binary absent).

- **`vectorstore.py`** — Chroma persistent client, one unified collection,
  cosine space. `ChunkRecord` dataclass = the ARCH §2 schema; `.metadata()`
  omits None keys and JSON-encodes `bbox` (Chroma rejects None/lists).
  `add_records` (upsert, batched), `query` (optional `modality` filter,
  decodes bbox back), `get_by_id` (for neighbour pulls), `collection_stats`.

- **`ingest.py`** — Phase 0 orchestrator. Walks `data/raw` recursively,
  stable `doc_id` from name+size+mtime (re-ingest = upsert), optional
  `theme` from `corpus_manifest.csv`. Builds `PendingRecord`s per file, then
  embeds **grouped by kind in batches**, then `Chroma.upsert`. Per-file
  try/except so one bad file doesn't abort the run. Unloads ImageBind +
  Whisper at the end. CLI: `--src --reset --limit --stats --smoke`.
  `--smoke` synthesises one DOCX/PDF/PNG/WAV, ingests, and queries back.

- **`query_pipeline.py`** — ARCH §3/§4. `Modality` enum, `QueryInput`
  (carries `origin` widget) / `RetrievedChunk` dataclasses. `route_input`
  with the **mic-vs-upload** split (`audio_is_spoken_query` → Whisper ASR →
  text query; else embed clip via audio tower). `preprocess_image` /
  `preprocess_audio_clip` / `transcribe_spoken_query`. `embed_query`
  per-modality (audio = mean-pooled window vecs). `search_unified_index`.
  `reciprocal_rank_fusion` **verbatim from ARCH §4** (fuses by `doc_id`).
  `group_and_rank` — dedupe to one representative chunk per doc unless
  `allow_multiple_per_doc`, tracks `matched_by` per composite input.
  `run_query_pipeline` = one search per input → RRF → group.

- **`generation.py`** — ARCH §5. `build_prompt` emits the exact
  `[n] (modality, "file", locator): text` block + `Question:` / `Answer:`.
  `SYSTEM_PROMPT` states the citation contract. `call_ollama` /
  `call_ollama_streaming` against the local Ollama server only.
  `extract_citations_used` regex-scans `[n]`, resolves to sources, returns
  `(used, dropped_hallucinated_numbers)`. `generate_grounded_answer` sets
  `cited_nothing` when the model cites nothing (surfaced to the user).

- **`app.py`** — Gradio `Blocks`, `USE_MOCK_BACKEND` flag (default on via
  `RAG_MOCK=1`). Chat history, textbox, image drag-drop, and **two audio
  tabs** making the ARCH §3 mic-vs-upload distinction explicit in UI copy.
  Citations panel shows every source with a cited / not-cited marker and the
  hallucinated-citation / cited-nothing warnings.

- **`build_corpus_manifest.py`** — ROADMAP Phase 2. Streams `ms_marco` (tries
  `v1.1`/`v2.1` configs + several passage-field shapes per TECH_STACK note),
  keyword-filters to an NTRO-relevant theme, writes
  `doc_id,theme,passage_text,image_file,audio_file,image_timestamp` CSV.
  `--emit-txt` also materialises passages as `.txt` under `data/raw/msmarco`.

- **`requirements.txt`** — tiered (light extraction / audio / ImageBind+torch
  / UI) with the ImageBind git-install and CUDA-torch index-url noted.

### Verified (pure-Python logic only — see blocker)

- Chunker: 4-sentence paragraph → 2 linked chunks, prev/next pointers correct,
  max token count within ceiling (heuristic counter).
- `reciprocal_rank_fusion` + `group_and_rank`: correct fused ordering
  (doc in both lists ranks first), `matched_by` accumulation correct.
- `build_prompt` output matches ARCH §5 byte-for-byte; `extract_citations_used`
  keeps `[1][2][3]`, drops hallucinated `[9]`.
- All 10 modules `ast.parse` clean.

### Broke / blocked on

- **HARD BLOCKER — Windows Smart App Control is ON and blocking every
  pip-installed native extension.** `VerifiedAndReputablePolicyState = 1`,
  usermode code-integrity enforced. Confirmed `ImportError: DLL load failed …
  An Application Control policy has blocked this file` for: `pymupdf`
  (`_extra`), `lxml` (python-docx), `pydantic_core` (chromadb), `PIL._imaging`,
  `regex` (nltk). This will equally block `torch`, `onnxruntime`, `imagebind`,
  `openai-whisper`. Only pure-Python wheels and the **pre-existing** numpy
  1.26.4 import. Smart App Control cannot be exempted per-app — it is
  all-or-nothing and, once disabled, only re-enables via a Windows reset.
  → Nothing that touches ImageBind / Chroma / PDF / OCR / Whisper can run in
     this Python environment. The **Phase 0 smoke test could not be run.**
  → Options for the user (their call):
     1. Disable Smart App Control (Windows Security → App & browser control →
        Smart App Control → Off), then `pip install -r requirements.txt`.
     2. Work inside **WSL2** — `wsl --install -d Ubuntu` (only the
        `docker-desktop` distro exists now); SAC doesn't police WSL. RTX 3050
        CUDA passthrough works. This is the recommended path for the whole ML
        stack anyway.
     3. Use Docker (Docker Desktop is installed, currently stopped).
- The nltk `punkt` data never downloaded (nltk import itself fails on
  `regex`). Chunker silently used its regex fallback. Fine for now.
- The heuristic token counter under-counts vs real CLIP BPE (a 3-sentence
  chunk measured 61 that BPE would likely push over 70 and re-split). Only
  matters while ImageBind is absent — `count_tokens` switches to the exact
  `SimpleTokenizer` once ImageBind installs. Re-check chunk sizes after that.
- Tesseract binary not installed and no Ollama binary — both needed later
  (Phase 0 OCR fallback, Phase 3 generation) but neither is the current
  blocker.
- Only ~16.6 GB free on C:. torch (CUDA) + ImageBind checkpoint (~4.5 GB) +
  Whisper will use ~8–10 GB. Tight but fits; worth watching.

### Next

**Resolve the environment blocker first — this gates all of Phase 0's
remaining checklist.** Recommended: set up WSL2 Ubuntu, clone/copy the repo
in, `python -m venv`, `pip install -r requirements.txt`, then
`pip install git+https://github.com/facebookresearch/ImageBind.git` and
install torch from the cu12x index-url. Also install the Tesseract binary
(`apt install tesseract-ocr`) and Ollama (`curl -fsSL https://ollama.com/install.sh | sh`,
then `ollama pull qwen2.5:3b-instruct`) while there.

Then, still Phase 0:
1. `python ingest.py --smoke` — this exercises extraction → chunk → ImageBind
   embed → Chroma add → query-back for all four modalities on synthetic
   files. First real test of `embeddings.py` (the ImageBind checkpoint
   auto-downloads to `.model_cache` on first call — needs network **once**).
2. Watch for: ImageBind's `data.load_and_transform_audio_data` signature
   (`clip_duration` / `clips_per_video` kwargs — verify they exist in the
   installed version; adjust `embeddings.embed_audio_windows` if not).
   ImageBind also hardcodes a `.checkpoints/` download path relative to CWD —
   if it ignores `MODEL_CACHE_DIR`, either run from repo root or symlink.
3. Re-check chunk token sizes now that `count_tokens` uses the real CLIP
   tokenizer; tune `CHUNK_TARGET_TOKENS` if chunks cluster too low/high.
4. Then tick the Phase 0 boxes in `ROADMAP.md` and move to Phase 1 (RRF is
   already implemented in `query_pipeline.py`; Phase 1 is mostly wiring +
   `transcribe_spoken_query` end-to-end, also already written).

---

## Session — 2026-08-28 (session 2)

### Built

- **Environment unblocked.** User disabled Windows Smart App Control (one-way;
  needs a Windows reset to re-enable). Native pip wheels now load.
- **Full stack installed into a venv on D:** (`C:` had <15 GB free). Layout:
  - `D:\SIH26_env\venv` — the venv (Python 3.11.6)
  - `D:\SIH26_env\{models,hf,whisper,ollama,pipcache,tmp}` — all model/weight
    caches + pip staging, kept off C:
  - `D:\SIH26_env\data\{raw,chroma}` — ingest inputs + Chroma store
  - torch **2.6.0+cu124** (CUDA works: RTX 3050 6 GB laptop, fp16),
    torchvision 0.21.0, torchaudio 2.6.0
  - ImageBind + pytorchvideo installed `--no-deps` from git (their pinned
    deps drag in cartopy/decord which fail on Windows and aren't needed);
    hand-picked deps: timm ftfy regex einops iopath fvcore av
  - openai-whisper 20250625, chromadb 1.5.9, gradio 6.26.0, pymupdf,
    python-docx, pytesseract, librosa, soundfile, nltk (+punkt/punkt_tab)
  - numpy resolved to **2.4.6** (torch pulled it; nothing broke — the
    `numpy<2` note in requirements.txt is now stale)
- **`env.ps1`** (new) — `. .\env.ps1` activates the venv and sets every cache
  path + `TEMP`/`TMP` + `RAG_*` data dirs to D:. Run it before any python.
- **`config.py`** — added `WHISPER_CACHE_DIR`, `IMAGEBIND_CKPT_URL`,
  `TESSERACT_CMD` (auto-locates the binary: env var → PATH → the standard
  `C:\Program Files\Tesseract-OCR\tesseract.exe`).
- **`embeddings.py`** — (1) checkpoint now downloaded to `MODEL_CACHE_DIR`
  and loaded via `imagebind_huge(pretrained=False)` + manual `load_state_dict`,
  bypassing ImageBind's hardcoded `.checkpoints/` path. (2) `_forward` now
  casts floating-point input tensors to the model dtype — fixes
  `Input type (FloatTensor) and weight type (HalfTensor)` on image/audio
  embedding under fp16.
- **`extraction.py`** — (1) `extract_audio` / `slice_audio_windows` decode
  WAV via `soundfile` and hand Whisper the ndarray directly, instead of
  letting Whisper shell out to a missing `ffmpeg` (`WinError 2`). (2)
  `import fitz` → `import pymupdf`. (3) `.txt`/`.md` support. (4) OCR sets
  `pytesseract.tesseract_cmd` from `config.TESSERACT_CMD`.
- **`ingest.py`** — smoke-test PDF builder uses `import pymupdf`.
- Tesseract installed by user (winget UB-Mannheim); Ollama install status
  unconfirmed (not on PATH in the shell checked — may need a fresh shell or
  may still be pending).

### Verified

- **`python ingest.py --smoke` PASSES end to end.** Synthetic DOCX/PDF/PNG/WAV
  → extract → chunk → ImageBind embed (fp16/CUDA) → Chroma. Collection:
  `{document: 2, image: 1, audio: 2}` = 5 chunks, metadata correct
  (page numbers, audio start/end times, prev/next pointers).
- Query "international development funding in 2024" ranks:
  `funding_note.pdf` d=0.378 > `dev_report_2024.docx` d=0.510 >
  `email_screenshot.png` d=0.828 > audio windows d=0.92/0.93 — i.e.
  text↔text tight, text↔image looser, text↔audio loosest, exactly as
  ARCHITECTURE.md §4 predicts (this is *why* RRF exists).
- OCR: Tesseract auto-located, `extract_image` returns real OCR text
  ("EMAIL SCREENSHOT1432 development budget" from a probe PNG).
- ImageBind checkpoint cached at `D:\SIH26_env\models\imagebind_huge.pth`
  (4.47 GB) — first download took ~27 min; subsequent loads ~35-45 s.

### Broke / blocked on

- **`numpy<2` in requirements.txt is stale** — actual env has numpy 2.4.6 and
  works. Update the pin or drop the comment.
- **Whisper needs ffmpeg for non-WAV audio** (mp3/m4a/ogg). WAV works without
  it. If the real corpus has compressed audio, `winget install Gyan.FFmpeg`.
- **Ollama not confirmed installed / on PATH.** Needed for Phase 3 only.
  User still to run `ollama pull qwen2.5:3b-instruct` (~2 GB → D: via the
  `OLLAMA_MODELS` env var that `setx` should have set).
- **`transcribe_spoken_query` path not tested end to end** — no real voice
  sample yet.
- Smoke test uses single-format synthetic files; multi-page PDFs, scanned
  PDFs, and real multi-sentence passages not yet exercised.
- PowerShell wraps the script's stderr as `NativeCommandError` noise around
  every run (cosmetic — `2>&1` on a native exe in PS 5.1). Ignore the
  "At line:1 char..." blocks in task output.

### Next

**Phase 0 is done.** Two ways forward, in priority order:

1. **Phase 2 — real demo corpus** (this is what makes the demo real):
   - `. .\env.ps1` then
     `python build_corpus_manifest.py --emit-txt` — pulls MS MARCO, filters
     to an NTRO theme, writes `D:\SIH26_env\data\corpus_manifest.csv` and
     materialises passages as `.txt` under `data\raw\msmarco\`. **Untested
     against real MS MARCO** — the `datasets` library isn't installed yet
     (`pip install datasets`), and the config/field-name probing in that
     script (v1.1 vs v2.1) needs a real run to confirm. Watch for HF needing
     one network call.
   - Then hand-create matching screenshots/audio for ~10-15 passages
     (ROADMAP Phase 2), fill the manifest's image_file/audio_file/
     image_timestamp columns, and `python ingest.py --src D:\SIH26_env\data\raw --reset`.

2. **Phase 3 — flip `app.py` to the real backend** once Ollama is up:
   - `ollama pull qwen2.5:3b-instruct`
   - `python -m ... ` run `app.py` with `RAG_MOCK=0` (or set in env.ps1)
   - gradio is v6, not v4 — `app.py` uses only stable APIs (`Blocks`,
     `Chatbot(type="messages")`, `Image`, `Audio`, `Tab`) but give it a
     click-through before relying on it.

Immediate first step for a fresh session: `cd` to the repo, `. .\env.ps1`,
`python ingest.py --stats` to confirm the collection still has the 5 smoke
chunks, then `pip install datasets` and start Phase 2.

---

## Session — 2026-08-28 (session 2, continued — autonomous)

User stepped out and asked me to push on without them. Covered Phases 1-4
(text) end to end.

### Built / changed

- **Ollama** — installed (winget), model dir set to `D:\SIH26_env\ollama` via
  user-level `OLLAMA_MODELS`, `qwen2.5:3b-instruct` (1.9 GB) pulled. `env.ps1`
  now auto-starts `ollama serve` if it's not already running.
- **`query_pipeline.reciprocal_rank_fusion`** — bug fix. It summed `1/(k+rank)`
  for *every* chunk of a doc in a single result list, so a file split into
  more chunks (e.g. 2 audio windows) out-scored genuinely more relevant
  single-chunk docs. Now dedupes `doc_id` **within** each list (best rank
  only); a doc still gets a separate contribution per *composite-query* list.
  This is faithful to ARCHITECTURE.md §4's stated intent.
- **`ingest.make_doc_id`** — now takes `manifest_ids`; if a file's stem is
  already a manifest `doc_id` (materialised MS MARCO passages), uses it
  verbatim so manifest / qrels / Chroma all agree. `load_theme_map` →
  `load_manifest` (returns themes + full rows). This fixed the eval scoring 0.
- **`ingest.smoke_test`** — now fully isolated: redirects `RAW_DATA_DIR` /
  `CHROMA_DIR` / `COLLECTION_NAME` to a temp workspace. Previously it did
  `reset=True` on the *real* collection and dumped 4 synthetic files into the
  real raw dir — destructive now that a real corpus exists.
- **`build_corpus_manifest.py`** — rewritten for `datasets` 5.x:
  - loads `microsoft/ms_marco` (the script-based `ms_marco` id is dead in v5)
  - chains v1.1 validation → v1.1 train to reach enough dev queries
  - `_extract_passages` returns `(text, is_selected)`; build prefers the
    query's selected passage so qrels are real judgments
  - also writes `<stem>_qrels.csv` (query_id, query, doc_id, is_relevant)
  - theme retargeted: **"energy and infrastructure"** (`DEFAULT_THEME` +
    `--theme`). The old "international development / foreign aid" keyword set
    produced junk (water parks, tourism) — general web QA has ~no NTRO-style
    content; energy/infra is a real coherent MS MARCO cluster.
  - `--emit-txt` now also prunes stale `.txt` files
- **`evaluate.py`** (new) — Recall@k / MRR@k against the qrels file. Text-only
  (PRD non-goal rules out image/audio metrics).
- **`generation.py`** — `_locator` returns "" (not "n/a") for pageless docs;
  `build_prompt` omits the empty locator; `SYSTEM_PROMPT` gained "write
  complete sentences / never reply with only a citation marker" after a 3B run
  produced a bare "[3]".
- **`app.py`** — `USE_MOCK_BACKEND` now defaults False (`RAG_MOCK=1` forces
  mock). Inline `_on_submit` closure → module-level `handle_submit` (testable).
  Removed `gr.Chatbot(type="messages")` — gradio 6 dropped that arg.
  Citations panel skips empty locator segments.
- **`env.ps1`** — added Ollama auto-start.
- **`requirements.txt`** — `gradio>=6.0`, added `datasets>=3.0` tier; dropped
  the stale `numpy<2` upper bound.
- **`config.py`** — `TESSERACT_CMD` auto-locator (env → PATH → Program Files).

### Verified

- **`python ingest.py --smoke`** still passes, now isolated (real 51-chunk
  collection untouched). Image chunk shows real OCR text.
- **Real corpus ingested**: 25 MS MARCO "energy & infrastructure" passages →
  **51 chunks** (`document` only) in Chroma at `D:\SIH26_env\data\chroma`.
- **`python evaluate.py`**: **Recall@5 = 0.96, MRR@10 = 0.825, 0 misses**
  over 25 dev queries. (Genuinely strong for ImageBind text retrieval.)
- **End-to-end through `app.py`**: e.g. "how is hydropower bad for the
  environment" → retrieves the hydropower passage → Qwen grounded answer with
  `[1]` resolving to `msmarco_3cfb9f6932f2.txt`. Gradio 6 UI builds.
- **Hallucinated citations**: 8 Qwen-3B runs → 0 bad `[n]`, 0 cited-nothing
  (1 degenerate bare-"[3]" answer, since mitigated in the system prompt).

### Broke / blocked on

- **Corpus is text-only.** Cross-modal linking — the PS's headline feature —
  can't be demoed until the matching images/audio exist. This is the main
  remaining gap and it needs the user (creative asset work).
- **Qwen2.5-3B is shaky**: occasionally front-loads citations (`[1][3] The
  report...`) or, once in ~8, emits a bare marker. Watch during the demo;
  the 7B upgrade (one line in `config.py` / `OLLAMA_MODEL`) is the fallback.
- **~2-4 of the 25 dev queries are marginally off-theme** ("where does lung
  cancer occur", "which organelle provides energy for translation") — matched
  via a loose keyword. Recall@5 is still 0.96 so not urgent; tighten
  `DEFAULT_KEYWORDS` or hand-prune the manifest if it bothers you.
- **C: down to ~11.8 GB free** (Windows churn + a stale ~640 MB in
  `%LOCALAPPDATA%\Temp` from the pre-redirect torch install). Safe to clear
  that Temp folder. Everything project-related is on D: (15.1 GB used).
- `datasets` streaming dropped the HF connection mid-`train` parquet twice
  (`peer closed connection`) — harmless, we had our 25 passages by then; a
  re-run with a bigger `--max-per-theme` might need `HF_HUB_DOWNLOAD_TIMEOUT`
  bumped or a retry.
- `transcribe_spoken_query` still not tested with a real voice clip.

### Next

**For the user (needs a human), in priority order:**
1. Create ~12-15 mock "report screenshot" images echoing the text of chosen
   passages from `D:\SIH26_env\data\corpus_manifest.csv` (a slide/doc with the
   passage's key sentence + a title). Tag 2-3 filenames with a capture time
   and put it in the `image_timestamp` column (e.g. `14:32`) to demo the PS's
   "screenshot at 14:32" line. Drop the PNGs in
   `D:\SIH26_env\data\raw\images\` and fill `image_file` in the manifest.
2. Create ~6-8 short audio clips (TTS or record) reading a passage summary →
   `D:\SIH26_env\data\raw\audio\`, fill `audio_file`.
3. `. .\env.ps1` then `python ingest.py --src D:\SIH26_env\data\raw --reset`
   to rebuild the collection with all three modalities.
4. `python app.py` → open the Gradio URL → click-through: text→image query
   ("show me the screenshot about <topic>"), image upload → related docs,
   the two audio tabs (mic vs upload), a composite text+image query.

**For a fresh Claude session (no user needed):**
- Nothing blocking. Could: tighten manifest keywords; add a `tests/`
  dir with the RRF / chunker / citation unit checks that are currently
  ad-hoc; wire the `run_query_pipeline` image/audio branches through
  `evaluate.py` as a manual spot-check harness once assets exist; implement
  the Phase 5 cross-encoder reranker.
- To run anything: `cd` to repo, `. .\env.ps1`, then `python <script>`.

---

## Session — 2026-08-28 (session 2, continued — project relocated)

### Built / changed

- **Whole project moved to `D:\SIH26\`** (was split: source on
  `C:\Users\dmade\Desktop\SIH'26`, everything else on `D:\SIH26_env`). Now one
  folder: source `.py`/`.md`, `venv\`, `models\` `hf\` `whisper\` `ollama\`
  `pipcache\` `tmp\`, `data\{raw,chroma}` + the manifest CSVs.
- **`env.ps1` rewritten** to be location-independent: derives every path from
  `$PSScriptRoot`, activates the venv manually (sets `VIRTUAL_ENV` + PATH,
  clears `PYTHONHOME`) instead of calling the path-baked `Activate.ps1`. The
  project can now be moved again with zero edits.
- venv `activate*` / `Activate.ps1` / `pyvenv.cfg` also patched (old path →
  new) for good measure; `python -m pip` and the console `.exe`s work.
- `OLLAMA_MODELS` user env var → `D:\SIH26\ollama` (`setx`); ollama server
  restarted; `qwen2.5:3b-instruct` still lists fine from the new location.
- `run.ps1` unchanged (already used `$PSScriptRoot`).

### Verified (all from `D:\SIH26`)

- `. .\env.ps1` → `python -> D:\SIH26\venv\Scripts\python.exe`
- all 11 modules import; `config.REPO_ROOT = D:\SIH26`, caches/data paths
  resolve under `D:\SIH26\`
- `python ingest.py --stats` → `{'total': 51, ...}` (collection intact)
- `python generation.py "Where is the Three Gorges Dam located?"` → correct
  grounded answer, `[1]` → `msmarco_6f7a297c15e7.txt`
- `python app.py` → HTTP 200 on http://127.0.0.1:7860

### Broke / blocked on

- The old `C:\Users\dmade\Desktop\SIH'26` folder still exists as an **empty
  husk** — a Claude Code session is cwd-pinned to it so it couldn't be
  deleted. Safe to remove manually after restarting Claude Code / the editor
  from `D:\SIH26`.
- `D:\SIH26_env\tmp\` also lingered (a few `.tmp` files held by a live handle
  — "Device or resource busy"). Harmless; clears on reboot. The rest of
  `D:\SIH26_env` moved cleanly and was removed.
- Everything else from the earlier session-2 entry still stands (text-only
  corpus is the real gap; Qwen-3B occasionally front-loads citations).

### Next

Unchanged from the previous entry — the user still needs to create the demo
image/audio assets (see that entry's "Next"), just now under
`D:\SIH26\data\raw\images\` and `...\audio\`. To run anything: open a terminal
in `D:\SIH26`, `. .\env.ps1`, then `python <script>` or `.\run.ps1 <cmd>`.

**Editor:** open the *folder* `D:\SIH26` (not a single file); set the Python
interpreter to `D:\SIH26\venv\Scripts\python.exe`.

---

## Session — 2026-08-28 (session 2, continued — move reverted)

Moved the whole project under `D:\SIH26`, then reverted it at the user's
request. Back to the original split layout:
- Source: `C:\Users\dmade\Desktop\SIH'26\`
- venv + model caches + Chroma data: `D:\SIH26_env\`

### Net changes kept from the round trip (all improvements, verified working)

- **`env.ps1`** — now activates the venv *manually* (`$env:VIRTUAL_ENV` +
  prepend `venv\Scripts` to PATH + clear `PYTHONHOME`) instead of calling the
  venv's path-baked `Activate.ps1`. More robust; unaffected by future moves of
  the venv. Still hardcodes `$envRoot = "D:\SIH26_env"`.
- **`config.py`** — `USE_MOCK_BACKEND` default flipped earlier still stands;
  no change here.
- venv `activate*` / `pyvenv.cfg` restored to `D:\SIH26_env\venv` paths.
- `OLLAMA_MODELS` user env var restored to `D:\SIH26_env\ollama`; ollama
  server restarted and lists `qwen2.5:3b-instruct` fine.
- Leftover empty `D:\SIH26` removed.

### Verified after revert (from `C:\Users\dmade\Desktop\SIH'26`)

- `. .\env.ps1` → `python -> D:\SIH26_env\venv\Scripts\python.exe`
- all 11 modules import; `config.REPO_ROOT = C:\Users\dmade\Desktop\SIH'26`,
  `CHROMA_DIR = D:\SIH26_env\data\chroma`
- `python ingest.py --stats` → `{'total': 51, ...}`
- `python generation.py "Why do fossil fuels cause pollution?"` → grounded
  answer citing `[3][4]` → real MS MARCO passages
- `python app.py` verified earlier this session → HTTP 200 on :7860

### Next

Unchanged: user creates the demo image/audio assets (see the earlier
"session 2, continued — autonomous" entry's Next section), under
`D:\SIH26_env\data\raw\images\` and `...\audio\`, then re-ingest.
Run anything with: `cd` to the C: repo, `. .\env.ps1`, then `.\run.ps1 <cmd>`
or `python <script>`.

---

## Session — 2026-08-28 (session 2, continued — query speed)

### Problem
Every query in the web UI took ~50 s because `run_query_pipeline` unloaded
ImageBind after each call, forcing a ~40 s reload of the 4.5 GB checkpoint
from disk on the next question.

### Fixed
- **`embeddings.offload_to_cpu()`** (new) — parks ImageBind's weights in system
  RAM and frees only the GPU. `_load_model()` now detects a CPU-parked model
  and moves it back to CUDA (~2 s) instead of reloading from disk.
- **`query_pipeline.run_query_pipeline(offload_after=...)`** — new flag; takes
  precedence over `unload_after`. `app.py`'s `_real_answer` now passes
  `unload_after=False, offload_after=True`.
- **`config.OLLAMA_KEEP_ALIVE`** (default `"30s"`, env `RAG_OLLAMA_KEEP_ALIVE`)
  — passed to both `call_ollama*`. Qwen releases VRAM ~30 s after a response so
  it doesn't collide with ImageBind on the 6 GB card.
- **GPU zombie cleanup** — found 3.3 GB of the 6 GB held by orphaned processes
  (a leftover system-python + two lingering `llama-server.exe`). That was
  causing a hard segfault (`0xC0000005`) when ImageBind tried to load. Killed
  them; GPU back to ~80 MB idle. **If queries start crashing/hanging again,
  check `nvidia-smi` for stuck `python.exe` / `llama-server.exe` and kill
  them.** Ctrl+C on `app.py` doesn't always release the GPU cleanly.

### Verified
3 sequential queries through `app.handle_submit` in one process:
`Q1 164 s` (cold: Python imports + checkpoint load + first Qwen load),
`Q2 9.0 s`, `Q3 7.8 s`. Logs show the CPU<->GPU swap working. In the real
`app.py` the heavy imports happen at launch, so the first *question* is
~40-50 s and every one after is **~8 s**.

### Note
D: is an NVMe SSD (WD SN810), not a slow disk — the first-load cost is
Python import + CUDA init + 4.5 GB state_dict load, all one-time per process.

---

## Session — 2026-08-28 (session 2, continued — packaged for sharing)

- **`SETUP.md`** (new) — full new-machine setup guide: prereqs, the Smart App
  Control / execution-policy / ImageBind-`--no-deps` gotchas, venv build,
  Ollama, smoke test, corpus ingest, run commands, WSL alternative.
- **`env.ps1`** made portable — defaults every path to the repo folder;
  reads an optional `env.local.ps1` for per-machine overrides
  (`$DataRoot`, `$OllamaExe`). Ollama found via `Get-Command` now.
- **`env.local.ps1`** (this machine, NOT shared) — sets `$DataRoot =
  "D:\SIH26_env"`. Plus `env.local.ps1.example` (shared template).
- **`.gitignore`** — venv/models/caches/chroma/`env.local.ps1`/scratch.
- **`run.ps1`** — `ingest` target now uses `$env:RAG_RAW_DATA_DIR` not a
  hardcoded path.
- Demo text corpus copied into the repo (`data/raw/msmarco/*.txt` +
  `data/corpus_manifest*.csv`) so it ships in the zip.
- **`C:\Users\dmade\Desktop\multimodal-rag.zip`** (84 KB) — source + docs +
  text corpus, no venv/models/index. Friend follows `SETUP.md`.

### Note for whoever imports the zip
`data/chroma/` is not included — run `python ingest.py --src data\raw --reset`
after install to build the index (~1 min once ImageBind is set up).

---

## Session — 2026-08-29 — multimodal demo corpus + retrieval work

### Built

- **`make_demo_assets.py`** (new) — generates a themed image+audio corpus
  tied to the 25 energy passages, fully offline:
  - 12 PNGs via Pillow (report covers, slides, a chart, an email, a
    steam-turbine diagram, and `dashboard_1432.png` with a visible **14:32**).
    Real text on every image so OCR + vision both have signal.
  - 6 WAV "briefing calls" via Windows `System.Speech` (no pip dep).
  - Matches each asset to a passage by scored whole-word keyword match, writes
    `image_file` / `audio_file` / `image_timestamp` back into the manifest
    (`;`-separated, position-aligned for images).
- **3 ingest hooks** for cross-modal:
  - `vectorstore.ChunkRecord` gained `image_timestamp` + `linked_passage`;
    both emitted to Chroma metadata.
  - `ingest.load_manifest` now also returns an **asset index**
    (`image/audio filename -> AssetLink{theme, passage_doc_id, image_timestamp}`)
    so screenshots/clips inherit their passage's theme and carry the link.
  - `build_image_records` / `build_audio_records` accept the link.
- **Text-space bridges for images and audio** (the retrieval fix that made
  multimodal actually work):
  - `build_image_records` also embeds the image's **OCR text** as extra
    image-modality chunks in the *text* space (ARCHITECTURE.md §2's
    "text-heavy screenshots match better on OCR").
  - `build_audio_records` also embeds the **Whisper transcript segments** as
    text-space chunks (ImageBind text↔audio is weak, §4) — with their
    start/end times.
  - Collection is now 160 chunks: `document 51, image 25, audio 84`.
- **`query_pipeline` changes:**
  - text query -> single unified search (now reaches docs + image-OCR +
    audio-transcript, all comparable in text space).
  - image/audio *clip* upload -> per-modality fan-out **plus a text bridge**:
    the clip is OCR'd / transcribed and also run as a text query, so related
    documents surface (fixed `transcribe_spoken_query`'s ffmpeg `WinError 2`
    the same way `extract_audio` was fixed).
  - `timestamp_in()` detects `HH:MM` in a query -> extra Chroma search filtered
    on `image_timestamp`, fused with **3x RRF weight** so an exact capture-time
    match outranks fuzzy hits. Serves the PS's "screenshot taken at 14:32".
  - `reciprocal_rank_fusion` gained per-list `weights`; `vectorstore.query`
    gained `where_extra` for metadata filters.
  - `RetrievedChunk` carries `image_timestamp` / `linked_passage` through.

### Verified (earlier iterations)

- All 12 images OCR'd + linked; all 6 clips transcribed + linked.
- "how does a steam turbine work" -> document + `diagram_steam_turbine.png` +
  the nuclear-call transcript segment about steam turbines. All 3 modalities,
  all on-topic.
- "email screenshot about solar cost" -> solar doc + `email_solar_review.png`
  + `call_solar_cost.wav`.
- Timestamp filter surfaces `dashboard_1432.png @14:32`.
- (Final combined re-test running as this entry is written.)

### Broke / worked around

- ImageBind cross-modal distances are on wildly different scales (text↔text
  ~0.2, text↔image ~0.8, text↔audio ~0.9) — a single unified search buries
  non-text hits. The **text-space bridge chunks** (OCR + transcript) are what
  fix this for text queries; per-modality fan-out + transcript/OCR bridge fix
  it for clip queries. Pure rank-RRF across 3 modality lists alone
  over-corrected (forced one irrelevant hit per modality).
- `report_cover_hydro` briefly inherited `@14:32` (shared its passage row with
  `dashboard_1432`) — fixed by making `image_timestamp` a position-aligned
  `;`-list.

### Next

- Re-copy `data/raw/{images,audio}` + updated `corpus_manifest.csv` into the
  repo `data/` for the zip.
- Update `app.py` citations panel to show `linked_passage` ("related sources")
  and render image thumbnails / audio players for image/audio hits.
- Then the FastAPI layer (`/ingest`, `/query`, `/documents`, streaming,
  offline fallback) per the friend's-system review.

---

## Session — 2026-08-29 (cont.) — expandable citations panel

### Built
- **`vectorstore.ChunkRecord.source_path`** — absolute path stored in Chroma
  metadata so the UI can render/play the actual file. Threaded through
  `ingest.py` (all 5 ChunkRecord sites), `query_pipeline.RetrievedChunk`,
  `generation.Source`.
- **`generation.Source`** also carries `image_timestamp` / `linked_passage`.
- **`vectorstore.get_passage_text(doc_id)`** — resolves an asset's
  `linked_passage` to readable text for the "related passage" line.
- **`app.py` rewritten** — the citations panel is now a `@gr.render` block of
  expandable `gr.Accordion` cards, one per source:
  - image hit -> the image rendered inline (`gr.Image`) + capture time
  - audio hit -> an audio player (`gr.Audio`) + the cited segment (start–end s)
  - document hit -> passage text + locator
  - every card shows the cross-modal "related passage" text
  - cited cards auto-expand; a flags line shows 🎤 transcribed query /
    cited-nothing / dropped-hallucination warnings
  - `launch(allowed_paths=[RAW_DATA_DIR], server_port=None)` so Gradio can
    serve the asset files and auto-pick a free port.
- **`run.ps1`** — added `stop` (kill stale python/llama-server/ollama holding
  the GPU or port) and `assets` (regenerate demo corpus) targets.
- **`app.py`** — dropped the hard `server_port=7860`; auto-scans 7860-7959.
- Synced `data/raw/{images,audio}` + manifests into the repo (45 files, 5.3 MB)
  so the demo corpus ships in the zip.

### Verified
- `handle_submit("...screenshot taken at 14:32")` -> answer names
  `dashboard_1432.png`; payload has it as source [1], `cited=True`,
  `image_timestamp=14:32`, `source_path` present, `linked_text` resolved.
- `app.py` launches under gradio 6, serves HTTP 200, `@gr.render` with
  Image/Audio inside Accordions works.
- Full re-ingest clean: 160 chunks (51 doc / 25 image / 84 audio).

### Next
- Browser click-through: image upload, the 🎤 mic tab (spoken query),
  audio-clip upload — confirm the panel renders images/players.
- Then the FastAPI layer (`/ingest`, `/query`, `/documents`, streaming,
  offline fallback) from the friend's-system review.
- Rebuild `multimodal-rag.zip` once the above is confirmed.

---

## Session — 2026-08-29 (cont.) — real-image test + PySide6 desktop UI

### Verified — vision modality works with arbitrary real images
Separate `phototest` Chroma collection (`RAG_COLLECTION` env override, new),
38 Pokémon sprites + 3 real photos:
- text→image: "orange dragon breathing fire"→charizard, "round pink
  creature"→jigglypuff, "blue sea serpent"→gyarados, "person wearing a face
  mask"→the mask photo, "hackathon with laptops"→the hackathon photo.
- image→image: charmander→other starters; squirtle→water types.
Conclusion: ImageBind text↔image / image↔image are solid. The earlier
roadmap-screenshot failures were OCR-of-dense-keyword-text hitting the weak
CLIP text tower — a known ImageBind tradeoff, not a vision-pipeline bug.
`group_and_rank` also fixed to keep up to 3 *materially different* chunks per
dense source (was collapsing a 10-topic slide to 1 fragment).

### Built — PySide6 desktop app ("RAG_OS"), replacing Gradio as primary UI
From the user's `rag_os_ui_pyside6_handoff` zip (UI-only shell). New package
`desktop/`:
- **`desktop/rag_os.qss`** — their dark terminal theme, extended for
  attachment chips, the media detail panel, status states.
- **`desktop/main_window.py`** — their layout + object names kept; added:
  📎 IMAGE / 🔊 AUDIO file-attach with clearable chips, a working ● MIC
  (QMediaRecorder → WAV → spoken query), a "spoken question" checkbox, and a
  SOURCE DETAIL drawer that renders the image (QPixmap), an audio ▶/■ player
  (QMediaPlayer), transcript/page text, capture time, and the cross-modal
  "related passage" line. `querySubmitted` now carries a dict
  `{text,image,audio,spoken}`. AI INSIGHTS + AUDIT LOG tabs get real data.
  Confidence = blend of top-hit closeness + citation coverage → % (or LOW).
- **`desktop/controller.py`** — `RagWorker(QObject)` on a `QThread` runs
  `run_query_pipeline` + `generate_grounded_answer` off the GUI thread
  (offload_after so ImageBind stays warm). `_sources_payload` = per-file
  cards. `_confidence` / `_insights` helpers.
- **`desktop/main.py`** — entry; puts repo root on sys.path, offline env vars.
- **`run.ps1`** — `.\run.ps1` now launches the desktop app; `.\run.ps1 web`
  is the Gradio fallback.
- **PySide6 6.11.2** installed; added to requirements.txt.

Offscreen construct + Controller wiring verified. End-to-end UI query test
running as this entry is written.

### Next
- Confirm the desktop end-to-end test (query → worker → add_result).
- Real launch on the user's display; click-through image/audio/mic.
- Restore `phototest` note: the demo collection `multimodal_rag` is untouched
  (194 chunks incl. the 2 WhatsApp roadmap images the user added).

---

## Session — 2026-08-29 (cont.) — corpus view + in-app ingest

### Built
- **`vectorstore.list_sources(collection)`** — every distinct ingested file
  (name, modality by majority vote, chunk count, doc_id, path, theme).
  **`vectorstore.delete_source(collection, name)`** — drop all chunks of a file.
- **`ingest._process_files()`** — extracted the per-file loop from `ingest()`
  so it's reusable. **`ingest.add_files_to_corpus(paths, progress=...)`** —
  copies user-picked files under `RAW_DATA_DIR/{docs,images,audio}/`, ingests
  into the *current* collection (no reset), keeps ImageBind warm
  (`offload_to_cpu`). Verified: adding `hackathon.jpg` -> 3 chunks, collection
  194→197, and "a photo of a hackathon…" then returns it #1.
- **`desktop/` CORPUS tab** — lists all indexed files with counts in the
  header; **+ ADD FILES** (multi-file picker → worker-thread ingest with a
  live progress line → auto-refresh), **↻** refresh, **DELETE** selected.
- **`desktop/controller.py`** — `RagWorker.ingest(paths)` slot +
  `ingestProgress` / `ingestDone` signals wired to the window.
- `MainWindow.ingestRequested = Signal(list)`.

So the app now supports the full loop the user asked for: **upload files while
running → they ingest → the RAG answers about them immediately.**

### Note
My `add_files_to_corpus` test left `hackathon.jpg` in the demo collection
(`multimodal_rag`, now ~46 files). Harmless — remove via the CORPUS tab's
DELETE or a `.\run.ps1 ingest` rebuild.

### Next
- User launch + click-through: + ADD FILES with a PDF and an image, then query.
- Optional: FastAPI layer (from the friend's-system review) still pending.

---

## Session — 2026-08-29 (cont.) — system design doc + perf plan

- Published the **as-built system design** as an artifact:
  https://claude.ai/code/artifact/508cf5de-2250-4454-a08c-4660d7302515
  Covers: component map, data model (chunk schema), ingestion + query
  workflows (with diagrams), the 6 design decisions that diverged from the
  original ARCHITECTURE.md (D1 text-space bridges, D2 routing, D3 timestamp
  filter, D4 chunks-per-doc, D5 model lifecycle, D6 in-app ingest), measured
  performance profile, and a 10-item speed roadmap.
- Renamed nothing yet — the CORPUS tab could become LIBRARY/INDEX (user asked
  why "corpus"; pending their pick).

### Perf roadmap (from the doc, priority order)
- **R1 pre-warm ImageBind at app startup** → first query ~60s → ~8s. low risk.
- **R2 drop the 2s audio-window embeddings** → audio ingest & clip queries
  faster, ~40% fewer audio chunks. transcript chunks already do the work.
- **R3 Ollama keep_alive 30s → 5m** → warm query −2-3s.
- R4 stream answer to UI · R5 faster-whisper · R6 trim prompt · R7 BM25 hybrid
  (the quality one) · R8 cross-encoder rerank · R9 1.5B speed model · R10
  keep ImageBind GPU-resident (OOM risk).

### Next
- Implement R1+R2+R3 (quick, low-risk) if the user wants.
- FastAPI layer still pending (from the friend's-system review).

---

## Session — 2026-08-29 (cont.) — R1/R2/R3 speed + BM25 hybrid

### Built
- **R3** — `config.OLLAMA_KEEP_ALIVE` `30s` → `5m`. Qwen stays resident across
  a read-then-ask gap.
- **R2** — `config.AUDIO_EMBED_WINDOWS` (default **off**). `build_audio_records`
  and `extract_audio` skip the 2 s window slicing + embedding; only Whisper
  transcript segments are indexed. `route_input` skips the audio-tower
  QueryInput for uploaded clips when off (transcript bridge handles them).
  Re-ingest: audio chunks **84 → 26**, total **197 → 139**.
- **R1** — `desktop/controller.py`: `RagWorker.warmup()` slot invoked on the
  worker thread at app start (`QMetaObject.invokeMethod`), loads torch +
  ImageBind + warms the BM25 index while the user reads the UI. Emits
  `warmupState`; top bar shows `● WARMING UP…` → `● LOCAL MODE · READY`.
- **BM25 hybrid** — new `lexical.py`: `BM25Okapi` over every chunk's
  `text_preview`, cached and auto-rebuilt when `collection.count()` changes.
  `lexical.search()` returns hit dicts shaped like `vectorstore.query()`.
  `query_pipeline` runs it for every text input and adds `<origin>:bm25` as
  another list into the same RRF. `config.USE_BM25` (default on),
  `BM25_SEARCH_K = 20`. `rank-bm25` added to requirements.

### Verified
BM25 verified: "what does the roadmap say about SQL and DBMS" -> the roadmap screenshot SQL chunks surface via `bm25` (d 0.089-0.28) where vector search alone missed them. matched_by shows hybrid fusion (chunks hit by both bm25 + text->unified rank top). Energy/timestamp regression check running.
query, OOP resources, practice hours, plus steam-turbine + 14:32 regression.)

### Next
- All 4 changes verified (see above). Desktop app launches, warmup loads ImageBind.
  queries; energy/timestamp queries must not regress.
- Re-ingest note: collection is now 139 chunks (incl. the 2 WhatsApp roadmap
  images + hackathon.jpg the user/test added).
- System-design artifact updated (perf table, D7, roadmap R1-R3+R7 marked shipped, lexical.py added).
- FastAPI layer still pending.

### Verified (2026-08-29) — R1/R2/R3 + BM25, no regression
- `python run_query_pipeline` (retrieval only, warm, offload_after=True):
  - "steam turbine condenser" 35.7s cold → **1.2s / 1.1s** warm (Q2, Q3)
  - "14:32" → `dashboard_1432.png @14:32` #1
  - "hydropower" → hydro doc #1 + hydro call (audio via transcript, windows dropped)
  - "SQL and DBMS" → roadmap screenshot SQL chunks via `bm25`
- `matched_by` shows hybrid fusion working: chunks hit by both
  `bm25|text->unified` rank top; pure-`bm25` chunks fill gaps vectors miss.
- Warm *retrieval* is now ~1.2s (was inside the ~8s figure). Full query +
  Qwen generation ≈ 5-6s warm now (keep_alive=5m → no LLM reload).
- The earlier "_bm.py exit 255" runs were PowerShell pipe/timeout artifacts,
  not code — `Start-Process -Wait` + file output ran clean.

---

## Session — 2026-08-29 (cont.) — desktop "Not Responding" fix

The RAG_OS window showed "(Not Responding)" at launch. Three GUI-thread
blockers, all fixed:
1. **`from PySide6.QtMultimedia import ...` at module top of main_window.py** —
   initializes the Windows media subsystem / probes audio devices, can block
   10-30 s. Now lazy: `_mm()` helper, imported only on mic/playback use.
2. **`reload_corpus()` called in `MainWindow.__init__`** — chromadb import +
   full metadata read before first paint. Now `QTimer.singleShot(80, ...)`.
3. **warmup queued via `QMetaObject.invokeMethod`** — replaced with the
   canonical `self.thread.started.connect(self.worker.warmup)` so it's
   unambiguously on the worker thread. `Controller.shutdown` also hardened
   (terminate fallback if the thread is mid model-load).

Verified: `python -m desktop.main` — window `Responding=True` at t+6s, t+18s,
t+43s (throughout the ImageBind warm-up). All `_HAS_MM` refs updated to `_mm()`.

### Next
- User confirms the launch on their display + clicks through.
- R4 (stream the answer to the UI) is the next perf item.
- FastAPI layer still pending.

---

## Session — 2026-08-29 (cont.) — backend split into its own process

The RAG_OS window froze ("Not Responding") because `import torch` on the
worker *thread* still holds the GIL. Fixed properly: the pipeline now runs in
a **separate process**.

### Built
- **`backend/server.py`** — FastAPI on `127.0.0.1:8077` (localhost only, still
  fully offline). Warms up ImageBind + BM25 on a background thread in its own
  process. Endpoints:
  - `GET /health` — `{ok, warmup: cold|loading|ready, ollama, collection}`
  - `POST /query` — full grounded result (shaped for the UI)
  - `POST /query/stream` — SSE: `meta` → `token`* → `done` (R4, streaming)
  - `GET /library`, `POST /library/delete`
  - `POST /ingest` — SSE progress → `done` summary
  - a `threading.Lock` serialises pipeline calls (ImageBind singleton)
- **`generation.stream_grounded_answer()`** — generator yielding
  `('sources'|'token'|'done', …)` for the SSE endpoint.
- **`config.py`** — `BACKEND_HOST/PORT/URL` (`RAG_BACKEND_*` env).
- **`desktop/api.py`** — thin httpx client (`query`, `query_stream`,
  `library`, `delete_source`, `ingest_stream`, `health`) + an SSE parser.
- **`desktop/controller.py`** — rewritten: `RagWorker` only makes blocking
  HTTP calls on its thread (GIL released during I/O). `wait_for_backend`
  polls `/health`. Streams tokens → `answerToken` signal.
- **`desktop/main_window.py`** — `on_answer_started/token`, `on_backend_state`;
  send button disabled until warm; `reload_corpus` / delete go through the API
  (no more `import vectorstore` in the UI process).
- **`desktop/main.py`** — spawns `python -m backend.server` (CREATE_NO_WINDOW)
  if `/health` is down, then shows the window *immediately*; backend left
  running on exit for a fast next launch.
- **`run.ps1`** — `.\run.ps1` launches UI (+ backend if needed);
  `.\run.ps1 server` runs just the backend.

### Verified
- Backend standalone: `/health` ok, warm-up ready ~50 s, `/library` → 46
  files, `/query` "steam turbine condenser" → grounded answer, confidence
  74 %, 8 sources (`diagram_steam_turbine.png` + doc cited).
- `/query/stream`: 125 tokens streamed, `done` with confidence 88 %.
- Desktop launch: **all python processes `Responding=True` throughout the
  ~50 s warm-up** — the UI never freezes. Window + backend are separate PIDs.

### Next
- User launch + click-through with the new streaming answer.
- Wire the streamed tokens visibly (they already append to the answer box).
- Update the system-design artifact for the 2-process architecture.

## Session — 2026-08-29 (cont.) — grounding gate: reject off-corpus questions

Backend-split test run surfaced a regression: "what is the capital of France"
answered **"Paris [1]"** (conf 64 %, no warning). BM25 always returns *some*
chunk, so the LLM always had a citation to hang a world-knowledge answer on,
and `cited_nothing` stayed False so nothing flagged it.

### Built
- **`config.py`**
  - `BM25_MIN_SCORE` (3.5) — drop BM25 hits below this raw Okapi score; a lone
    common word ("capital") no longer counts as a match.
  - `BM25_STRONG_SCORE` (8.0) — a hit at/above this is a real rare-term match
    ("SQL", "DBMS") and overrides the grounding gate.
  - `RETRIEVAL_MAX_DISTANCE` (0.40) — cosine-distance gate. Measured: good
    in-domain text queries land 0.18–0.27; off-corpus ones 0.42–0.53.
- **`lexical.py`** — `search()` stops at `BM25_MIN_SCORE` instead of `<= 0`.
- **`query_pipeline.py`** — `QueryResult` gains `weak: bool` + `best_distance`.
  A text query is `weak` when the closest **vector** (text-space) chunk is
  past `RETRIEVAL_MAX_DISTANCE` **and** no BM25 hit clears `BM25_STRONG_SCORE`.
  Image/audio-clip queries skip the gate (cross-modal distances).
- **`generation.py`**
  - `SYSTEM_PROMPT` rule 3 hardened: answer only from Sources, never own
    knowledge; if nothing matches reply exactly `REFUSAL`.
  - `generate_grounded_answer(..., weak=True)` / `stream_grounded_answer` now
    **short-circuit the LLM**: return `REFUSAL` ("The corpus does not contain
    information on this.") with sources still populated (user sees what came
    closest) — a famous fact like "Paris" can't leak from the 3B weights.
- **`backend/server.py`** — `_confidence` / `_warning` take `weak` +
  `best_distance`: `weak` → `LOW` + warning; BM25 pseudo-distances
  (`1/(1+score)`) excluded from the closeness score via `_is_bm25_only`;
  `cited_nothing` only forces LOW when `best_distance > 0.33` (tight match +
  missing `[n]` marker is just the 3B model skipping formatting).
  `_insights` shows a `match  best_dist 0.xxx` line.

### Verified (8-query batch, /query and /query/stream)
| query | result |
|---|---|
| steam turbine condenser | 74 %, diagram + doc cited |
| nuclear power costs | 63 %, doc cited |
| solar panel economics | 62 %, doc cited |
| SQL and DBMS (noisy OCR) | 66 % — BM25 strong-score override kept it |
| **capital of France** | **"corpus does not contain…", LOW + warning** |
| who won the 2022 world cup | refused, LOW + warning |
| how do I bake sourdough bread | refused, LOW + warning |
| screenshot at 14:32 | 64 %, dashboard_1432.png cited (weak synthesis, known) |

### Next
- `evaluate.py` re-run to confirm Recall@5 held after the BM25 min-score floor
  (was 0.96).
- User launch + click-through of the desktop app.
- Q2 ("screenshot at 14:32") synthesis is still soft — the dashboard image has
  no report reference in it, so the honest answer is roughly right anyway.

### Fixed — UI crash / freeze right after an answer finishes

`Controller` was a plain `class`, not a `QObject`. Its `_on_result` method was
connected to the worker's `resultReady` signal — but a signal→plain-method
connection has no thread affinity, so Qt invoked it **directly on the worker
thread**. `_on_result` → `window.add_result()` writes to the answer box and
the audit-log `QTextEdit` from the worker thread → *"QObject: Cannot create
children for a parent that is in a different thread"* → hard crash or
"(Not Responding)" the moment streaming completed.

- **`desktop/controller.py`** — `Controller(QObject)` + `super().__init__()`;
  `_on_result` decorated `@Slot(dict)`. Now delivered back to the GUI thread
  via a queued connection. `reload_corpus()` deferred with
  `QTimer.singleShot(0, …)` so the answer paints before the `/library` refresh.
- **`desktop/main.py`** — `faulthandler.enable()` for real tracebacks on any
  future hard crash.

Verified: backend `/query/stream` returns a grounded answer (conf 67 %, 8
sources); window opens (`RAG_OS — Offline Multimodal RAG`), Responding=True.
User to confirm no crash/freeze after the answer completes in the UI.

## Session — 2026-08-29 (cont.) — repo restructured into a `rag/` package

Flat pile of 15 modules in the root → a real package layout (user request).

### Moved
| was (root) | now |
|---|---|
| `config.py` | `rag/config.py` |
| `embeddings.py` | `rag/core/embeddings.py` |
| `vectorstore.py` | `rag/core/vectorstore.py` |
| `extraction.py` | `rag/ingest/extraction.py` |
| `chunking.py` | `rag/ingest/chunking.py` |
| `ingest.py` | `rag/ingest/corpus.py` |
| `build_corpus_manifest.py` | `rag/ingest/manifest.py` |
| `query_pipeline.py` | `rag/retrieval/pipeline.py` |
| `lexical.py` | `rag/retrieval/lexical.py` |
| `generation.py` | `rag/generation/answer.py` |
| `app.py` | `web/app.py` |
| `evaluate.py` | `scripts/evaluate.py` |
| `make_demo_assets.py` | `scripts/make_demo_assets.py` |
| `PRD/ARCHITECTURE/TECH_STACK/ROADMAP/SETUP/KICKOFF_PROMPT .md` | `docs/` |

`backend/` and `desktop/` packages unchanged in place.

### Rewired
- ~55 imports rewritten to absolute package paths (`from rag.core import
  embeddings`, `from rag.retrieval import pipeline`, …). Renamed-module call
  sites updated (`query_pipeline.` → `pipeline.`, `generation.` → `answer.`,
  `ing.` → `corpus.`).
- `backend/server.py`, `desktop/main.py` already put the repo root on
  `sys.path`; added the same 1-liner to `web/app.py`, `scripts/evaluate.py`,
  `scripts/make_demo_assets.py`.
- `run.ps1`: `web` → `web/app.py`, `eval` → `scripts/evaluate.py`,
  `stats`/`ingest` → `python -m rag.ingest.corpus`, `ask` → `python -m
  rag.generation.answer`, `assets` → `scripts/make_demo_assets.py`.
- `CLAUDE.md` rewritten with the new layout + a repo-map section;
  `README.md` added; `docs/TECH_STACK.md` reference-design section marked
  superseded with a module map.
- Stale `python ingest.py` / `query_pipeline.py` refs in docstrings updated.

### Verified
- All 15 modules + `backend.server` + `desktop.main` + `web/app.py` import
  cleanly (fresh interpreter).
- Backend warm-up OK (~45 s); logger names now `rag.core.embeddings` etc.
- `/query` batch: steam turbine 74 % (2 cited), "capital of France" refused,
  SQL/DBMS screenshot 66 % — grounding fix intact through the move.
- Desktop app launches, window responsive, no ui.err output.

### Next
- `scripts/evaluate.py` run (Recall@5 was 0.96) — still blocked by GPU
  contention with the running backend; run with `.\run.ps1 stop` first.
- User click-through of the desktop app on the new structure.

### Shareable zip built

`Desktop/SIH26-multimodal-rag.zip` (4 MB, 86 files) for handing to a
collaborator. Contains all source (`rag/ backend/ desktop/ web/ scripts/`),
docs, `requirements.txt`, `env.ps1`/`run.ps1`, `env.local.ps1.example`, and
the **full demo corpus** (`data/raw/` 25 txt + 12 img + 6 wav +
`corpus_manifest*.csv`). Excludes venv, model weights, `.model_cache/`
(4.8 GB ImageBind ckpt), `data/chroma/`, `env.local.ps1`, `__pycache__`.
Verified: extracts to `SIH26-multimodal-rag/`, all modules import from a
clean path. `docs/SETUP.md` refreshed for the `rag/` layout + new run
commands + "corpus now includes images/audio".

## Session — 2026-08-29 (cont.) — themed UI dropped in

Friend restyled the desktop UI (from `Downloads/SIH26-multimodal-rag-themed.zip`,
built on top of our latest — same `rag/` package, same controller QObject fix,
same backend). Only two files differed:

- `desktop/main_window.py` (577 → 739 lines) — new layout: left rail owns
  inputs/actions (new chat, drag-drop, upload, image, audio, mic, sources),
  bigger right-side answer workspace, persistent bottom instrument bar
  (AI ANALYTICS / CORPUS / TIMELINE / AUDIT LOG).
- `desktop/rag_os.qss` (58 → 230 lines) — deep navy-black `#070B10`, Inter
  body + JetBrains Mono for labels/data, cyan accent `#69C7FF`, green status,
  amber warnings.

Signal/slot surface unchanged (`querySubmitted`, `ingestRequested`,
`on_answer_*`, `add_result`, `reload_corpus`, …) — clean drop-in against our
`controller.py` / `api.py`. Originals saved to scratchpad as `*.pre-theme.bak`.

Verified: imports + compiles; backend warm-up OK; desktop app launches
(window responsive, no ui.err); `/query` "steam turbine" → 74 %, 8 sources.

## Session — 2026-08-29 (cont.) — corpus migration: MS MARCO → AMI Meeting Corpus

### Why
The demo's hardest claim (PRD "judge angle" #2) is cross-format linking —
transcript segment ↔ cited paragraph ↔ on-screen slide. The MS MARCO plan
needed *manufactured* image/audio pairs per passage. The **AMI Meeting
Corpus** gives genuinely simultaneous, same-content audio / transcript /
slides / documents for the same meetings — nothing paired after the fact.
GovReport and MS MARCO both dropped.

### Verified before committing (per the migration brief)
- **License: CC BY 4.0** — confirmed on the official page
  (`groups.inf.ed.ac.uk/ami/corpus/license.shtml`) and the HF mirror
  metadata (`license:cc-by-4.0`). Attribution only, commercial use allowed.
  The older "CC BY-NC-SA" description is stale. Attribution block to add to
  README when AMI files land.
- **Slides exist** for almost all meetings (`datapresent.shtml`); JPEGs are
  at `AMICorpusMirror/amicorpus/<mid>/slidesBackUp/<start>__<end>.jpg` — the
  filename IS the slide's display window in seconds. No OCR .txt ships → our
  own pytesseract pass in ingestion produces it (same as any image).
- **DOC/PDF leg is REAL — step 1 of the decision succeeds.** Each scenario
  meeting's `shared-doc/` has participant-produced
  `ProjectDocuments.Final.Report.doc`, `Minutes.of.{Kickoff,2nd,3rd,4th}-meeting.doc`,
  plus `.ppt` design decks and `*sum.txt` post-meeting summary emails. No
  synthetic content needed. `.doc` → text via **antiword** (already on PATH
  at `/mingw64/bin/antiword`); `.ppt` skipped (the slide JPEGs already are
  screenshots of those decks); `*sum.txt` used as-is.
- **Sources:** audio+rough transcript on HF `edinburghcstr/ami` (ihm, 36
  meetings, ALL-CAPS no-punctuation `text` field). Clean transcripts +
  slide-change timing come from the official
  `AMICorpusAnnotations/ami_public_manual_1.6.2.zip` (22.9 MB, all meetings).
  Meeting audio: mirror `<mid>/audio/<mid>.Mix-Headset.wav` (~40 MB each).

### Docs updated
`PRD.md` (judge-angle #2, eval bullet), `TECH_STACK.md` (corpus row +
doc-extraction row), `ROADMAP.md` (status block, Phase 2 rewritten, Phase 4
eval), `ARCHITECTURE.md` (chunk-size note). Chunking / RRF / citation
contract unchanged — only the corpus source moved.

### Open decisions for next step
1. **Meeting set.** Default plan: `ES2002a–d` — one complete scenario project
   arc (kickoff → final report), has slides + whiteboard + real minutes +
   final report, and is in the HF ihm set. ~220 MB download, ~2 h of audio.
   Alternative: add `ES2003a–d` (8 meetings).
2. **Transcript source.** Official manual annotations (proper casing +
   punctuation, +parsing code) vs HF `edinburghcstr/ami` (trivial to load
   but ALL-CAPS, no punctuation — reads badly in cited output). Leaning
   official.
3. **Audio handling.** A 30-min meeting per file makes our Whisper-on-ingest
   step redundant (we have the real transcript) and slow. Plan: let the
   manifest supply a pre-made transcript so `extract_audio` skips Whisper,
   and still slice the audio into ImageBind windows. Small `extraction.py` /
   `corpus.py` tweak — "what flows in", not the core design.

### Next
- Write the AMI builder (`rag/ingest/manifest.py`) + update
  `corpus.py load_manifest` for the AMI manifest schema
  (`doc_id`=`meeting_id`, `modality`, `source_file`, `text_preview`,
  `start_time`, `end_time`).
- Run it for the chosen meetings; ingest; rebuild the dev-query/qrels set.

### Built — AMI corpus builder + pipeline wiring (2026-08-29 cont.)

Decisions taken: **ES2002a + ES2002b**, **official manual transcripts**,
code-only this session (user runs the download + full ingest).

- **`rag/ingest/manifest.py`** rewritten as the AMI builder. `python -m
  rag.ingest.manifest` (defaults ES2002a,b) downloads per meeting into
  `data/raw/ami/<mid>/`:
  - `<mid>.Mix-Headset.wav` + `<mid>.Mix-Headset.wav.segments.json` — the real
    time-aligned transcript, parsed from the manual `words`+`segments` NXT XML
    in `ami_public_manual_1.6.2.zip` (proper casing + punctuation; verified:
    ES2002a → 236 segments, ~2600 words).
  - `slides/<mid>.<start>__<end>.jpg` from the mirror `slidesBackUp/` (filename
    = on-screen window in seconds).
  - `docs/*.txt` — real minutes (`.doc` → text via **antiword**, auto-located
    incl. the Git-for-Windows path) + the 4 per-role summary emails;
    `_shared/ES2002.final-report.txt` (series-level).
  Writes the manifest CSV `doc_id,meeting_id,modality,source_file,text_preview,
  start_time,end_time`. `doc_id` = meeting id (final report = series `ES2002`).
  Prints the CC BY 4.0 attribution block.
- **`rag/ingest/corpus.py`**: `load_manifest()` auto-detects the AMI schema
  (has a `modality` column) vs the legacy MS MARCO one. New `AssetLink`
  fields `start_time`/`end_time`. The ingest loop takes `doc_id` from the
  manifest link when present. `build_image_records` / `build_document_records`
  now prefix chunk ids with a filename slug so many files under one meeting
  `doc_id` don't collide. `iter_source_files` skips `_cache/`.
- **`rag/ingest/extraction.py`**: `extract_audio` reads a
  `<wav>.segments.json` sidecar and skips Whisper entirely when present.
- **`rag/generation/answer.py`**: image `_locator` says "on screen mm:ss-mm:ss"
  for AMI slide windows, "captured HH:MM" for screenshots.
- **`rag/config.py`**: `AMI_*` constants. **Also fixed a restructure bug** —
  `REPO_ROOT` was `Path(__file__).parent` (= `rag/` since config.py moved
  into the package); now `parents[1]`. Was masked on this machine because
  `env.ps1` sets every `RAG_*` path to D:.
- **`run.ps1`**: `ami` target; `ingest` now points at `data/raw/ami`.

Verified: builder ran end-to-end for ES2002a (`--slide-limit 3`) — audio,
transcript JSON, slides, 6 docs, manifest all correct; `load_manifest()`
resolves every file to its meeting `doc_id`; `compileall` clean.
`data/raw/ami/ES2002a/` is now partially populated (test download, cached).

### Next
- `python -m rag.ingest.manifest` (full ES2002a + ES2002b, ~250 MB).
- `python -m rag.ingest.corpus --src data/raw/ami --reset` — first AMI index.
- Hand-write ~20–30 AMI dev queries + qrels; re-run `scripts/evaluate.py`.
- Add the AMI CC BY 4.0 attribution to `README.md`.
- Old MS MARCO files still under `data/raw/{msmarco,images,audio}` +
  `data/corpus_manifest_qrels.csv` — remove once AMI is the live corpus.

### Corpus consolidated — AMI only, one location (2026-08-29 cont.)

User ran `python -m rag.ingest.manifest`; a first pass was interrupted before
the `.doc` fetch (no minutes/report), re-ran to completion.

**Single corpus location: `D:\SIH26_env\data\raw\ami\`** (env.ps1 puts data on
D:; C: is at ~1.6 GB free). 41 files / 113 MB:
- ES2002a: audio + transcript sidecar + 17 slides + kickoff minutes + 4 role
  summaries
- ES2002b: audio + transcript sidecar + 11 slides + 2nd-meeting minutes + 4
  role summaries
- `_shared/ES2002.final-report.txt` (series doc_id `ES2002`)
- `_cache/` — the 22 MB annotation zip (reused, gitignored)

Manifest `D:\SIH26_env\data\corpus_manifest.csv`: 41 rows
(audio:2, image:28, document:11).

Old MS MARCO + synthetic corpus archived to
`D:\SIH26_env\data\_archive_pre_ami\` (msmarco/, images/, audio/, old qrels) —
reversible. Repo `data/` on C: emptied to `data/raw/.gitkeep`.
`.gitignore` now excludes `data/raw/` and `data/corpus_manifest*.csv` — the
corpus is download-on-demand, not committed.

### Next
- Stop the running desktop app/backend (they hold the old Chroma collection),
  then `python -m rag.ingest.corpus --src data\raw\ami --reset` for the first
  AMI index.
- Also kill the stale backend+desktop from 09:12 (duplicate of the 10:57 set).

### Corpus moved into the project folder (2026-08-29 cont.)

User wants the corpus visible in the editor alongside the code, not split
onto D:. Done — and it fit because a **duplicate 4.5 GB ImageBind checkpoint**
was sitting in `C:\...\SIH'26\.model_cache\` (from an early run without
env.ps1). Deleted it — the live one is on D: — freeing C: from 3.0 → 7.3 GB.

- Corpus + manifest moved `D:\SIH26_env\data\` → `C:\...\SIH'26\data\`:
  `data\raw\ami\` (135 MB), `data\corpus_manifest.csv`. The Chroma index will
  build into `data\chroma\` on next ingest.
- **`env.ps1`**: new `$PipelineDataRoot = $RepoRoot`; `RAG_RAW_DATA_DIR` /
  `RAG_CHROMA_DIR` / `RAG_MANIFEST` now derive from it, not `$DataRoot`. So the
  corpus + index always live with the code; `env.local.ps1`'s `$DataRoot`
  still moves only the heavy stuff (venv, models, whisper, ollama, hf) to D:.
- `env.local.ps1` + `.example` comments updated.
- Old `D:\SIH26_env\data\{chroma,_archive_pre_ami}` now orphaned (env doesn't
  point there) — harmless; `_archive_pre_ami` is the intentional MS MARCO
  backup.

Verified: `. .\env.ps1` → all `RAG_*` data paths resolve under the repo;
`load_manifest()` reads the AMI manifest (3 doc_ids, 82 asset keys).

### Next
- `python -m rag.ingest.corpus --src data\raw\ami --reset` — first AMI index,
  now lands in `data\chroma\` in the project.

### First AMI index built + old corpus purged (2026-08-29 cont.)

`python -m rag.ingest.corpus --src data\raw\ami --reset` →
**866 chunks: document 91 / image 107 / audio 668** (audio = the real
time-aligned transcript segments; ES2002b alone is 432). Index at
`data\chroma\` in the project (6.8 MB). No Whisper ran — the `.segments.json`
sidecars were used.

Smoke query "what is the project about and who are the team members" →
grounded answer citing **both** a summary doc `[7]` and an audio transcript
segment `[1] @ 02:12-02:22` — cross-modal citation working on real data.

Purged (user: don't need the old corpus):
`D:\SIH26_env\data\` deleted entirely (archive + orphaned old index); D: now
holds only venv / models / whisper / ollama / hf. `data\raw\ami\_cache\`
(22 MB annotation zip) also deleted — re-running `rag.ingest.manifest`
re-fetches it.

**AMI corpus = 43 files, 113 MB, at `C:\...\SIH'26\data\`:**
2 audio + 2 transcript sidecars + 28 slides + 11 docs (2 minutes, 8 role
summaries, 1 final report) + `corpus_manifest.csv` (41 entries).

### Refreshed shareable zip (2026-08-29 cont.)

`Desktop/SIH26-multimodal-rag.zip` rebuilt — **66 MB / 85 files**
(32 code, 9 docs, 44 corpus). Now includes the **full AMI corpus**
(`data/raw/ami/` — 43 files, 2 wavs, transcripts, slides, docs) +
`corpus_manifest.csv`, so a collaborator runs only
`python -m rag.ingest.corpus --src data\raw\ami --reset` — no download step.
Excludes venv / models / `.model_cache` / `data/chroma` /
`data/raw/ami/_cache` (22 MB annotation zip) / `__pycache__` /
`env.local.ps1` / logs / `*.bak`.

`requirements.txt` updated: added `fastapi` / `uvicorn` / `httpx` (backend
split), noted the `antiword` binary dependency, dropped `datasets` (AMI
builder is stdlib-only). `README.md` gained the AMI CC BY 4.0 attribution.
Verified: extracts to `SIH26-multimodal-rag/`, all modules import from a
clean path, corpus + manifest intact.

### Source-only zip + full requirements.txt (2026-08-29 cont.)

- **`Desktop/SIH26-multimodal-rag-src.zip`** — 138 KB, 47 files. Code + docs
  only (no `data/`, no `deploy/hf-space/data/`, no media). For sharing under a
  10 MB cap; recipient fetches the corpus with `python -m rag.ingest.manifest`.
  (`SIH26-multimodal-rag.zip`, 66 MB, still exists — that one bundles the AMI
  corpus for a turnkey setup.)
- **`requirements.txt` rewritten** as a self-contained install guide: a header
  block with the system prereqs (Python 3.11, NVIDIA driver, Tesseract,
  Ollama, antiword, ffmpeg), the 4 manual steps (torch via the CUDA index,
  ImageBind `--no-deps` from git, nltk punkt, `ollama pull`), then every
  pip package grouped with a one-line reason. Added `fastapi`/`uvicorn`/
  `httpx`; `gradio>=5.0`.
- `.gitignore`: added `.gradio/`.

### README.md rewritten as the full guide (2026-08-29 cont.)

`README.md` is now a complete front-door: what it does (capability table),
how it works (ASCII pipeline + the 4 key design decisions), repo layout,
a copy-paste quick start (prereqs → venv → torch → ImageBind → corpus →
index → run), a table of every `run.ps1` target + the raw module commands,
corpus management (AMI builder flags, adding your own files, the manifest
schema), evaluation, the temporary/permanent demo-link options, hard
constraints, and a docs index. `SIH26-multimodal-rag-src.zip` rebuilt with
it (still ~140 KB).
