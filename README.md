# 🔍 Offline Multimodal RAG — SIH 2026 · PS 25231 (NTRO)

Ask one plain-language question and get a **grounded, cited answer** drawn from
**documents, images and audio recordings at once** — with every citation
resolving to the exact page, slide, or audio second it came from, and an
explicit refusal when the corpus can't answer. Everything runs **locally with
no network** once the models are cached: retrieval, generation, all of it.

![Inference](https://img.shields.io/badge/Inference-100%25%20Offline-2ea44f)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![Embeddings](https://img.shields.io/badge/Embeddings-ImageBind%201024--d-orange)
![Vector store](https://img.shields.io/badge/Vector%20store-Chroma-5636d3)
![LLM](https://img.shields.io/badge/LLM-Qwen2.5%203B%20local-black)
![API](https://img.shields.io/badge/API-FastAPI-009688)
![Desktop](https://img.shields.io/badge/Desktop-PySide6-41cd52)
![SIH 2026](https://img.shields.io/badge/SIH%202026-PS%2025231-ff6f00)

- **Problem statement:** design an offline multimodal RAG system that ingests,
  indexes, and queries DOC / PDF / images / voice recordings in one unified
  semantic space, with cross-format links (transcript segment ↔ paragraph ↔
  screenshot). Full text in [`docs/PRD.md`](docs/PRD.md).
- **Design & the reasoning behind every non-obvious choice:**
  [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).
- **Session-by-session build log:** [`PROGRESS.md`](PROGRESS.md).

## 🎥 Technical overview video

A walkthrough of the architecture and pipeline:

https://github.com/AnishMudili23/SIH26-multimodal-rag/raw/main/docs/technical-overview.mp4

(If the player doesn't load inline, [download the video](docs/technical-overview.mp4).)

---

## ✨ What it does

| Capability | How |
|---|---|
| Ingest 4 formats | `.docx`/`.doc` · `.pdf` (text layer + OCR fallback) · images (OCR + vision embed) · audio (Whisper transcript + timestamps) |
| One semantic index | Every chunk — text, image, audio-transcript — embedded by **ImageBind** into one 1024-d space, stored in one **Chroma** collection |
| Natural-language query | Type, speak (mic → Whisper), or drop a file/image; results span all modalities |
| Grounded answer | A local **Qwen2.5** writes the answer *strictly* from retrieved passages; every factual sentence ends with a `[n]` citation |
| Navigable citations | Each `[n]` → file + page / audio timestamp / slide window; the UI opens the source |
| Refuses when unsure | A relevance gate makes it reply *"the corpus does not contain information on this"* instead of guessing |
| Cross-format linking | A transcript segment, the slide on screen at that second, and the meeting's minutes are all reachable from one another (shared `doc_id`) |

---

## 🧠 How it works

```
INGESTION   raw files ─▶ extract / OCR / transcribe ─▶ chunk (≤60 tok)
                     ─▶ ImageBind embed ─▶ Chroma (one unified index)

QUERY       question (type / speak / upload) ─▶ route + embed
                     ─▶ search per modality ─▶ fuse: semantic + BM25 (RRF)
                     ─▶ rank + group by source ─▶ grounding gate

GENERATION  top passages ─▶ numbered-source prompt ─▶ Qwen2.5 (local)
                     ─▶ parse + verify [n] citations ─▶ answer + sources

UI          PySide6 desktop app  ◀── HTTP ──▶  FastAPI backend (127.0.0.1:8077)
            web/app.py (Gradio)   runs the pipeline in-process
```

Key decisions (details in `docs/ARCHITECTURE.md`):

- **One encoder, one index.** ImageBind gives a genuine joint space, so a text
  query really ranks an image or an audio moment against a paragraph — not in a
  separate silo. Cost: the CLIP text tower's **hard 77-token limit**, which is
  why chunks are 40–60 tokens (a correctness requirement, not a style choice).
- **Hybrid retrieval.** Per-modality vector search **+** a BM25 keyword pass,
  fused with Reciprocal Rank Fusion — keeps recall high on names, codes and
  noisy OCR where the CLIP text tower is weak.
- **Separate backend process.** `import torch` on a worker *thread* holds
  Python's GIL and froze the Qt UI, so the pipeline runs in its own process
  (`backend/server.py`); the desktop app is a thin HTTP client.
- **Grounding gate.** If the closest real match is far *and* no strong keyword
  hit, the query is off-corpus → generation returns a flat refusal instead of
  letting the LLM answer from its own weights.

---

## 📁 Repo layout

```
rag/                     core library — import as `from rag... import ...`
├── config.py            every tunable + path (single source of truth)
├── core/
│   ├── embeddings.py     ImageBind unified encoder (text / image / audio → 1024-d)
│   └── vectorstore.py    ChromaDB collection + the chunk schema
├── ingest/
│   ├── extraction.py     DOC/PDF/image/audio → text, OCR, transcript
│   ├── chunking.py       sentence-boundary ≤60-token chunks
│   ├── corpus.py         build / add-to the index   `python -m rag.ingest.corpus`
│   └── manifest.py       AMI Meeting Corpus downloader + manifest builder
├── retrieval/
│   ├── pipeline.py       route → embed → per-modality search → RRF → group → gate
│   └── lexical.py        BM25 keyword search (the lexical half of hybrid)
└── generation/
    └── answer.py         numbered-source prompt → local Ollama → parsed cited answer

backend/server.py         FastAPI on 127.0.0.1:8077 — the pipeline in its own process
desktop/                  PySide6 desktop app (main.py, main_window.py, controller.py, api.py)
web/app.py                Gradio UI — single-process fallback + the shareable demo
scripts/
├── evaluate.py           Recall@k / MRR@k against a qrels file
└── make_demo_assets.py   (legacy) the retired synthetic image/audio corpus generator
deploy/hf-space/          a CPU-only Gradio build for hosting on Hugging Face Spaces
docs/                     PRD · ARCHITECTURE · TECH_STACK · ROADMAP · SETUP
data/                     raw/ami/  +  corpus_manifest.csv  +  chroma/   (gitignored)
```

`backend/`, `desktop/`, `web/` and `scripts/` each put the repo root on
`sys.path`, so `import rag...` resolves however they're launched.

---

## 🚀 Quick start

Full machine-by-machine setup is in **[`docs/SETUP.md`](docs/SETUP.md)** and the
dependency list with every step is in **[`requirements.txt`](requirements.txt)**.
Short version (Windows + NVIDIA GPU):

```powershell
# 0. prerequisites: Python 3.11, an NVIDIA driver, Tesseract, Ollama, antiword
#    (Git for Windows bundles antiword).  See requirements.txt.

# 1. environment
python -m venv venv
.\venv\Scripts\python -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 `
    --index-url https://download.pytorch.org/whl/cu124
.\venv\Scripts\python -m pip install -r requirements.txt
.\venv\Scripts\python -m pip install --no-deps git+https://github.com/facebookresearch/pytorchvideo.git
.\venv\Scripts\python -m pip install --no-deps git+https://github.com/facebookresearch/ImageBind.git
.\venv\Scripts\python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
ollama pull qwen2.5:3b-instruct

# 2. activate (every new terminal) — or just use run.ps1, which does this for you
. .\env.ps1

# 3. get the demo corpus (skip if data/raw/ami already ships in your copy)
python -m rag.ingest.manifest              # downloads 2 AMI meetings (~250 MB)

# 4. build the index
python -m rag.ingest.corpus --src data\raw\ami --reset
python -m rag.ingest.corpus --stats        # expect ~800-900 chunks

# 5. run
.\run.ps1                                   # opens the desktop app
```

First run downloads the ImageBind checkpoint (~4.5 GB) and Whisper small
(~460 MB) once — after that it is fully offline.

### If C: is short on space

`env.ps1` reads an optional `env.local.ps1` (gitignored). Copy
`env.local.ps1.example` → `env.local.ps1` and set `$DataRoot` to a folder on
another drive — the venv, model caches and Ollama models move there. The
**corpus and Chroma index stay in the repo** (`$PipelineDataRoot`, left as
default) so they're visible next to the code.

---

## ⚡ Everything you can run

All entry points go through `run.ps1` (it activates the env, ensures Ollama is
up, then runs the target):

| Command | What it does |
|---|---|
| `.\run.ps1` | the **desktop app** — spawns the backend if it isn't already up |
| `.\run.ps1 server` | just the backend (FastAPI, `127.0.0.1:8077`) |
| `.\run.ps1 web` | the **Gradio** UI instead (browser) |
| `.\run.ps1 ask "..."` | ask one question from the terminal, no UI |
| `.\run.ps1 ami [--meetings ...]` | download AMI meetings + rebuild the manifest |
| `.\run.ps1 ingest` | rebuild the index from `data/raw/ami` (`--reset`) |
| `.\run.ps1 stats` | Chroma collection stats |
| `.\run.ps1 eval` | retrieval metrics (needs a `*_qrels.csv`) |
| `.\run.ps1 stop` | kill the backend + Ollama + anything holding the GPU |

Direct module invocations (what `run.ps1` calls under the hood):
`python -m desktop.main` · `python -m backend.server` ·
`python -m rag.ingest.corpus --src <dir> --reset` ·
`python -m rag.ingest.manifest` · `python -m rag.generation.answer "<q>"` ·
`python web/app.py` · `python scripts/evaluate.py`.

---

## 📚 Managing the corpus

**Rebuild / extend the AMI demo corpus:**

```powershell
python -m rag.ingest.manifest                                   # ES2002a, ES2002b (default)
python -m rag.ingest.manifest --meetings ES2002a ES2002b ES2003a
python -m rag.ingest.manifest --skip-download                   # rebuild manifest from existing files
```

It downloads per meeting into `data/raw/ami/<id>/`: the audio, a time-aligned
transcript sidecar (`*.segments.json` — used instead of running Whisper), the
projected slides (filename = on-screen seconds), and the meeting's own minutes
+ summary emails (`.doc` → text via `antiword`). Then it writes
`data/corpus_manifest.csv`.

**Add your own files** — drop `.docx`/`.pdf`/images/`.wav` anywhere under
`data/raw/`, then `.\run.ps1 ingest` (or use the desktop app's **+ Add files**
button, which ingests without a full rebuild).

**Manifest schema** (`data/corpus_manifest.csv`):
`doc_id, meeting_id, modality, source_file, text_preview, start_time, end_time`
— `doc_id` groups every chunk of one meeting so cross-modal links are queryable.

---

## 📊 Evaluation

```powershell
python scripts\evaluate.py --qrels data\corpus_manifest_qrels.csv
```

Reads a `query, doc_id, is_relevant` CSV, runs each query through the pipeline,
and reports **Recall@5 / MRR@10**. The prior MS MARCO corpus scored
Recall@5 = 0.96; an AMI dev-query set is a to-do (see `docs/ROADMAP.md` §4).

---

## 🔗 Sharing a demo

- **Temporary link** — run the full system on your GPU and tunnel it:
  ```powershell
  $env:RAG_SHARE = "1"; python web/app.py     # prints a *.gradio.live URL (~1 week)
  ```
- **Permanent link** — `deploy/hf-space/` is a CPU-only Gradio build for a free
  Hugging Face Space (MiniLM instead of ImageBind, Qwen2.5-1.5B instead of 3B;
  same retrieval + grounding logic). See `deploy/hf-space/README.md`.
- The **PySide desktop app can't be hosted** — it's a native window. Record it
  for the demo video; link the Gradio version.

---

## 🚧 Hard constraints — do not violate these

- **Offline at inference.** No external API calls once running. Weights are
  downloaded/cached ahead of time; inference works with the network cable out.
- **ImageBind text tower = hard 77-token limit** (silent truncation). Chunks
  stay 40–60 tokens. Don't enlarge without re-reading `docs/ARCHITECTURE.md` §2.
- **Finite VRAM** (table in `docs/TECH_STACK.md`). ImageBind + Whisper + the
  LLM don't all fit resident on a 6 GB card — the backend offloads ImageBind
  to system RAM between queries.

---

## 📖 Documentation

| File | What's in it |
|---|---|
| [`docs/PRD.md`](docs/PRD.md) | the problem statement, goals, success criteria, the judge angle |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | the full pipeline design + the reasoning behind every non-obvious choice |
| [`docs/TECH_STACK.md`](docs/TECH_STACK.md) | model/library choices, the VRAM budget |
| [`docs/ROADMAP.md`](docs/ROADMAP.md) | phased task list with checkboxes |
| [`docs/SETUP.md`](docs/SETUP.md) | install walkthrough for a fresh machine |
| [`PROGRESS.md`](PROGRESS.md) | append-only session log — read the last entry to see where work stopped |

---

## 📝 Demo corpus — attribution

The demo corpus is the **AMI Meeting Corpus** (meetings ES2002a, ES2002b),
<http://groups.inf.ed.ac.uk/ami/corpus/>, licensed **CC BY 4.0**.
J. Carletta et al., *"The AMI Meeting Corpus"*, 2005.

---

<p align="center">Built for <b>Smart India Hackathon 2026</b> · Problem Statement 25231 (NTRO)</p>
