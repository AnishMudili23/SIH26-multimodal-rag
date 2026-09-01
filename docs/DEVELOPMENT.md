# Development notes

Orientation for anyone working in this repo: the overview, what to read
first, the layout, and the constraints that must not be broken.

## Project overview

Offline multimodal RAG system for SIH problem statement #25231 (NTRO) —
ingests DOC/PDF/images/audio, indexes them in a unified ImageBind vector
space, and answers natural-language queries with grounded, cited responses
via a local LLM. Fully offline at inference time.

## Read these before doing anything else

- `docs/PRD.md` — what we're building and why; the actual PS requirements and
what "success" means.
- `docs/ARCHITECTURE.md` — the full pipeline design and the reasoning behind
every non-obvious decision (unified embedding space, chunk size, merge
strategy, citation contract). If a design choice looks arbitrary, it
isn't — check here before changing it.
- `docs/TECH_STACK.md` — exact model/library choices and VRAM budget.
- `docs/ROADMAP.md` — phased task list.
- `README.md` — the layout of the repo and how to run each entry point.

## Repo layout

```
rag/            the core library — import as `from rag... import ...`
  config.py         all tunables + paths (single source of truth)
  core/             embeddings.py (ImageBind), vectorstore.py (Chroma)
  ingest/           extraction, chunking, corpus.py (build/add), manifest.py
  retrieval/        pipeline.py (route→embed→search→RRF), lexical.py (BM25)
  generation/       answer.py (prompt → local Ollama → cited answer)
backend/server.py   FastAPI on 127.0.0.1:8077 (own process, own GIL)
desktop/            PySide6 app — thin HTTP client of the backend
web/app.py          Gradio UI (in-process fallback)
scripts/            evaluate.py (Recall@k), make_demo_assets.py
docs/               PRD, ARCHITECTURE, TECH_STACK, ROADMAP, SETUP
data/               corpus manifest + Chroma index (index is gitignored)
```

Entry points all go through `run.ps1` (`.\run.ps1`, `.\run.ps1 server`,
`.\run.ps1 eval`, …). `backend/`, `desktop/`, `web/` and `scripts/` put the
repo root on `sys.path` so `import rag...` resolves.



## Hard constraints — do not violate these

- **Fully offline at inference.** No calls to external APIs (no OpenAI,
no HuggingFace inference API, no cloud anything) once the system is
running. Model weights are downloaded/cached ahead of time; inference
itself must work with no network.
- **ImageBind's text tower has a 77-token hard limit** (CLIP-based) and
truncates silently rather than erroring. This is why chunks are 40-60
tokens — do not "simplify" chunking back to a larger size without
re-reading `ARCHITECTURE.md` §2 first.
- **GPU budget is finite.** See the VRAM table in `TECH_STACK.md`. Don't
assume ImageBind + Whisper + the LLM can all sit resident in VRAM
simultaneously on smaller GPUs — check before adding anything that loads
another model.

