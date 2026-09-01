# SETUP — getting this running on a new machine

Offline multimodal RAG (SIH #25231). Windows + NVIDIA GPU assumed (the target
demo machine is a 6 GB RTX 3050 laptop). Linux/WSL works too and is smoother
for the ML stack — see the note at the end.

Read `README.md` then `docs/DEVELOPMENT.md` for the project overview and repo layout,
then this.

---

## 0. What's in this zip / what's not

**Included:** all source (`rag/`, `backend/`, `desktop/`, `web/`, `scripts/`),
the design docs (`docs/`, `README.md`),
`requirements.txt`, `env.ps1` / `run.ps1`, and the **full demo corpus** —
25 MS MARCO passages + 12 generated images + 6 briefing-call WAVs
(`data/raw/` + `data/corpus_manifest*.csv`).

**NOT included** (you build/download these locally):
- `venv/` — the Python environment (machine-specific, ~6 GB)
- `models/`, `whisper/`, `ollama/`, `hf/` — model weights (~9 GB total; the
  ImageBind checkpoint auto-downloads on first run)
- `data/raw/ami/` + `data/corpus_manifest.csv` — the demo corpus, fetched by
  `python -m rag.ingest.manifest` (~135 MB, AMI Meeting Corpus, CC BY 4.0)
- `data/chroma/` — the vector index (rebuilt by one `ingest` command)
- `env.local.ps1` — the original author's machine paths (copy the `.example`)

The corpus + index live in the repo's `data/` (with the code). Only the heavy
caches (venv, models, whisper, ollama) go on another drive — set `$DataRoot`
in `env.local.ps1` if C: is full; leave `$PipelineDataRoot` alone.

Disk needed after full install: **~12 GB** (mostly the venv + model weights).

---

## 1. Prerequisites (install these first)

| Tool | How | Notes |
|---|---|---|
| **Python 3.11** | python.org | 3.12/3.13 may fight `pytorchvideo`; 3.11 is tested |
| **NVIDIA driver** | GeForce Experience / nvidia.com | recent enough for CUDA 12.x |
| **Tesseract OCR** | `winget install UB-Mannheim.TesseractOCR` | for scanned PDFs / screenshot text |
| **Ollama** | `winget install Ollama.Ollama` | the local LLM runtime |
| **ffmpeg** *(optional)* | `winget install Gyan.FFmpeg` | only needed for non-WAV audio (mp3/m4a) |

### Windows gotcha — Smart App Control
If `pip install` succeeds but `import` fails with
`DLL load failed ... An Application Control policy has blocked this file`,
Smart App Control is on. Turn it **off**: Windows Security → App & browser
control → Smart App Control → Off (one-way; needs a Windows reset to re-enable).
Or use WSL2 instead (see bottom).

### PowerShell execution policy
If `. .\env.ps1` says "running scripts is disabled":
```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned   # answer Y
```

---

## 2. Python environment

From the repo folder:

```powershell
python -m venv venv

# torch FIRST, matched to your CUDA (cu124 build works on driver 12.x+):
.\venv\Scripts\python -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 `
    --index-url https://download.pytorch.org/whl/cu124

# the rest:
.\venv\Scripts\python -m pip install -r requirements.txt

# ImageBind — NOT on PyPI, install without its deps (they drag in cartopy /
# decord which fail to build on Windows and aren't needed):
.\venv\Scripts\python -m pip install --no-deps git+https://github.com/facebookresearch/pytorchvideo.git
.\venv\Scripts\python -m pip install --no-deps git+https://github.com/facebookresearch/ImageBind.git

# NLTK sentence data:
.\venv\Scripts\python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

Verify CUDA:
```powershell
.\venv\Scripts\python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### If C: is too full for the venv + models
Put them on another drive. Create `env.local.ps1` (copy from
`env.local.ps1.example`) and set `$DataRoot` to a folder on that drive, then
create the venv there: `python -m venv D:\somewhere\venv`.

---

## 3. The LLM

```powershell
ollama pull qwen2.5:3b-instruct        # ~1.9 GB
```
If you moved things to another drive, also:
`setx OLLAMA_MODELS "D:\somewhere\ollama"` then restart Ollama.

---

## 4. Activate + smoke test

```powershell
. .\env.ps1                              # every new terminal needs this
python -m rag.ingest.corpus --smoke      # synthesises 1 file/format, ingests, queries back
```
On first run this downloads the ImageBind checkpoint (~4.5 GB, once). Expect
"Smoke test PASSED".

---

## 5. Build the demo corpus (AMI Meeting Corpus)

The corpus is the **AMI Meeting Corpus** (CC BY 4.0). It is *not* in the zip —
one command downloads it (~250 MB for the two default meetings):

```powershell
python -m rag.ingest.manifest                          # ES2002a + ES2002b
#   downloads audio + real transcripts + slides + the meetings' own
#   minutes/report into data\raw\ami\, and writes data\corpus_manifest.csv
python -m rag.ingest.corpus --src data\raw\ami --reset # build the Chroma index
python -m rag.ingest.corpus --stats
python scripts\evaluate.py                             # once AMI qrels exist
```

`.doc` minutes are converted with `antiword` (bundled with Git for Windows;
`apt install antiword` on Linux). More/other meetings:
`python -m rag.ingest.manifest --meetings ES2002a ES2002b ES2003a`.

The earlier MS MARCO + synthetic-asset corpus (`scripts/make_demo_assets.py`,
`data/raw/{msmarco,images,audio}`) is superseded.

---

## 6. Run it

```powershell
.\run.ps1                                # desktop app (PySide6) — starts the backend
.\run.ps1 web                            # OR the Gradio web UI at http://127.0.0.1:7860
.\run.ps1 ask "how is hydropower bad for the environment?"
.\run.ps1 eval
```

The desktop app spawns `backend/server.py` (FastAPI on :8077) as a separate
process and talks to it over HTTP — the window opens instantly and stays
responsive while the backend warms up (~45 s the first time). After that every
query is ~1 s retrieval + a few seconds of generation.

If a query crashes or hangs, the GPU probably has leftover processes:
```powershell
.\run.ps1 stop                        # kills python / llama-server / ollama, shows GPU
nvidia-smi                            # confirm it's clear, then relaunch
```

---

## 7. Where the work stands

See `docs/ROADMAP.md` (checkboxes). Short
version: **text + image + audio retrieval and grounded cited generation all
work offline, end-to-end.** The desktop app (PySide6) + separate FastAPI
backend are built and stable. A grounding gate rejects out-of-corpus questions
instead of hallucinating. Open items are roadmap perf tuning (faster-whisper,
cross-encoder rerank, a smaller/faster model).

---

## WSL2 / Linux alternative

Smoother for this stack (no Smart App Control, cleaner CUDA). In Ubuntu:
```bash
sudo apt install -y python3-venv python3-pip tesseract-ocr ffmpeg
python3 -m venv venv && source venv/bin/activate
pip install torch torchvision torchaudio            # default index has CUDA on Linux
pip install -r requirements.txt
pip install --no-deps git+https://github.com/facebookresearch/pytorchvideo.git
pip install --no-deps git+https://github.com/facebookresearch/ImageBind.git
curl -fsSL https://ollama.com/install.sh | sh && ollama pull qwen2.5:3b-instruct
```
Then use `env.sh`-equivalents or just export the same env vars `env.ps1` sets.
(No `.sh` version is written yet — port `env.ps1` if you go this route.)
