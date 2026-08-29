"""Shared configuration for the offline multimodal RAG system.

Single source of truth for paths, model names, and the tuning constants that
ARCHITECTURE.md pins down (chunk token bounds, audio window length, Chroma
collection name). Imported by both the ingestion pipeline and rag/retrieval/pipeline.py
so the embedding space and schema stay identical on both sides.
"""

from __future__ import annotations

import os
from pathlib import Path

# --- Paths -----------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]   # this file is rag/config.py

# Where raw source files to ingest live (DOC/PDF/images/audio), one flat dir
# or nested — rag/ingest/corpus.py walks it recursively.
RAW_DATA_DIR = Path(os.environ.get("RAG_RAW_DATA_DIR", REPO_ROOT / "data" / "raw"))

# Persistent Chroma store (offline, on-disk).
CHROMA_DIR = Path(os.environ.get("RAG_CHROMA_DIR", REPO_ROOT / "data" / "chroma"))

# ImageBind checkpoint cache (the ~4.5 GB .pth). Kept out of the repo — on this
# machine C: is nearly full, so env.ps1 points this at D:.
MODEL_CACHE_DIR = Path(os.environ.get("RAG_MODEL_CACHE", REPO_ROOT / ".model_cache"))

# Whisper model files (~460 MB for "small").
WHISPER_CACHE_DIR = Path(
    os.environ.get("RAG_WHISPER_CACHE", MODEL_CACHE_DIR / "whisper")
)

IMAGEBIND_CKPT_URL = "https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth"

CORPUS_MANIFEST_CSV = Path(
    os.environ.get("RAG_MANIFEST", REPO_ROOT / "data" / "corpus_manifest.csv")
)

# --- AMI Meeting Corpus (rag/ingest/manifest.py) ---------------------------
# The demo corpus source (CC BY 4.0). Downloaded meeting files land under
# RAW_DATA_DIR/ami/<meeting_id>/ ; big source archives cache under _cache/.
AMI_MEETINGS = [
    m.strip() for m in
    os.environ.get("RAG_AMI_MEETINGS", "ES2002a,ES2002b").split(",") if m.strip()
]
AMI_MIRROR = "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus"
AMI_ANNOTATIONS_URL = (
    "https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip"
)
AMI_AUDIO_STREAM = "Mix-Headset"          # mixed 4-headset track, one wav/meeting
AMI_LICENSE = "CC BY 4.0"

# --- Chroma --------------------------------------------------------------------

# One unified collection for every modality (ARCHITECTURE.md §1/§2).
# Overridable so a throwaway test corpus doesn't disturb the demo collection.
COLLECTION_NAME = os.environ.get("RAG_COLLECTION", "multimodal_rag")

# ImageBind embeddings are cosine-compared.
CHROMA_DISTANCE = "cosine"

# --- Chunking (ARCHITECTURE.md §2) --------------------------------------------

# ImageBind's text tower is CLIP-based: 77-token context including SOT/EOT,
# so ~75 usable. It truncates silently. Chunks are kept well under that.
CLIP_CONTEXT_LENGTH = 77

CHUNK_TARGET_TOKENS = 55   # accumulate sentences until we reach ~this
CHUNK_MAX_TOKENS = 70      # anything above this is hard re-split
CHUNK_MIN_TOKENS = 8       # below this, fold into a neighbour rather than emit

# --- Audio (ARCHITECTURE.md §1/§2) ------------------------------------------

# ImageBind's audio tower expects fixed ~2s windows (mel-spectrogram input).
# Longer recordings are sliced into these for embedding; Whisper transcribes
# the whole file separately for citation display.
AUDIO_WINDOW_SECONDS = 2.0
AUDIO_WINDOW_HOP_SECONDS = 2.0   # no overlap between embedding windows
AUDIO_SAMPLE_RATE = 16000

# --- Models -----------------------------------------------------------------

WHISPER_MODEL = os.environ.get("RAG_WHISPER_MODEL", "small")


def _find_tesseract() -> str | None:
    """Locate the Tesseract binary — env var, PATH, or the standard Windows
    install location (winget/UB-Mannheim doesn't always update PATH)."""
    env = os.environ.get("RAG_TESSERACT_CMD")
    if env:
        return env
    from shutil import which

    if which("tesseract"):
        return "tesseract"
    for cand in (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ):
        if Path(cand).exists():
            return cand
    return None


TESSERACT_CMD = _find_tesseract()

# Local RAG backend (backend/server.py) — the desktop UI is an HTTP client so
# the heavy pipeline never shares a GIL with the Qt event loop. Localhost only.
BACKEND_HOST = os.environ.get("RAG_BACKEND_HOST", "127.0.0.1")
BACKEND_PORT = int(os.environ.get("RAG_BACKEND_PORT", "8077"))
BACKEND_URL = f"http://{BACKEND_HOST}:{BACKEND_PORT}"

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("RAG_OLLAMA_MODEL", "qwen2.5:3b-instruct")
# How long Ollama keeps the LLM resident in VRAM after a response. ImageBind is
# parked on the CPU during generation, so keeping Qwen resident across a normal
# read-then-ask gap is free (a fresh load costs ~2-3 s). It briefly co-resides
# with ImageBind on the next query's embed — fine on 6 GB with fp16 ImageBind.
OLLAMA_KEEP_ALIVE = os.environ.get("RAG_OLLAMA_KEEP_ALIVE", "5m")

# Device for ImageBind / Whisper. "cuda" if available else "cpu" — resolved
# lazily in embeddings.py so importing config never touches torch.
DEVICE = os.environ.get("RAG_DEVICE", "auto")

# --- Retrieval ---------------------------------------------------------------

RRF_K = 60                 # ARCHITECTURE.md §4
DEFAULT_TOP_K = 8          # chunks handed to generation
PER_MODALITY_SEARCH_K = 20 # pulled from Chroma per input before RRF

# Hybrid retrieval: a BM25 lexical index over chunk text, fused into the RRF
# alongside the vector search. Fixes exact-term queries and noisy-OCR docs
# where ImageBind's CLIP text tower is weak. ~1 ms per query, no GPU.
USE_BM25 = os.environ.get("RAG_USE_BM25", "1") == "1"
BM25_SEARCH_K = 20
# A lone common term ("capital", "report") can score a few points against an
# unrelated chunk. Drop BM25 hits below this raw Okapi score so an off-topic
# query doesn't get handed a spurious "match" to answer around.
BM25_MIN_SCORE = float(os.environ.get("RAG_BM25_MIN_SCORE", "3.5"))
# A BM25 hit at or above this raw score is a genuine rare-term keyword match
# (e.g. "SQL", "DBMS") and overrides the grounding gate below even when the
# vector search is loose — that's the OCR/keyword case hybrid search is for.
BM25_STRONG_SCORE = float(os.environ.get("RAG_BM25_STRONG_SCORE", "8.0"))

# Grounding gate: if the best *vector* (cosine) distance across a query's hits
# is worse than this, the corpus probably doesn't cover the question. We still
# return the chunks (for transparency) but tell the LLM they're a weak match
# and it must not answer from its own knowledge. Good in-domain text queries
# land ~0.2-0.45 in the ImageBind text sub-space; unrelated ones sit ~0.55+.
RETRIEVAL_MAX_DISTANCE = float(os.environ.get("RAG_RETRIEVAL_MAX_DISTANCE", "0.40"))

# Audio: embed the fixed 2 s windows via ImageBind's audio tower. Off by
# default — audio<->text alignment is weak, so the Whisper transcript segments
# (always indexed) do the real retrieval work, and skipping the windows makes
# audio ingestion ~30 % faster and the index smaller. Turn on only if you
# need pure "sounds like this clip" audio-to-audio matching.
AUDIO_EMBED_WINDOWS = os.environ.get("RAG_AUDIO_WINDOWS", "0") == "1"


def ensure_dirs() -> None:
    """Create the on-disk directories the pipeline writes to."""
    for d in (
        RAW_DATA_DIR,
        CHROMA_DIR,
        MODEL_CACHE_DIR,
        WHISPER_CACHE_DIR,
        CORPUS_MANIFEST_CSV.parent,
    ):
        d.mkdir(parents=True, exist_ok=True)
