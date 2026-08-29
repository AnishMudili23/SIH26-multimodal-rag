"""ImageBind embedding wrapper — the single joint text/image/audio space.

ARCHITECTURE.md §1: everything (documents, images, audio windows, and every
query) is embedded through ImageBind so it all lands in one comparable vector
space and one Chroma collection. This module is imported by both the ingestion
pipeline (rag/ingest/corpus.py) and rag/retrieval/pipeline.py's embed_query/preprocessing.

The model is loaded lazily and cached process-wide. On <=8 GB GPUs ImageBind
and the LLM should not both be resident (TECH_STACK.md VRAM budget) — call
`unload()` after an ingestion run or after embedding a query batch, before
handing off to Ollama.
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from rag import config

logger = logging.getLogger(__name__)

EMBED_DIM = 1024  # imagebind_huge output width

_model = None          # cached imagebind_model
_device: str | None = None
_tokenizer = None      # cached CLIP SimpleTokenizer for token counting


# --------------------------------------------------------------------------
# Device / model lifecycle
# --------------------------------------------------------------------------

def resolve_device() -> str:
    global _device
    if _device is not None:
        return _device
    if config.DEVICE != "auto":
        _device = config.DEVICE
        return _device
    try:
        import torch
        _device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        _device = "cpu"
    return _device


def _load_model():
    """Load imagebind_huge once and cache it. If it's cached but parked on the
    CPU (see offload_to_cpu), move it back to the GPU — a couple of seconds vs.
    ~40 s to reload the 4.5 GB checkpoint from disk."""
    global _model
    if _model is not None:
        import torch

        device = resolve_device()
        if device == "cuda" and next(_model.parameters()).device.type != "cuda":
            logger.info("Moving cached ImageBind CPU->GPU ...")
            _model.to("cuda")
        return _model

    try:
        import torch
        from imagebind.models import imagebind_model
    except ImportError as e:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "ImageBind is not installed. Install with:\n"
            "  pip install git+https://github.com/facebookresearch/ImageBind.git\n"
            f"(underlying import error: {e})"
        ) from e

    config.MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    device = resolve_device()

    # imagebind_huge(pretrained=True) hardcodes a download to
    # .checkpoints/imagebind_huge.pth relative to cwd. Instead we build the
    # bare model and load a checkpoint from MODEL_CACHE_DIR (env.ps1 -> D:),
    # downloading it there once if absent.
    ckpt = config.MODEL_CACHE_DIR / "imagebind_huge.pth"
    if not ckpt.exists():
        logger.info("Downloading ImageBind checkpoint -> %s (~4.5 GB, one time)", ckpt)
        torch.hub.download_url_to_file(config.IMAGEBIND_CKPT_URL, str(ckpt), progress=True)

    logger.info("Loading ImageBind (imagebind_huge) on %s ...", device)
    model = imagebind_model.imagebind_huge(pretrained=False)
    state = torch.load(str(ckpt), map_location="cpu")
    model.load_state_dict(state)
    model.eval().to(device)
    if device == "cuda":
        model.half()  # ~5 GB fp16, per TECH_STACK.md
    _model = model
    return _model


def unload() -> None:
    """Free the ImageBind model from (V)RAM entirely. Safe to call when not
    loaded. Use after an ingestion run; the next use reloads from disk (~40 s)."""
    global _model
    if _model is None:
        return
    _model = None
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    logger.info("ImageBind unloaded")


def offload_to_cpu() -> None:
    """Park the model in system RAM and free the GPU (for Ollama's LLM), but
    keep the weights so the next query only pays a ~2 s CPU->GPU move, not a
    ~40 s disk reload. Used by the long-running Gradio app between queries."""
    global _model
    if _model is None:
        return
    try:
        import torch

        if next(_model.parameters()).device.type == "cuda":
            _model.to("cpu")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("ImageBind offloaded GPU->CPU (weights kept)")
    except ImportError:
        pass


# --------------------------------------------------------------------------
# Token counting (ARCHITECTURE.md §2 — 77-token CLIP limit drives chunking)
# --------------------------------------------------------------------------

def _get_tokenizer():
    global _tokenizer
    if _tokenizer is not None:
        return _tokenizer
    try:
        from imagebind.models.multimodal_preprocessors import SimpleTokenizer
        import imagebind

        bpe = Path(imagebind.__file__).parent / "bpe" / "bpe_simple_vocab_16e6.txt.gz"
        if bpe.exists():
            _tokenizer = SimpleTokenizer(bpe_path=str(bpe))
    except Exception as e:  # noqa: BLE001 - any failure -> heuristic fallback
        logger.debug("CLIP tokenizer unavailable (%s); using heuristic counter", e)
        _tokenizer = None
    return _tokenizer


def count_tokens(text: str) -> int:
    """Approximate the CLIP BPE token count for `text`.

    Uses ImageBind's own SimpleTokenizer when available (exact); otherwise a
    deliberately slightly-pessimistic word heuristic so the chunker errs on the
    side of shorter chunks rather than silent truncation.
    """
    text = text.strip()
    if not text:
        return 0
    tok = _get_tokenizer()
    if tok is not None:
        try:
            # SimpleTokenizer.encode -> list[int] of BPE ids (no SOT/EOT)
            return len(tok.encode(text)) + 2  # + SOT/EOT
        except Exception:  # noqa: BLE001
            pass
    # Heuristic fallback (ImageBind not installed): CLIP BPE averages ~1.3
    # tokens/word on clean prose but spikes on digits, hyphens, and
    # punctuation. Bias pessimistic so the chunker splits shorter rather than
    # risking silent truncation.
    words = text.split()
    punct = sum(text.count(c) for c in ",.;:!?()[]\"'/-")
    digits = sum(c.isdigit() for c in text)
    return int(round(len(words) * 1.5 + punct * 0.5 + digits * 0.3)) + 2


def fits_text_tower(text: str) -> bool:
    return count_tokens(text) <= config.CLIP_CONTEXT_LENGTH


# --------------------------------------------------------------------------
# Embedding calls
# --------------------------------------------------------------------------

def _forward(inputs: dict) -> dict:
    import torch

    model = _load_model()
    device = resolve_device()

    # On CUDA the model is fp16 (model.half()), but ImageBind's transforms
    # return fp32 tensors -> "Input type ... and weight type ... should be the
    # same". Cast only the floating-point inputs (token-id tensors stay Long).
    param_dtype = next(model.parameters()).dtype
    cast: dict = {}
    for k, v in inputs.items():
        if torch.is_tensor(v):
            v = v.to(device)
            if v.is_floating_point():
                v = v.to(param_dtype)
        cast[k] = v

    with torch.no_grad():
        out = model(cast)
    return {k: v.float().cpu().numpy() for k, v in out.items()}


def _normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def embed_texts(texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
    """Embed a list of short strings. Returns (N, 1024) L2-normalized array."""
    from imagebind import data
    from imagebind.models.imagebind_model import ModalityType

    texts = [t if t.strip() else " " for t in texts]
    device = resolve_device()
    chunks: list[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        batch = list(texts[i : i + batch_size])
        inputs = {ModalityType.TEXT: data.load_and_transform_text(batch, device)}
        chunks.append(_forward(inputs)[ModalityType.TEXT])
    return _normalize(np.vstack(chunks)) if chunks else np.zeros((0, EMBED_DIM))


def embed_images(image_paths: Sequence[str | Path], batch_size: int = 16) -> np.ndarray:
    """Embed image files via ImageBind's vision tower. Returns (N, 1024)."""
    from imagebind import data
    from imagebind.models.imagebind_model import ModalityType

    paths = [str(p) for p in image_paths]
    device = resolve_device()
    chunks: list[np.ndarray] = []
    for i in range(0, len(paths), batch_size):
        batch = paths[i : i + batch_size]
        inputs = {
            ModalityType.VISION: data.load_and_transform_vision_data(batch, device)
        }
        chunks.append(_forward(inputs)[ModalityType.VISION])
    return _normalize(np.vstack(chunks)) if chunks else np.zeros((0, EMBED_DIM))


def embed_audio_windows(
    window_paths: Sequence[str | Path], batch_size: int = 16
) -> np.ndarray:
    """Embed pre-sliced ~2s audio window files (one clip each).

    rag/ingest/corpus.py / rag/retrieval/pipeline.py slice longer recordings into fixed windows on
    disk first (ARCHITECTURE.md §2) so each embedding maps to an exact
    start/end time for citation display.
    """
    from imagebind import data
    from imagebind.models.imagebind_model import ModalityType

    paths = [str(p) for p in window_paths]
    device = resolve_device()
    chunks: list[np.ndarray] = []
    for i in range(0, len(paths), batch_size):
        batch = paths[i : i + batch_size]
        inputs = {
            ModalityType.AUDIO: data.load_and_transform_audio_data(
                batch,
                device,
                clip_duration=config.AUDIO_WINDOW_SECONDS,
                clips_per_video=1,
                sample_rate=config.AUDIO_SAMPLE_RATE,
            )
        }
        chunks.append(_forward(inputs)[ModalityType.AUDIO])
    return _normalize(np.vstack(chunks)) if chunks else np.zeros((0, EMBED_DIM))
