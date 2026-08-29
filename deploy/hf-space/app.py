"""Offline Multimodal RAG — hosted preview (SIH 2026, PS 25231 / NTRO).

The full system runs air-gapped on a GPU workstation: ImageBind for a unified
text/image/audio embedding space and a local Qwen2.5 via Ollama. This public
preview keeps the same pipeline shape — hybrid retrieval, grounded + refusing
generation, navigable citations — but swaps in CPU-friendly models so it runs
on a free Space. Code + demo video: see the About tab.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import gradio as gr

from pipeline import Retriever
from llm import answer as generate

RET: Retriever | None = None

EXAMPLES = [
    "What was decided about the target group and the batteries?",
    "What did the marketing expert say about remote controls?",
    "What was on the closing slide of the kick-off meeting?",
    "Who was present at the kick-off meeting and when was it held?",
    "How much did the final remote control cost?",
    "What is the capital of France?",          # grounding — should refuse
]

REPO_URL = "https://github.com/USER/REPO"      # <-- set before submitting
VIDEO_URL = "https://youtu.be/XXXXXXXX"        # <-- set before submitting


def _boot():
    global RET
    if RET is None:
        RET = Retriever()
    return RET


def _audio_clip(path: str, start: float, end: float) -> str | None:
    try:
        import soundfile as sf
        data, sr = sf.read(path)
        a, b = int(max(0, start - 0.3) * sr), int((end + 0.3) * sr)
        out = Path(tempfile.mkdtemp()) / "clip.wav"
        sf.write(out, data[a:b], sr)
        return str(out)
    except Exception:
        return path if Path(path).exists() else None


def ask(question: str):
    question = (question or "").strip()
    if not question:
        return "Ask a question about the meetings.", "", []
    ret = _boot()
    units, weak, sim, lex = ret.search(question)
    ga = generate(question, units, weak)

    if ga.refused:
        conf = "LOW &mdash; not in the corpus"
    else:
        pct = max(10, min(95, round(100 * min(1.0, sim * 1.6))))
        conf = f"{pct}%"

    meta = (f"**Confidence:** {conf} &nbsp;&nbsp; "
            f"`retrieved {len(units)} chunks` &nbsp; `best sim {sim:.2f}` &nbsp; "
            f"`bm25 {lex:.1f}` &nbsp; `model qwen2.5-1.5b (cpu)`")

    cited = set(ga.cited)
    cards = []
    for s in ga.sources:
        u = s.unit
        mark = "cited" if s.n in cited else "not cited"
        head = f"### [{s.n}] {u.source_file}  \n`{u.modality}`"
        if u.locator():
            head += f" &middot; `{u.locator()}`"
        head += f" &middot; _{mark}_ &middot; meeting `{u.doc_id}`"
        body = [head, "", f"> {u.text}"]
        img = audio = None
        if u.modality == "image" and Path(u.source_path).exists():
            img = u.source_path
        if u.modality == "audio" and Path(u.source_path).exists():
            audio = _audio_clip(u.source_path, u.start, u.end)
        cards.append((("\n".join(body)), img, audio))

    return ga.text, meta, cards


CSS = """
.gradio-container{max-width:1080px!important}
#answer textarea{font-size:1.05rem;line-height:1.6}
.srccard{border:1px solid var(--border-color-primary);border-radius:8px;padding:.6rem .8rem;margin-bottom:.5rem}
footer{display:none!important}
"""

with gr.Blocks(title="PS 25231 — Multimodal RAG (preview)", css=CSS,
               theme=gr.themes.Soft(primary_hue="amber", neutral_hue="stone")) as demo:
    gr.Markdown(
        "# Offline Multimodal RAG &mdash; hosted preview\n"
        "**SIH 2026 &middot; PS 25231 &middot; NTRO.** One question &rarr; a cited answer drawn "
        "from meeting **audio, slides and minutes** at once. Every citation resolves to a "
        "file + timestamp; the system **refuses** when the corpus can't answer.\n\n"
        "> Corpus: 2 real meetings from the AMI Meeting Corpus (CC BY 4.0). "
        "This preview uses CPU models (MiniLM + Qwen2.5-1.5B); the full system uses "
        "ImageBind + Qwen2.5-3B on a GPU, fully offline. First answer after a cold "
        "start takes ~1 min while models load."
    )

    with gr.Tab("Ask"):
        with gr.Row():
            q = gr.Textbox(label="Question", scale=5, autofocus=True,
                           placeholder="e.g. what was decided about the batteries?")
            btn = gr.Button("Ask", variant="primary", scale=1)
        gr.Examples(EXAMPLES, inputs=q, label="Try one")
        ans = gr.Textbox(label="Grounded answer", elem_id="answer", lines=5,
                         interactive=False, show_copy_button=True)
        info = gr.Markdown()
        gr.Markdown("### Retrieved sources")
        with gr.Column() as src_col:
            src_md = [gr.Markdown(visible=False) for _ in range(18)]
            src_img = [gr.Image(visible=False, height=160, show_label=False) for _ in range(18)]
            src_aud = [gr.Audio(visible=False, show_label=False) for _ in range(18)]

    with gr.Tab("About"):
        gr.Markdown(
            f"""
**What this is.** A retrieval-augmented generation system for mixed archives —
documents, images and recordings in one semantic index — that answers in
natural language with a citation for every claim, and runs with no network.

**This preview vs. the full system**

| | Preview (here) | Full system |
|---|---|---|
| Embedding | MiniLM (text only, CPU) | ImageBind unified text/image/audio (GPU) |
| LLM | Qwen2.5-1.5B, llama.cpp, CPU | Qwen2.5-3B via Ollama |
| Slides | matched by OCR text | OCR text **+** vision embedding |
| Runs | free CPU Space | air-gapped GPU workstation |

Retrieval shape is identical: per-modality search, reciprocal rank fusion with
a BM25 keyword pass, `doc_id` grouping for cross-modal linking, and a grounding
gate that refuses off-corpus questions.

**Code:** {REPO_URL}  &nbsp;•&nbsp;  **Demo video:** {VIDEO_URL}

Demo corpus: AMI Meeting Corpus, meetings ES2002a/ES2002b — CC BY 4.0.
J. Carletta et al., "The AMI Meeting Corpus", 2005.
"""
        )

    def route(question):
        text, meta, cards = ask(question)
        upd = [gr.update(value=text), gr.update(value=meta)]
        for i in range(18):
            if i < len(cards):
                md, img, aud = cards[i]
                upd.append(gr.update(value=md, visible=True))
                upd.append(gr.update(value=img, visible=img is not None))
                upd.append(gr.update(value=aud, visible=aud is not None))
            else:
                upd += [gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)]
        return upd

    outs = [ans, info]
    for i in range(18):
        outs += [src_md[i], src_img[i], src_aud[i]]
    btn.click(route, q, outs)
    q.submit(route, q, outs)

if __name__ == "__main__":
    demo.queue(max_size=12).launch()
