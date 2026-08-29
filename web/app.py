"""Gradio UI — the single chat-style interface (ARCHITECTURE.md §6).

Text box + image drag-drop + audio (upload AND microphone) + an expandable
citations panel that renders each source in place: the image itself for image
hits, an audio player cued to the cited segment for audio hits, the passage +
page for documents, and the cross-modal "related passage" link. This is the
PS's citation-transparency requirement ("citations expand to open the original
document, view full transcript segments, or inspect image metadata").

USE_MOCK_BACKEND (RAG_MOCK=1) serves a canned answer for UI-only work.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root on path

from rag import config  # noqa: E402

# Offline constraint (PRD.md): kill every startup network call gradio makes
# before gradio is imported anywhere. On a locked-down network these otherwise
# stall the server for minutes.
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("GRADIO_SERVER_NAME", "127.0.0.1")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

USE_MOCK_BACKEND = os.environ.get("RAG_MOCK", "0") == "1"


# --------------------------------------------------------------------------
# Backend
# --------------------------------------------------------------------------

def _mock_answer(question, image_path, audio_path, spoken):
    from rag.generation.answer import GroundedAnswer, Source

    srcs = [
        Source(1, "document", "hydro_review.pdf", "page 2",
               "Hydropower is emission-free but flow regulation is the main "
               "environmental impact; fish-stocking is the mitigation.",
               doc_id="d1"),
        Source(2, "image", "dashboard_1432.png", "captured 14:32",
               "Hydro Generation dashboard — fish-stocking budget 62% used.",
               doc_id="d1", image_timestamp="14:32"),
    ]
    ans = ("Hydropower generation is emission-free, but regulating river flow is "
           "the main environmental impact, mitigated by fish-stocking [1]. A "
           "dashboard captured at 14:32 shows 62% of that budget used [2].")
    return GroundedAnswer(answer=ans, sources=srcs, citations_used=srcs, model="mock")


def _real_answer(question, image_path, audio_path, spoken):
    from rag.generation.answer import generate_grounded_answer
    from rag.retrieval.pipeline import run_query_pipeline

    res = run_query_pipeline(
        text=question or None,
        image_path=image_path,
        audio_path=audio_path,
        audio_is_spoken_query=spoken,
        unload_after=False,
        offload_after=True,          # park ImageBind on CPU between queries
    )
    q_for_llm = res.transcribed_query or question or "(non-text query)"
    ga = generate_grounded_answer(q_for_llm, res.chunks, weak=getattr(res, "weak", False))
    ga._transcribed = res.transcribed_query
    return ga


def answer_query(question, image_path, audio_path, spoken):
    return (_mock_answer if USE_MOCK_BACKEND else _real_answer)(
        question, image_path, audio_path, spoken
    )


# --------------------------------------------------------------------------
# Turn a GroundedAnswer into plain dicts for the render callback
# --------------------------------------------------------------------------

def _linked_passage_text(doc_id: str) -> str:
    if not doc_id:
        return ""
    try:
        from rag.core import vectorstore

        return vectorstore.get_passage_text(vectorstore.get_collection(), doc_id)
    except Exception:  # noqa: BLE001
        return ""


def _sources_payload(ga) -> list[dict]:
    """One card per distinct source file (a dense slide/PDF can yield several
    chunks — merge them), cited if any of its chunks was cited."""
    cited_nums = {s.number for s in ga.citations_used}
    by_file: dict[str, dict] = {}
    for s in ga.sources:
        card = by_file.get(s.source_file)
        if card is None:
            card = by_file[s.source_file] = {
                "number": s.number,
                "numbers": [],
                "cited": False,
                "modality": s.modality,
                "source_file": s.source_file,
                "locator": s.locator,
                "texts": [],
                "source_path": s.source_path,
                "start_time": s.start_time,
                "end_time": s.end_time,
                "image_timestamp": s.image_timestamp,
                "linked_text": _linked_passage_text(getattr(s, "linked_passage", "") or ""),
            }
        card["numbers"].append(s.number)
        card["cited"] = card["cited"] or (s.number in cited_nums)
        if s.text and s.text not in card["texts"]:
            card["texts"].append(s.text)
        if card["start_time"] is None and s.start_time is not None:
            card["start_time"], card["end_time"] = s.start_time, s.end_time
        if not card["image_timestamp"] and s.image_timestamp:
            card["image_timestamp"] = s.image_timestamp

    out = []
    for card in by_file.values():
        card["text"] = "  …  ".join(card["texts"])
        out.append(card)
    out.sort(key=lambda c: (not c["cited"], min(c["numbers"])))
    return out


def _flags(ga) -> str:
    bits = []
    if getattr(ga, "_transcribed", None):
        bits.append(f"🎤 heard: *{ga._transcribed}*")
    if ga.cited_nothing:
        bits.append("⚠️ the answer cited **no** source — retrieved context may "
                    "not actually answer this")
    if ga.dropped_citation_numbers:
        bits.append(f"⚠️ dropped invented citation(s) {ga.dropped_citation_numbers}")
    return "  \n".join(bits)


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------

def build_demo():
    import gradio as gr

    with gr.Blocks(title="Offline Multimodal RAG (SIH #25231)",
                   analytics_enabled=False) as demo:
        gr.Markdown(
            "# Offline Multimodal RAG &nbsp;·&nbsp; SIH #25231 (NTRO)\n"
            "Ask in plain language. Attach an image, or an audio clip to match. "
            "Every answer is grounded — expand a citation to see the source."
            + ("\n\n**⚠️ MOCK BACKEND** (`RAG_MOCK=1`)" if USE_MOCK_BACKEND else "")
        )
        sources_state = gr.State([])

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(height=420, label="Conversation")
                flags = gr.Markdown()
                question = gr.Textbox(
                    placeholder='e.g. "the report that references the screenshot taken at 14:32"',
                    label="Question", lines=1,
                )
                with gr.Row():
                    image_in = gr.Image(type="filepath", height=120,
                                        label="Attach image (image → related docs)",
                                        sources=["upload", "clipboard"])
                    with gr.Column():
                        with gr.Tab("🎤 Ask aloud"):
                            mic_in = gr.Audio(sources=["microphone"], type="filepath",
                                              label="Spoken question — transcribed, then searched")
                        with gr.Tab("🔊 Match a clip"):
                            audio_up = gr.Audio(sources=["upload"], type="filepath",
                                                label="Audio content — matched + transcript-bridged")
                submit = gr.Button("Ask", variant="primary")

            with gr.Column(scale=2):
                gr.Markdown("### Sources")

                @gr.render(inputs=[sources_state])
                def render_sources(sources):
                    if not sources:
                        gr.Markdown("_Sources for the answer appear here — click to expand._")
                        return
                    for s in sources:
                        tag = "✅ cited" if s["cited"] else "not cited"
                        loc = f" · {s['locator']}" if s["locator"] else ""
                        nums = "".join(f"[{n}]" for n in s["numbers"])
                        title = f"{nums} {s['source_file']} · {s['modality']}{loc} · {tag}"
                        with gr.Accordion(title, open=s["cited"]):
                            if s["modality"] == "image" and s["source_path"]:
                                gr.Image(value=s["source_path"], show_label=False,
                                         height=300)
                            elif s["modality"] == "audio" and s["source_path"]:
                                gr.Audio(value=s["source_path"], show_label=False)
                                if s["start_time"] is not None:
                                    gr.Markdown(f"cited segment **{s['start_time']:.0f}s – "
                                                f"{s['end_time']:.0f}s**")
                            body = f"> {s['text']}" if s["text"] else ""
                            if s["image_timestamp"]:
                                body += f"\n\n**captured:** {s['image_timestamp']}"
                            if s["linked_text"]:
                                body += ("\n\n**related passage:** "
                                         + s["linked_text"][:220].strip() + "…")
                            if body:
                                gr.Markdown(body)

        io = dict(inputs=[question, image_in, mic_in, audio_up, chatbot],
                  outputs=[chatbot, sources_state, flags, question, mic_in, audio_up])
        submit.click(handle_submit, **io)
        question.submit(handle_submit, **io)

    return demo


def handle_submit(q, img, mic, aud, history):
    """mic = spoken query (ASR), aud = audio clip to match against the corpus."""
    history = list(history or [])
    spoken = mic is not None
    audio_path = mic or aud
    if not (q or img or audio_path):
        history.append({"role": "assistant",
                        "content": "Type a question or attach an image / audio clip first."})
        return history, [], "", "", None, None

    ga = answer_query(q, img, audio_path, spoken)
    user_msg = q or ("(spoken question)" if spoken else "(audio clip)" if aud else "(image)")
    history.append({"role": "user", "content": user_msg})
    history.append({"role": "assistant", "content": ga.answer})
    return history, _sources_payload(ga), _flags(ga), "", None, None


if __name__ == "__main__":
    demo = build_demo()
    print("\nStarting server — watch for the 'Running on local URL' line.\n")
    # RAG_SHARE=1  ->  also open a temporary public *.gradio.live tunnel
    # (72 h, only while this process runs) so a teammate can test remotely.
    _share = os.environ.get("RAG_SHARE") == "1"
    demo.launch(
        server_name="0.0.0.0" if _share else "127.0.0.1",
        server_port=None,          # scan 7860-7959 for a free port
        share=_share,
        inbrowser=not _share,
        show_error=True,
        allowed_paths=[str(config.RAW_DATA_DIR)],
    )
