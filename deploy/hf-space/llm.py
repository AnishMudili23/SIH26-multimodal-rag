"""Grounded generation for the hosted preview.

A small quantised Qwen2.5 runs on CPU via llama.cpp. Same contract as the
desktop app: numbered sources in, an answer where every factual sentence ends
with a bracketed citation, hallucinated numbers dropped, and a hard refusal
when retrieval was flagged off-corpus.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

REPO = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
FILE = "qwen2.5-1.5b-instruct-q4_k_m.gguf"

REFUSAL = "The corpus does not contain information on this."

SYSTEM = (
    "You answer strictly from the numbered Sources provided. Rules:\n"
    "1. Write 1-4 complete sentences. End every factual sentence with a "
    "bracketed citation like [1], or [1][3] for multiple. Never reply with "
    "only a citation marker.\n"
    "2. Use only citation numbers that appear in the Sources list.\n"
    "3. Answer only from the Sources, never from your own knowledge. If no "
    'Source addresses the question, reply exactly: "' + REFUSAL + '"'
)

_CITE = re.compile(r"\[(\d+)\]")
_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        from huggingface_hub import hf_hub_download
        from llama_cpp import Llama
        path = hf_hub_download(repo_id=REPO, filename=FILE)
        _llm = Llama(model_path=path, n_ctx=4096, n_threads=4, verbose=False)
    return _llm


@dataclass
class Source:
    n: int
    unit: object                       # pipeline.Unit


@dataclass
class Answer:
    text: str
    sources: list[Source]
    cited: list[int] = field(default_factory=list)
    refused: bool = False


def _prompt(question: str, units: list) -> tuple[str, list[Source]]:
    lines = ["Sources:"]
    srcs: list[Source] = []
    for i, u in enumerate(units, 1):
        loc = u.locator()
        label = f'{u.modality}, "{u.source_file}"' + (f", {loc}" if loc else "")
        lines.append(f"[{i}] ({label}): {u.text}")
        srcs.append(Source(i, u))
    lines += ["", f"Question: {question}", "Answer:"]
    return "\n".join(lines), srcs


def answer(question: str, units: list, weak: bool) -> Answer:
    if not units or weak:
        _, srcs = _prompt(question, units)
        return Answer(REFUSAL, srcs, [], refused=True)

    prompt, srcs = _prompt(question, units)
    llm = _get_llm()
    resp = llm.create_chat_completion(
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": prompt}],
        temperature=0.2, max_tokens=320,
    )
    text = resp["choices"][0]["message"]["content"].strip()

    valid = {s.n for s in srcs}
    seen, cited = set(), []
    for m in _CITE.finditer(text):
        k = int(m.group(1))
        if k in seen:
            continue
        seen.add(k)
        if k in valid:
            cited.append(k)
    # drop hallucinated markers from the visible text
    text = _CITE.sub(lambda m: m.group(0) if int(m.group(1)) in valid else "", text)
    return Answer(text, srcs, cited, refused=not cited and REFUSAL.lower() in text.lower())
