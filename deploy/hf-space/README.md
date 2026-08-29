---
title: Multimodal RAG (PS 25231)
emoji: 🗂️
colorFrom: yellow
colorTo: gray
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
license: cc-by-4.0
short_description: Offline multimodal RAG over meeting audio, slides and minutes — cited, refusing, air-gapped.
---

# Offline Multimodal RAG — hosted preview

**SIH 2026 · Problem Statement 25231 · NTRO.**

Ask one plain-language question and get an answer drawn from meeting **audio,
slides and minutes at once**, with a navigable citation for every claim — and
an explicit refusal when the corpus does not cover the question.

This Space is a **CPU preview**. The full system runs air-gapped on a GPU
workstation with ImageBind (one unified text/image/audio embedding space) and a
local Qwen2.5-3B via Ollama. Here, ImageBind is replaced by a small text
encoder (MiniLM) and the LLM by Qwen2.5-1.5B on llama.cpp. The retrieval
pipeline — per-modality search, reciprocal rank fusion, a BM25 keyword pass,
`doc_id` grouping for cross-modal linking, and the grounding gate — is the same.

**Corpus:** two meetings (ES2002a, ES2002b) from the
[AMI Meeting Corpus](http://groups.inf.ed.ac.uk/ami/corpus/), CC BY 4.0.
J. Carletta et al., "The AMI Meeting Corpus", 2005.

First answer after a cold start takes ~1 minute while the models download and load.

## Run locally

```bash
pip install -r requirements.txt
python app.py
```
