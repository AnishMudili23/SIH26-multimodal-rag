# PRD — Multimodal RAG System (SIH #25231, NTRO)

## Source problem statement

**ID:** 25231 | **Category:** Software | **Theme:** Smart Automation | **Difficulty:** Hard
**Organisation:** National Technical Research Organisation (NTRO)

> Design and build a multimodal Retrieval-Augmented Generation (RAG) system
> leveraging a Large Language Model (LLM) for OFFLINE mode that can ingest,
> index, and query diverse data formats such as DOC, PDF, Images and voice
> recordings within a unified semantic retrieval framework.

**Background (verbatim intent):** the target organisation routinely handles
diverse data — PDF, DOC, images, screenshots, recorded calls, free-form notes.
Traditional search tools isolate text/image/audio search. RAG grounds LLM
output in retrieved data; multimodal RAG extends this across images and audio
for richer, context-aware intelligence.

**Required capabilities (from the PS description):**
- Ingest multimodal inputs: text from DOCX/PDF, image embeddings, speech-to-text for audio.
- Index all modalities in a shared vector space for semantic retrieval.
- Support natural-language queries returning text, images, and audio segments.
- Generate grounded summaries/answers integrating retrieved context via an LLM.
- Establish cross-format links (e.g. audio transcript segment ↔ cited paragraph ↔ screenshot).

**Expected solution (from the PS, verbatim examples we are building toward):**
- Unified query interface: plain-language chat/search box.
  Example query given in the PS itself: *"Show me the report that has a
  description about international development in 2024 OR show the report
  that references the screenshot taken at 14:32."*
- Optional: non-text input — upload DOCX/PDF, drag-and-drop images, attach
  audio, or speak the query.
- Text-to-image search: e.g. "email screenshot" retrieves matching images
  alongside text passages.
- Image-to-text search: upload/select an image, surface semantically related
  documents or audio transcript snippets.
- Optional: audio-to-others — play a clip, retrieve related text/docs/images.
- Citation transparency: every answer has numbered citations linking back to
  source files; citations expand to open the original document, view full
  transcript segments, or inspect image metadata.

**Stakeholders:** Government agencies, industry/enterprises.
**Impact area:** Efficiency improvement, productivity, transparency.
**Data types:** Image, text, audio.
**Solution type:** Web solution.

## Goals (this build)

1. Fully offline pipeline — ingestion, retrieval, and generation all run
   locally, no external API calls at inference time.
2. Ingest and index DOC, PDF, images, and audio recordings into one unified
   retrieval index.
3. Support the PS's own example queries verbatim: theme + date style
   ("international development in 2024") and timestamp-anchored screenshot
   references ("screenshot taken at 14:32").
4. Cross-modal search in both directions: text→image, image→text, and
   spoken query→any modality.
5. Grounded, cited answers — every generated claim traceable to a numbered
   source with enough metadata (file, page, timestamp) to navigate back to it.
6. A single chat-style interface supporting text, image, and audio input.

## Non-goals (explicitly out of scope for this build)

- Cloud/online mode — the PS variant we are answering is OFFLINE mode only.
- Formal quantitative evaluation of image or audio retrieval quality — no
  ground-truth relevance judgments exist for those modalities in our corpus;
  validated by manual spot-check against a known manifest instead.
- Multi-user auth, access control, or production-scale deployment.
- Full production-grade ingestion of arbitrary file volumes — the target is
  a working, demonstrable pipeline at hackathon/demo scale (tens of docs).

## Success criteria

- A judge can type (or speak) a natural-language question and get a grounded
  answer with numbered citations, offline, on demo hardware.
- The PS's literal example query pattern works: a themed query returns the
  right report, and a timestamp-style query returns the right screenshot.
- Text-to-image and image-to-text search both demonstrably work on the demo
  corpus (e.g. querying "email screenshot" surfaces the right image).
- Every citation is clickable/expandable to its source file, page number, or
  audio timestamp range.
- The system runs entirely on local hardware (GPU laptop/desktop) with no
  network calls during the demo.
- Retrieval quality is backed by a real number (Recall@5 / MRR@10 against a
  hand-built dev-query set over the AMI Meeting Corpus), not just anecdote.

## Judge/evaluation angle

Two things are likely to matter most to judges given the PS's own emphasis:

1. **Citation transparency** is called out explicitly and repeatedly in the
   PS — this is a differentiator to demo confidently, not an afterthought.
   Expandable citations that open the real source (page/timestamp) should be
   treated as a first-class feature, not a nice-to-have.
2. **Cross-format linking** ("connecting an audio transcript segment to a
   cited paragraph and screenshot") is the hardest-looking claim in the PS
   and the one most likely to be probed live. This is why the demo corpus is
   the **AMI Meeting Corpus** (see `TECH_STACK.md`): for a given meeting its
   audio, time-aligned transcript, on-screen slides, and the meeting's own
   minutes/report are genuinely about the same content, captured together —
   not paired after the fact. A random or single-modality corpus has nothing
   to cross-link and the demo would visibly fail on exactly the thing judges
   will try first.

Everything else (raw retrieval metrics, model size, architecture elegance)
is secondary to those two demo moments working reliably.
