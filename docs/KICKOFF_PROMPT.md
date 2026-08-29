Paste this into Claude Code to start the session.

---

Before doing anything else, read these five files in the repo root, in
order: CLAUDE.md, PRD.md, ARCHITECTURE.md, TECH_STACK.md, ROADMAP.md.

They contain the full locked design for this project — problem statement,
architecture decisions and the reasoning behind them, exact tech choices,
and a phased task list. Do not propose alternative architectures or
re-litigate decisions already made in these docs (e.g. ImageBind as the
embedding space, chunk size, Reciprocal Rank Fusion for merging) unless you
hit a concrete technical blocker that makes the documented approach
actually unworkable — if that happens, stop and explain the blocker rather
than silently deviating.

No code exists in this repo yet. TECH_STACK.md includes the intended
structure for query_pipeline.py, generation.py, app.py, and
build_corpus_manifest.py — build them fresh following that structure and
ARCHITECTURE.md's design exactly, rather than improvising a different
approach.

Start on Phase 0 from ROADMAP.md — the ingestion pipeline. This is the
current blocker: nothing else can be tested against real data until raw
DOC/PDF/image/audio files can actually be extracted, chunked, embedded, and
written into a Chroma collection. Work through Phase 0's checklist in
order.

## Session logging — maintain PROGRESS.md

Keep a file called PROGRESS.md at the repo root. If it doesn't exist yet,
create it. At the end of this session — and every future session — append
a new dated entry (do not overwrite previous entries) in this format:

```
## Session — <date>

### Built
- what was implemented or changed, specifically enough that it's clear
  without re-reading the diff

### Broke / blocked on
- anything that didn't work, any error not yet resolved, any assumption
  from the docs that turned out to be wrong

### Next
- the specific next step — which ROADMAP.md checklist item to pick up,
  and any context a fresh session would need to start there immediately
  without asking the user to re-explain anything
```

Before ending the session, confirm PROGRESS.md has been updated with this
session's entry. If you run out of context mid-task, still write a
PROGRESS.md entry for whatever was completed before stopping — a partial
entry is much better than none.

If ROADMAP.md checkboxes need updating based on what was completed this
session, update them directly in ROADMAP.md as well, not just in
PROGRESS.md.
