"""Text retrieval evaluation — Recall@k / MRR@k against the MS MARCO qrels
(`ROADMAP.md` Phase 4).

Reads `<manifest>_qrels.csv` (query, doc_id, is_relevant) produced by
`rag/ingest/manifest.py`, runs each query through `rag.retrieval.pipeline`, and
scores whether the known-relevant passage's `doc_id` is retrieved.

Text-only by design — `PRD.md` non-goals rule out a formal image/audio metric
(no ground truth); those are spot-checked manually against the manifest.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root on path

from rag import config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("eval")


def load_qrels(path: Path) -> dict[str, set[str]]:
    """query -> set of relevant doc_ids (is_relevant truthy)."""
    rel: dict[str, set[str]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if str(row.get("is_relevant", "")).strip() not in ("1", "true", "True"):
                continue
            rel.setdefault(row["query"].strip(), set()).add(row["doc_id"].strip())
    return rel


def evaluate(qrels_path: Path, k_recall: int = 5, k_mrr: int = 10) -> dict:
    from rag.retrieval import pipeline

    qrels = load_qrels(qrels_path)
    if not qrels:
        raise SystemExit(f"No relevant qrels found in {qrels_path}")

    top_k = max(k_recall, k_mrr)
    hits_at_r = 0
    rr_sum = 0.0
    n = 0
    misses: list[str] = []

    for query, rel_docs in qrels.items():
        n += 1
        res = pipeline.run_query_pipeline(
            text=query, top_k=top_k, modality_filter="document", unload_after=False
        )
        ranked_docs: list[str] = []
        for c in res.chunks:
            if c.doc_id not in ranked_docs:
                ranked_docs.append(c.doc_id)

        if any(d in rel_docs for d in ranked_docs[:k_recall]):
            hits_at_r += 1
        rr = 0.0
        for rank, d in enumerate(ranked_docs[:k_mrr], start=1):
            if d in rel_docs:
                rr = 1.0 / rank
                break
        rr_sum += rr
        if rr == 0.0:
            misses.append(query)

    try:
        from rag.core import embeddings
        embeddings.unload()
    except Exception:  # noqa: BLE001
        pass

    result = {
        "n_queries": n,
        f"recall@{k_recall}": round(hits_at_r / n, 4),
        f"mrr@{k_mrr}": round(rr_sum / n, 4),
        "n_misses": len(misses),
        "misses": misses,
    }
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--qrels",
        type=Path,
        default=config.CORPUS_MANIFEST_CSV.with_name(
            config.CORPUS_MANIFEST_CSV.stem + "_qrels.csv"
        ),
    )
    ap.add_argument("--k-recall", type=int, default=5)
    ap.add_argument("--k-mrr", type=int, default=10)
    args = ap.parse_args(argv)

    res = evaluate(args.qrels, args.k_recall, args.k_mrr)
    print(f"\nqueries:      {res['n_queries']}")
    print(f"Recall@{args.k_recall}:    {res[f'recall@{args.k_recall}']}")
    print(f"MRR@{args.k_mrr}:      {res[f'mrr@{args.k_mrr}']}")
    print(f"misses ({res['n_misses']}):")
    for q in res["misses"]:
        print(f"  - {q}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
