"""ChromaDB wrapper — the single unified collection (ARCHITECTURE.md §1/§2).

One persistent collection holds every chunk of every modality. Embeddings are
ImageBind outputs, so they are directly comparable regardless of whether the
chunk came from a document, an image, or an audio window. rag/retrieval/pipeline.py
reads from the same collection this module writes.

Chroma metadata values must be str/int/float/bool (no None, no lists), so the
optional schema fields (page_number, start_time, bbox, prev/next pointers) are
encoded here: absent keys are simply omitted, bbox is JSON-encoded.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

from rag import config

logger = logging.getLogger(__name__)


@dataclass
class ChunkRecord:
    """One row of the unified collection (ARCHITECTURE.md §2 schema)."""

    id: str
    embedding: Sequence[float]
    modality: str                       # "document" | "image" | "audio"
    doc_id: str
    source_file: str                    # basename, for display
    text_preview: str
    source_path: str | None = None      # absolute path, for the UI to render/play
    page_number: int | None = None
    start_time: float | None = None
    end_time: float | None = None
    bbox: list[float] | None = None
    prev_chunk_id: str | None = None
    next_chunk_id: str | None = None
    theme: str | None = None
    image_timestamp: str | None = None   # "HH:MM" capture time (images), from manifest
    linked_passage: str | None = None    # doc_id of the text passage this asset echoes

    def metadata(self) -> dict[str, Any]:
        md: dict[str, Any] = {
            "modality": self.modality,
            "doc_id": self.doc_id,
            "source_file": self.source_file,
            "text_preview": self.text_preview,
        }
        if self.source_path:
            md["source_path"] = self.source_path
        if self.page_number is not None:
            md["page_number"] = int(self.page_number)
        if self.start_time is not None:
            md["start_time"] = float(self.start_time)
        if self.end_time is not None:
            md["end_time"] = float(self.end_time)
        if self.bbox is not None:
            md["bbox"] = json.dumps(self.bbox)
        if self.prev_chunk_id:
            md["prev_chunk_id"] = self.prev_chunk_id
        if self.next_chunk_id:
            md["next_chunk_id"] = self.next_chunk_id
        if self.theme:
            md["theme"] = self.theme
        if self.image_timestamp:
            md["image_timestamp"] = self.image_timestamp
        if self.linked_passage:
            md["linked_passage"] = self.linked_passage
        return md


def _client():
    try:
        import chromadb
    except ImportError as e:
        raise RuntimeError("chromadb not installed (`pip install chromadb`)") from e
    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(config.CHROMA_DIR))


def get_collection(reset: bool = False):
    client = _client()
    if reset:
        try:
            client.delete_collection(config.COLLECTION_NAME)
            logger.info("Deleted existing collection %s", config.COLLECTION_NAME)
        except Exception:  # noqa: BLE001 - didn't exist
            pass
    return client.get_or_create_collection(
        name=config.COLLECTION_NAME,
        metadata={"hnsw:space": config.CHROMA_DISTANCE},
    )


def add_records(collection, records: Sequence[ChunkRecord], batch_size: int = 256) -> int:
    """Upsert ChunkRecords into the collection. Returns count written."""
    if not records:
        return 0
    written = 0
    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        collection.upsert(
            ids=[r.id for r in batch],
            embeddings=[list(map(float, r.embedding)) for r in batch],
            metadatas=[r.metadata() for r in batch],
            documents=[r.text_preview for r in batch],
        )
        written += len(batch)
    return written


def query(
    collection,
    embedding: Sequence[float],
    n_results: int = config.PER_MODALITY_SEARCH_K,
    modality: str | None = None,
    where_extra: dict | None = None,
):
    """Nearest-neighbour search against the unified collection.

    Returns a list of dicts: id, distance, and the flattened metadata
    (including text_preview). `modality` scopes the search; `where_extra`
    adds further metadata equality filters (e.g. {"image_timestamp": "14:32"}).
    """
    clauses = []
    if modality:
        clauses.append({"modality": modality})
    for k, v in (where_extra or {}).items():
        clauses.append({k: v})
    where = clauses[0] if len(clauses) == 1 else ({"$and": clauses} if clauses else None)
    res = collection.query(
        query_embeddings=[list(map(float, embedding))],
        n_results=n_results,
        where=where,
        include=["metadatas", "distances", "documents"],
    )
    out: list[dict[str, Any]] = []
    ids = res.get("ids", [[]])[0]
    dists = res.get("distances", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    for cid, dist, meta in zip(ids, dists, metas):
        row = dict(meta or {})
        if "bbox" in row and isinstance(row["bbox"], str):
            try:
                row["bbox"] = json.loads(row["bbox"])
            except json.JSONDecodeError:
                pass
        row["id"] = cid
        row["distance"] = dist
        out.append(row)
    return out


def get_by_id(collection, chunk_id: str) -> dict[str, Any] | None:
    res = collection.get(ids=[chunk_id], include=["metadatas", "documents"])
    ids = res.get("ids", [])
    if not ids:
        return None
    row = dict(res["metadatas"][0] or {})
    row["id"] = ids[0]
    row["text_preview"] = (res.get("documents") or [row.get("text_preview", "")])[0]
    return row


def get_passage_text(collection, doc_id: str) -> str:
    """First document chunk's text for a doc_id — used to show what an
    image/audio asset's `linked_passage` refers to."""
    try:
        res = collection.get(where={"doc_id": doc_id}, include=["documents", "metadatas"])
        pairs = list(zip(res.get("metadatas", []), res.get("documents", [])))
        docs = [d for m, d in pairs if (m or {}).get("modality") == "document"]
        return (docs or [d for _, d in pairs] or [""])[0] or ""
    except Exception:  # noqa: BLE001
        return ""


def collection_stats(collection) -> dict[str, Any]:
    total = collection.count()
    by_modality: dict[str, int] = {}
    for m in ("document", "image", "audio"):
        try:
            by_modality[m] = len(collection.get(where={"modality": m})["ids"])
        except Exception:  # noqa: BLE001
            by_modality[m] = -1
    return {"total": total, "by_modality": by_modality}


def list_sources(collection) -> list[dict[str, Any]]:
    """Every distinct ingested file: name, modality, chunk count, doc_id, path.
    Powers a 'what's in the index' view."""
    from collections import Counter

    res = collection.get(include=["metadatas"])
    files: dict[str, dict[str, Any]] = {}
    for m in res.get("metadatas", []):
        m = m or {}
        name = m.get("source_file", "?")
        f = files.setdefault(name, {
            "source_file": name, "chunks": 0, "_mods": Counter(),
            "doc_id": m.get("doc_id", ""), "source_path": "", "theme": "",
        })
        f["chunks"] += 1
        f["_mods"][m.get("modality", "?")] += 1
        f["source_path"] = f["source_path"] or m.get("source_path", "")
        f["theme"] = f["theme"] or m.get("theme", "")
    for f in files.values():
        f["modality"] = f.pop("_mods").most_common(1)[0][0]
    return sorted(files.values(), key=lambda x: (x["modality"], x["source_file"].lower()))


def delete_source(collection, source_file: str) -> int:
    """Remove every chunk of one file from the index. Returns count removed."""
    got = collection.get(where={"source_file": source_file})
    ids = got.get("ids", [])
    if ids:
        collection.delete(ids=ids)
    return len(ids)
