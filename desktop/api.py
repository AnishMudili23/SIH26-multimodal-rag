"""Thin HTTP client for the local RAG backend (backend/server.py).

All calls are blocking; the desktop runs them on its worker thread, where the
GIL is released during network I/O — so the Qt UI stays responsive.
"""

from __future__ import annotations

import json

import httpx

from rag import config

BASE = config.BACKEND_URL
_LONG = httpx.Timeout(600.0, connect=5.0)   # model load / generation can be slow


def health(timeout: float = 3.0) -> dict:
    r = httpx.get(f"{BASE}/health", timeout=timeout)
    r.raise_for_status()
    return r.json()


def query(payload: dict) -> dict:
    r = httpx.post(f"{BASE}/query", json=payload, timeout=_LONG)
    r.raise_for_status()
    return r.json()


def query_stream(payload: dict):
    """Yields ('meta'|'token'|'done'|'error', data) as the server streams."""
    with httpx.stream("POST", f"{BASE}/query/stream", json=payload, timeout=_LONG) as r:
        r.raise_for_status()
        yield from _sse(r)


def library() -> list[dict]:
    r = httpx.get(f"{BASE}/library", timeout=15.0)
    r.raise_for_status()
    return r.json().get("files", [])


def delete_source(source_file: str) -> int:
    r = httpx.post(f"{BASE}/library/delete", json={"source_file": source_file}, timeout=30.0)
    r.raise_for_status()
    return r.json().get("removed", 0)


def ingest_stream(paths: list[str]):
    """Yields ('progress'|'done'|'error', data)."""
    with httpx.stream("POST", f"{BASE}/ingest", json={"paths": paths}, timeout=_LONG) as r:
        r.raise_for_status()
        yield from _sse(r)


def _sse(response):
    event = "message"
    for line in response.iter_lines():
        if not line:
            continue
        if line.startswith("event:"):
            event = line.split(":", 1)[1].strip()
        elif line.startswith("data:"):
            try:
                yield (event, json.loads(line.split(":", 1)[1].strip()))
            except json.JSONDecodeError:
                pass
            event = "message"
