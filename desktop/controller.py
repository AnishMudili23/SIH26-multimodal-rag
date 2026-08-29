"""Wires the RAG_OS window to the local backend over HTTP.

The pipeline lives in a separate process (backend/server.py), so no amount of
torch/ImageBind loading can freeze the Qt UI. This worker only makes blocking
HTTP calls on its own thread — the GIL is released during network I/O.
"""

from __future__ import annotations

import time

from PySide6.QtCore import QObject, QThread, QTimer, Signal, Slot

from desktop import api


class RagWorker(QObject):
    resultReady = Signal(dict)          # final {question, answer, sources, confidence, insights, warning}
    answerToken = Signal(str)           # streamed token
    answerStarted = Signal(str)         # question (clears the answer box)
    failed = Signal(str)
    ingestProgress = Signal(str)
    ingestDone = Signal(dict)
    warmupState = Signal(str)           # "loading" | "ready" | "error: ..."
    backendState = Signal(str)          # "up" | "down"

    @Slot()
    def wait_for_backend(self):
        """Poll /health until the server is up and its models are warm."""
        last = None
        for _ in range(600):            # up to ~5 min of first-run model download
            try:
                h = api.health()
                if last != "up":
                    self.backendState.emit("up"); last = "up"
                w = h.get("warmup")
                self.warmupState.emit(w if w != "cold" else "loading")
                if w == "ready":
                    return
                if w == "error":
                    self.warmupState.emit("error: " + h.get("warmup_error", "?"))
                    return
            except Exception:  # noqa: BLE001
                if last != "down":
                    self.backendState.emit("down"); last = "down"
            time.sleep(0.6)

    @Slot(dict)
    def run(self, payload: dict):
        try:
            self.answerStarted.emit(payload.get("text") or "(query)")
            final = None
            for ev, data in api.query_stream(payload):
                if ev == "meta":
                    if data.get("question"):
                        pass
                elif ev == "token":
                    self.answerToken.emit(data.get("t", ""))
                elif ev == "done":
                    final = data
                elif ev == "error":
                    self.failed.emit(data.get("message", "query failed"))
                    return
            if final is None:
                self.failed.emit("no response from backend")
                return
            final["question"] = payload.get("text") or "(query)"
            self.resultReady.emit(final)
        except Exception as ex:  # noqa: BLE001
            self.failed.emit(str(ex))

    @Slot(list)
    def ingest(self, paths: list):
        try:
            summary = None
            for ev, data in api.ingest_stream(paths):
                if ev == "progress":
                    self.ingestProgress.emit(
                        f"ingesting {data.get('i', 0)}/{data.get('n', 0)} — {data.get('name', '')}")
                elif ev == "done":
                    summary = data
                elif ev == "error":
                    summary = {"written": 0, "ok": [], "failed": [("ingest", data.get("message", "?"))]}
            self.ingestDone.emit(summary or {"written": 0, "ok": [], "failed": []})
        except Exception as ex:  # noqa: BLE001
            self.ingestDone.emit({"written": 0, "ok": [], "failed": [("ingest", str(ex))]})


class Controller(QObject):
    """Owns the worker thread and relays its signals to the window.

    Must be a QObject: signals connected to a plain-object method have no
    thread affinity, so Qt calls them *directly on the worker thread* — and
    `_on_result` touches Qt widgets (the answer box, the audit log), which
    crashes / freezes the UI if it isn't on the GUI thread. As a QObject
    created on the main thread, `_on_result` is delivered back to the main
    thread via a queued connection.
    """

    def __init__(self, window):
        super().__init__()
        self.window = window
        self.thread = QThread()
        self.thread.setObjectName("rag-io")
        self.worker = RagWorker()
        self.worker.moveToThread(self.thread)

        window.querySubmitted.connect(self.worker.run)
        window.ingestRequested.connect(self.worker.ingest)
        self.worker.answerStarted.connect(window.on_answer_started)
        self.worker.answerToken.connect(window.on_answer_token)
        self.worker.resultReady.connect(self._on_result)
        self.worker.failed.connect(window.show_error)
        self.worker.ingestProgress.connect(window.on_ingest_progress)
        self.worker.ingestDone.connect(window.on_ingest_done)
        self.worker.warmupState.connect(window.on_warmup_state)
        self.worker.backendState.connect(window.on_backend_state)

        self.thread.started.connect(self.worker.wait_for_backend)
        self.thread.start()

    @Slot(dict)
    def _on_result(self, r: dict):
        self.window.add_result(
            r["question"], r["answer"], r.get("sources"),
            r.get("confidence", "—"), r.get("insights", ""), r.get("warning", ""),
        )
        # let the answer paint before the (blocking) /library refresh
        QTimer.singleShot(0, self.window.reload_corpus)

    def shutdown(self):
        self.thread.quit()
        if not self.thread.wait(3000):
            self.thread.terminate()
            self.thread.wait(1000)
