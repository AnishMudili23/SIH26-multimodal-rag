"""RAG_OS desktop UI.

Design: dark technical analyst workspace.  The left rail owns the primary
input/actions (new chat, drag/drop, upload, image, audio, mic and sources),
while the right side is the larger answer workspace.  A persistent bottom
instrument bar exposes AI ANALYTICS, CORPUS, TIMELINE and AUDIT LOG without
letting those tools disappear when the main content scrolls.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from PySide6.QtCore import Qt, Signal, QUrl, QTimer
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QFrame, QScrollArea, QTextEdit, QSplitter, QTabWidget,
    QListWidget, QListWidgetItem, QFileDialog, QCheckBox, QSizePolicy,
)

_MM = None


def _mm():
    global _MM
    if _MM is None:
        try:
            from PySide6 import QtMultimedia as m
            _MM = m
        except Exception:  # noqa: BLE001
            _MM = False
    return _MM


ROOT = Path(__file__).resolve().parent
IMG_EXTS = "Images (*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff)"
AUD_EXTS = "Audio (*.wav *.mp3 *.m4a *.flac *.ogg)"
SUPPORTED = {".pdf", ".docx", ".txt", ".md", ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".wav", ".mp3", ".m4a", ".flac", ".ogg"}


class MainWindow(QMainWindow):
    querySubmitted = Signal(dict)
    newChatRequested = Signal()
    sourceSelected = Signal(dict)
    ingestRequested = Signal(list)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("RAG_OS — Offline Multimodal RAG")
        self.resize(1480, 940)
        self.setMinimumSize(1100, 720)
        self.setAcceptDrops(True)
        self.history: list[dict] = []
        self._image_path: str | None = None
        self._audio_path: str | None = None
        self._recorder = None
        self._rec_path: str | None = None
        self._player = None
        self._audio_out = None
        self._build_ui()
        self._load_style()
        QTimer.singleShot(80, self.reload_corpus)

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(12, 10, 12, 10)
        outer.setSpacing(8)

        # TOP HEADER ----------------------------------------------------
        top = QHBoxLayout()
        brand = QLabel("RAG_OS")
        brand.setObjectName("brand")
        node = QLabel("NODE_01  /  OFFLINE ANALYST WORKSPACE")
        node.setObjectName("muted")
        self.status = QLabel("● LOCAL MODE")
        self.status.setObjectName("status")
        self.session = QLabel("SESSION / UNTITLED")
        self.session.setObjectName("session")
        self.new_chat = QPushButton("＋ NEW CHAT")
        self.new_chat.setObjectName("primaryButton")
        self.new_chat.clicked.connect(self._new_chat)
        top.addWidget(brand)
        top.addSpacing(12)
        top.addWidget(node)
        top.addStretch()
        top.addWidget(self.status)
        top.addSpacing(12)
        top.addWidget(self.session)
        top.addSpacing(8)
        top.addWidget(self.new_chat)
        outer.addLayout(top)

        # MAIN WORKSPACE ------------------------------------------------
        main = QSplitter(Qt.Horizontal)
        main.setChildrenCollapsible(False)
        main.setHandleWidth(4)

        # LEFT CONTROL RAIL
        left = QFrame()
        left.setObjectName("leftRail")
        ll = QVBoxLayout(left)
        ll.setContentsMargins(10, 10, 10, 10)
        ll.setSpacing(8)

        ll.addWidget(self._label("INPUT / SOURCES"))

        self.drop_zone = QFrame()
        self.drop_zone.setObjectName("dropZone")
        dz = QVBoxLayout(self.drop_zone)
        dz.setContentsMargins(12, 14, 12, 14)
        drop_title = QLabel("DROP FILES HERE")
        drop_title.setObjectName("dropTitle")
        drop_sub = QLabel("PDF · DOCX · IMAGE · AUDIO")
        drop_sub.setObjectName("muted")
        drop_sub.setAlignment(Qt.AlignCenter)
        dz.addWidget(drop_title)
        dz.addWidget(drop_sub)
        ll.addWidget(self.drop_zone)

        self.btn_upload = QPushButton("＋ UPLOAD / INDEX FILES")
        self.btn_upload.setObjectName("uploadButton")
        self.btn_upload.clicked.connect(self._pick_ingest_files)
        ll.addWidget(self.btn_upload)

        media_row = QHBoxLayout()
        self.btn_image = QPushButton("▣ IMAGE")
        self.btn_image.setObjectName("toolButton")
        self.btn_image.clicked.connect(self._pick_image)
        self.btn_audio = QPushButton("◈ AUDIO")
        self.btn_audio.setObjectName("toolButton")
        self.btn_audio.clicked.connect(self._pick_audio)
        media_row.addWidget(self.btn_image)
        media_row.addWidget(self.btn_audio)
        ll.addLayout(media_row)

        self.btn_mic = QPushButton("●  MIC / VOICE QUERY")
        self.btn_mic.setObjectName("toolButton")
        self.btn_mic.setCheckable(True)
        self.btn_mic.toggled.connect(self._toggle_mic)
        ll.addWidget(self.btn_mic)

        self.spoken_cb = QCheckBox("treat attached audio as a spoken question")
        ll.addWidget(self.spoken_cb)

        self.chip_image = QPushButton("")
        self.chip_image.setObjectName("chip")
        self.chip_image.clicked.connect(lambda: self._clear_attach("image"))
        self.chip_image.hide()
        self.chip_audio = QPushButton("")
        self.chip_audio.setObjectName("chip")
        self.chip_audio.clicked.connect(lambda: self._clear_attach("audio"))
        self.chip_audio.hide()
        ll.addWidget(self.chip_image)
        ll.addWidget(self.chip_audio)

        ll.addWidget(self._label("RETRIEVED SOURCES"))
        self.sources_header = self._label("0 RESULTS")
        self.sources = QListWidget()
        self.sources.setObjectName("sourcesList")
        self.sources.currentItemChanged.connect(self._on_source_selected)
        ll.addWidget(self.sources_header)
        ll.addWidget(self.sources, 1)

        self.ingest_status = QLabel("")
        self.ingest_status.setObjectName("warn")
        self.ingest_status.setWordWrap(True)
        self.ingest_status.hide()
        ll.addWidget(self.ingest_status)
        main.addWidget(left)

        # RIGHT ANSWER WORKSPACE
        right = QFrame()
        right.setObjectName("answerWorkspace")
        rl = QVBoxLayout(right)
        rl.setContentsMargins(12, 10, 12, 10)
        rl.setSpacing(8)

        # query stays with the output so the analyst can ask/refine without
        # leaving the main workspace.
        qlabel = self._label("QUERY / MULTIMODAL SEARCH")
        rl.addWidget(qlabel)
        qrow = QHBoxLayout()
        self.query = QLineEdit()
        self.query.setPlaceholderText("Ask anything about your indexed evidence…")
        self.query.setMinimumHeight(52)
        self.query.returnPressed.connect(self._submit)
        qrow.addWidget(self.query, 1)
        self.btn_send = QPushButton("RUN  ➤")
        self.btn_send.setObjectName("sendButton")
        self.btn_send.clicked.connect(self._submit)
        self.btn_send.setEnabled(False)
        qrow.addWidget(self.btn_send)
        rl.addLayout(qrow)

        attach_note = QLabel("TEXT + IMAGE + AUDIO  ·  local retrieval  ·  local generation")
        attach_note.setObjectName("muted")
        rl.addWidget(attach_note)

        # Large answer panel
        answer_header = QHBoxLayout()
        answer_header.addWidget(self._label("AI SYNTHESIS / GROUNDED RESPONSE"))
        answer_header.addStretch()
        answer_header.addWidget(self._label("CONFIDENCE"))
        self.confidence = QLabel("—")
        self.confidence.setObjectName("confidence")
        answer_header.addWidget(self.confidence)
        rl.addLayout(answer_header)

        self.answer = QTextEdit()
        self.answer.setObjectName("answerBox")
        self.answer.setReadOnly(True)
        self.answer.setPlaceholderText("Your grounded response will appear here. Retrieved evidence is shown in the left rail.")
        rl.addWidget(self.answer, 1)

        self.warn = QLabel("")
        self.warn.setObjectName("warn")
        self.warn.setWordWrap(True)
        self.warn.hide()
        rl.addWidget(self.warn)

        # source detail is compact; the main answer remains dominant.
        detail = QFrame()
        detail.setObjectName("detail")
        dl = QVBoxLayout(detail)
        dl.setContentsMargins(8, 6, 8, 6)
        dl.addWidget(self._label("SOURCE DETAIL"))
        self.detail_scroll = QScrollArea()
        self.detail_scroll.setWidgetResizable(True)
        self.detail_scroll.setFixedHeight(150)
        self.detail_scroll.setFrameShape(QFrame.NoFrame)
        self.detail_body = QWidget()
        self.detail_layout = QVBoxLayout(self.detail_body)
        self.detail_layout.setContentsMargins(0, 0, 0, 0)
        self.detail_layout.setSpacing(5)
        self._detail_placeholder()
        self.detail_scroll.setWidget(self.detail_body)
        dl.addWidget(self.detail_scroll)
        rl.addWidget(detail)
        main.addWidget(right)
        main.setSizes([340, 1100])
        outer.addWidget(main, 1)

        # SESSION HISTORY — compact horizontal cards above the persistent bar
        hh = QHBoxLayout()
        hh.addWidget(self._label("RECENT SESSIONS"))
        clear = QPushButton("CLEAR")
        clear.setObjectName("smallButton")
        clear.clicked.connect(self._clear_history)
        hh.addWidget(clear)
        hh.addStretch()
        outer.addLayout(hh)
        self.history_scroll = QScrollArea()
        self.history_scroll.setWidgetResizable(True)
        self.history_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.history_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.history_container = QWidget()
        self.history_layout = QHBoxLayout(self.history_container)
        self.history_layout.setContentsMargins(0, 0, 0, 0)
        self.history_layout.setSpacing(8)
        self.history_scroll.setWidget(self.history_container)
        self.history_scroll.setFixedHeight(74)
        outer.addWidget(self.history_scroll)

        # BOTTOM INSTRUMENT BAR — always visible; this is the "task bar".
        self.tabs = QTabWidget()
        self.tabs.tabBar().hide()
        self.tabs.setObjectName("toolDrawer")
        self.tab_insights = QTextEdit()
        self.tab_insights.setReadOnly(True)
        self.tab_insights.setPlainText("Retrieved chunks · formats · latency · model")
        self.tab_audit = QTextEdit()
        self.tab_audit.setReadOnly(True)
        self.tab_audit.setPlainText("Query and retrieval events appear here.")

        corpus_tab = QWidget()
        corpus_l = QVBoxLayout(corpus_tab)
        crow = QHBoxLayout()
        self.corpus_header = self._label("INDEX / LOADING…")
        crow.addWidget(self.corpus_header)
        crow.addStretch()
        self.btn_add_files = QPushButton("＋ ADD FILES")
        self.btn_add_files.setObjectName("outlineButton")
        self.btn_add_files.clicked.connect(self._pick_ingest_files)
        btn_refresh = QPushButton("↻")
        btn_refresh.setObjectName("smallButton")
        btn_refresh.clicked.connect(self.reload_corpus)
        self.btn_del_source = QPushButton("DELETE")
        self.btn_del_source.setObjectName("smallButton")
        self.btn_del_source.clicked.connect(self._delete_selected_source)
        crow.addWidget(self.btn_add_files)
        crow.addWidget(btn_refresh)
        crow.addWidget(self.btn_del_source)
        corpus_l.addLayout(crow)
        self.corpus_list = QListWidget()
        corpus_l.addWidget(self.corpus_list)

        timeline = self._placeholder("Temporal evidence will appear here when the retrieval pipeline returns dated evidence.")
        self.tabs.addTab(self.tab_insights, "AI ANALYTICS")
        self.tabs.addTab(corpus_tab, "CORPUS")
        self.tabs.addTab(timeline, "TIMELINE")
        self.tabs.addTab(self.tab_audit, "AUDIT LOG")

        # hidden drawer; task-bar buttons toggle it.
        self.tool_drawer = QFrame()
        self.tool_drawer.setObjectName("toolDrawerFrame")
        td = QVBoxLayout(self.tool_drawer)
        td.setContentsMargins(0, 0, 0, 0)
        td.addWidget(self.tabs)
        self.tool_drawer.setFixedHeight(0)
        self.tool_drawer.hide()
        outer.addWidget(self.tool_drawer)

        bar = QFrame()
        bar.setObjectName("taskBar")
        bl = QHBoxLayout(bar)
        bl.setContentsMargins(4, 4, 4, 4)
        bl.setSpacing(6)
        self.tool_buttons = []
        for idx, title in enumerate(("AI ANALYTICS", "CORPUS", "TIMELINE", "AUDIT LOG")):
            b = QPushButton(f"{idx + 1:02d}  {title}")
            b.setObjectName("taskButton")
            b.setCheckable(True)
            b.clicked.connect(lambda checked=False, i=idx: self._show_tool(i))
            bl.addWidget(b, 1)
            self.tool_buttons.append(b)
        outer.addWidget(bar)

    # ------------------------------------------------------------ helpers
    def _label(self, text):
        w = QLabel(text)
        w.setObjectName("sectionLabel")
        return w

    def _placeholder(self, text):
        w = QTextEdit()
        w.setReadOnly(True)
        w.setPlainText(text)
        return w

    def _clear_layout(self, layout):
        while layout.count():
            it = layout.takeAt(0)
            if it.widget():
                it.widget().deleteLater()

    # ------------------------------------------------------------ task bar
    def _show_tool(self, index):
        for i, b in enumerate(self.tool_buttons):
            b.setChecked(i == index)
        self.tabs.setCurrentIndex(index)
        if self.tool_drawer.isVisible() and self.tool_drawer.height() > 0:
            self.tool_drawer.setFixedHeight(0)
            self.tool_drawer.hide()
            for b in self.tool_buttons:
                b.setChecked(False)
        else:
            self.tool_drawer.show()
            self.tool_drawer.setFixedHeight(190)

    # ------------------------------------------------------- drag & drop
    def dragEnterEvent(self, event):
        paths = [u.toLocalFile() for u in event.mimeData().urls() if u.isLocalFile()]
        if any(Path(p).suffix.lower() in SUPPORTED for p in paths):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        paths = [u.toLocalFile() for u in event.mimeData().urls() if u.isLocalFile()]
        paths = [p for p in paths if Path(p).suffix.lower() in SUPPORTED]
        if paths:
            self._start_ingest(paths)
            event.acceptProposedAction()

    # ------------------------------------------------------- attachments
    def _pick_image(self):
        p, _ = QFileDialog.getOpenFileName(self, "Attach image", "", IMG_EXTS)
        if p:
            self._image_path = p
            self.chip_image.setText(f"▣ {Path(p).name}   ×")
            self.chip_image.show()

    def _pick_audio(self):
        p, _ = QFileDialog.getOpenFileName(self, "Attach audio clip", "", AUD_EXTS)
        if p:
            self._audio_path = p
            self.chip_audio.setText(f"◈ {Path(p).name}   ×")
            self.chip_audio.show()

    def _clear_attach(self, kind):
        if kind == "image":
            self._image_path = None
            self.chip_image.hide()
        else:
            self._audio_path = None
            self.chip_audio.hide()

    def _toggle_mic(self, on):
        mm = _mm()
        if not mm:
            self.warn.setText("Mic capture unavailable in this Qt build — use AUDIO instead.")
            self.warn.show()
            self.btn_mic.setChecked(False)
            return
        if on:
            try:
                self._rec_path = str(Path(tempfile.mkdtemp(prefix="ragos_mic_")) / "q.wav")
                self._capture = mm.QMediaCaptureSession()
                self._audio_in = mm.QAudioInput()
                self._capture.setAudioInput(self._audio_in)
                self._recorder = mm.QMediaRecorder()
                fmt = mm.QMediaFormat(mm.QMediaFormat.FileFormat.Wave)
                self._recorder.setMediaFormat(fmt)
                self._capture.setRecorder(self._recorder)
                self._recorder.setOutputLocation(QUrl.fromLocalFile(self._rec_path))
                self._recorder.record()
                self.btn_mic.setText("■  STOP & ASK")
                self.status.setText("● RECORDING")
                self.status.setObjectName("statusBusy")
                self._restyle(self.status)
            except Exception as ex:  # noqa: BLE001
                self.warn.setText(f"Mic error: {ex} — use AUDIO instead.")
                self.warn.show()
                self.btn_mic.setChecked(False)
        else:
            try:
                if self._recorder:
                    self._recorder.stop()
                self.btn_mic.setText("●  MIC / VOICE QUERY")
                self.status.setText("● LOCAL MODE")
                self.status.setObjectName("status")
                self._restyle(self.status)
                QTimer.singleShot(400, self._submit_spoken)
            except Exception:  # noqa: BLE001
                pass

    def _submit_spoken(self):
        if self._rec_path and Path(self._rec_path).exists():
            self.querySubmitted.emit({"text": self.query.text().strip() or None,
                                      "image_path": None, "audio_path": self._rec_path,
                                      "spoken": True})
            self._set_busy(True)

    def _restyle(self, w):
        w.style().unpolish(w)
        w.style().polish(w)

    # --------------------------------------------------------- submit
    def _submit(self):
        text = self.query.text().strip()
        if not (text or self._image_path or self._audio_path):
            return
        self.querySubmitted.emit({"text": text or None,
                                  "image_path": self._image_path,
                                  "audio_path": self._audio_path,
                                  "spoken": self.spoken_cb.isChecked()})
        self._set_busy(True)

    def _set_busy(self, busy):
        self.btn_send.setEnabled(not busy)
        self.status.setText("● WORKING…" if busy else "● LOCAL MODE")
        self.status.setObjectName("statusBusy" if busy else "status")
        self._restyle(self.status)

    def _new_chat(self):
        self.query.clear()
        self.answer.clear()
        self.confidence.setText("—")
        self.sources.clear()
        self.warn.hide()
        self._image_path = self._audio_path = None
        self.chip_image.hide()
        self.chip_audio.hide()
        self._detail_placeholder()
        self.session.setText("SESSION / UNTITLED")
        self.newChatRequested.emit()

    # --------------------------------------------------- results in
    def add_result(self, question, answer, sources=None, confidence="—", insights="", warning=""):
        self._set_busy(False)
        self.answer.setPlainText(answer)
        self.confidence.setText(str(confidence))
        self.confidence.setObjectName("confidenceLow" if _low_conf(confidence) else "confidence")
        self._restyle(self.confidence)
        if warning:
            self.warn.setText("⚠ " + warning)
            self.warn.show()
        else:
            self.warn.hide()

        sources = sources or []
        self.sources.clear()
        for s in sources:
            loc = f"  ·  {s['locator']}" if s.get("locator") else ""
            mark = "✓" if s.get("cited") else "·"
            item = QListWidgetItem(f"{mark}  {s['source_file']}\n   {s['modality']}{loc}")
            item.setData(Qt.UserRole, s)
            self.sources.addItem(item)
        self.sources_header.setText(f"{len(sources)} RESULTS")
        self.tab_insights.setPlainText(insights or "")
        self._append_audit(question, len(sources), confidence)

        self.history.append({"question": question, "answer": answer, "confidence": confidence,
                             "sources": sources, "insights": insights, "warning": warning})
        self._refresh_history()
        self.session.setText(f"SESSION / Q{len(self.history):02d}")
        if sources:
            self.sources.setCurrentRow(0)

    def show_error(self, message):
        self._set_busy(False)
        self.warn.setText("⚠ " + message)
        self.warn.show()

    def on_answer_started(self, question):
        self.answer.setPlainText("")
        self.warn.hide()
        self.confidence.setText("…")

    def on_answer_token(self, tok):
        cur = self.answer.textCursor()
        cur.movePosition(cur.MoveOperation.End)
        cur.insertText(tok)
        self.answer.setTextCursor(cur)

    def on_backend_state(self, state):
        if state == "down":
            self.status.setText("● BACKEND STARTING…")
            self.status.setObjectName("statusBusy")
        self._restyle(self.status)

    def on_warmup_state(self, state: str):
        if state in ("loading", "cold"):
            self.status.setText("● WARMING UP…")
            self.status.setObjectName("statusBusy")
        elif state == "ready":
            self.status.setText("● LOCAL MODE · READY")
            self.status.setObjectName("status")
            self.btn_send.setEnabled(True)
            self.reload_corpus()
        else:
            self.status.setText("● LOCAL MODE")
            self.status.setObjectName("status")
            self.warn.setText("⚠ warm-up: " + state)
            self.warn.show()
        self._restyle(self.status)

    # ------------------------------------------------ source detail
    def _detail_placeholder(self):
        self._clear_layout(self.detail_layout)
        lbl = QLabel("Select a source from the left rail to inspect its image, audio, transcript or page text.")
        lbl.setWordWrap(True)
        lbl.setObjectName("muted")
        self.detail_layout.addWidget(lbl)
        self.detail_layout.addStretch()

    def _on_source_selected(self, cur, _prev):
        if cur is None:
            return
        s = cur.data(Qt.UserRole)
        self.sourceSelected.emit(s)
        self._render_detail(s)

    def _render_detail(self, s: dict):
        self._clear_layout(self.detail_layout)
        head = QLabel(f"[{','.join(str(n) for n in s.get('numbers', []))}] {s['source_file']} — {s['modality']}" +
                      ("   ✓ cited" if s.get("cited") else "   (not cited)"))
        head.setObjectName("sectionLabel")
        self.detail_layout.addWidget(head)
        path = s.get("source_path")
        if s["modality"] == "image" and path and Path(path).exists():
            pic = QLabel()
            pic.setObjectName("mediaImage")
            pm = QPixmap(path)
            if not pm.isNull():
                pic.setPixmap(pm.scaledToHeight(120, Qt.SmoothTransformation))
            self.detail_layout.addWidget(pic)
        elif s["modality"] == "audio" and path and Path(path).exists():
            row = QHBoxLayout()
            play = QPushButton("▶ PLAY")
            play.setObjectName("smallButton")
            play.clicked.connect(lambda: self._play_audio(path))
            stop = QPushButton("■")
            stop.setObjectName("smallButton")
            stop.clicked.connect(self._stop_audio)
            row.addWidget(play)
            row.addWidget(stop)
            row.addStretch()
            self.detail_layout.addLayout(row)
            if s.get("start_time") is not None:
                self.detail_layout.addWidget(QLabel(f"cited segment: {s['start_time']:.0f}s – {s['end_time']:.0f}s"))
        if s.get("text"):
            body = QLabel(s["text"])
            body.setWordWrap(True)
            body.setTextInteractionFlags(Qt.TextSelectableByMouse)
            self.detail_layout.addWidget(body)
        if s.get("linked_text"):
            rel = QLabel("related passage:  " + s["linked_text"][:280] + "…")
            rel.setWordWrap(True)
            rel.setObjectName("muted")
            self.detail_layout.addWidget(rel)
        self.detail_layout.addStretch()

    def _play_audio(self, path):
        mm = _mm()
        if not mm:
            return
        if self._player is None:
            self._player = mm.QMediaPlayer()
            self._audio_out = mm.QAudioOutput()
            self._player.setAudioOutput(self._audio_out)
        self._player.setSource(QUrl.fromLocalFile(path))
        self._player.play()

    def _stop_audio(self):
        if self._player:
            self._player.stop()

    # ------------------------------------------------------- history
    def _refresh_history(self):
        self._clear_layout(self.history_layout)
        for i, h in reversed(list(enumerate(self.history))):
            b = QPushButton(f"Q{i + 1:02d}  ·  {h['question'][:34]}\n{h['confidence']}  •  click to restore")
            b.setObjectName("historyCard")
            b.setFixedSize(260, 56)
            b.clicked.connect(lambda _=False, idx=i: self._restore(idx))
            self.history_layout.addWidget(b)
        self.history_layout.addStretch()

    def _restore(self, idx):
        h = self.history[idx]
        self.query.setText(h["question"])
        self.answer.setPlainText(h["answer"])
        self.confidence.setText(str(h["confidence"]))
        self.sources.clear()
        for s in h.get("sources", []):
            it = QListWidgetItem(f"{'✓' if s.get('cited') else '·'}  {s['source_file']}\n   {s['modality']}")
            it.setData(Qt.UserRole, s)
            self.sources.addItem(it)
        self.tab_insights.setPlainText(h.get("insights", ""))
        if h.get("warning"):
            self.warn.setText("⚠ " + h["warning"])
            self.warn.show()
        else:
            self.warn.hide()
        self.session.setText(f"SESSION / Q{idx + 1:02d}")

    def _clear_history(self):
        self.history.clear()
        self._refresh_history()
        self.session.setText("SESSION / UNTITLED")

    # -------------------------------------------------------- corpus
    def reload_corpus(self):
        try:
            from desktop import api
            rows = api.library()
        except Exception as ex:  # noqa: BLE001
            self.corpus_header.setText("INDEX / BACKEND NOT READY")
            self.corpus_list.clear()
            self.corpus_list.addItem(QListWidgetItem(f"({ex})"))
            return
        self.corpus_list.clear()
        icon = {"document": "▢", "image": "▣", "audio": "◈"}
        counts: dict[str, int] = {}
        for r in rows:
            counts[r["modality"]] = counts.get(r["modality"], 0) + 1
            it = QListWidgetItem(f"{icon.get(r['modality'], '·')} {r['source_file']}\n   {r['modality']} · {r['chunks']} chunk(s)" +
                                 (f" · {r['theme']}" if r["theme"] else ""))
            it.setData(Qt.UserRole, r)
            self.corpus_list.addItem(it)
        summary = "  ".join(f"{k}:{v}" for k, v in sorted(counts.items()))
        self.corpus_header.setText(f"INDEX / {len(rows)} FILES  —  {summary}")

    def _delete_selected_source(self):
        it = self.corpus_list.currentItem()
        if it is None:
            return
        r = it.data(Qt.UserRole)
        try:
            from desktop import api
            n = api.delete_source(r["source_file"])
            self._append_audit(f"delete {r['source_file']!r}", 0, f"-{n} chunks")
        except Exception as ex:  # noqa: BLE001
            self.warn.setText(f"⚠ delete failed: {ex}")
            self.warn.show()
        self.reload_corpus()

    def _pick_ingest_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Add files to the corpus", "",
            "All supported (*.pdf *.docx *.txt *.md *.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff *.wav *.mp3 *.m4a *.flac *.ogg);;All files (*)")
        if paths:
            self._start_ingest(paths)

    def _start_ingest(self, paths):
        self.btn_upload.setEnabled(False)
        self.btn_add_files.setEnabled(False)
        self.ingest_status.setText(f"INDEXING {len(paths)} FILE(S)…")
        self.ingest_status.show()
        self.ingestRequested.emit(list(paths))

    def on_ingest_progress(self, msg: str):
        self.ingest_status.setText(msg)

    def on_ingest_done(self, summary: dict):
        self.btn_upload.setEnabled(True)
        self.btn_add_files.setEnabled(True)
        n_ok = len(summary.get("ok", []))
        n_fail = len(summary.get("failed", []))
        parts = [f"added {n_ok} file(s), {summary.get('written', 0)} chunks"]
        if n_fail:
            parts.append("failed: " + ", ".join(f"{n} ({why})" for n, why in summary["failed"]))
        self.ingest_status.setText("  ·  ".join(parts))
        self._append_audit("ingest", n_ok, f"+{summary.get('written', 0)} chunks")
        self.reload_corpus()

    def _append_audit(self, q, n, conf):
        from datetime import datetime
        line = f"[{datetime.now():%H:%M:%S}]  query={q!r}  sources={n}  confidence={conf}"
        self.tab_audit.setPlainText((self.tab_audit.toPlainText() + "\n" + line).strip())

    # --------------------------------------------------------- style
    def _load_style(self):
        qss = ROOT / "rag_os.qss"
        if qss.exists():
            self.setStyleSheet(qss.read_text(encoding="utf-8"))


def _low_conf(pct) -> bool:
    try:
        return float(str(pct).rstrip("%")) < 45
    except (ValueError, TypeError):
        return False
